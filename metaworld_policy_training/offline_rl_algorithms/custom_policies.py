# the main difference in this file is to add the layer normalization to the actor and critic networks without needing to
# overwrite SB3 classes directly/create a new fork
import torch as th
import numpy as np
from gym import spaces
from stable_baselines3.common.distributions import (
    SquashedDiagGaussianDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.preprocessing import get_action_dim
from torch import nn
from typing import Optional, Union, Type, Dict, Any, List, Tuple

from stable_baselines3.common.policies import (
    BasePolicy,
    BaseModel,
    ContinuousCritic,
)
from stable_baselines3.sac.policies import (
    SACPolicy,
    get_actor_critic_arch,
    Actor,
    LOG_STD_MAX,
    LOG_STD_MIN,
)
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    NatureCNN,
    CombinedExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
import math

import copy

# Type definitions
TensorDict = dict[str, th.Tensor]
PyTorchObs = Union[th.Tensor, TensorDict]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(
            th.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
    with_bias: bool = True,
    use_layer_norm: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :param with_bias: If set to False, the layers will not learn an additive bias
    :param use_layer_norm: Whether to use Layer Normalization or not
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0], bias=with_bias), activation_fn()]
        if use_layer_norm:
            modules.append(nn.LayerNorm(net_arch[0]))
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))
        if use_layer_norm:
            modules.append(nn.LayerNorm(net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class CustomActor(Actor):
    """
    Actor network (policy) for SAC.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param use_layer_norm: Whether to use layer normalization or not
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        use_layer_norm: bool = False,
    ):
        BasePolicy.__init__(
            self,
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        action_dim = get_action_dim(self.action_space)
        latent_pi_net = create_mlp(
            features_dim, -1, net_arch, activation_fn, use_layer_norm=use_layer_norm
        )
        self.use_layer_norm = use_layer_norm
        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim,
                full_std=full_std,
                use_expln=use_expln,
                learn_features=True,
                squash_output=True,
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim,
                latent_sde_dim=last_layer_dim,
                log_std_init=log_std_init,
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(
                    self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean)
                )
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                use_layer_norm=self.use_layer_norm,
            )
        )
        return data

    def __deepcopy__(self, memo):
        obj = type(self).__new__(self.__class__)
        output_dict = {}
        for key, value in self.__dict__.items():
            # if key in ["action_space", "observation_space"]:
            #     output_dict[key] = value
            # else:
            #     output_dict[key] = copy.deepco/copy(value)
            try:
                output_dict[key] = copy.deepcopy(value)
            except:
                print("Actor failed to deep copy key:", key)
                output_dict[key] = copy.copy(value)
        obj.__dict__ = output_dict
        return obj


class ActionSequenceActor(CustomActor):
    # uses an RNN to output an action sequence

    action_space: spaces.Box

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        action_sequence_length: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        use_layer_norm: bool = False,
    ):
        BasePolicy.__init__(
            self,
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        assert not use_sde, "ActionSequenceActor does not support gSDE"
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        action_dim = get_action_dim(self.action_space)
        latent_pi_net = create_mlp(
            features_dim, -1, net_arch, activation_fn, use_layer_norm=use_layer_norm
        )
        self.use_layer_norm = use_layer_norm

        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        self.action_dist = SquashedDiagGaussianDistribution(action_dim)  # type: ignore[assignment]
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=last_layer_dim, nhead=8, batch_first=True
        )
        self.action_transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=1, norm=nn.LayerNorm(last_layer_dim)
        )
        self.triangular_mask = th.triu(
            th.ones(action_sequence_length, action_sequence_length) * float("-inf"),
            diagonal=1,
        )
        self.position_embedding = PositionalEncoding(
            last_layer_dim, max_len=action_sequence_length
        )
        # self.mu = GRU(last_layer_dim, last_layer_dim, num_layers=1, batch_first=True)
        # self.mu_processor = nn.Linear(last_layer_dim, action_dim)

        self.mu_processor = nn.Sequential(
            nn.Linear(last_layer_dim, last_layer_dim // 2),
            activation_fn(),
            nn.Linear(last_layer_dim // 2, action_dim),
        )
        # self.log_std = nn.GRU(
        #    last_layer_dim, last_layer_dim, num_layers=1, batch_first=True
        # )
        self.log_std_processor = nn.Sequential(
            nn.Linear(last_layer_dim, last_layer_dim // 2),
            activation_fn(),
            nn.Linear(last_layer_dim // 2, action_dim),
        )
        self.action_sequence_length = action_sequence_length

        # Print action transformer parameter count
        print(
            f"ActionTransformer has {sum(p.numel() for p in self.action_transformer.parameters())} parameters"
        )

    def get_action_dist_params(
        self, obs: PyTorchObs
    ) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
            Mean, log_std will be of shape (batch_size * sequence_length, action_dim)
        """
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        # run the rnn  for self.action_sequence_length for mu and log_std
        # mean_actions_intermediate, _ = self.mu(
        #    latent_pi.unsqueeze(1).repeat(1, self.action_sequence_length, 1)
        # )
        mean_actions_intermediate = self.position_embedding(
            latent_pi.unsqueeze(1).repeat(1, self.action_sequence_length, 1)
        )

        mean_actions_intermediate = self.action_transformer(
            mean_actions_intermediate,
            memory=mean_actions_intermediate,
        )
        # tgt_mask=self.triangular_mask,

        # mean_actions_intermediate = self.action_transformer(
        #     mean_actions_intermediate, memory=None, tgt_mask=self.triangular_mask
        # )
        # mean_actions_intermediate = mean_actions_intermediate.reshape(
        #     -1, mean_actions_intermediate.shape[-1]
        # )
        mean_actions = self.mu_processor(mean_actions_intermediate)

        # Add batch back
        # mean_actions = mean_actions.reshape(
        #    -1, self.action_sequence_length, mean_actions.shape[-1]
        # )

        # log_std_intermediate, _ = self.log_std(
        #     latent_pi.unsqueeze(1).repeat(1, self.action_sequence_length, 1)
        # )
        # log_std_intermediate = log_std_intermediate.reshape(
        #     -1, log_std_intermediate.shape[-1]
        # )
        # log_std = self.log_std_processor(self.activation_fn()(log_std_intermediate))

        log_std_intermediate = mean_actions_intermediate  # Adjust as needed
        log_std = self.log_std_processor(log_std_intermediate)

        # Add batch back
        # log_std = log_std.reshape(-1, self.action_sequence_length, log_std.shape[-1])

        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        # mean_actions = mean_actions.reshape(
        #    -1, self.action_sequence_length, mean_actions.shape[-1]
        # )
        # log_std = log_std.reshape(-1, self.action_sequence_length, log_std.shape[-1])
        return mean_actions, log_std, {}

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        # reshape everything to be (batch_size * action_sequence_length, action_dim)
        batch_size = mean_actions.shape[0] // self.action_sequence_length
        action_dim = mean_actions.shape[-1]
        # try:
        #     mean_actions = mean_actions.reshape(
        #         batch_size * self.action_sequence_length, action_dim
        #     )
        # except:
        #     breakpoint()
        # log_std = log_std.reshape(batch_size * self.action_sequence_length, action_dim)

        actions = self.action_dist.actions_from_params(
            mean_actions, log_std, deterministic=deterministic, **kwargs
        )

        # actions = actions.reshape(batch_size, self.action_sequence_length, action_dim)
        return actions

    def action_log_prob(self, obs: PyTorchObs) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        batch_size = mean_actions.shape[0] // self.action_sequence_length
        # action_dim = mean_actions.shape[-1]
        # try:
        #     mean_actions = mean_actions.reshape(
        #         batch_size * self.action_sequence_length, action_dim
        #     )
        # except:
        #     breakpoint()
        # return action and associated log prob
        actions, log_prob = self.action_dist.log_prob_from_params(
            mean_actions, log_std, **kwargs
        )

        return actions, log_prob
        # return actions.reshape(
        #     batch_size, self.action_sequence_length, action_dim
        # ), log_prob.reshape(batch_size, self.action_sequence_length)

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> th.Tensor:
        return self(observation, deterministic)


class RecurrentQNetwork(nn.Module):
    def __init__(self, action_dim, features_dim, activation_fn, q_network):
        super().__init__()
        self.action_feature_extractor = nn.Linear(action_dim, 128)
        # self.activation_fn = activation_fn()
        # self.recurrent_action_processor = nn.GRU(
        #    features_dim, features_dim, batch_first=True
        # )

        # self.downprojector = nn.Linear(features_dim, 128)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, batch_first=True
        )
        self.transformer_action_processor = nn.TransformerEncoder(
            encoder_layer, num_layers=1, norm=nn.LayerNorm(128)
        )
        self.position_embedding = PositionalEncoding(128, max_len=100)
        self.q_network = q_network

    def forward(self, q_input: Tuple[th.Tensor, th.Tensor]) -> th.Tensor:
        obs, actions = q_input
        # reshape actions to be (batch_size * action_sequence_length, action_dim)
        batch_size = actions.shape[0]
        action_dim = actions.shape[-1]
        chunk_size = actions.shape[1]

        orig_actions_dim = actions.dim()
        if orig_actions_dim == 3:
            actions = actions.reshape(batch_size * actions.shape[1], action_dim)

        action_features = self.action_feature_extractor(actions)
        # action_features = self.activation_fn(action_features)
        # action_features = self.downprojector(action_features)
        # action_features = action_features.reshape(batch_size, actions.shape[1], -1)
        # action_features, _ = self.recurrent_action_processor(action_features)
        if orig_actions_dim == 3:
            action_features = action_features.reshape(batch_size, chunk_size, -1)
        else:  # action chunk size is 1
            action_features = action_features.reshape(batch_size, 1, -1)

        action_features = self.position_embedding(action_features)
        action_features = self.transformer_action_processor(action_features)
        action_features = action_features.mean(dim=1)
        q_input = th.cat([obs, action_features], dim=1)
        return self.q_network(q_input)


class CustomContinuousCritic(ContinuousCritic):
    """
    Critic network(s) for DDPG/SAC/TD3.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    :param use_layer_norm: Whether to use layer normalization or not
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        use_layer_norm: bool = True,
        parallelize: bool = False,
        recurrent_action: bool = False,
    ):
        BaseModel.__init__(
            self,
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        self.recurrent_action = recurrent_action
        for idx in range(n_critics):
            q_net = create_mlp(
                features_dim + action_dim
                if not recurrent_action
                else features_dim + 128,
                1,
                net_arch,
                activation_fn,
                use_layer_norm=use_layer_norm,
            )
            q_net = nn.Sequential(*q_net)
            if recurrent_action:
                q_net = RecurrentQNetwork(
                    action_dim,
                    features_dim,
                    activation_fn,
                    q_net,
                )
            if not parallelize:
                self.add_module(f"qf{idx}", q_net)

            self.q_networks.append(q_net)

        self.parallelize = parallelize

        if parallelize:
            self.base_model = copy.deepcopy(self.q_networks[0])
            params, buffers = th.func.stack_module_state(self.q_networks)

            self.params = nn.ParameterList([nn.Parameter(p) for p in params.values()])

            for k, v in buffers.items():
                k = k.replace(".", "_")
                self.register_buffer(k, v)

    def forward(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        critic_indices: th.Tensor = None,
    ) -> Tuple[th.Tensor, ...]:
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        if self.recurrent_action:
            qvalue_input = (features, actions)
        else:
            qvalue_input = th.cat([features, actions], dim=1)

        if not self.parallelize:
            if critic_indices is not None:
                return tuple(
                    self.q_networks[idx](qvalue_input) for idx in critic_indices
                )
            else:
                return tuple(q_net(qvalue_input) for q_net in self.q_networks)
        else:
            if critic_indices is not None:
                params = {
                    f"param_{i}": p[critic_indices] for i, p in enumerate(self.params)
                }
                output = th.vmap(self._fmodel, in_dims=(0, None, None))(
                    params,
                    dict(self.named_buffers()),
                    qvalue_input,
                )
                return tuple(output)
            else:
                params_dict = {f"param_{i}": p for i, p in enumerate(self.params)}
                output = th.vmap(self._fmodel, in_dims=(0, None, None))(
                    params_dict,
                    dict(self.named_buffers()),
                    qvalue_input,
                )
                return tuple(output)

    def _fmodel(
        self, params: Dict[str, th.Tensor], buffers: Dict[str, th.Tensor], x: th.Tensor
    ) -> th.Tensor:
        return th.func.functional_call(self.base_model, (params, buffers), x)


class CustomSACPolicy(SACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        policy_layer_norm: bool = False,
        critic_layer_norm: bool = False,
    ):
        BasePolicy.__init__(
            self,
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [256, 256]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()

        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)

        # add layer norm
        self.actor_kwargs.update({"use_layer_norm": policy_layer_norm})

        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
                "use_layer_norm": critic_layer_norm,
            }
        )

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                share_features_extractor=self.share_features_extractor,
                critic_layer_norm=self.critic_kwargs["use_layer_norm"],
                policy_layer_norm=self.actor_kwargs["use_layer_norm"],
            )
        )
        return data

    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> CustomActor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        return CustomActor(**actor_kwargs).to(self.device)

    def make_critic(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )
        return CustomContinuousCritic(**critic_kwargs).to(self.device)


class CustomRNNSACPolicy(CustomSACPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        policy_layer_norm: bool = False,
        critic_layer_norm: bool = False,
        action_sequence_length: int = 3,
    ):
        self.action_sequence_length = action_sequence_length

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
            policy_layer_norm,
            critic_layer_norm,
        )

    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> ActionSequenceActor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        actor_kwargs.update({"action_sequence_length": self.action_sequence_length})
        return ActionSequenceActor(**actor_kwargs).to(self.device)

    def make_critic(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> CustomContinuousCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )
        critic_kwargs.update({"recurrent_action": True})
        return CustomContinuousCritic(**critic_kwargs).to(self.device)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)
        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

        # now reshape it back to (batch_size, action_sequence_length, action_dim)
        actions = actions.reshape(
            -1, self.action_sequence_length, *self.action_space.shape
        )

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)
        return actions, state

    def __deepcopy__(self, memo):
        obj = type(self).__new__(self.__class__)
        output_dict = {}
        for key, value in self.__dict__.items():
            # if key in ["action_space", "observation_space"]:
            #     output_dict[key] = value
            # else:
            #     output_dict[key] = copy.deepco/copy(value)
            try:
                output_dict[key] = copy.deepcopy(value)
            except:
                # print("Policy failed to deep copy key:", key)
                output_dict[key] = copy.copy(value)
        obj.__dict__ = output_dict
        return obj


CustomMlpPolicy = CustomSACPolicy
CustomRNNMlpPolicy = CustomRNNSACPolicy


class CustomCnnPolicy(CustomSACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    :param policy_layer_norm: Whether to use layer normalization in the policy network
    :param critic_layer_norm: Whether to use layer normalization in the critic network
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        policy_layer_norm: bool = False,
        critic_layer_norm: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
            policy_layer_norm,
            critic_layer_norm,
        )


class CustomMultiInputPolicy(CustomSACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    :param policy_layer_norm: Whether to use layer normalization in the policy network
    :param critic_layer_norm: Whether to use layer normalization in the critic network
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        policy_layer_norm: bool = False,
        critic_layer_norm: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
            policy_layer_norm,
            critic_layer_norm,
        )


if __name__ == "__main__":
    # test that it works
    import gym

    env = gym.make("Pendulum-v1")
    policy_test = CustomSACPolicy(
        env.observation_space,
        env.action_space,
        lambda x: 1e-3,
    )
