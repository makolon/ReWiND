import torch as th
import torch.nn as nn
from gym import spaces

from typing import List, Type

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FlatRangeFeaturesExtractor(BaseFeaturesExtractor):
    """
    The input observation space is a flattened list. we want to project certain ranges of this list

    Args:
        observation_space: The observation space of the environment
        dim_ranges: List of ranges for each key. Ex. [12, 768, 768]
        projection_dims: List of dimensions for each item. Ex. [128, 256, 256]
        activation_fn: Activation function to use
        normalize_images: Whether to normalize images
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        dim_ranges: List[int],
        projection_dims: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        self.activation_fn = activation_fn
        self.normalize_images = normalize_images

        self.dim_ranges = dim_ranges
        self.projection_dims = projection_dims

        super().__init__(
            observation_space,
            sum(projection_dims),
        )

        # We simply project each key's inputs to the shape defined in dims_per_key
        self.projectors = {}
        concat_size = 0
        for dim_range, projection_dim in zip(dim_ranges, projection_dims):
            self.projectors[str(concat_size)] = nn.Linear(dim_range, projection_dim)
            concat_size += projection_dim

        self.projectors = nn.ModuleDict(self.projectors)

    def forward(self, x):
        # Project each key's inputs to the shape defined in dims_per_key
        projected = []
        concat_size = 0
        for dim_range, projection_dim in zip(self.dim_ranges, self.projection_dims):
            projected.append(
                self.projectors[str(concat_size)](
                    x[:, concat_size : concat_size + dim_range]
                )
            )
            concat_size += projection_dim
        return th.cat(projected, dim=1)
