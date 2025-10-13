from reward_model import BaseRewardModel
# pavel test sorry
import os
import torch
import abc
import numpy as np
import joblib
from typing import List, Union
import torch.nn.functional as F

from reward_model.models.ReWiND_transformer import ReWiNDTransformer
from reward_model.reward_utils import dino_load_image, mean_pooling
from transformers import AutoTokenizer, AutoModel




class ReWiNDRewardModel(BaseRewardModel):
    def __init__(
        self,
        model_load_path: Union[str, List[str]],
        camera_names: List[str],
        device: str = "cuda",
        batch_size=64,
        reward_at_every_step: bool = False,
        success_bonus: float = 10.0,
    ):
        """
        Initializes the ReWiNDRewardModel.
        :param model_load_path: Path to the model checkpoint.
        :param device: Device to run the model on (default: 'cuda').
        :param batch_size: Batch size to use for encoding data (default: 64).
        :param reward_at_every_step: Whether to calculate rewards at every step (default: False).
        """
        super().__init__(device, batch_size, success_bonus=success_bonus)
        self.reward_at_every_step = reward_at_every_step

        self.camera_names = camera_names
        if isinstance(model_load_path, list):
            self.multiple_cameras = True
            self.model_load_path = model_load_path
            self.rewind_model = [self._load_model(path) for path in model_load_path]
        else:
            self.multiple_cameras = False
            self.model_load_path = model_load_path
            self.rewind_model = self._load_model(model_load_path)

        # for the text embedding, we use minilm
        self.minilm_tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L12-v2"
        )
        self.minilm_model = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L12-v2"
        ).to(device)

        # for the image embedding, we use dino
        self.dino_vits14 = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14"
        ).to(device)

        self.dino_batch_size = 64

    def _load_model(self, model_load_path: str):
        """
        Loads the pretrained ReWiNDTransformer model from the provided path.
        :param model_load_path: Path to the pretrained model file.
        :return: Loaded model.
        """

        model_dict = torch.load(model_load_path, map_location=self.device, weights_only=False)
        args = model_dict["args"]
        self.args = args
        print(args)
        model = ReWiNDTransformer(
            args=args,
            video_dim=self.img_output_dim,  # Original video embedding dimension
            text_dim=self.text_output_dim,  # Original text embedding dimension
            hidden_dim=512,  # Common dimension for transformer processing
        ).to(self.device)

        model.load_state_dict(model_dict["model_state_dict"])
        model.eval()

        # load ema model
        # model.load_state_dict(model_dict["ema_model"])
        # model.eval()
        return model

    def _encode_text_batch(self, text: List[str]) -> np.ndarray:
        """
        Encodes a batch of text data into a representation.
        :param text: A list of text data to be encoded.
        :return: Encoded representation of the text.
        """
        with torch.no_grad():
            encoded_input = self.minilm_tokenizer(
                text, padding=False, truncation=True, return_tensors="pt"
            ).to(self.device)
            model_output = self.minilm_model(**encoded_input)
            text_embeddings = (
                mean_pooling(model_output, encoded_input["attention_mask"])
                .cpu()
                .numpy()
            )

        return text_embeddings

    def _encode_image_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encodes a batch of video frames into an image representation.
        :param images: A batch of video frames to be encoded. The shape of the input should be (batch_size, num_images, height, width, channels).
        :return: Encoded representation of each frame.
        """
        assert images.shape[0] == 1, "ReWiND doesn't support batch > 1"
        images = images.squeeze(0)
        with torch.inference_mode():
            episode_images_dino = [
                dino_load_image(
                    img.to("cpu").numpy().transpose(1, 2, 0).astype(np.uint8)
                )
                for img in images
            ]
            episode_images_dino = [
                torch.concatenate(episode_images_dino[i : i + self.dino_batch_size])
                for i in range(0, len(episode_images_dino), self.dino_batch_size)
            ]
            embedding_list = []
            for batch in episode_images_dino:
                episode_image_embeddings = (
                    self.dino_vits14(batch.to(self.device)).squeeze().detach().cpu()
                )
                embedding_list.append(episode_image_embeddings)
            episode_image_embeddings = torch.concat(embedding_list)

        return episode_image_embeddings.unsqueeze(0)

    def calculate_rewards(
        self,
        encoded_texts: Union[np.ndarray, torch.Tensor],
        encoded_videos: Union[np.ndarray, torch.Tensor],
        camera_name: str = None,
    ) -> np.ndarray:
        """
        Calculates the rewards for given text and video representations.
        :param encoded_texts: Encoded text representations.
        :param encoded_videos: Encoded video representations.
        :return: Reward values for each text-video pair.
        """
        assert len(encoded_texts) == len(encoded_videos), (
            "The number of text and video representations should be the same."
        )
        for i in range(0, len(encoded_videos), self.batch_size):
            batch_texts = encoded_texts[i : i + self.batch_size]
            batch_videos = encoded_videos[i : i + self.batch_size]
            if isinstance(encoded_texts, np.ndarray):
                batch_texts = torch.tensor(batch_texts, dtype=torch.float32)
            if isinstance(encoded_videos, np.ndarray):
                batch_videos = torch.tensor(batch_videos, dtype=torch.float32)
            rewards = self._calculate_reward_batch(
                batch_texts.to(self.device), batch_videos.to(self.device), camera_name
            )
            if i == 0:
                rewards_all = rewards
            else:
                rewards_all = np.concatenate((rewards_all, rewards))
        return rewards_all

    def _calculate_reward_batch(
        self,
        encoded_texts: torch.Tensor,
        encoded_videos: torch.Tensor,
        camera_name: str = None,
    ) -> torch.Tensor:
        """
        Calculates the rewards for a batch of text and video representations.
        :param encoded_texts: Encoded text representations.
        :param encoded_videos: Encoded video representations. Shape: (batch_size, num_images, embedding_dim).
        :return: Reward values for each text-video pair.
        """
        if self.multiple_cameras:
            model = self.rewind_model[self.camera_names.index(camera_name)]
        else:
            model = self.rewind_model

        with torch.no_grad():
            # remove batch dimension for video_encoding_model not supported and then only
            encoded_texts = encoded_texts.squeeze(0)
            encoded_videos = encoded_videos.squeeze(0)
            if self.args.subsample_video:
                encoded_videos = self.padding_video(
                    encoded_videos, self.args.max_length
                ).unsqueeze(0)
            reward = (
                model(encoded_videos.float(), encoded_texts.float()).cpu().numpy()
            )

        # print(reward)
        # return the last reward
        reward = reward[:, -1, 0]
        # print(reward)
        return reward

    @property
    def img_output_dim(self) -> int:
        """
        Returns the output dimension of the image encoder. Used to determine the observation space of a policy.
        """
        return 768

    @property
    def text_output_dim(self) -> int:
        """
        Returns the output dimension of the text encoder. Used to determine the observation space of a policy.
        """
        return 384

    @property
    def name(self) -> str:
        """
        Returns the name of the encoder class.
        """
        return "ReWiNDRewardModel"

    def padding_video(self, video_frames, max_length):
        video_length = len(video_frames)
        if isinstance(video_frames, np.ndarray):
            video_frames = torch.tensor(video_frames)
        if video_length < max_length:
            # padding last frame
            padding_length = max_length - video_length
            # first_frame = video_frames[0].unsqueeze(0)
            last_frame = video_frames[-1].unsqueeze(0)
            padding_frames = last_frame.repeat(padding_length, 1)
            video_frames = torch.cat([video_frames, padding_frames], dim=0)

        elif video_length > max_length:
            frame_idx = np.linspace(0, video_length - 1, max_length).astype(int)
            video_frames = video_frames[frame_idx]

        return video_frames
