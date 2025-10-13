import torch
import abc
import numpy as np
from typing import List, Union

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class BaseRewardModel(abc.ABC):
    def __init__(
        self,
        device: str = "cuda",
        batch_size=64,
        success_bonus=10.0,
        reward_divisor=1.0,
    ):
        """
        Initialize the encoder. Subclasses can implement specific initialization as needed.
        """
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.success_bonus = success_bonus

        # Note: this is handled in the wrapper or offline replay buffer
        self.reward_divisor = reward_divisor

        # Load minilm-12v2
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L12-v2"
        )
        self.model = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L12-v2"
        )

    def encode_text(self, text: Union[str, List]) -> np.ndarray:
        """
        Encodes a text input into a representation.
        :param text: Text data to be encoded. If a list of strings is provided, it will be batch encoded.
        :return: Encoded representation of the text.
        """
        if isinstance(text, list):
            for i in range(0, len(text), self.batch_size):
                batch_text = text[i : i + self.batch_size]
                encoded_text = self._encode_text_batch(batch_text)
                if i == 0:
                    encoded_text_all = encoded_text
                else:
                    encoded_text_all = np.concatenate((encoded_text_all, encoded_text))
        else:
            encoded_text_all = self._encode_text_batch([text])

        # ensure the output is a numpy array
        if isinstance(encoded_text_all, torch.Tensor):
            encoded_text_all = encoded_text_all.detach().cpu().numpy()

        return encoded_text_all

    def encode_text_for_policy(self, text: Union[str, List]) -> np.ndarray:
        """
        Encodes a text input into a representation for policy training.
        :param text: Text data to be encoded. If a list of strings is provided, it will be batch encoded.
        :return: Encoded representation of the text.
        """

        encoded_input = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            text_embeddings = mean_pooling(
                model_output, encoded_input["attention_mask"]
            )

        # normalize the embeddings
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

        return text_embeddings.detach().cpu().numpy()

    def encode_images(self, images: np.ndarray) -> np.ndarray:
        """
        Encodes a video input (sequence of frames) into an image representation.
        :param images: A sequence of video frames to be encoded. The shape of the input should be (num_vids, num_frames, *).
        :return: Encoded representation of each frame.
        """
        assert len(images.shape) == 5, "The input should be a sequence of video frames."
        # ensure the channels are first
        if images.shape[-1] == 3 and not images.shape[2] == 3:
            images = np.transpose(images, (0, 1, 4, 2, 3))
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i : i + self.batch_size]
            batch_images = torch.tensor(batch_images, dtype=torch.float32).to(
                self.device
            )
            encoded_images = self._encode_image_batch(batch_images).cpu().numpy()
            if i == 0:
                encoded_images_all = encoded_images
            else:
                encoded_images_all = np.concatenate(
                    (encoded_images_all, encoded_images)
                )
        return encoded_images_all

    @abc.abstractmethod
    def _encode_image_batch(self, images: torch.Tensor) -> np.ndarray:
        """
        Encodes a batch of video frames into an image representation.
        :param images: A batch of video frames to be encoded. The shape of the input should be (batch_size, num_frames, height, width, channels).
        :return: Encoded representation of each frame.
        """
        pass

    def calculate_rewards(
        self,
        encoded_texts: Union[np.ndarray, torch.Tensor],
        encoded_videos: Union[np.ndarray, torch.Tensor],
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
                batch_texts.to(self.device), batch_videos.to(self.device)
            )
            if i == 0:
                rewards_all = rewards
            else:
                rewards_all = np.concatenate((rewards_all, rewards))
        return rewards_all

    def set_success_bonus(self, success_bonus: float):
        self.success_bonus = success_bonus

    def set_reward_divisor(self, reward_divisor: float):
        self.reward_divisor = reward_divisor

    @abc.abstractmethod
    def _calculate_reward_batch(
        self, encoded_texts: np.ndarray, encoded_videos: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the rewards for a batch of text and video representations.
        :param encoded_texts: Encoded text representations.
        :param encoded_videos: Encoded video representations.
        :return: Reward values for each text-video pair.
        """
        pass

    @property
    def img_output_dim(self) -> int:
        """
        Returns the output dimension of the image encoder. Used to determine the observation space of a policy.
        """
        pass

    @property
    def text_output_dim(self) -> int:
        """
        Returns the output dimension of the text encoder. Used to determine the observation space of a policy.
        """
        pass

    @property
    def policy_text_output_dim(self) -> int:
        """
        Returns the output dimension of the text encoder. Used to determine the observation space of a policy.
        """
        return 384  # for MiniLM

    @property
    def name(self) -> str:
        """
        Returns the name of the encoder class.
        """
        pass
