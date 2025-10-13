import torch
import numpy as np

import abc
from typing import Union, List
from reward_model.base_reward_model import BaseRewardModel

from transformers import AutoTokenizer, AutoModel
from reward_model.reward_utils import dino_load_image, mean_pooling

class EnvRewardModel(BaseRewardModel):
    def __init__(self, reward_type: str="dense", model_path: str = "", device: str = "cuda", reward_at_every_step: bool = False, success_bonus: float = 10.) -> None:
        """
        Env reward model simply passes the reward from the simulator.
        Initializes a observation encoder with a pretrained model 
        for image and text encoding.
        :param model_path: Path to the model file.
        :param device: Device to run the model on.
        """
        super().__init__(device, success_bonus=success_bonus)

        self.reward_type = reward_type

        # TODO: Turn this into a cfg option and a param in every constructor
        self.reward_at_every_step = reward_at_every_step


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
        assert images.shape[0] == 1, "Env reward model doesn't support batch > 1"
        images = images.squeeze(0)

        with torch.inference_mode():
            episode_images_dino = [
                dino_load_image(
                    (img.to("cpu").numpy().transpose(1, 2, 0)).astype(np.uint8)
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

    def _calculate_reward_batch(self, encoded_texts, encoded_videos):
        """
        Calculates the reward for a batch of encoded texts and videos.
        :param encoded_texts: Encoded text representations.
        :param encoded_videos: Encoded video representations.
        :return: Reward for the batch.
        """
        return 0 # Always return 0 reward
    
    @property
    def name(self) -> str:
        """
        Returns the name of the encoder class.
        """
        return self.reward_type
    
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
