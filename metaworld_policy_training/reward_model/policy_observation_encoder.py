import torch
import numpy as np
from typing import List, Union
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from reward_model.reward_utils import dino_load_image, mean_pooling


class PolicyObservationEncoder:
    """
    Independent policy observation encoder that uses ReWiND's encoding method to provide input features for policy.
    Image: 768-dim (DINO), Text: 384-dim (MiniLM)
    """
    
    def __init__(self, device: str = "cuda", batch_size: int = 64):
        """
        Initialize policy observation encoder
        :param device: Computing device
        :param batch_size: Batch size
        """
        self.device = torch.device(device)
        self.batch_size = batch_size
        
        # Text encoder - use MiniLM (same as ReWiND)
        self.minilm_tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L12-v2"
        )
        self.minilm_model = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L12-v2"
        ).to(device)
        
        # Image encoder - use DINO (same as ReWiND)
        self.dino_vits14 = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14"
        ).to(device)
        
        self.dino_batch_size = 64
    
    def _encode_text_batch(self, text: List[str]) -> np.ndarray:
        """
        Encode text batch (using MiniLM, fully consistent with ReWiND)
        :param text: List of texts
        :return: Encoded text features (384-dim)
        """
        # Implementation fully consistent with ReWiND
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
    
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text (for policy)
        :param text: Text or list of texts
        :return: Encoded text features
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
            
        return encoded_text_all
    
    def _encode_image_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode image batch (using DINO, fully consistent with ReWiND)
        :param images: Image tensor (batch_size, num_images, height, width, channels)
        :return: Encoded image features (768-dim)
        """
        # Implementation fully consistent with ReWiND
        assert images.shape[0] == 1, "Policy observation encoder doesn't support batch > 1"
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
    
    def encode_images(self, images: np.ndarray) -> np.ndarray:
        """
        Encode image sequence (for policy)
        :param images: Image sequence (num_vids, num_frames, *)
        :return: Encoded image features
        """
        assert len(images.shape) == 5, "The input should be a sequence of video frames."
        # Ensure channels are in first dimension
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
    
    @property
    def img_output_dim(self) -> int:
        """Policy observation image encoder output dimension (DINO: 768)"""
        return 768
    
    @property
    def text_output_dim(self) -> int:
        """Policy observation text encoder output dimension (MiniLM: 384)"""
        return 384
