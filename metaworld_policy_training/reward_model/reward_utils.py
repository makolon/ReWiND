import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T




device = "cuda" if torch.cuda.is_available() else "cpu"


dino_transform_image = T.Compose(
    [T.ToTensor(), T.CenterCrop(224), T.Normalize([0.5], [0.5])]
)



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


def dino_load_image(img: np.ndarray) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    """
    img = Image.fromarray(img)

    transformed_img = dino_transform_image(img)[:3].unsqueeze(0)
    
    return transformed_img





def compute_similarity(text_embeddings, image_embeddings):
    cosine_similarity = F.cosine_similarity(text_embeddings, image_embeddings)
    return cosine_similarity




