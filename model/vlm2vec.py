import torch
from torch import nn
from typing import List, Dict
from transformers import CLIPProcessor, CLIPModel

from PIL import Image
from data.preprocessing import preprocess_image, clean_text


class VLM2Vec(nn.Module):
    """
    Wrapper around a pretrained Vision-Language Model (e.g., CLIP-based like LLaVA)
    to return:
    - contextual token/image embeddings (sequence-level)
    - final hidden state (document-level embedding)
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", freeze: bool = False):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, text: str, images: List[Image.Image]):
        """
        Args:
            text (str): Cleaned input text
            images (List[PIL.Image]): List of associated images

        Returns:
            - contextual_embeddings: (seq_len, hidden_dim)
            - doc_embedding: (hidden_dim,)
        """
        # === Preprocess images ===
        processed_images = [preprocess_image(img) for img in images]
        image_tensor = torch.stack(processed_images)  # (num_images, 3, H, W)

        # === Tokenize + Process ===
        inputs = self.processor(text=[text], images=images, return_tensors="pt", padding=True, truncation=True)

        outputs = self.model(**inputs, output_hidden_states=True)

        # === Extract contextual embeddings ===
        text_embeddings = outputs.text_model_output.last_hidden_state[0]   # (seq_len, hidden_dim)
        image_embeddings = outputs.vision_model_output.last_hidden_state.mean(dim=1)[0]  # (hidden_dim,)
        contextual_embeddings = torch.cat([text_embeddings, image_embeddings.unsqueeze(0)], dim=0)

        # === Final document embedding ===
        doc_embedding = text_embeddings[-1]  # last token embedding

        return contextual_embeddings, doc_embedding
