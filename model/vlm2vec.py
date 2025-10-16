import torch
from torch import nn
from typing import List, Dict, Optional
from transformers import CLIPProcessor, CLIPModel

from PIL import Image
from data.preprocessing import preprocess_image, clean_text


class VLM2Vec(nn.Module):
    """
    Wrapper around a pretrained Vision-Language Model (LLaVA or CLIP)
    to return:
    - contextual token/image embeddings (sequence-level)
    - final hidden state (document-level embedding)
    
    Supports:
    - VLM2Vec-LLaVA models (4096 dimensions): "TIGER-Lab/VLM2Vec-LLaVa-Next" (default)
    - LLaVA models (4096 dimensions): "llava-hf/llava-1.5-7b-hf", "llava-hf/llava-v1.6-mistral-7b-hf"
    - CLIP models (768 dimensions): "openai/clip-vit-large-patch14"
    """

    def __init__(self, model_name: str = "TIGER-Lab/VLM2Vec-LLaVa-Next", freeze: bool = False):
        super().__init__()
        self.model_name = model_name
        self.is_llava = "llava" in model_name.lower() or "vlm2vec" in model_name.lower()
        
        if self.is_llava:
            # For LLaVA/VLM2Vec models, we need the language model's hidden states
            try:
                from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
                self.processor = LlavaNextProcessor.from_pretrained(model_name)
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto"  # Automatically handle device placement
                )
                print(f"✓ Loaded VLM2Vec/LLaVA model: {model_name}")
                print(f"✓ Hidden dimension: {self.model.config.text_config.hidden_size}")
            except ImportError:
                raise ImportError(
                    "LLaVA/VLM2Vec models require transformers with LLaVA support. "
                    "Install with: pip install transformers>=4.37.0 accelerate"
                )
        else:
            # Fallback to CLIP
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            print(f"✓ Loaded CLIP model: {model_name}")
            print(f"✓ Hidden dimension: {self.model.config.text_config.hidden_size}")

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    @property
    def hidden_dim(self) -> int:
        """Return the hidden dimension of the model"""
        if self.is_llava:
            return self.model.config.text_config.hidden_size  # 4096 for LLaVA-7B
        else:
            return self.model.config.text_config.hidden_size  # 768 for CLIP-large

    def forward(self, text: str, images: List[Image.Image]):
        """
        Args:
            text (str): Cleaned input text
            images (List[PIL.Image]): List of associated images

        Returns:
            - contextual_embeddings: (seq_len, hidden_dim)
            - doc_embedding: (hidden_dim,)
        """
        if self.is_llava:
            return self._forward_llava(text, images)
        else:
            return self._forward_clip(text, images)
    
    def _forward_clip(self, text: str, images: List[Image.Image]):
        """Forward pass for CLIP models"""
        # === Tokenize + Process ===
        inputs = self.processor(text=[text], images=images, return_tensors="pt", padding=True, truncation=True)
        
        # Move to same device as model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_hidden_states=True)

        # === Extract contextual embeddings ===
        text_embeddings = outputs.text_model_output.last_hidden_state[0]   # (seq_len, hidden_dim)
        image_embeddings = outputs.vision_model_output.last_hidden_state.mean(dim=1)[0]  # (hidden_dim,)
        contextual_embeddings = torch.cat([text_embeddings, image_embeddings.unsqueeze(0)], dim=0)

        # === Final document embedding (last token) ===
        doc_embedding = text_embeddings[-1]  # last token embedding

        return contextual_embeddings, doc_embedding
    
    def _forward_llava(self, text: str, images: List[Image.Image]):
        """
        Forward pass for LLaVA/VLM2Vec models.
        Uses the final token embeddings from the language model backbone.
        """
        # Prepare conversation format for LLaVA
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Process inputs
        inputs = self.processor(
            images=images[0] if images else None, 
            text=prompt, 
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Get hidden states without generating text
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        
        # Extract language model hidden states (these are 4096-dim for LLaVA-7B)
        # The hidden states from the language model backbone contain the final token embeddings
        hidden_states = outputs.hidden_states[-1][0]  # (seq_len, 4096)
        
        # === Final document embedding (last token) ===
        # This is the key: we use the final token embedding as mentioned in VLM2Vec
        doc_embedding = hidden_states[-1]  # (4096,)
        
        return hidden_states, doc_embedding
