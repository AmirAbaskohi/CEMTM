"""
Custom collate functions for handling multimodal data with PIL images in batches.
"""

import torch
from PIL import Image
from typing import List, Dict, Any


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function that handles PIL.Image objects and creates proper batches.
    
    Args:
        batch: List of samples, each containing:
            - "id": str
            - "text": str
            - "images": List[PIL.Image.Image] or []
    
    Returns:
        Dictionary with:
            - "id": List[str]
            - "text": List[str]
            - "images": List[List[PIL.Image.Image]]
    """
    ids = []
    texts = []
    images_list = []
    
    for sample in batch:
        ids.append(sample.get("id", ""))
        texts.append(sample.get("text", ""))
        
        # Handle images - can be empty list or list of PIL.Image objects
        images = sample.get("images", [])
        if isinstance(images, list):
            # Filter out None values and ensure all are PIL Images
            images = [img for img in images if img is not None and isinstance(img, Image.Image)]
        images_list.append(images)
    
    return {
        "id": ids,
        "text": texts,
        "images": images_list,  # List of image lists, one per sample in batch
    }
