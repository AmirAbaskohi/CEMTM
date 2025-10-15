import os
from typing import List, Dict, Callable, Optional
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
import json
import csv

class MultimodalDataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class LazyMultimodalDataset(IterableDataset):
    """
    Lazy-loading dataset that loads data on-the-fly instead of loading everything into memory.
    Uses an iterable approach to avoid memory overflow with large datasets.
    """
    def __init__(self, name: str, path: str, batch_size: int = 4):
        self.name = name.lower()
        self.path = path
        self.batch_size = batch_size
        
        if self.name not in DATASET_LOADERS_LAZY:
            raise ValueError(f"Unknown dataset: {self.name}")
        
        # Get dataset size without loading all data
        self.length = self._get_dataset_length()
    
    def _get_dataset_length(self) -> int:
        """Get the total number of items in the dataset without loading all data."""
        if self.name == "mscoco14":
            with open(os.path.join(self.path, "annotations", "captions_train2014.json"), "r", encoding="utf-8") as f:
                return len(json.load(f)["annotations"])
        elif self.name == "wikiweb2m":
            with open(os.path.join(self.path, "wikiweb2m.jsonl"), "r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        elif self.name == "spiqa":
            with open(os.path.join(self.path, "metadata.json"), "r", encoding="utf-8") as f:
                return len(json.load(f))
        elif self.name == "tqa":
            with open(os.path.join(self.path, "tqa.json"), "r", encoding="utf-8") as f:
                tqa_data = json.load(f)
                return sum(len(lesson["questions"]) for lesson in tqa_data["lessons"])
        elif self.name == "fhm":
            with open(os.path.join(self.path, "hateful_memes.jsonl"), "r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        elif self.name == "vist":
            with open(os.path.join(self.path, "stories.json"), "r", encoding="utf-8") as f:
                return len(json.load(f)["stories"])
        elif self.name == "t4sa":
            with open(os.path.join(self.path, "t4sa.csv"), "r", encoding="utf-8") as f:
                return sum(1 for _ in f) - 1  # Subtract header row
        return 0
    
    def __len__(self):
        return self.length
    
    def __iter__(self):
        """Iterate through the dataset lazily."""
        loader_func = DATASET_LOADERS_LAZY[self.name]
        return loader_func(self.path)


# === Dataset Loaders (Load all at once - legacy) ===

def load_wikiweb2m(path: str) -> List[Dict]:
    data = []
    with open(os.path.join(path, "wikiweb2m.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            images = []
            for img_path in item.get("image_paths", []):
                try:
                    img = Image.open(os.path.join(path, img_path)).convert("RGB")
                    images.append(img)
                except Exception:
                    continue
            data.append({
                "id": item["id"],
                "text": item["text"],
                "images": images
            })
    return data



def load_spiqa(path: str) -> List[Dict]:
    data = []
    with open(os.path.join(path, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
        for item in metadata:
            images = []
            for img_name in item.get("image_names", []):
                try:
                    img = Image.open(os.path.join(path, "images", img_name)).convert("RGB")
                    images.append(img)
                except Exception:
                    continue
            data.append({
                "id": item["id"],
                "text": item["paragraph"],
                "images": images
            })
    return data



def load_tqa(path: str) -> List[Dict]:
    data = []
    with open(os.path.join(path, "tqa.json"), "r", encoding="utf-8") as f:
        tqa_data = json.load(f)
        for lesson in tqa_data["lessons"]:
            for qa in lesson["questions"]:
                images = []
                for img_name in qa.get("image_names", []):
                    try:
                        img = Image.open(os.path.join(path, "images", img_name)).convert("RGB")
                        images.append(img)
                    except Exception:
                        continue
                data.append({
                    "id": qa["id"],
                    "text": qa["question_text"],
                    "images": images
                })
    return data


def load_fhm(path: str) -> List[Dict]:
    data = []
    with open(os.path.join(path, "hateful_memes.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            try:
                img = Image.open(os.path.join(path, "img", item["img"]).convert("RGB"))
            except Exception:
                img = None
            data.append({
                "id": str(item["id"]),
                "text": item["text"],
                "images": [img] if img else []
            })
    return data


def load_mscoco14(path: str) -> List[Dict]:
    data = []
    with open(os.path.join(path, "annotations", "captions_train2014.json"), "r", encoding="utf-8") as f:
        annotations = json.load(f)["annotations"]
        for ann in annotations:
            img_filename = f"COCO_train2014_{ann['image_id']:012d}.jpg"
            try:
                img = Image.open(os.path.join(path, "train2014", img_filename)).convert("RGB")
            except Exception:
                img = None
            data.append({
                "id": str(ann["id"]),
                "text": ann["caption"],
                "images": [img] if img else []
            })
    return data


def load_vist(path: str) -> List[Dict]:
    data = []
    with open(os.path.join(path, "stories.json"), "r", encoding="utf-8") as f:
        stories = json.load(f)["stories"]
        for story in stories:
            images = []
            for img_name in story.get("image_names", []):
                try:
                    img = Image.open(os.path.join(path, "images", img_name)).convert("RGB")
                    images.append(img)
                except Exception:
                    continue
            data.append({
                "id": story["story_id"],
                "text": " ".join(story["sentences"]),
                "images": images
            })
    return data

def load_t4sa(path: str) -> List[Dict]:
    data = []
    with open(os.path.join(path, "t4sa.csv"), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                img = Image.open(os.path.join(path, "images", row["image_name"])).convert("RGB")
            except Exception:
                img = None
            data.append({
                "id": row["id"],
                "text": row["tweet_text"],
                "images": [img] if img else []
            })
    return data


# === Lazy Dataset Loaders (Load on-the-fly) ===

def load_wikiweb2m_lazy(path: str):
    """Generator that yields items one at a time."""
    with open(os.path.join(path, "wikiweb2m.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            images = []
            for img_path in item.get("image_paths", []):
                try:
                    img = Image.open(os.path.join(path, img_path)).convert("RGB")
                    images.append(img)
                except Exception:
                    continue
            yield {
                "id": item["id"],
                "text": item["text"],
                "images": images
            }


def load_spiqa_lazy(path: str):
    """Generator that yields items one at a time."""
    with open(os.path.join(path, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
        for item in metadata:
            images = []
            for img_name in item.get("image_names", []):
                try:
                    img = Image.open(os.path.join(path, "images", img_name)).convert("RGB")
                    images.append(img)
                except Exception:
                    continue
            yield {
                "id": item["id"],
                "text": item["paragraph"],
                "images": images
            }


def load_tqa_lazy(path: str):
    """Generator that yields items one at a time."""
    with open(os.path.join(path, "tqa.json"), "r", encoding="utf-8") as f:
        tqa_data = json.load(f)
        for lesson in tqa_data["lessons"]:
            for qa in lesson["questions"]:
                images = []
                for img_name in qa.get("image_names", []):
                    try:
                        img = Image.open(os.path.join(path, "images", img_name)).convert("RGB")
                        images.append(img)
                    except Exception:
                        continue
                yield {
                    "id": qa["id"],
                    "text": qa["question_text"],
                    "images": images
                }


def load_fhm_lazy(path: str):
    """Generator that yields items one at a time."""
    with open(os.path.join(path, "hateful_memes.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            try:
                img = Image.open(os.path.join(path, "img", item["img"])).convert("RGB")
            except Exception:
                img = None
            yield {
                "id": str(item["id"]),
                "text": item["text"],
                "images": [img] if img else []
            }


def load_mscoco14_lazy(path: str):
    """Generator that yields items one at a time."""
    with open(os.path.join(path, "annotations", "captions_train2014.json"), "r", encoding="utf-8") as f:
        annotations = json.load(f)["annotations"]
        for ann in annotations:
            img_filename = f"COCO_train2014_{ann['image_id']:012d}.jpg"
            try:
                img = Image.open(os.path.join(path, "train2014", img_filename)).convert("RGB")
            except Exception:
                img = None
            yield {
                "id": str(ann["id"]),
                "text": ann["caption"],
                "images": [img] if img else []
            }


def load_vist_lazy(path: str):
    """Generator that yields items one at a time."""
    with open(os.path.join(path, "stories.json"), "r", encoding="utf-8") as f:
        stories = json.load(f)["stories"]
        for story in stories:
            images = []
            for img_name in story.get("image_names", []):
                try:
                    img = Image.open(os.path.join(path, "images", img_name)).convert("RGB")
                    images.append(img)
                except Exception:
                    continue
            yield {
                "id": story["story_id"],
                "text": " ".join(story["sentences"]),
                "images": images
            }


def load_t4sa_lazy(path: str):
    """Generator that yields items one at a time."""
    with open(os.path.join(path, "t4sa.csv"), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                img = Image.open(os.path.join(path, "images", row["image_name"])).convert("RGB")
            except Exception:
                img = None
            yield {
                "id": row["id"],
                "text": row["tweet_text"],
                "images": [img] if img else []
            }


# === Dataset Dispatch ===

DATASET_LOADERS: Dict[str, Callable[[str], List[Dict]]] = {
    "wikiweb2m": load_wikiweb2m,
    "spiqa": load_spiqa,
    "tqa": load_tqa,
    "fhm": load_fhm,
    "mscoco14": load_mscoco14,
    "t4sa": load_t4sa,
    "vist": load_vist,
}

DATASET_LOADERS_LAZY: Dict[str, Callable[[str], any]] = {
    "wikiweb2m": load_wikiweb2m_lazy,
    "spiqa": load_spiqa_lazy,
    "tqa": load_tqa_lazy,
    "fhm": load_fhm_lazy,
    "mscoco14": load_mscoco14_lazy,
    "t4sa": load_t4sa_lazy,
    "vist": load_vist_lazy,
}


def get_dataset(name: str, path: str, lazy: bool = False, batch_size: int = 4):
    """
    Get a dataset instance.
    
    Args:
        name: Dataset name
        path: Path to dataset files
        lazy: If True, use lazy loading (memory-efficient). If False, load all data at once.
        batch_size: Batch size for lazy loading
    
    Returns:
        Dataset instance
    """
    name = name.lower()
    
    if lazy:
        return LazyMultimodalDataset(name, path, batch_size)
    else:
        if name not in DATASET_LOADERS:
            raise ValueError(f"Unknown dataset: {name}")
        data = DATASET_LOADERS[name](path)
        return MultimodalDataset(data)

