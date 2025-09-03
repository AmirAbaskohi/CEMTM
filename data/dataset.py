import os
from typing import List, Dict, Callable
from PIL import Image
from torch.utils.data import Dataset
import json
import csv

class MultimodalDataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# === Dataset Loaders ===

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


def get_dataset(name: str, path: str) -> MultimodalDataset:
    name = name.lower()
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {name}")
    data = DATASET_LOADERS[name](path)
    return MultimodalDataset(data)
