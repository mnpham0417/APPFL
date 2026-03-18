"""
CLIP + LoRA dataset loader for dLoRA AB_SVD federated experiments.

Wraps clip_lora's dataset infrastructure to produce per-client
torch.utils.data.Dataset objects compatible with APPFL's data pipeline.

Supports all clip_lora benchmark datasets:
    caltech101, dtd, eurosat, fgvc, food101, imagenet,
    oxford_flowers, oxford_pets, stanford_cars, sun397, ucf101.

The returned dataset objects carry two extra attributes that
CLIPLoRATrainer reads to build text features:
    .classnames (list[str]): human-readable class labels.
    .template   (list[str]): text template(s), e.g. ["a photo of a {}."]

IID partitioning: each client receives a class-balanced subset of the
few-shot training data (matching clip_lora's dlora_data.py logic).

Non-IID partitioning: classes are split across clients; each client
exclusively owns certain classes.

Usage in client_dlora_ab_svd.yaml:
    data_configs:
      dataset_path: "./resources/dataset/clip_lora_dataset.py"
      dataset_name: "get_clip_lora_dataset"
      dataset_kwargs:
        dataset_name: "dtd"
        root_path: "/path/to/data"
        shots: 16
        backbone: "ViT-B/16"
        num_clients: 5
        client_id: 0          # overridden per client by MPI script
        data_dist: "iid"
        clip_lora_root: "/path/to/clip_lora"
"""

import sys
import os
import random
from collections import defaultdict
from typing import List, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image


# ---------------------------------------------------------------------------
# Internal dataset wrapper
# ---------------------------------------------------------------------------

class _CLIPDatasetWrapper(Dataset):
    """
    Wraps a list of clip_lora Datum objects into a standard
    (image, label) torch Dataset with CLIP preprocessing.

    Extra attributes:
        classnames (list[str]): class labels.
        template   (list[str]): text template strings.
    """

    def __init__(self, data_source, transform, classnames, template):
        self._data = data_source
        self._transform = transform
        self.classnames = classnames
        self.template = template

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        datum = self._data[idx]
        img = Image.open(datum.impath).convert("RGB")
        if self._transform is not None:
            img = self._transform(img)
        return img, datum.label


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------

def get_clip_lora_dataset(
    dataset_name: str,
    root_path: str,
    shots: int,
    backbone: str,
    num_clients: int,
    client_id: int,
    data_dist: str = "iid",
    clip_lora_root: Optional[str] = None,
    seed: int = 1,
    **kwargs,
):
    """
    Return (train_dataset, val_dataset) for a given client.

    The train_dataset is an IID or non-IID partition of the few-shot
    training split.  The val_dataset is the full validation split
    (shared across all clients for evaluation).

    Args:
        dataset_name:   One of the clip_lora benchmark names.
        root_path:      Root directory where raw datasets are stored.
        shots:          Number of few-shot training samples per class.
        backbone:       CLIP backbone string (for CLIP preprocessing).
        num_clients:    Total number of federated clients.
        client_id:      Zero-based index of this client.
        data_dist:      "iid" or "non-iid".
        clip_lora_root: Path to the clip_lora project directory.
        seed:           Random seed for data partitioning.

    Returns:
        (train_subset, val_dataset): both are _CLIPDatasetWrapper instances.
    """
    # --- Ensure clip_lora is importable ---
    _add_clip_lora_to_path(clip_lora_root)

    import clip
    from datasets import build_dataset
    import torchvision.transforms as transforms

    # CLIP image preprocessing (training augmentation)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=224,
                scale=(0.08, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    # CLIP val/test preprocessing (from clip.load's default preprocess)
    _, clip_preprocess = clip.load(backbone, device="cpu")

    # Build the full dataset using clip_lora's infrastructure
    full_dataset = build_dataset(dataset_name, root_path, shots, clip_preprocess)

    classnames = full_dataset.classnames
    template = full_dataset.template

    # --- Partition training data ---
    train_x = full_dataset.train_x  # list of Datum objects
    client_data = _partition(train_x, num_clients, client_id, data_dist, seed)

    train_dataset = _CLIPDatasetWrapper(
        client_data, train_transform, classnames, template
    )
    val_dataset = _CLIPDatasetWrapper(
        full_dataset.val, clip_preprocess, classnames, template
    )

    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_clip_lora_to_path(clip_lora_root: Optional[str]):
    """Insert clip_lora into sys.path so its packages are importable."""
    if clip_lora_root is not None:
        if clip_lora_root not in sys.path:
            sys.path.insert(0, clip_lora_root)
        return

    # Auto-detect based on expected repo layout
    candidate = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "clip_lora")
    )
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)


def _partition(
    data_source,
    num_clients: int,
    client_id: int,
    data_dist: str,
    seed: int,
) -> list:
    """
    Partition data_source (list of Datum) for a single client.

    IID:     Each class is split evenly across all clients.
    Non-IID: Classes are assigned exclusively to individual clients
             (round-robin over shuffled class list).
    """
    rng = random.Random(seed)

    class_to_data = defaultdict(list)
    for item in data_source:
        class_to_data[item.label].append(item)

    if data_dist == "iid":
        client_data = []
        for cls_items in class_to_data.values():
            shuffled = cls_items[:]
            rng.shuffle(shuffled)
            n = len(shuffled)
            base = n // num_clients
            remainder = n % num_clients
            start = client_id * base + min(client_id, remainder)
            end = start + base + (1 if client_id < remainder else 0)
            client_data.extend(shuffled[start:end])
        return client_data

    elif data_dist == "non-iid":
        classes = list(class_to_data.keys())
        rng.shuffle(classes)
        client_data = []
        for idx, cls in enumerate(classes):
            if idx % num_clients == client_id:
                client_data.extend(class_to_data[cls])
        return client_data

    else:
        raise ValueError(
            f"data_dist must be 'iid' or 'non-iid', got '{data_dist}'"
        )
