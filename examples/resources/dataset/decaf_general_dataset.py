"""
General-purpose dataset loader for DeCaF (CLIP + LoRA) federated experiments.

Wraps standard torchvision datasets (MNIST, CIFAR-10, CIFAR-100) for use
with DeCaFTrainer. Applies CLIP preprocessing, attaches class names and text
templates, and supports few-shot subsampling and IID/non-IID partitioning.

Supported dataset_name values:
    "mnist"    — 10 digit classes, grayscale converted to RGB
    "cifar10"  — 10 classes
    "cifar100" — 100 fine-grained classes

Usage in client_decaf_cifar10.yaml:
    data_configs:
      dataset_path: "./resources/dataset/decaf_general_dataset.py"
      dataset_name: "get_decaf_general_dataset"
      dataset_kwargs:
        dataset_name: "cifar10"
        root_path: "./datasets/RawData"
        backbone: "ViT-B/16"
        shots: 40           # None = use all training data
        data_dist: "iid"    # "iid" or "non-iid"
        num_clients: 5
        client_id: 0
        seed: 1
"""

import random
from collections import defaultdict
from typing import List, Optional

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Class names
# ---------------------------------------------------------------------------

_MNIST_CLASSES = [str(i) for i in range(10)]

_CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

_CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
    "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman",
    "worm",
]

_CLASSNAMES = {
    "mnist":    _MNIST_CLASSES,
    "cifar10":  _CIFAR10_CLASSES,
    "cifar100": _CIFAR100_CLASSES,
}

_TEMPLATES = {
    "mnist":    ["a photo of the number: '{}'."],
    "cifar10":  ["a photo of a {}."],
    "cifar100": ["a photo of a {}."],
}


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class _CLIPDatasetWrapper(Dataset):
    """
    Wraps a list of (image_tensor, label) pairs pre-processed for CLIP.

    Attributes:
        classnames (list[str]): human-readable class labels.
        template   (list[str]): CLIP text template(s).
    """

    def __init__(self, samples, classnames: List[str], template: List[str]):
        self._samples = samples       # list of (tensor, int)
        self.classnames = classnames
        self.template = template

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------

def get_decaf_general_dataset(
    dataset_name: str,
    num_clients: int,
    client_id: int,
    backbone: str = "ViT-B/16",
    shots: Optional[int] = None,
    data_dist: str = "iid",
    root_path: str = "./datasets/RawData",
    seed: int = 1,
    **kwargs,
):
    """
    Return (train_dataset, val_dataset) for a given client.

    The train dataset is an IID or non-IID partition of the (optionally
    few-shot) training split.  The val dataset is the full test split
    shared across all clients.

    Args:
        dataset_name:  "mnist", "cifar10", or "cifar100".
        num_clients:   Total number of federated clients.
        client_id:     Zero-based index of this client.
        backbone:      CLIP backbone string (used to load CLIP preprocessing).
        shots:         Max training samples per class. None = use all data.
        data_dist:     "iid" or "non-iid".
        root_path:     Directory for torchvision to download/cache data.
        seed:          Random seed for partitioning and shot sampling.

    Returns:
        (train_subset, val_dataset): both are _CLIPDatasetWrapper instances.
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in _CLASSNAMES:
        raise ValueError(
            f"dataset_name must be one of {list(_CLASSNAMES)}, got '{dataset_name}'"
        )

    classnames = _CLASSNAMES[dataset_name]
    template = _TEMPLATES[dataset_name]

    # Build CLIP preprocessing transforms
    train_transform, val_transform = _build_transforms(backbone, dataset_name)

    # Load raw torchvision datasets (download if needed)
    tv_train, tv_val = _load_torchvision(dataset_name, root_path)

    # Apply CLIP preprocessing eagerly (avoids repeated transforms in the loop)
    train_samples = _preprocess(tv_train, train_transform)
    val_samples   = _preprocess(tv_val,   val_transform)

    # Few-shot subsampling on the full training set (before partitioning)
    if shots is not None:
        train_samples = _sample_shots(train_samples, shots, seed)

    # Partition training samples across clients
    client_samples = _partition(train_samples, num_clients, client_id, data_dist, seed)

    train_dataset = _CLIPDatasetWrapper(client_samples, classnames, template)
    val_dataset   = _CLIPDatasetWrapper(val_samples,    classnames, template)

    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_transforms(backbone: str, dataset_name: str):
    """Return (train_transform, val_transform) with CLIP normalization."""
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std  = (0.26862954, 0.26130258, 0.27577711)

    # MNIST is grayscale — convert to RGB first
    to_rgb = [T.Grayscale(num_output_channels=3)] if dataset_name == "mnist" else []

    train_transform = T.Compose(
        to_rgb + [
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=clip_mean, std=clip_std),
        ]
    )

    val_transform = T.Compose(
        to_rgb + [
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=clip_mean, std=clip_std),
        ]
    )

    return train_transform, val_transform


def _load_torchvision(dataset_name: str, root_path: str):
    """Load train and val splits as raw torchvision datasets.

    Uses a per-dataset file lock so that when multiple MPI client processes
    start simultaneously only one downloads while the others wait.
    """
    import fcntl
    import os

    loaders = {
        "mnist":    torchvision.datasets.MNIST,
        "cifar10":  torchvision.datasets.CIFAR10,
        "cifar100": torchvision.datasets.CIFAR100,
    }
    cls = loaders[dataset_name]

    os.makedirs(root_path, exist_ok=True)
    lock_path = os.path.join(root_path, f".{dataset_name}_download.lock")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            tv_train = cls(root_path, train=True,  download=True, transform=None)
            tv_val   = cls(root_path, train=False, download=True, transform=None)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    return tv_train, tv_val


def _preprocess(tv_dataset, transform) -> List:
    """Apply transform to every sample and return list of (tensor, label)."""
    return [(transform(img), label) for img, label in tv_dataset]


def _sample_shots(samples: List, shots: int, seed: int) -> List:
    """Keep at most `shots` samples per class."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for item in samples:
        by_class[item[1]].append(item)
    result = []
    for cls_items in by_class.values():
        shuffled = cls_items[:]
        rng.shuffle(shuffled)
        result.extend(shuffled[:shots])
    return result


def _partition(samples: List, num_clients: int, client_id: int,
               data_dist: str, seed: int) -> List:
    """
    Partition samples for a single client.

    IID:     Each class is split evenly across all clients.
    Non-IID: Classes are assigned exclusively to clients (round-robin).
    """
    rng = random.Random(seed)

    by_class = defaultdict(list)
    for item in samples:
        by_class[item[1]].append(item)

    if data_dist == "iid":
        client_data = []
        for cls_items in by_class.values():
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
        classes = list(by_class.keys())
        rng.shuffle(classes)
        client_data = []
        for idx, cls in enumerate(classes):
            if idx % num_clients == client_id:
                client_data.extend(by_class[cls])
        return client_data

    else:
        raise ValueError(
            f"data_dist must be 'iid' or 'non-iid', got '{data_dist}'"
        )
