"""
Flower102 dataset loader for decentralized learning experiments.

Reads the Zhou-split JSON file (split_zhou_OxfordFlowers.json) which
contains train / val / test splits in the format:
    [[filename, label_0indexed, classname], ...]

The training split is partitioned across clients using class-balanced IID
partitioning (each class distributed equally across all clients).

Args to get_flower102():
    num_clients  (int) : total number of FL/decentralized clients
    client_id    (int) : 0-based index of this client
    data_path    (str) : absolute path to the Flower102 directory
                         (should contain jpg/ and split_zhou_OxfordFlowers.json)
    split        (str) : "iid" (default) uses class-balanced equal partition.
                         "noniid" uses class-exclusive partition (each client
                         gets a disjoint subset of classes).

Returns:
    (train_dataset, test_dataset) — both are torch.utils.data.Dataset objects.
"""

import json
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# -----------------------------------------------------------------------
# Low-level dataset class
# -----------------------------------------------------------------------

class Flower102Dataset(Dataset):
    """
    Lazy-loading dataset for Flower102.
    Each item in ``samples`` is (image_path, label).
    """

    def __init__(self, samples, transform=None):
        self.samples = samples  # list of (abs_path, int_label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# -----------------------------------------------------------------------
# Public interface
# -----------------------------------------------------------------------

def get_flower102(
    num_clients: int,
    client_id: int,
    data_path: str = "/work/mech-ai-scratch/nsaadati/projects/dlora/others/CLIP-LoRA/data/Flower102",
    split: str = "iid",
    **kwargs,
):
    """
    Return the Flower102 dataset partition for the given client.

    Parameters
    ----------
    num_clients : int
        Total number of clients.
    client_id : int
        0-based client index.
    data_path : str
        Path to the Flower102 root directory (contains jpg/ and the JSON).
    split : str
        "iid"    – class-balanced IID partition (default).
        "noniid" – each client receives a disjoint set of classes.

    Returns
    -------
    (train_dataset, test_dataset)
    """
    image_dir = os.path.join(data_path, "jpg")
    split_file = os.path.join(data_path, "split_zhou_OxfordFlowers.json")

    with open(split_file, "r") as f:
        splits = json.load(f)

    # ImageNet-style normalisation (Flower102 is a natural image dataset)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )

    # ----------------------------------------------------------------
    # Build full train + test sample lists
    # ----------------------------------------------------------------
    def _to_samples(items):
        """Convert JSON items → list of (abs_path, label)."""
        return [
            (os.path.join(image_dir, fname), int(label))
            for fname, label, _ in items
        ]

    all_train_samples = _to_samples(splits["train"])
    test_samples = _to_samples(splits["test"])

    # ----------------------------------------------------------------
    # Partition train samples across clients
    # ----------------------------------------------------------------
    if split == "noniid":
        client_samples = _noniid_partition(
            all_train_samples, num_clients, client_id
        )
    else:
        client_samples = _iid_partition(
            all_train_samples, num_clients, client_id
        )

    train_dataset = Flower102Dataset(client_samples, transform=train_transform)
    test_dataset = Flower102Dataset(test_samples, transform=test_transform)

    return train_dataset, test_dataset


# -----------------------------------------------------------------------
# Partition helpers
# -----------------------------------------------------------------------

def _iid_partition(samples, num_clients, client_id):
    """
    Class-balanced IID partition: each class is divided evenly across
    all clients, so every client sees a proportional share of every class.
    """
    # Group sample indices by class label
    label_to_indices = {}
    for idx, (_, label) in enumerate(samples):
        label_to_indices.setdefault(label, []).append(idx)

    chosen = []
    for label, indices in sorted(label_to_indices.items()):
        indices = np.array(indices)
        samples_per_client = len(indices) // num_clients
        start = client_id * samples_per_client
        end = start + samples_per_client
        # Last client absorbs any remainder
        if client_id == num_clients - 1:
            end = len(indices)
        chosen.extend(indices[start:end].tolist())

    return [samples[i] for i in chosen]


def _noniid_partition(samples, num_clients, client_id):
    """
    Class-exclusive non-IID partition: classes are divided into
    ``num_clients`` disjoint groups and each client gets one group.
    """
    all_labels = sorted(set(label for _, label in samples))
    num_classes = len(all_labels)
    classes_per_client = max(1, num_classes // num_clients)

    start_cls = client_id * classes_per_client
    if client_id == num_clients - 1:
        end_cls = num_classes
    else:
        end_cls = start_cls + classes_per_client

    client_classes = set(all_labels[start_cls:end_cls])
    return [(path, label) for path, label in samples if label in client_classes]
