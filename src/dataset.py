from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class WindowItem:
    """Represents a single EEG window item with its path, metadata, and label."""
    path: str
    meta: str
    label: str


class EEGWindowDataset(Dataset):
    """
    PyTorch Dataset for loading EEG windows from preprocessed .npz files.

    This dataset loads EEG signal windows and their corresponding labels for training
    generative models like WGAN-GP or DDPM. It supports different normalization schemes
    and handles class indexing for multi-class classification.

    Args:
        root (str): Root directory containing the processed data (e.g., 'data/processed').
        split (str): Data split to load ('train', 'val', or 'test').
        normalization (str): Normalization method ('none', 'wgan_minmax', or 'zscore').
    """

    def __init__(self, root: str, split: str, normalization: str = "none", length: int = 250) -> None:
        self.root = Path(root)
        self.split = split
        self.normalization = normalization
        self.length = length

        self.items: List[WindowItem] = []
        # Load manifest file to get list of items for the specified split
        manifest_path = Path("results/manifest.json")
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for r in data:
                if r.get("split") == split:
                    self.items.append(WindowItem(r["path"], r["meta"], r["label"]))

        # Normalize labels to display names
        self.label_map = {
            "musc": "Muscle",
            "eyem": "Eye movement",
            "elec": "Electrode",
            "chew": "Chewing",
            "shiv": "Shiver",
            "Muscle": "Muscle",
            "Eye movement": "Eye movement",
            "Electrode": "Electrode",
            "Chewing": "Chewing",
            "Shiver": "Shiver",
        }
        for item in self.items:
            item.label = self.label_map.get(item.label, item.label)

        # Build label to index mapping for classification
        labels = sorted({it.label for it in self.items})
        self.class_to_idx = {c: i for i, c in enumerate(labels)}

    def __len__(self) -> int:
        """Returns the number of items in the dataset."""
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Loads and returns a single EEG window and its label.

        Args:
            idx (int): Index of the item to load.

        Returns:
            Tuple[torch.Tensor, int]: EEG signal tensor (C, T) and class index.
        """
        it = self.items[idx]
        # Load the .npz file and extract the 'array' key containing the EEG data
        with np.load(it.path) as data:
            x = data['array']  # Shape: (C, T), where C=channels, T=time samples
        # Load metadata from JSON file
        with open(it.meta, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Convert to torch tensor
        x = torch.from_numpy(x).float()

        # Pad or truncate to fixed length
        target_length = self.length
        if x.size(-1) < target_length:
            # Pad with zeros
            pad_size = target_length - x.size(-1)
            x = F.pad(x, (0, pad_size))
        elif x.size(-1) > target_length:
            # Truncate
            x = x[:, :target_length]

        # Apply the specified normalization
        if self.normalization == "wgan_minmax":
            # Min-max normalization to [-1, 1] for WGAN
            mn = meta.get("min", x.min().item())
            mx = meta.get("max", x.max().item())
            denom = (mx - mn) if (mx - mn) != 0 else 1.0
            x = 2.0 * (x - mn) / denom - 1.0
        elif self.normalization == "zscore":
            # Z-score normalization for DDPM
            mu = x.mean(dim=-1, keepdim=True)
            sd = x.std(dim=-1, keepdim=True) + 1e-8  # Add epsilon to avoid division by zero
            x = (x - mu) / sd

        # Convert label to class index
        y = self.class_to_idx[it.label]
        return x, int(y)
