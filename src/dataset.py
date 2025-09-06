from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class WindowItem:
    path: str
    meta: str
    label: str


class EEGWindowDataset(Dataset):
    def __init__(self, root: str, split: str, normalization: str = "none") -> None:
        self.root = Path(root)
        self.split = split
        self.normalization = normalization

        self.items: List[WindowItem] = []
        manifest_path = Path("results/manifest.json")
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for r in data:
                if r.get("split") == split:
                    self.items.append(WindowItem(r["path"], r["meta"], r["label"]))

        # Build label mapping
        labels = sorted({it.label for it in self.items})
        self.class_to_idx = {c: i for i, c in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        it = self.items[idx]
        x = np.load(it.path)  # (C, T)
        with open(it.meta, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Apply normalization variants
        if self.normalization == "wgan_minmax":
            mn = meta.get("min", float(np.min(x)))
            mx = meta.get("max", float(np.max(x)))
            denom = (mx - mn) if (mx - mn) != 0 else 1.0
            x = 2.0 * (x - mn) / denom - 1.0
        elif self.normalization == "zscore":
            mu = x.mean(axis=-1, keepdims=True)
            sd = x.std(axis=-1, keepdims=True) + 1e-8
            x = (x - mu) / sd

        y = self.class_to_idx[it.label]
        return torch.from_numpy(x).float(), int(y)
