from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import yaml


def parse_args():
    ap = argparse.ArgumentParser(description="Utility classifier eval: train on real vs real+synthetic and evaluate on real test")
    ap.add_argument("--config", required=True, help="Path to YAML config used for training models")
    ap.add_argument("--ckpt-ddpm", default=None, help="Path to DDPM checkpoint for generation")
    ap.add_argument("--ckpt-wgan", default=None, help="Path to WGAN-GP generator checkpoint for generation")
    ap.add_argument("--n-synth-per-class", type=int, default=500, help="Max synthetic samples per class to add")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_manifest() -> List[dict]:
    m = Path("results/manifest.json")
    if not m.exists():
        raise FileNotFoundError("results/manifest.json not found. Run preprocessing to build manifest.")
    with open(m, "r", encoding="utf-8") as f:
        return json.load(f)


def build_class_map(items: List[dict]) -> Tuple[Dict[str, int], List[str]]:
    labels = sorted({r["label"] for r in items})
    class_to_idx = {c: i for i, c in enumerate(labels)}
    idx_to_class = labels
    return class_to_idx, idx_to_class


class RealWindows(Dataset):
    def __init__(self, records: List[dict], class_to_idx: Dict[str, int], length: int):
        self.records = records
        self.class_to_idx = class_to_idx
        self.length = length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i: int):
        r = self.records[i]
        with np.load(r["path"]) as d:
            x = d["array"].astype(np.float32)  # (C,T)
        # z-score per channel
        x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8)
        # pad/trim
        if x.shape[1] < self.length:
            pad = self.length - x.shape[1]
            x = np.pad(x, ((0, 0), (0, pad)))
        elif x.shape[1] > self.length:
            x = x[:, : self.length]
        y = self.class_to_idx[r["label"]]
        return torch.from_numpy(x), int(y)


class SyntheticWindows(Dataset):
    def __init__(self, xs: np.ndarray, ys: np.ndarray, length: int):
        assert xs.ndim == 3, "xs must be (N,C,T)"
        self.xs = xs.astype(np.float32)
        self.ys = ys.astype(np.int64)
        self.length = length

    def __len__(self):
        return self.xs.shape[0]

    def __getitem__(self, i: int):
        x = self.xs[i]
        # z-score per channel
        x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8)
        if x.shape[1] < self.length:
            pad = self.length - x.shape[1]
            x = np.pad(x, ((0, 0), (0, pad)))
        elif x.shape[1] > self.length:
            x = x[:, : self.length]
        y = int(self.ys[i])
        return torch.from_numpy(x), y


class SmallCNN1D(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(64), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(128), nn.MaxPool1d(2),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T)
        h = self.net(x)
        h = h.mean(dim=-1)  # GAP
        return self.head(h)


def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs, dim=0)
    ys = torch.tensor(ys, dtype=torch.long)
    return xs, ys


def train_classifier(model: nn.Module, dl: DataLoader, dv: DataLoader, device: torch.device, epochs: int, lr: float) -> Tuple[float, float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0
    best_f1 = 0.0
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        # eval
        acc, f1 = eval_classifier(model, dv, device)
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
    return best_acc, best_f1


def eval_classifier(model: nn.Module, dl: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            ps.append(preds)
            ys.append(yb.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return float(acc), float(f1)


def generate_synthetic(cfg: dict, ckpt: str, model_kind: str, device: torch.device, per_class: int, num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    from src.eval.generate import generate_samples
    xs_all = []
    ys_all = []
    for c in range(num_classes):
        xs, ys = generate_samples(cfg, ckpt, device, model_kind, n=per_class, class_id=c)
        xs_all.append(xs)
        ys_all.append(ys)
    xs_all = np.concatenate(xs_all, axis=0)
    ys_all = np.concatenate(ys_all, axis=0)
    return xs_all, ys_all


def make_loaders(cfg: dict, mix: str, ddpm_ckpt: str | None, wgan_ckpt: str | None, per_class: int, device: torch.device):
    items = load_manifest()
    class_to_idx, idx_to_class = build_class_map(items)
    num_classes = len(idx_to_class)
    length = int(cfg['model']['length'])

    train_records = [r for r in items if r.get("split") == "train"]
    test_records = [r for r in items if r.get("split") == "test"]
    if not train_records or not test_records:
        raise RuntimeError("Manifest must contain train and test splits")

    real_train = RealWindows(train_records, class_to_idx, length)
    real_test = RealWindows(test_records, class_to_idx, length)

    in_ch = int(cfg['model']['channels'])

    # Build training dataset according to mix
    if mix == 'real':
        ds_train = real_train
    elif mix == 'real+ddpm':
        if ddpm_ckpt is None:
            raise ValueError("--ckpt-ddpm required for mix=real+ddpm")
        xs, ys = generate_synthetic(cfg, ddpm_ckpt, 'ddpm', device, per_class, num_classes)
        ds_train = torch.utils.data.ConcatDataset([real_train, SyntheticWindows(xs, ys, length)])
    elif mix == 'real+wgan':
        if wgan_ckpt is None:
            raise ValueError("--ckpt-wgan required for mix=real+wgan")
        xs, ys = generate_synthetic(cfg, wgan_ckpt, 'wgan_gp', device, per_class, num_classes)
        ds_train = torch.utils.data.ConcatDataset([real_train, SyntheticWindows(xs, ys, length)])
    else:
        raise ValueError(f"Unknown mix {mix}")

    dl_train = DataLoader(ds_train, batch_size=cfg['training'].get('batch_size', 64), shuffle=True, num_workers=cfg['training'].get('num_workers', 4), collate_fn=collate_fn)
    dl_test = DataLoader(real_test, batch_size=cfg['training'].get('batch_size', 64), shuffle=False, num_workers=cfg['training'].get('num_workers', 4), collate_fn=collate_fn)
    return dl_train, dl_test, in_ch, len(idx_to_class)


def main():
    args = parse_args()
    set_seed(args.seed)
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    mixes = [('real', None), ('real+wgan', args.ckpt_wgan), ('real+ddpm', args.ckpt_ddpm)]
    rows = []
    for mix, ck in mixes:
        try:
            dl_train, dl_test, in_ch, num_classes = make_loaders(cfg, mix, args.ckpt_ddpm, args.ckpt_wgan, args.n_synth_per_class, device)
        except Exception as e:
            print(f"Skipping {mix}: {e}")
            continue
        model = SmallCNN1D(in_ch=in_ch, num_classes=num_classes).to(device)
        acc, f1 = train_classifier(model, dl_train, dl_test, device, args.epochs, args.lr)
        print(f"{mix}: acc={acc:.4f} macroF1={f1:.4f}")
        rows.append({"mix": mix, "acc": acc, "macro_f1": f1})

    if rows:
        df = pd.DataFrame(rows)
        out_dir = Path('results')
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / 'utility_classifier.csv', index=False)
        with open(out_dir / 'table_utility.tex', 'w', encoding='utf-8') as f:
            f.write(df.to_latex(index=False, float_format=lambda v: f"{v:.3f}"))
        print("Wrote results/utility_classifier.csv and results/table_utility.tex")
    else:
        print("No rows produced — check checkpoints and manifest.")


if __name__ == '__main__':
    main()
