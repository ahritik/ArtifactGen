from __future__ import annotations

"""
Quick, zero-training plots from existing artifacts:
 - split_counts.png: bar chart of subjects per split (from results/split_summary.json)
 - checkpoint_counts.png: bar chart of checkpoint file counts grouped by prefix (from results/checkpoints)

Outputs are written under paper/figs/.
"""

import json
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "paper" / "figs"


def split_counts_bar() -> Path:
    data_path = ROOT / "results" / "split_summary.json"
    counts = {"train": 0, "val": 0, "test": 0}
    if data_path.exists():
        try:
            data = json.loads(data_path.read_text(encoding="utf-8"))
            counts.update({k: int(v) for k, v in data.get("per_split_counts", {}).items()})
        except Exception:
            pass
    names = ["train", "val", "test"]
    values = [counts[k] for k in names]
    plt.figure(figsize=(4, 3))
    bars = plt.bar(names, values, color=["#4CAF50", "#FF9800", "#2196F3"])
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, str(v), ha="center", va="bottom", fontsize=9)
    plt.ylabel("subjects")
    plt.title("Subjects per split")
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "split_counts.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def checkpoint_counts_bar() -> Path:
    ckpt_root = ROOT / "results" / "checkpoints"
    counts: Counter[str] = Counter()
    if ckpt_root.exists():
        for p in ckpt_root.glob("*.pth"):
            prefix = p.name.split("_")[0]
            counts[prefix] += 1
    labels = list(counts.keys()) or ["(none)"]
    values = [counts[k] for k in labels] or [0]
    plt.figure(figsize=(5, 3))
    bars = plt.bar(labels, values, color="#607D8B")
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, str(v), ha="center", va="bottom", fontsize=9)
    plt.ylabel("files")
    plt.title("Checkpoint files by prefix")
    plt.xticks(rotation=30, ha="right")
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "checkpoint_counts.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def main() -> None:
    s = split_counts_bar()
    c = checkpoint_counts_bar()
    print("Wrote:")
    print(" -", s)
    print(" -", c)


if __name__ == "__main__":
    main()
