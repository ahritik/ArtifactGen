from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f) if yaml is not None else {}
    manifest = Path("results/manifest.json")
    if not manifest.exists():
        print("No manifest found.")
        return
    with open(manifest, "r", encoding="utf-8") as f:
        items = json.load(f)
    print(f"Loaded {len(items)} items; compute KID/MMD/PRD via encoder features here…")


if __name__ == "__main__":
    main()
