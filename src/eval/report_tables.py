from __future__ import annotations

"""
Generate small, fast LaTeX tables from existing configs and results (no retraining):

Outputs (written under paper/figs/):
 - table_methods_summary.tex    # WGAN vs DDPM config side-by-side
 - table_splits.tex             # Subject counts per split
 - table_checkpoints.tex        # Checkpoint inventory summary

Usage (PowerShell):
  python -m src.eval.report_tables
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
PAPER_FIGS = ROOT / "paper" / "figs"


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        try:
            return yaml.safe_load(text) or {}
        except Exception:
            pass
    # Minimal fallback: extremely naive key lookup (only what we need)
    out: Dict[str, Any] = {}
    for line in text.splitlines():
        if ":" in line and not line.strip().startswith("#"):
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _write_tex(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def make_methods_summary(ddpm_cfg: Dict[str, Any], wgan_cfg: Dict[str, Any]) -> str:
    # Extract fields
    def safe(d: Dict[str, Any], path: Tuple[str, ...], default: Any = "—") -> Any:
        cur: Any = d
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    rows = []
    rows.append((
        "WGAN-GP",
        str(safe(wgan_cfg, ("data", "window_seconds"), "1.0")),
        str(safe(wgan_cfg, ("model", "length"), "250")),
        "minmax_per_window" if bool(safe(wgan_cfg, ("data", "store_minmax"), True)) else "—",
        str(len(safe(wgan_cfg, ("data", "channels"), [])) or safe(wgan_cfg, ("model", "channels"), "8")),
        str(safe(wgan_cfg, ("model", "num_classes"), "5")),
        "—",
        "—",
    ))
    sampler = str(safe(ddpm_cfg, ("sampling", "sampler"), "ddim"))
    steps = str(safe(ddpm_cfg, ("sampling", "steps"), "50"))
    rows.append((
        "DDPM",
        str(safe(ddpm_cfg, ("data", "window_seconds"), "2.0")),
        str(safe(ddpm_cfg, ("model", "length"), "500")),
        str(safe(ddpm_cfg, ("data", "normalization"), "zscore_per_recording")),
        str(len(safe(ddpm_cfg, ("data", "channels"), [])) or safe(ddpm_cfg, ("model", "channels"), "8")),
        str(safe(ddpm_cfg, ("model", "num_classes"), "5")),
        f"{sampler} / {steps}",
        str(safe(ddpm_cfg, ("sampling", "guidance_scale"), "3.5")),
    ))

    # LaTeX table
    lines = [
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Model & Win (s) & Length & Normalization & Ch & Classes & Sampler/Steps & CFG \\\\",
        "\\midrule",
    ]
    for r in rows:
        line = f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]} & {r[5]} & {r[6]} & {r[7]} \\"
        lines.append(line)
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def make_split_table(split_json: Path) -> str:
    counts = {"train": "—", "val": "—", "test": "—"}
    if split_json.exists():
        try:
            data = json.loads(split_json.read_text(encoding="utf-8"))
            counts.update({k: str(v) for k, v in data.get("per_split_counts", {}).items()})
        except Exception:
            pass
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        " & Train & Val & Test \\\\",
        "\\midrule",
        f"Subjects & {counts['train']} & {counts['val']} & {counts['test']} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines)


def make_checkpoint_table(ckpt_dir: Path) -> str:
    # Summarize any known checkpoint files by prefix
    entries: list[tuple[str, str]] = []
    if ckpt_dir.exists():
        files = sorted([p.name for p in ckpt_dir.glob("*.pth")])
        # Group basic prefixes
        by_prefix: Dict[str, list[str]] = {}
        for f in files:
            prefix = f.split("_")[0]
            by_prefix.setdefault(prefix, []).append(f)
        for prefix, group in by_prefix.items():
            summary = f"n={len(group)}"
            entries.append((prefix, summary))

    if not entries:
        entries.append(("(none)", "—"))

    lines = [
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Prefix & Summary \\\\",
        "\\midrule",
    ]
    for k, v in entries:
        lines.append(f"{k} & {v} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    ddpm_cfg = _read_yaml(ROOT / "configs" / "ddpm_raw.yaml")
    wgan_cfg = _read_yaml(ROOT / "configs" / "wgan_raw.yaml")

    # methods summary
    methods_tex = make_methods_summary(ddpm_cfg, wgan_cfg)
    _write_tex(PAPER_FIGS / "table_methods_summary.tex", methods_tex)

    # split table
    split_tex = make_split_table(ROOT / "results" / "split_summary.json")
    _write_tex(PAPER_FIGS / "table_splits.tex", split_tex)

    # checkpoint inventory (root-level results/checkpoints)
    ckpt_tex = make_checkpoint_table(ROOT / "results" / "checkpoints")
    _write_tex(PAPER_FIGS / "table_checkpoints.tex", ckpt_tex)

    print("Wrote:")
    for f in ("table_methods_summary.tex", "table_splits.tex", "table_checkpoints.tex"):
        print(" -", PAPER_FIGS / f)


if __name__ == "__main__":
    main()
