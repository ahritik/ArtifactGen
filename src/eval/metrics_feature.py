from __future__ import annotations

"""Feature-space metrics for EEG artifact generation.

Computes:
    - Fréchet Distance (FID) between real and synthetic embeddings.
    - Multi-kernel MMD (RBF) as a two-sample distance.
    - Precision / Recall (PRD-style) via coarse PCA bin overlap.
    - Diversity proxy (1 - mean pairwise cosine) on a feature subset.
    - Mean DTW baseline (real-real) as a structural lower bound (optional).

Outputs appended / created as:
    CSV:   results/feature_metrics.csv
    LaTeX: paper/figs/table_metrics.tex

Each CSV row includes the model kind (ddpm / wgan_gp) and sample counts.
"""

import argparse
import json
from pathlib import Path
from typing import Tuple
import math

import numpy as np

try:  # optional yaml
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyFeatureNet(nn.Module):
    """Lightweight 1D Conv encoder -> 128-D embedding."""
    def __init__(self, in_ch: int = 8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 64, 7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, 5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.proj = nn.Linear(256, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,C,T)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = x.mean(-1)
        return self.proj(x)


def frechet_distance(mu1: np.ndarray, mu2: np.ndarray, cov1: np.ndarray, cov2: np.ndarray) -> float:
    diff = mu1 - mu2
    cov_prod = cov1 @ cov2
    eigvals, eigvecs = np.linalg.eigh(cov_prod)
    eigvals = np.clip(eigvals, 0, None)
    sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals + 1e-12)) @ eigvecs.T
    return float(diff.dot(diff) + np.trace(cov1 + cov2 - 2 * sqrt_cov))


def mmd_rbf(x: np.ndarray, y: np.ndarray, gammas: Tuple[float,...] = (0.5,1.0,2.0)) -> float:
    def _k(a,b,g):
        d2 = ((a[:,None,:]-b[None,:,:])**2).sum(-1)
        return np.exp(-g * d2)
    m, n = x.shape[0], y.shape[0]
    acc = 0.0
    for g in gammas:
        k_xx = _k(x,x,g); k_yy = _k(y,y,g); k_xy = _k(x,y,g)
        np.fill_diagonal(k_xx, 0.0); np.fill_diagonal(k_yy, 0.0)
        mmd2 = k_xx.sum()/(m*(m-1)) + k_yy.sum()/(n*(n-1)) - 2*k_xy.mean()
        acc += mmd2
    return float(acc/len(gammas))


def prd_precision_recall(x: np.ndarray, y: np.ndarray, n_bins: int = 20) -> Tuple[float,float]:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=0)
    both = np.concatenate([x,y],0)
    coords = pca.fit_transform(both)
    xr, yr = coords[:len(x)], coords[len(x):]
    mn0,mx0 = xr[:,0].min(), xr[:,0].max(); mn1,mx1 = xr[:,1].min(), xr[:,1].max()
    def _bin(z0,z1):
        i = np.clip(((z0 - mn0)/(mx0-mn0+1e-8)*n_bins).astype(int),0,n_bins-1)
        j = np.clip(((z1 - mn1)/(mx1-mn1+1e-8)*n_bins).astype(int),0,n_bins-1)
        return i,j
    bx_i,bx_j = _bin(xr[:,0],xr[:,1]); by_i,by_j = _bin(yr[:,0],yr[:,1])
    real_bins = set(zip(bx_i,bx_j)); synth_bins = set(zip(by_i,by_j))
    inter = len(synth_bins & real_bins)
    precision = inter / (len(synth_bins)+1e-8)
    recall    = inter / (len(real_bins)+1e-8)
    return precision, recall


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    C, T = a.shape
    total = 0.0
    for c in range(C):
        x = a[c]; y = b[c]
        D = np.full((T+1,T+1), np.inf); D[0,0] = 0.0
        for i in range(1,T+1):
            xi = x[i-1]
            for j in range(1,T+1):
                cost = (xi - y[j-1])**2
                D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
        total += D[T,T]
    return total / C


def extract_real_windows(items, max_per_class: int | None = None):
    by_label: dict[str, list[str]] = {}
    for it in items:
        lbl = it.get('label')
        by_label.setdefault(lbl, []).append(it['path'])
    paths: list[str] = []
    for _, plist in by_label.items():
        if max_per_class is not None:
            plist = plist[:max_per_class]
        paths.extend(plist)
    return paths

def parse_args():
    ap = argparse.ArgumentParser(description="Feature-level metrics: FID / MMD / PRD / diversity / DTW baseline")
    ap.add_argument('--config', required=True)
    ap.add_argument('--real-cache', default='results/real_features.npy')
    ap.add_argument('--synth-cache', default=None)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--model-kind', default='ddpm', choices=['ddpm','wgan_gp'])
    ap.add_argument('--n', type=int, default=1024)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--out-csv', default='results/feature_metrics.csv')
    ap.add_argument('--out-tex', default='paper/figs/table_metrics.tex')
    ap.add_argument('--dtw-subsample', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--bootstrap', type=int, default=200, help='Bootstrap repetitions for CI estimates')
    ap.add_argument('--ci', type=float, default=0.95, help='Confidence interval level for bootstrapped metrics')
    return ap.parse_args()

def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) if yaml is not None else {}


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f'Manifest not found: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_or_extract_real(args, items, encoder, device, batch: int) -> np.ndarray:
    if Path(args.real_cache).exists():
        feats = np.load(args.real_cache)
        print(f'Loaded cached real features: {feats.shape}')
        return feats
    real_paths = extract_real_windows(items, max_per_class=2000)
    real_data = load_windows(real_paths)
    if real_data.shape[0] == 0:
        raise RuntimeError('No real windows loaded.')
    feats = encode_batches(real_data, encoder, device, batch)
    np.save(args.real_cache, feats)
    print(f'Extracted real features: {feats.shape}')
    return feats


def get_or_extract_synth(args, cfg, encoder, device, batch: int) -> np.ndarray:
    if args.synth_cache and Path(args.synth_cache).exists():
        feats = np.load(args.synth_cache)
        print(f'Loaded cached synthetic features: {feats.shape}')
        return feats
    if args.ckpt is None:
        raise RuntimeError('Provide --ckpt or --synth-cache for synthetic evaluation.')
    from src.eval.generate import generate_samples
    synth = generate_samples(cfg, args.ckpt, device, args.model_kind, n=args.n)
    feats = encode_batches(np.asarray(synth), encoder, device, batch)
    if args.synth_cache:
        np.save(args.synth_cache, feats)
    print(f'Extracted synthetic features: {feats.shape}')
    return feats


def _bootstrap_stat(x: np.ndarray, y: np.ndarray, func, B: int, rng: np.random.Generator):
    vals = []
    m, n = x.shape[0], y.shape[0]
    for _ in range(B):
        xi = x[rng.integers(0, m, size=m)]
        yi = y[rng.integers(0, n, size=n)]
        vals.append(func(xi, yi))
    return np.asarray(vals)


def _ci_interval(samples: np.ndarray, level: float):
    lo = ((1-level)/2)*100
    hi = (1-(1-level)/2)*100
    return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))


def _one_nn_accuracy(real_feats: np.ndarray, synth_feats: np.ndarray) -> float:
    from sklearn.metrics import accuracy_score
    X = np.concatenate([real_feats, synth_feats], axis=0)
    y = np.array([0]*len(real_feats) + [1]*len(synth_feats))
    d2 = ((X[:,None,:]-X[None,:,:])**2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn_idx = d2.argmin(axis=1)
    y_pred = y[nn_idx]
    return float(accuracy_score(y, y_pred))


def _c2st_logistic(real_feats: np.ndarray, synth_feats: np.ndarray, seed: int) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    rng = np.random.default_rng(seed)
    X = np.concatenate([real_feats, synth_feats], axis=0)
    y = np.array([0]*len(real_feats) + [1]*len(synth_feats))
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    split = int(0.7*len(X))
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)
    return float(accuracy_score(yte, yhat))


def compute_metrics(real_feats: np.ndarray, synth_feats: np.ndarray, seed: int, bootstrap: int, ci_level: float) -> dict:
    mu_r, mu_s = real_feats.mean(0), synth_feats.mean(0)
    cov_r = np.cov(real_feats, rowvar=False)
    cov_s = np.cov(synth_feats, rowvar=False)
    fid = frechet_distance(mu_r, mu_s, cov_r, cov_s)
    mmd = mmd_rbf(real_feats, synth_feats)
    prec, rec = prd_precision_recall(real_feats, synth_feats)
    f_beta = (1+1.0**2) * prec * rec / (1.0**2 * prec + rec + 1e-8)
    rng = np.random.default_rng(seed)
    idx = rng.choice(synth_feats.shape[0], min(512, synth_feats.shape[0]), replace=False)
    sf = synth_feats[idx]
    sf_norm = sf / (np.linalg.norm(sf, axis=1, keepdims=True)+1e-8)
    cos = sf_norm @ sf_norm.T
    dproxy = 1 - (np.tril(cos, k=-1).sum() / ((cos.shape[0]*(cos.shape[0]-1))/2 + 1e-8))
    one_nn = _one_nn_accuracy(real_feats, synth_feats)
    c2st = _c2st_logistic(real_feats, synth_feats, seed)
    rng = np.random.default_rng(seed)
    if bootstrap > 0:
        mmd_samples = _bootstrap_stat(real_feats, synth_feats, lambda a,b: mmd_rbf(a,b), bootstrap, rng)
        mmd_lo, mmd_hi = _ci_interval(mmd_samples, ci_level)
    else:
        mmd_lo = mmd_hi = mmd
    return dict(fid=fid, mmd=mmd, mmd_ci_lo=mmd_lo, mmd_ci_hi=mmd_hi, prd_precision=prec, prd_recall=rec, prd_f1=f_beta, diversity_proxy=dproxy, one_nn=one_nn, c2st_acc=c2st)


def compute_dtw_baseline(items, subsample: int, seed: int) -> float:
    if subsample <= 0:
        return math.nan
    rng = np.random.default_rng(seed)
    rs_idx = rng.choice(len(items), size=min(subsample, len(items)), replace=False)
    dtw_vals: list[float] = []
    for j in range(len(rs_idx)-1):
        pa = items[rs_idx[j]]['path']; pb = items[rs_idx[j+1]]['path']
        try:
            wa = np.load(pa); wb = np.load(pb)
        except (OSError, ValueError):
            continue
        if wa.shape == wb.shape:
            dtw_vals.append(dtw_distance(wa, wb))
    return float(np.mean(dtw_vals)) if dtw_vals else math.nan


def write_outputs(args, metrics: dict, real_feats: np.ndarray, synth_feats: np.ndarray, mean_dtw: float):
    from pandas import DataFrame, read_csv, concat
    row = {
        'model_kind': args.model_kind,
        **metrics,
        'mean_dtw_real_baseline': mean_dtw,
        'n_real_feat': real_feats.shape[0],
        'n_synth_feat': synth_feats.shape[0]
    }
    out_dir = Path(args.out_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if Path(args.out_csv).exists():
        df = read_csv(args.out_csv)
        df = concat([df, DataFrame([row])], ignore_index=True)
    else:
        df = DataFrame([row])
    df.to_csv(args.out_csv, index=False)
    with open(args.out_tex, 'w', encoding='utf-8') as f:
        f.write('\n% Auto-generated feature metrics table\n')
        f.write('\\begin{tabular}{lrrrrrrrrrr}\\toprule\n')
        f.write('Model & FID & MMD & MMD CI$_{lo}$ & MMD CI$_{hi}$ & PRD$_p$ & PRD$_r$ & PRD$_{F1}$ & 1-NN & C2ST & DivProxy \\ \\midrule\n')
        f.write(f"{row['model_kind']} & {row['fid']:.3f} & {row['mmd']:.4f} & {row['mmd_ci_lo']:.4f} & {row['mmd_ci_hi']:.4f} & {row['prd_precision']:.3f} & {row['prd_recall']:.3f} & {row['prd_f1']:.3f} & {row['one_nn']:.3f} & {row['c2st_acc']:.3f} & {row['diversity_proxy']:.3f} \\ \\bottomrule\n")
        f.write('\\end{tabular}\n')
    print(f"Wrote metrics to {args.out_csv} and LaTeX table to {args.out_tex}")


def main():  # pragma: no cover orchestrator
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = load_config(args.config)
    items = load_manifest(Path('results/manifest.json'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_ch = cfg.get('channels', 8) if isinstance(cfg.get('channels'), int) else 8
    encoder = TinyFeatureNet(in_ch=in_ch).to(device); encoder.eval()
    real_feats = get_or_extract_real(args, items, encoder, device, args.batch)
    synth_feats = get_or_extract_synth(args, cfg, encoder, device, args.batch)
    metrics = compute_metrics(real_feats, synth_feats, args.seed, args.bootstrap, args.ci)
    mean_dtw = compute_dtw_baseline(items, args.dtw_subsample, args.seed)
    write_outputs(args, metrics, real_feats, synth_feats, mean_dtw)


def load_windows(paths):
    arrs: list[np.ndarray] = []
    for p in paths:
        try:
            x = np.load(p)
            if x.ndim == 2:  # (C,T)
                arrs.append(x)
        except (OSError, ValueError):
            continue
    return np.stack(arrs) if arrs else np.empty((0,8,250))


def encode_batches(data: np.ndarray, encoder: nn.Module, device: torch.device, batch: int) -> np.ndarray:
    feats: list[np.ndarray] = []
    tensor = torch.from_numpy(data).float().to(device)
    for i in range(0, tensor.shape[0], batch):
        with torch.no_grad():
            feats.append(encoder(tensor[i:i+batch]).cpu().numpy())
    return np.concatenate(feats) if feats else np.empty((0,128))


if __name__ == '__main__':  # pragma: no cover
    main()
