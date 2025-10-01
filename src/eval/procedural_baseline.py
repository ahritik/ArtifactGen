"""Procedural baseline artifact synthesizer.

Implements simple parametric generators for artifact-like EEG windows
as a sanity-check baseline versus learned models (WGAN-GP, DDPM).

Generates:
  - Eye (blink): low-frequency biphasic transient prominent in frontal channels
  - Muscle: band-limited high-frequency noise bursts (20-40 Hz)
  - Electrode: slow drift + sporadic spikes
  - Chewing: quasi-rhythmic bursts 2-3 Hz amplitude-modulated by jaw motion
  - Shiver: narrowband 6-10 Hz tremor-like oscillation with low amplitude

Each window is length T, channels C. Channel topographies are approximated
with static weights to mimic spatial emphasis (e.g., frontal vs central).

Outputs an array shaped (N, C, T) and corresponding label indices.

Use this to populate a "procedural" model_kind column in evaluation tables.
"""
from __future__ import annotations

from typing import Tuple, List
import numpy as np

ARTIFACT_CLASSES = ["muscle", "eye", "electrode", "chewing", "shiver"]


def _topography_weights(c: int, kind: str) -> np.ndarray:
    # simple deterministic spatial weighting (normalize to sum=1)
    rng = np.random.default_rng(hash(kind) & 0xFFFF)
    w = rng.uniform(0.2, 1.0, size=c)
    if kind == "eye":
        # emphasize first two frontal channels
        w[0:2] *= 2.0
    elif kind == "muscle":
        w[2:6] *= 1.5
    elif kind == "electrode":
        w *= 0.8
    elif kind == "chewing":
        w[1:4] *= 1.3
    elif kind == "shiver":
        w[4:] *= 1.4
    w = w / (w.sum() + 1e-8)
    return w


def gen_eye(c: int, t: int, rng: np.random.Generator) -> np.ndarray:
    signal = np.zeros((c, t))
    pos = rng.integers(int(0.2*t), int(0.8*t))
    width = rng.integers(int(0.03*t), int(0.06*t))
    pulse = np.concatenate([
        np.linspace(0, 1, width//2, endpoint=False),
        np.linspace(1, -0.6, width - width//2)
    ])
    weights = _topography_weights(c, "eye")[:, None]
    start = max(0, pos - width//2)
    end = min(t, start + len(pulse))
    signal[:, start:end] += weights[:, 0:1] * pulse[: end - start]
    signal += 0.05 * rng.standard_normal(signal.shape)
    return signal


def gen_muscle(c: int, t: int, rng: np.random.Generator) -> np.ndarray:
    signal = 0.05 * rng.standard_normal((c, t))
    n_bursts = rng.integers(1, 3)
    for _ in range(n_bursts):
        start = rng.integers(0, int(0.7*t))
        dur = rng.integers(int(0.1*t), int(0.3*t))
        dur = min(dur, t - start)
        burst = rng.standard_normal((c, dur))
        # Band-limit by simple cumulative sum diff (highpass-ish)
        burst = burst - burst.mean(axis=-1, keepdims=True)
        signal[:, start:start+dur] += 0.3 * burst
    return signal


def gen_electrode(c: int, t: int, rng: np.random.Generator) -> np.ndarray:
    drift = rng.normal(0, 0.002, size=(c,))
    base = np.cumsum(drift[:, None] + 0.01 * rng.standard_normal((c, t)), axis=1)
    # inject spikes
    for _ in range(rng.integers(1, 4)):
        ch = rng.integers(0, c)
        pos = rng.integers(0, t-5)
        amp = rng.uniform(0.5, 1.5)
        base[ch, pos:pos+5] += amp * np.hanning(5)
    return base


def gen_chewing(c: int, t: int, rng: np.random.Generator) -> np.ndarray:
    base_freq = rng.uniform(2.0, 3.5)  # Hz modulation
    sr = 250
    time = np.arange(t) / sr
    mod = 0.5 * (1 + np.sin(2 * np.pi * base_freq * time))
    carrier = np.sin(2 * np.pi * rng.uniform(6, 10) * time)
    signal = (mod * carrier)[None, :] * _topography_weights(c, "chewing")[:, None]
    signal += 0.05 * rng.standard_normal((c, t))
    return signal


def gen_shiver(c: int, t: int, rng: np.random.Generator) -> np.ndarray:
    sr = 250
    freq = rng.uniform(6.0, 10.0)
    time = np.arange(t) / sr
    base = np.sin(2 * np.pi * freq * time)[None, :] * _topography_weights(c, "shiver")[:, None]
    trem = base + 0.02 * rng.standard_normal((c, t))
    return trem


GEN_FUNCS = {
    "eye": gen_eye,
    "muscle": gen_muscle,
    "electrode": gen_electrode,
    "chewing": gen_chewing,
    "shiver": gen_shiver,
}


def generate_procedural(n: int, c: int = 8, t: int = 250, seed: int = 0) -> Tuple[np.ndarray, List[int]]:
    rng = np.random.default_rng(seed)
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for i in range(n):
        cls = ARTIFACT_CLASSES[i % len(ARTIFACT_CLASSES)]
        arr = GEN_FUNCS[cls](c, t, rng)
        xs.append(arr)
        ys.append(ARTIFACT_CLASSES.index(cls))
    return np.stack(xs), ys


if __name__ == "__main__":  # simple smoke test
    X, y = generate_procedural(10)
    print("Procedural batch:", X.shape, y)
