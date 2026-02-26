"""
Data augmentation for rep angle sequences.

All public functions accept/return np.ndarray of shape (90, 9) — a single rep.
`augment_batch` operates on a full dataset (N, 90, 9).

Augmentations preserve the physical plausibility of the angle sequences
and keep scores approximately stable (so the augmented copy reuses the
original GT scores).

Available transforms:
  time_warp        — smooth random time-axis distortion
  add_noise        — Gaussian noise on angle values
  amplitude_scale  — scale deviations from channel mean (ROM scaling)
  mirror           — swap left-right bilateral pairs
  augment_batch    — apply random combinations to a full dataset
"""

import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import N_FRAMES, N_ANGLES, BILATERAL_PAIRS


# ── Individual transforms ─────────────────────────────────────────────────────

def time_warp(sequence: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """
    Random time warping: compress/expand different time segments.

    Generates a smooth displacement field via cubic spline through
    a small set of randomly displaced knots, then resamples the sequence.

    Args:
        sequence : (90, 9) angle array in degrees
        sigma    : warp strength as fraction of total length (default 0.05)

    Returns:
        Warped sequence, same shape.
    """
    n = len(sequence)
    n_knots = 5

    knot_x = np.linspace(0, n - 1, n_knots)
    knot_y = knot_x + np.random.normal(0.0, sigma * n, n_knots)
    # Anchor endpoints so the warp doesn't shift the sequence boundaries
    knot_y[0]  = 0.0
    knot_y[-1] = float(n - 1)
    # Ensure monotonically increasing (soft clip)
    knot_y = np.maximum.accumulate(knot_y)
    knot_y = np.clip(knot_y, 0.0, float(n - 1))

    warp_fn  = interp1d(knot_x, knot_y, kind='cubic')
    warped_x = warp_fn(np.arange(n))

    result = np.zeros_like(sequence)
    for j in range(sequence.shape[1]):
        fn = interp1d(np.arange(n), sequence[:, j], kind='linear',
                      bounds_error=False, fill_value='extrapolate')
        result[:, j] = np.clip(fn(warped_x), 0.0, 180.0)

    return result.astype(np.float32)


def add_noise(sequence: np.ndarray, noise_std: float = 0.5) -> np.ndarray:
    """
    Add independent Gaussian noise to each angle channel.

    Simulates sensor jitter and minor landmark detection noise.

    Args:
        sequence  : (90, 9) angle array in degrees
        noise_std : standard deviation of noise in degrees (default 0.5)

    Returns:
        Noisy sequence clipped to [0, 180].
    """
    noise = np.random.normal(0.0, noise_std, sequence.shape).astype(np.float32)
    return np.clip(sequence + noise, 0.0, 180.0)


def amplitude_scale(
    sequence: np.ndarray,
    scale_range: tuple[float, float] = (0.85, 1.15),
) -> np.ndarray:
    """
    Scale angle deviations from each channel's mean.

    Preserves the resting position but stretches or compresses the ROM.
    A scale > 1 increases movement range; < 1 reduces it.

    Args:
        sequence    : (90, 9) angle array in degrees
        scale_range : (min, max) uniform sample range for the scale factor

    Returns:
        Scaled sequence clipped to [0, 180].
    """
    scale = float(np.random.uniform(*scale_range))
    mean  = sequence.mean(axis=0, keepdims=True)
    return np.clip(mean + scale * (sequence - mean), 0.0, 180.0).astype(np.float32)


def mirror(sequence: np.ndarray) -> np.ndarray:
    """
    Mirror left-right: swap bilateral angle pairs.

    Bilateral pairs (left_idx, right_idx):
      (0, 1) — knees
      (2, 3) — hips
      (4, 5) — elbows
      (6, 7) — shoulders
    Spine (channel 8) is unchanged.

    Useful for generating left-leg-forward lunge variants, etc.

    Args:
        sequence : (90, 9) angle array in degrees

    Returns:
        Mirrored copy.
    """
    mirrored = sequence.copy()
    for l_idx, r_idx in BILATERAL_PAIRS:
        mirrored[:, l_idx] = sequence[:, r_idx]
        mirrored[:, r_idx] = sequence[:, l_idx]
    return mirrored


# ── Batch augmentation ────────────────────────────────────────────────────────

def _augment_one(sequence: np.ndarray) -> np.ndarray:
    """Apply a random subset of transforms to a single sequence."""
    seq = sequence.copy()
    r   = np.random.random

    if r() < 0.5:
        seq = time_warp(seq, sigma=np.random.uniform(0.02, 0.07))
    if r() < 0.6:
        seq = add_noise(seq, noise_std=np.random.uniform(0.2, 1.2))
    if r() < 0.35:
        seq = amplitude_scale(seq, scale_range=(0.88, 1.12))
    if r() < 0.35:
        seq = mirror(seq)

    return seq


def augment_batch(
    sequences: np.ndarray,
    scores: np.ndarray,
    factor: int = 2,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Augment a full dataset, returning original + augmented copies.

    GT scores are reused unchanged for augmented copies because the
    transforms are label-preserving by design (small perturbations,
    mirror-symmetric scoring).

    Args:
        sequences : (N, 90, 9) original sequences
        scores    : (N, 6) GT scores in [0, 100]
        factor    : number of augmented copies added per original sample
        seed      : optional random seed for reproducibility

    Returns:
        aug_sequences : (N * (1 + factor), 90, 9)
        aug_scores    : (N * (1 + factor), 6)
    """
    if seed is not None:
        np.random.seed(seed)

    all_seqs   = [sequences]
    all_scores = [scores]

    for _ in range(factor):
        aug = np.stack([_augment_one(sequences[i]) for i in range(len(sequences))])
        all_seqs.append(aug)
        all_scores.append(scores)

    return (
        np.concatenate(all_seqs,   axis=0),
        np.concatenate(all_scores, axis=0),
    )
