"""
Unified dataset loader for the form scorer training pipeline.

Handles both synthetic (generated) and real (expert-annotated) data.
Applies feature extraction (angles → angles + velocities) and
train / val / test splits with stratification by score quartile.

Key functions:
  compute_features(sequences)          — (N,90,9) → (N,90,18) model-ready
  load_synthetic(exercise)             — load generated .npy files
  load_real(exercise)                  — load expert-annotated .npy files
  get_splits(exercise, ...)            — stratified train/val/test split
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    N_FRAMES, N_ANGLES, N_FEATURES, EXERCISES, SCORE_DIMS,
    ANGLE_NORM, VEL_NORM,
)

# ── Data directories ──────────────────────────────────────────────────────────
_ROOT       = Path(__file__).parent
SYNTH_DIR   = _ROOT / 'generated'   # output of synthetic_generator.py
REAL_DIR    = _ROOT / 'real'        # output of annotation_tool.py


# ── Feature extraction ────────────────────────────────────────────────────────

def compute_features(sequences: np.ndarray) -> np.ndarray:
    """
    Convert raw angle sequences to model input features.

    Matches FeatureExtractor.dart exactly:
      Channels  0–8 : angle / 180          → [0, 1]
      Channels 9–17 : frame-diff / 30      → [-1, 1]  (velocity)
      Frame 0 velocity is zero-padded.

    Args:
        sequences : (N, 90, 9) raw angle array in degrees

    Returns:
        features  : (N, 90, 18) float32, clipped to [-1, 1]
    """
    n = len(sequences)
    features = np.zeros((n, N_FRAMES, N_FEATURES), dtype=np.float32)

    # Normalised angles
    features[:, :, :N_ANGLES] = (sequences / ANGLE_NORM).astype(np.float32)

    # Frame-to-frame velocities (central difference approximation)
    velocities = np.diff(sequences, axis=1).astype(np.float32)  # (N, 89, 9)
    features[:, 1:, N_ANGLES:] = velocities / VEL_NORM
    # features[:, 0, N_ANGLES:] stays zero (boundary)

    return np.clip(features, -1.0, 1.0)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_synthetic(exercise: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load synthetic sequences and scores for one exercise.

    Returns:
        sequences : (N, 90, 9) float32 in degrees
        scores    : (N, 6) float32 in [0, 1]   ← normalised for model output
    """
    ex_dir = SYNTH_DIR / exercise
    seq_path   = ex_dir / 'sequences.npy'
    score_path = ex_dir / 'scores.npy'

    if not seq_path.exists():
        raise FileNotFoundError(
            f'Synthetic data not found for "{exercise}". '
            f'Run: python data/synthetic_generator.py --exercise {exercise}'
        )

    sequences = np.load(seq_path).astype(np.float32)
    scores    = np.load(score_path).astype(np.float32) / 100.0  # → [0, 1]
    return sequences, scores


def load_real(exercise: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Load real expert-annotated sequences and scores for one exercise.

    Returns (None, None) if no real data exists yet (pre-collection phase).

    Returns:
        sequences : (N, 90, 9) float32 in degrees  or None
        scores    : (N, 6) float32 in [0, 1]        or None
    """
    ex_dir     = REAL_DIR / exercise
    seq_path   = ex_dir / 'sequences.npy'
    score_path = ex_dir / 'scores.npy'

    if not seq_path.exists() or not score_path.exists():
        return None, None

    sequences = np.load(seq_path).astype(np.float32)
    scores    = np.load(score_path).astype(np.float32) / 100.0
    return sequences, scores


# ── Train / val / test splits ─────────────────────────────────────────────────

def get_splits(
    exercise: str,
    val_ratio: float  = 0.15,
    test_ratio: float = 0.15,
    include_real: bool = True,
    augment: bool = False,
    augment_factor: int = 2,
    seed: int = 42,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """
    Load data, optionally combine synthetic + real, then split.

    Stratification is by overall score quartile to ensure all quality
    tiers appear in train / val / test.

    Args:
        exercise       : one of EXERCISES
        val_ratio      : fraction held out for validation
        test_ratio     : fraction held out for final test
        include_real   : merge real annotated data when available
        augment        : apply augmentation to training set only
        augment_factor : number of augmented copies per original (if augment)
        seed           : random seed for reproducibility

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
        X* : (N, 90, 18) float32  — features ready for model input
        y* : (N, 6) float32       — scores in [0, 1]
    """
    # --- Load data ---
    seqs, scores = load_synthetic(exercise)

    if include_real:
        real_seqs, real_scores = load_real(exercise)
        if real_seqs is not None:
            seqs   = np.concatenate([seqs,   real_seqs],   axis=0)
            scores = np.concatenate([scores, real_scores], axis=0)

    # --- Feature extraction ---
    features = compute_features(seqs)

    # --- Stratified split by overall score quartile ---
    overall   = scores[:, 5]  # overall is last column
    quartiles = np.digitize(overall, bins=[0.25, 0.50, 0.75])

    X_train, X_temp, y_train, y_temp = train_test_split(
        features, scores,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        stratify=quartiles,
    )

    temp_quartiles = np.digitize(y_temp[:, 5], bins=[0.25, 0.50, 0.75])
    relative_test  = test_ratio / (val_ratio + test_ratio)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test,
        random_state=seed,
        stratify=temp_quartiles,
    )

    # --- Optional augmentation on training set only ---
    if augment:
        from data.augmentation import augment_batch
        # Augmentation works on raw sequences; rebuild features after augment
        raw_train = X_train[:, :, :9] * 180.0  # reverse angle normalisation
        aug_raw, y_train = augment_batch(raw_train, y_train,
                                         factor=augment_factor, seed=seed)
        X_train = compute_features(aug_raw)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def dataset_info(exercise: str) -> None:
    """Print a summary of available data for one exercise."""
    try:
        seqs, scores = load_synthetic(exercise)
        print(f'{exercise}  synthetic: {len(seqs):>5} samples  '
              f'overall={scores[:,5].mean()*100:.1f}±{scores[:,5].std()*100:.1f}')
    except FileNotFoundError:
        print(f'{exercise}  synthetic: NOT FOUND')

    real_seqs, _ = load_real(exercise)
    if real_seqs is not None:
        print(f'{exercise}  real:      {len(real_seqs):>5} samples')
    else:
        print(f'{exercise}  real:      (none yet)')


if __name__ == '__main__':
    print('Dataset summary\n' + '-' * 50)
    for ex in EXERCISES:
        dataset_info(ex)
