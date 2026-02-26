"""
Three-way comparison: Rule-based vs DL-only vs Hybrid scoring.

Hybrid blend mirrors ScoringOrchestrator.dart exactly:
  hybrid = 0.30 × rule_score + 0.70 × dl_score

Rule-based formulas mirror rule_based_scorer.dart exactly so the
comparison is apples-to-apples with the Flutter production code.

Results are exported to evaluation/results_comparison.csv and printed
as a readable table.

Usage:
    python evaluation/compare_approaches.py              # phase 3 models
    python evaluation/compare_approaches.py --phase 1
"""

import sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    EXERCISES, SCORE_DIMS, BILATERAL_PAIRS,
    IDX_LEFT_HIP, IDX_RIGHT_HIP, IDX_SPINE,
    STABILITY_DENOM, SYMMETRY_DENOM, TEMPO_GOOD_VEL, TEMPO_BAD_VEL,
)
from data.dataset_loader import get_splits
from data.synthetic_generator import EXERCISE_PROFILES

MODELS_DIR  = Path(__file__).parent.parent / 'models' / 'saved'
RESULTS_DIR = Path(__file__).parent

RULE_WEIGHT   = 0.30   # matches ScoringOrchestrator.dart
HYBRID_WEIGHT = 1.0 - RULE_WEIGHT


# ── Rule-based scorer (matches rule_based_scorer.dart) ────────────────────────

def rule_based_score(sequence: np.ndarray, exercise: str) -> np.ndarray:
    """
    Apply rule-based scoring to a single (90, 9) angle sequence.

    Implements the exact same formulas as rule_based_scorer.dart:
      depth      = actualROM / idealROM × 100
      stability  = (1 − velocityVariance / 25) × 100
      symmetry   = (1 − avgBilateralDiff / 20) × 100
      tempo      = linear scale: ≤5 °/frame → 100, ≥25 °/frame → 0
      alignment  = (1 − errorFrameCount / totalFrames) × 100
      overall    = mean of all five

    Returns:
        (6,) float32 array in [0, 100]
    """
    profile     = EXERCISE_PROFILES[exercise]
    primary_idx = profile['primary_joint']
    ideal_rom   = float(profile['ideal_rom'])

    # Depth
    actual_rom = abs(float(sequence[:, primary_idx].min()) -
                     float(sequence[0, primary_idx]))
    depth = float(np.clip(actual_rom / max(ideal_rom, 1.0) * 100.0, 0.0, 100.0))

    # Stability
    velocities = np.diff(sequence, axis=0)
    vel_var    = float(np.var(np.abs(velocities)))
    stability  = float(np.clip((1.0 - vel_var / STABILITY_DENOM) * 100.0, 0.0, 100.0))

    # Symmetry
    diffs = np.concatenate([
        np.abs(sequence[:, l] - sequence[:, r]) for l, r in BILATERAL_PAIRS
    ])
    avg_bilateral = float(diffs.mean())
    symmetry = float(np.clip((1.0 - avg_bilateral / SYMMETRY_DENOM) * 100.0,
                             0.0, 100.0))

    # Tempo
    peak_vel = float(np.abs(velocities).max())
    if peak_vel <= TEMPO_GOOD_VEL:
        tempo = 100.0
    elif peak_vel >= TEMPO_BAD_VEL:
        tempo = 0.0
    else:
        tempo = float((1.0 - (peak_vel - TEMPO_GOOD_VEL) /
                       (TEMPO_BAD_VEL - TEMPO_GOOD_VEL)) * 100.0)

    # Alignment (exercise-specific posture check)
    spine = sequence[:, IDX_SPINE]
    if exercise in ('squat', 'lunge', 'bicep_curl'):
        errors = int((spine > 35.0).sum())
    elif exercise == 'pushup':
        errors = int((spine > 20.0).sum())
    else:  # plank
        errors = int(
            ((sequence[:, IDX_LEFT_HIP]  < 150.0) |
             (sequence[:, IDX_RIGHT_HIP] < 150.0)).sum()
        )
    alignment = float(np.clip((1.0 - errors / len(sequence)) * 100.0, 0.0, 100.0))

    overall = (depth + stability + symmetry + tempo + alignment) / 5.0
    return np.array([depth, stability, symmetry, tempo, alignment, overall],
                    dtype=np.float32)


def rule_based_batch(sequences: np.ndarray, exercise: str) -> np.ndarray:
    """Apply rule_based_score to a batch of sequences."""
    return np.stack([rule_based_score(sequences[i], exercise)
                     for i in range(len(sequences))])


# ── Hybrid blend ──────────────────────────────────────────────────────────────

def hybrid_score(rule: np.ndarray, dl: np.ndarray) -> np.ndarray:
    """30% rule + 70% DL — matches ScoringOrchestrator.dart."""
    return RULE_WEIGHT * rule + HYBRID_WEIGHT * dl


# ── Per-exercise comparison ───────────────────────────────────────────────────

def _mean_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rs = []
    for i in range(y_true.shape[1]):
        if np.std(y_true[:, i]) > 1e-6 and np.std(y_pred[:, i]) > 1e-6:
            r, _ = stats.pearsonr(y_true[:, i], y_pred[:, i])
            rs.append(float(r))
    return float(np.mean(rs)) if rs else 0.0


def compare_exercise(
    exercise: str,
    model: tf.keras.Model,
    verbose: bool = True,
) -> dict:
    """
    Run 3-way comparison for one exercise on the held-out test split.

    Returns:
        dict with keys: exercise, rule_mae, rule_pearson,
                        dl_mae, dl_pearson, hybrid_mae, hybrid_pearson
    """
    _, _, test   = get_splits(exercise)
    X_test, y_test = test

    # Reconstruct raw angle sequences from normalised features
    raw_sequences = X_test[:, :, :9] * 180.0   # (N, 90, 9)
    y_true_100    = y_test * 100.0

    # 1. Rule-based
    y_rule = rule_based_batch(raw_sequences, exercise)

    # 2. DL only
    y_dl = model.predict(X_test, verbose=0) * 100.0

    # 3. Hybrid
    y_hybrid = hybrid_score(y_rule, y_dl)

    mae_r, r_r = float(np.mean(np.abs(y_true_100 - y_rule))),   _mean_pearson(y_true_100, y_rule)
    mae_d, r_d = float(np.mean(np.abs(y_true_100 - y_dl))),     _mean_pearson(y_true_100, y_dl)
    mae_h, r_h = float(np.mean(np.abs(y_true_100 - y_hybrid))), _mean_pearson(y_true_100, y_hybrid)

    if verbose:
        print(f'\n{exercise.upper()}  (n_test={len(X_test)})')
        print(f'{"Approach":<14} {"MAE":>6} {"Pearson":>9}')
        print('─' * 34)
        print(f'{"Rule-based":<14} {mae_r:>6.2f} {r_r:>9.3f}')
        print(f'{"DL only":<14} {mae_d:>6.2f} {r_d:>9.3f}')
        print(f'{"Hybrid (30/70)":<14} {mae_h:>6.2f} {r_h:>9.3f}')

    return {
        'exercise':      exercise,
        'rule_mae':      mae_r, 'rule_pearson':   r_r,
        'dl_mae':        mae_d, 'dl_pearson':     r_d,
        'hybrid_mae':    mae_h, 'hybrid_pearson': r_h,
    }


# ── All-exercise comparison ───────────────────────────────────────────────────

def compare_all(phase: int = 3) -> pd.DataFrame:
    rows = []

    for exercise in EXERCISES:
        model_path = MODELS_DIR / f'{exercise}_phase{phase}_final.keras'
        if not model_path.exists():
            model_path = MODELS_DIR / f'{exercise}_phase{phase}_best.keras'
        if not model_path.exists():
            print(f'[SKIP] {exercise}: model not found')
            continue

        model = tf.keras.models.load_model(str(model_path))
        rows.append(compare_exercise(exercise, model, verbose=True))

    if not rows:
        print('No models found.')
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / 'results_comparison.csv'
    df.to_csv(out_path, index=False)

    # ── Summary table ──────────────────────────────────────────────────────
    print(f'\n{"=" * 55}')
    print('Overall Summary (mean across all exercises)')
    print(f'{"=" * 55}')
    print(f'{"Approach":<16} {"Mean MAE":>9} {"Mean Pearson":>13}')
    print('─' * 42)
    for label, mae_col, r_col in [
        ('Rule-based',     'rule_mae',   'rule_pearson'),
        ('DL only',        'dl_mae',     'dl_pearson'),
        ('Hybrid (30/70)', 'hybrid_mae', 'hybrid_pearson'),
    ]:
        print(f'{label:<16} {df[mae_col].mean():>9.2f} {df[r_col].mean():>13.3f}')

    print(f'\nResults → {out_path}')
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare rule-based vs DL vs hybrid scoring.')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3], default=3)
    args = parser.parse_args()
    compare_all(phase=args.phase)
