"""
Model evaluation: MAE, Pearson correlation, ICC(2,1).

Evaluates the trained form scorer on the held-out test split
for each exercise, both per-dimension and overall.

Results are printed to stdout and exported to evaluation/results_evaluation.csv.

Usage:
    python evaluation/evaluate_model.py              # phase 3 models
    python evaluation/evaluate_model.py --phase 1
    python evaluation/evaluate_model.py --exercise squat --phase 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EXERCISES, SCORE_DIMS
from data.dataset_loader import get_splits

MODELS_DIR  = Path(__file__).parent.parent / 'models' / 'saved'
RESULTS_DIR = Path(__file__).parent


# ── Metric helpers ────────────────────────────────────────────────────────────

def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) < 1e-6 or np.std(y_pred) < 1e-6:
        return 0.0
    r, _ = stats.pearsonr(y_true, y_pred)
    return float(r)


def _icc_21(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    ICC(2,1) — two-way random-effects, single measures, absolute agreement.

    Equivalent to the formula used in exercise science reliability studies.
    """
    n  = len(y_true)
    if n < 3:
        return 0.0

    grand_mean = (y_true.mean() + y_pred.mean()) / 2.0
    ss_between = n * ((y_true.mean() - grand_mean) ** 2 +
                      (y_pred.mean() - grand_mean) ** 2)
    ss_within  = np.sum((y_true - y_true.mean()) ** 2 +
                        (y_pred - y_pred.mean()) ** 2)
    ss_error   = np.sum((y_true - y_pred) ** 2) / 2.0

    ms_between = ss_between / (n - 1)
    ms_error   = ss_error   / (n - 1)

    denom = ms_between + ms_error
    if denom < 1e-10:
        return 1.0
    return float(np.clip((ms_between - ms_error) / denom, 0.0, 1.0))


# ── Per-exercise evaluation ───────────────────────────────────────────────────

def evaluate_exercise(
    exercise: str,
    model: tf.keras.Model,
    verbose: bool = True,
) -> dict:
    """
    Evaluate model on the test split for one exercise.

    Args:
        exercise : exercise name
        model    : loaded Keras model
        verbose  : print results to stdout

    Returns:
        dict: {dimension: {mae, rmse, pearson, icc}}
        All error metrics in [0, 100] scale.
    """
    _, _, test   = get_splits(exercise)
    X_test, y_test = test

    y_pred = model.predict(X_test, verbose=0)   # [0, 1]

    y_true_100 = y_test * 100.0
    y_pred_100 = y_pred * 100.0

    results = {}
    for i, dim in enumerate(SCORE_DIMS):
        t = y_true_100[:, i]
        p = y_pred_100[:, i]
        results[dim] = {
            'mae':     float(np.mean(np.abs(t - p))),
            'rmse':    float(np.sqrt(np.mean((t - p) ** 2))),
            'pearson': _pearson(t, p),
            'icc':     _icc_21(t, p),
        }

    if verbose:
        print(f'\n{exercise.upper()}  (n_test={len(X_test)})')
        print(f'{"Dimension":<12} {"MAE":>6} {"RMSE":>7} {"Pearson":>9} {"ICC":>7}')
        print('─' * 46)
        for dim, m in results.items():
            print(f'{dim:<12} {m["mae"]:>6.2f} {m["rmse"]:>7.2f} '
                  f'{m["pearson"]:>9.3f} {m["icc"]:>7.3f}')

    return results


# ── All-exercise evaluation ───────────────────────────────────────────────────

def evaluate_all(phase: int = 3, verbose: bool = True) -> pd.DataFrame:
    """
    Evaluate all exercises, export CSV, print summary.

    Args:
        phase   : which saved model phase to load (1, 2, or 3)
        verbose : print per-exercise tables

    Returns:
        DataFrame with columns:
          exercise, dimension, mae, rmse, pearson, icc
    """
    rows = []

    for exercise in EXERCISES:
        model_path = MODELS_DIR / f'{exercise}_phase{phase}_final.keras'
        if not model_path.exists():
            # Try best checkpoint as fallback
            model_path = MODELS_DIR / f'{exercise}_phase{phase}_best.keras'
        if not model_path.exists():
            print(f'[SKIP] {exercise}: model not found ({model_path.name})')
            continue

        model   = tf.keras.models.load_model(str(model_path))
        results = evaluate_exercise(exercise, model, verbose=verbose)

        for dim, metrics in results.items():
            rows.append({'exercise': exercise, 'dimension': dim, **metrics})

    if not rows:
        print('No models found. Train models first.')
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / 'results_evaluation.csv'
    df.to_csv(out_path, index=False)

    # ── Summary ────────────────────────────────────────────────────────────
    overall_df = df[df['dimension'] == 'overall']
    print(f'\n{"=" * 50}')
    print(f'Summary — Phase {phase} models (overall dimension)')
    print(f'{"=" * 50}')
    print(f'  Mean MAE:     {overall_df["mae"].mean():.2f}  (target < 8)')
    print(f'  Mean Pearson: {overall_df["pearson"].mean():.3f}  (target > 0.85)')
    print(f'  Mean ICC:     {overall_df["icc"].mean():.3f}')
    print(f'\nFull results → {out_path}')

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate form scorer model.')
    parser.add_argument('--exercise', choices=EXERCISES + ['all'], default='all')
    parser.add_argument('--phase',    type=int, choices=[1, 2, 3],  default=3)
    args = parser.parse_args()

    if args.exercise == 'all':
        evaluate_all(phase=args.phase)
    else:
        model_path = MODELS_DIR / f'{args.exercise}_phase{args.phase}_final.keras'
        if not model_path.exists():
            model_path = MODELS_DIR / f'{args.exercise}_phase{args.phase}_best.keras'
        if not model_path.exists():
            print(f'Model not found: {model_path}')
            sys.exit(1)
        model = tf.keras.models.load_model(str(model_path))
        evaluate_exercise(args.exercise, model, verbose=True)
