"""
Prediction visualisations for thesis figures.

Generates and saves:
  scatter_<exercise>.png       — predicted vs actual for all 6 dimensions
  error_dist_<exercise>.png    — prediction error histograms
  hybrid_comparison.png        — bar chart: rule vs DL vs hybrid MAE & Pearson

All figures are saved to evaluation/figures/.

Usage:
    python evaluation/visualize_predictions.py              # phase 3
    python evaluation/visualize_predictions.py --phase 1
    python evaluation/visualize_predictions.py --exercise squat --phase 3
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # non-interactive backend for script execution
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EXERCISES, SCORE_DIMS
from data.dataset_loader import get_splits
from evaluation.compare_approaches import compare_all

MODELS_DIR  = Path(__file__).parent.parent / 'models' / 'saved'
FIGURES_DIR = Path(__file__).parent / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)


# ── Scatter: predicted vs actual ─────────────────────────────────────────────

def scatter_predicted_vs_actual(
    exercise: str,
    model: tf.keras.Model,
    save: bool = True,
) -> plt.Figure:
    """
    6-panel scatter plot of predicted vs actual scores for each dimension.

    Args:
        exercise : exercise name
        model    : loaded Keras model
        save     : save PNG to figures/

    Returns:
        matplotlib Figure
    """
    _, _, test   = get_splits(exercise)
    X_test, y_test = test

    y_pred = model.predict(X_test, verbose=0) * 100.0
    y_true = y_test * 100.0

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        f'Predicted vs Actual — {exercise.replace("_", " ").title()}',
        fontsize=14, fontweight='bold',
    )

    for idx, (ax, dim) in enumerate(zip(axes.flat, SCORE_DIMS)):
        t = y_true[:, idx]
        p = y_pred[:, idx]

        ax.scatter(t, p, alpha=0.25, s=8, color='steelblue', rasterized=True)
        ax.plot([0, 100], [0, 100], 'r--', linewidth=1.2, label='Perfect')

        r, _  = stats.pearsonr(t, p)
        mae   = np.mean(np.abs(t - p))

        ax.set_title(f'{dim}   r={r:.3f}  MAE={mae:.1f}', fontsize=10)
        ax.set_xlabel('Actual', fontsize=9)
        ax.set_ylabel('Predicted', fontsize=9)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.tick_params(labelsize=8)

    plt.tight_layout()

    if save:
        path = FIGURES_DIR / f'scatter_{exercise}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved: {path}')

    return fig


# ── Error distribution histograms ────────────────────────────────────────────

def error_distribution(
    exercise: str,
    model: tf.keras.Model,
    save: bool = True,
) -> plt.Figure:
    """
    6-panel histogram of prediction errors (predicted − actual) per dimension.

    Args:
        exercise : exercise name
        model    : loaded Keras model
        save     : save PNG to figures/

    Returns:
        matplotlib Figure
    """
    _, _, test   = get_splits(exercise)
    X_test, y_test = test

    y_pred  = model.predict(X_test, verbose=0) * 100.0
    errors  = y_pred - y_test * 100.0

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle(
        f'Prediction Error Distribution — {exercise.replace("_", " ").title()}',
        fontsize=14, fontweight='bold',
    )

    for idx, (ax, dim) in enumerate(zip(axes.flat, SCORE_DIMS)):
        e = errors[:, idx]
        ax.hist(e, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(0,        color='red',    linestyle='--', linewidth=1.2)
        ax.axvline(e.mean(), color='orange', linestyle=':',  linewidth=1.2,
                   label=f'bias={e.mean():.1f}')
        ax.set_title(f'{dim}   σ={e.std():.1f}  bias={e.mean():.1f}', fontsize=10)
        ax.set_xlabel('Error (pred − actual)', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.tick_params(labelsize=8)

    plt.tight_layout()

    if save:
        path = FIGURES_DIR / f'error_dist_{exercise}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved: {path}')

    return fig


# ── Hybrid comparison bar chart ───────────────────────────────────────────────

def hybrid_comparison_bar(phase: int = 3, save: bool = True) -> plt.Figure:
    """
    Side-by-side bar chart comparing Rule-based / DL-only / Hybrid
    on MAE and Pearson correlation across all exercises.

    Thesis key figure demonstrating hybrid superiority.

    Args:
        phase : which model phase to compare
        save  : save PNG to figures/

    Returns:
        matplotlib Figure
    """
    df = compare_all(phase=phase)
    if df.empty:
        print('No comparison data — run compare_approaches.py first.')
        return plt.figure()

    exercises = df['exercise'].tolist()
    x     = np.arange(len(exercises))
    width = 0.25

    colors = {
        'Rule-based': '#e07b54',
        'DL only':    '#5b9bd5',
        'Hybrid':     '#70ad47',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Scoring Approach Comparison — All Exercises',
                 fontsize=14, fontweight='bold')

    # MAE (lower = better)
    ax1.bar(x - width, df['rule_mae'],   width, label='Rule-based', color=colors['Rule-based'])
    ax1.bar(x,         df['dl_mae'],     width, label='DL only',    color=colors['DL only'])
    ax1.bar(x + width, df['hybrid_mae'], width, label='Hybrid',     color=colors['Hybrid'])
    ax1.axhline(8.0, color='black', linestyle='--', linewidth=1.2, label='Target MAE < 8')
    ax1.set_title('Mean Absolute Error (lower = better)')
    ax1.set_ylabel('MAE (0–100 scale)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([e.replace('_', '\n') for e in exercises])
    ax1.legend(fontsize=9)

    # Pearson (higher = better)
    ax2.bar(x - width, df['rule_pearson'],   width, label='Rule-based', color=colors['Rule-based'])
    ax2.bar(x,         df['dl_pearson'],     width, label='DL only',    color=colors['DL only'])
    ax2.bar(x + width, df['hybrid_pearson'], width, label='Hybrid',     color=colors['Hybrid'])
    ax2.axhline(0.85, color='black', linestyle='--', linewidth=1.2, label='Target r > 0.85')
    ax2.set_title('Pearson Correlation (higher = better)')
    ax2.set_ylabel('Pearson r')
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks(x)
    ax2.set_xticklabels([e.replace('_', '\n') for e in exercises])
    ax2.legend(fontsize=9)

    plt.tight_layout()

    if save:
        path = FIGURES_DIR / f'hybrid_comparison_phase{phase}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved: {path}')

    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate prediction visualisations.')
    parser.add_argument('--exercise', choices=EXERCISES + ['all'], default='all')
    parser.add_argument('--phase',    type=int, choices=[1, 2, 3],  default=3)
    args = parser.parse_args()

    exercises = EXERCISES if args.exercise == 'all' else [args.exercise]

    for exercise in exercises:
        model_path = MODELS_DIR / f'{exercise}_phase{args.phase}_final.keras'
        if not model_path.exists():
            model_path = MODELS_DIR / f'{exercise}_phase{args.phase}_best.keras'
        if not model_path.exists():
            print(f'[SKIP] {exercise}: model not found')
            continue

        model = tf.keras.models.load_model(str(model_path))
        scatter_predicted_vs_actual(exercise, model)
        error_distribution(exercise, model)

    if args.exercise == 'all':
        hybrid_comparison_bar(phase=args.phase)

    print('\nAll figures saved to evaluation/figures/')
