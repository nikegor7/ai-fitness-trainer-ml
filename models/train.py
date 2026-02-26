"""
Three-phase training pipeline for the exercise form scorer.

Phase 1 — Pre-train on synthetic data
  25 000 samples (5 exercises × 5 000), 50 epochs, lr = 1e-3
  Target: val MAE < 15 on [0, 100] scale

Phase 2 — Fine-tune on real data  (requires expert-annotated data)
  Freeze Conv1D layers, lr = 1e-4, 30 epochs
  Target: val MAE < 10

Phase 3 — Full fine-tune on combined data
  Unfreeze all layers, lr = 1e-5, 20 epochs
  Target: val MAE < 8, Pearson > 0.85

Usage:
    python models/train.py --exercise squat --phase 1
    python models/train.py --exercise all   --phase 1
    python models/train.py --exercise squat --phase 2
    python models/train.py --exercise squat --phase 3
    python models/train.py --exercise squat --phase 1 --epochs 100 --batch 32
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import EXERCISES, SCORE_DIMS
from models.form_scorer_model import (
    build_form_scorer,
    compile_model,
    freeze_conv_layers,
    unfreeze_all_layers,
)
from data.dataset_loader import get_splits

# ── Output directories ────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent / 'saved'
LOGS_DIR   = Path(__file__).parent / 'logs'


# ── Callbacks ─────────────────────────────────────────────────────────────────

def get_callbacks(exercise: str, phase: int) -> list:
    run_name = f'{exercise}_phase{phase}'
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(MODELS_DIR / f'{run_name}_best.keras'),
            monitor='val_mae',
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_mae',
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(LOGS_DIR / run_name),
            histogram_freq=0,
        ),
        tf.keras.callbacks.CSVLogger(
            str(LOGS_DIR / f'{run_name}_history.csv'),
        ),
    ]


# ── Phase 1 — pre-train on synthetic data ────────────────────────────────────

def phase1_pretrain(
    exercise: str,
    epochs: int     = 50,
    batch_size: int = 64,
) -> tf.keras.Model:
    print(f'\n{"=" * 60}')
    print(f'Phase 1 — Pre-train  |  {exercise}')
    print(f'{"=" * 60}')

    train, val, _ = get_splits(exercise, include_real=False, augment=True)
    X_train, y_train = train
    X_val,   y_val   = val
    print(f'Train: {len(X_train):,}   Val: {len(X_val):,}')

    model = build_form_scorer()
    model = compile_model(model, learning_rate=1e-3)
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(exercise, phase=1),
        verbose=1,
    )

    save_path = MODELS_DIR / f'{exercise}_phase1_final.keras'
    model.save(str(save_path))
    best_mae = min(history.history['val_mae']) * 100
    print(f'\nSaved → {save_path}')
    print(f'Best val MAE: {best_mae:.2f}  (target < 15)')
    return model


# ── Phase 2 — fine-tune on real data ─────────────────────────────────────────

def phase2_finetune(
    exercise: str,
    epochs: int     = 30,
    batch_size: int = 32,
) -> tf.keras.Model:
    print(f'\n{"=" * 60}')
    print(f'Phase 2 — Fine-tune  |  {exercise}')
    print(f'{"=" * 60}')

    model_path = MODELS_DIR / f'{exercise}_phase1_final.keras'
    if not model_path.exists():
        raise FileNotFoundError(
            f'Phase 1 model not found. Run phase 1 first.\n  {model_path}'
        )

    train, val, _ = get_splits(exercise, include_real=True)
    X_train, y_train = train
    X_val,   y_val   = val
    print(f'Train: {len(X_train):,}   Val: {len(X_val):,}')

    model = tf.keras.models.load_model(str(model_path))
    freeze_conv_layers(model)
    model = compile_model(model, learning_rate=1e-4)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(exercise, phase=2),
        verbose=1,
    )

    save_path = MODELS_DIR / f'{exercise}_phase2_final.keras'
    model.save(str(save_path))
    best_mae = min(history.history['val_mae']) * 100
    print(f'\nSaved → {save_path}')
    print(f'Best val MAE: {best_mae:.2f}  (target < 10)')
    return model


# ── Phase 3 — full fine-tune on combined data ─────────────────────────────────

def phase3_full_finetune(
    exercise: str,
    epochs: int     = 20,
    batch_size: int = 32,
) -> tf.keras.Model:
    print(f'\n{"=" * 60}')
    print(f'Phase 3 — Full fine-tune  |  {exercise}')
    print(f'{"=" * 60}')

    # Prefer phase 2 checkpoint; fall back to phase 1
    for phase_label in (2, 1):
        model_path = MODELS_DIR / f'{exercise}_phase{phase_label}_final.keras'
        if model_path.exists():
            break
    else:
        raise FileNotFoundError('No phase 1 or 2 model found. Run earlier phases first.')

    train, val, _ = get_splits(exercise, include_real=True)
    X_train, y_train = train
    X_val,   y_val   = val
    print(f'Train: {len(X_train):,}   Val: {len(X_val):,}')

    model = tf.keras.models.load_model(str(model_path))
    unfreeze_all_layers(model)
    model = compile_model(model, learning_rate=1e-5)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(exercise, phase=3),
        verbose=1,
    )

    save_path = MODELS_DIR / f'{exercise}_phase3_final.keras'
    model.save(str(save_path))
    best_mae = min(history.history['val_mae']) * 100
    print(f'\nSaved → {save_path}')
    print(f'Best val MAE: {best_mae:.2f}  (target < 8)')
    return model


# ── Multi-exercise runner ─────────────────────────────────────────────────────

def train_all(phase: int, **kwargs) -> None:
    """Train all exercises sequentially for the given phase."""
    dispatch = {1: phase1_pretrain, 2: phase2_finetune, 3: phase3_full_finetune}
    fn = dispatch[phase]
    for ex in EXERCISES:
        try:
            fn(ex, **kwargs)
        except FileNotFoundError as e:
            print(f'[SKIP] {ex}: {e}')


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train form scorer model.')
    parser.add_argument('--exercise', choices=EXERCISES + ['all'], default='all')
    parser.add_argument('--phase',    type=int, choices=[1, 2, 3],  default=1)
    parser.add_argument('--epochs',   type=int, default=None,
                        help='Override default epoch count for the phase.')
    parser.add_argument('--batch',    type=int, default=None,
                        help='Override default batch size.')
    args = parser.parse_args()

    # Build kwargs, only passing overrides when specified
    kwargs = {}
    if args.epochs is not None:
        kwargs['epochs']     = args.epochs
    if args.batch is not None:
        kwargs['batch_size'] = args.batch

    dispatch = {1: phase1_pretrain, 2: phase2_finetune, 3: phase3_full_finetune}
    fn = dispatch[args.phase]

    if args.exercise == 'all':
        train_all(args.phase, **kwargs)
    else:
        fn(args.exercise, **kwargs)
