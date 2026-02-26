"""
Form scorer model: Conv1D + Bidirectional LSTM.

Architecture
────────────
Input  : (batch, 90, 18)  — 90 frames × 18 features (9 angles + 9 velocities)
Output : (batch, 6)       — sigmoid [0, 1]
                            multiply × 100 in Dart → RepScore dimensions

Layers
  Conv1D(64, k=5) → Conv1D(64, k=3) → MaxPool(2) → Dropout(0.3)
  → BiLSTM(64, seq=True) → BiLSTM(64, seq=False)
  → Dense(128) → Dropout(0.3) → Dense(6, sigmoid)

Parameters: ~200 K trainable

Score dimensions (order matches SCORE_DIMS in config.py and RepScore.dart):
  0 depth  1 stability  2 symmetry  3 tempo  4 alignment  5 overall
"""

import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Input, Model, layers

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import N_FRAMES, N_FEATURES, N_OUTPUTS, SCORE_DIMS


def build_form_scorer(
    input_shape: tuple[int, int] = (N_FRAMES, N_FEATURES),
    n_outputs: int = N_OUTPUTS,
) -> Model:
    """
    Build and return the uncompiled form scorer model.

    Args:
        input_shape : (frames, features) — default (90, 18)
        n_outputs   : number of score dimensions — default 6

    Returns:
        Keras Model (uncompiled).
    """
    inp = Input(shape=input_shape, name='angle_sequence')

    # ── Local temporal feature extraction ─────────────────────────────────
    x = layers.Conv1D(64, kernel_size=5, activation='relu',
                      padding='same', name='conv1')(inp)
    x = layers.Conv1D(64, kernel_size=3, activation='relu',
                      padding='same', name='conv2')(x)
    x = layers.MaxPooling1D(pool_size=2, name='pool')(x)   # 90 → 45 frames
    x = layers.Dropout(0.3, name='drop1')(x)

    # ── Sequence modelling ────────────────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True), name='bilstm1')(x)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False), name='bilstm2')(x)

    # ── Classifier head ───────────────────────────────────────────────────
    x   = layers.Dense(128, activation='relu', name='dense1')(x)
    x   = layers.Dropout(0.3, name='drop2')(x)
    out = layers.Dense(n_outputs, activation='sigmoid', name='scores')(x)

    return Model(inputs=inp, outputs=out, name='form_scorer')


def compile_model(model: Model, learning_rate: float = 1e-3) -> Model:
    """
    Compile with Adam + MSE loss + MAE metric.

    MSE on [0,1] outputs is equivalent to MSE on [0,100] scaled by 1/10000,
    so the model optimises the same objective as MAE in 0-100 space.

    Args:
        model         : uncompiled Keras Model
        learning_rate : Adam learning rate

    Returns:
        Compiled model (in-place modification + return).
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')],
    )
    return model


def freeze_conv_layers(model: Model) -> None:
    """
    Freeze Conv1D layers for Phase 2 fine-tuning.

    Call before re-compiling the model to keep early features fixed
    while adapting the LSTM + Dense layers to real data.
    """
    for layer in model.layers:
        if layer.name.startswith('conv'):
            layer.trainable = False


def unfreeze_all_layers(model: Model) -> None:
    """Unfreeze all layers for Phase 3 full fine-tuning."""
    for layer in model.layers:
        layer.trainable = True


if __name__ == '__main__':
    model = build_form_scorer()
    model.summary()
    total = model.count_params()
    print(f'\nTotal parameters: {total:,}')
    print(f'Score dimensions: {SCORE_DIMS}')
