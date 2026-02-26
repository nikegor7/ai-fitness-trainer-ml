"""
Convert trained Keras models to float16 quantised TFLite.

Output: form_scorer_<exercise>_v1.tflite  (~500 KB per exercise)

The output files activate DLFormScorer in the Flutter app.
Copy them to:  ai_fitness_trainer/assets/models/

Usage:
    python export/convert_tflite.py --exercise squat
    python export/convert_tflite.py --all
    python export/convert_tflite.py --all --output-dir /path/to/flutter/assets/models
    python export/convert_tflite.py --all --phase 1   (use phase 1 models)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EXERCISES, N_FRAMES, N_FEATURES
from data.dataset_loader import get_splits

MODELS_DIR  = Path(__file__).parent.parent / 'models' / 'saved'
DEFAULT_OUT = Path(__file__).parent / 'tflite'


# ── Conversion ────────────────────────────────────────────────────────────────

def convert_one(
    exercise: str,
    model_path: Path,
    output_dir: Path,
) -> Path:
    """
    Convert a single Keras model to float16 quantised TFLite.

    Args:
        exercise   : exercise name (used in output filename)
        model_path : path to .keras model file
        output_dir : directory to save .tflite file

    Returns:
        Path to the generated .tflite file.
    """
    print(f'\n  Converting {exercise}  ←  {model_path.name}')

    model     = tf.keras.models.load_model(str(model_path))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # float16 quantisation — ~2× size reduction with negligible accuracy loss
    converter.optimizations          = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_bytes = converter.convert()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f'form_scorer_{exercise}_v1.tflite'
    out_path.write_bytes(tflite_bytes)

    size_kb = len(tflite_bytes) / 1024
    print(f'  → {out_path.name}  ({size_kb:.1f} KB)')

    return out_path


# ── Verification ──────────────────────────────────────────────────────────────

def verify_one(tflite_path: Path, exercise: str) -> None:
    """
    Load the converted model, run one test inference, and print I/O shapes.

    Args:
        tflite_path : path to .tflite file
        exercise    : exercise name (used to load a test sample)
    """
    print(f'  Verifying {tflite_path.name} …')

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load one real sample from test split
    try:
        _, _, test = get_splits(exercise)
        X_test, _ = test
        sample = X_test[:1].astype(np.float32)
    except FileNotFoundError:
        # No data yet — use random input for shape check only
        sample = np.random.rand(1, N_FRAMES, N_FEATURES).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    scores = (output[0] * 100).tolist()

    print(f'  Input  shape : {input_details[0]["shape"]}  '
          f'dtype: {input_details[0]["dtype"].__name__}')
    print(f'  Output shape : {output_details[0]["shape"]}  '
          f'dtype: {output_details[0]["dtype"].__name__}')
    print(f'  Test scores  : {[f"{s:.1f}" for s in scores]}')
    print(f'  ✓ Inference OK')


# ── Batch conversion ──────────────────────────────────────────────────────────

def convert_all(
    phase: int      = 3,
    output_dir: Path = DEFAULT_OUT,
    verify: bool    = True,
) -> list[Path]:
    """
    Convert all exercise models for the given phase.

    Args:
        phase      : model phase to load (1, 2, or 3)
        output_dir : where to save .tflite files
        verify     : run inference verification after each conversion

    Returns:
        List of paths to generated .tflite files.
    """
    print(f'TFLite conversion — Phase {phase} models')
    print(f'Output directory: {output_dir}\n')

    outputs = []
    total_kb = 0.0

    for exercise in EXERCISES:
        model_path = MODELS_DIR / f'{exercise}_phase{phase}_final.keras'
        if not model_path.exists():
            model_path = MODELS_DIR / f'{exercise}_phase{phase}_best.keras'
        if not model_path.exists():
            print(f'  [SKIP] {exercise}: model not found')
            continue

        tflite_path = convert_one(exercise, model_path, output_dir)
        total_kb   += tflite_path.stat().st_size / 1024
        outputs.append(tflite_path)

        if verify:
            verify_one(tflite_path, exercise)

    if outputs:
        print(f'\n{"─" * 50}')
        print(f'Converted {len(outputs)}/{len(EXERCISES)} models')
        print(f'Total size: {total_kb:.1f} KB  '
              f'({total_kb / 1024:.2f} MB)  '
              f'[target ~{len(outputs) * 500:.0f} KB]')
        print(f'\nCopy to Flutter project:')
        print(f'  {output_dir}/*.tflite')
        print(f'  → ai_fitness_trainer/assets/models/')

    return outputs


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert Keras models to float16 TFLite.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--exercise', choices=EXERCISES,
                       help='Convert a single exercise.')
    group.add_argument('--all', action='store_true',
                       help='Convert all exercises.')
    parser.add_argument('--phase',      type=int, choices=[1, 2, 3], default=3)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--no-verify',  action='store_true',
                        help='Skip inference verification.')
    args = parser.parse_args()

    if args.all:
        convert_all(
            phase=args.phase,
            output_dir=args.output_dir,
            verify=not args.no_verify,
        )
    else:
        # Single exercise
        model_path = MODELS_DIR / f'{args.exercise}_phase{args.phase}_final.keras'
        if not model_path.exists():
            model_path = MODELS_DIR / f'{args.exercise}_phase{args.phase}_best.keras'
        if not model_path.exists():
            print(f'Model not found: {model_path}')
            sys.exit(1)

        tflite_path = convert_one(args.exercise, model_path, args.output_dir)
        if not args.no_verify:
            verify_one(tflite_path, args.exercise)
