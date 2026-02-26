# AI Fitness Trainer — ML Training Pipeline

Python training pipeline for the Deep Learning form scoring models used in the [AI Fitness Trainer](https://github.com/nikegor7/ai-fitness-trainer) Flutter app.

Part of the Master's thesis:
> *"Development of an Intelligent System for Monitoring the Performance of Physical Exercises Using Computer Vision and Deep Learning"*

---

## Overview

The Flutter app uses a **three-layer hybrid scoring system** to evaluate exercise form in real time:

| Layer | Where | What |
|---|---|---|
| **Layer 1** — Rule-based | Flutter (Dart) | Angle thresholds → instant per-frame feedback |
| **Layer 2** — DL model | Flutter (TFLite) | Conv1D + BiLSTM → per-rep multi-dimensional score |
| **Layer 3** — Adaptive | Flutter (Dart) | User-specific baseline adjustment |

This repository handles **Layer 2**: training the Keras models and exporting them as `.tflite` files that the app loads at runtime.

---

## Supported Exercises

| Exercise | Primary Joint | Ideal ROM |
|---|---|---|
| Squat | Knee | 85° |
| Push-up | Elbow | 80° |
| Plank | Spine (isometric) | — |
| Bicep Curl | Elbow | 120° |
| Lunge | Front Knee | 80° |

---

## Score Dimensions

Each rep is scored on **6 dimensions** in [0, 100]:

| # | Dimension | Formula |
|---|---|---|
| 0 | **depth** | `actualROM / idealROM × 100` |
| 1 | **stability** | `(1 − velocityVariance / 25) × 100` |
| 2 | **symmetry** | `(1 − avgBilateralDiff / 20) × 100` |
| 3 | **tempo** | linear: ≤ 5 °/frame → 100, ≥ 25 °/frame → 0 |
| 4 | **alignment** | `(1 − errorFrameCount / totalFrames) × 100` |
| 5 | **overall** | mean of all five |

Formulas match `rule_based_scorer.dart` exactly — so the DL model learns a generalisation of the rule-based scorer.

---

## Model Architecture

```
Input  (batch, 90, 18)
  │
  ├─ Conv1D(64, k=5, relu, same)
  ├─ Conv1D(64, k=3, relu, same)
  ├─ MaxPooling1D(2)               → (batch, 45, 64)
  ├─ Dropout(0.3)
  │
  ├─ BiLSTM(64, return_sequences=True)
  ├─ BiLSTM(64, return_sequences=False)
  │
  ├─ Dense(128, relu)
  ├─ Dropout(0.3)
  └─ Dense(6, sigmoid)

Output (batch, 6)   ×100 in Dart → RepScore
~200 K trainable parameters
```

**Input features** (18 channels, 90 frames):
- Channels 0–8: joint angles ÷ 180 → [0, 1]
- Channels 9–17: frame-to-frame velocity ÷ 30 → [−1, 1]

**9 joint angles** (same order as `FeatureExtractor.dart`):
`leftKnee · rightKnee · leftHip · rightHip · leftElbow · rightElbow · leftShoulder · rightShoulder · spineAngle`

---

## Project Structure

```
ai_fitness_trainer_ml/
│
├── config.py                      # Shared constants (angles, exercises, thresholds)
├── requirements.txt
│
├── data/
│   ├── synthetic_generator.py     # Generate 5 000 synthetic reps/exercise with GT scores
│   ├── augmentation.py            # Time warp · noise · amplitude scale · mirror
│   ├── dataset_loader.py          # Load synthetic + real data, compute features, splits
│   └── annotation_tool.py        # tkinter GUI for expert labeling of real reps
│
├── models/
│   ├── form_scorer_model.py       # Keras model definition
│   └── train.py                   # Three-phase training loop
│
├── evaluation/
│   ├── evaluate_model.py          # MAE · Pearson · ICC(2,1) per exercise and dimension
│   ├── compare_approaches.py      # Rule-based vs DL vs Hybrid comparison
│   └── visualize_predictions.py  # Scatter plots · error histograms · bar charts
│
├── export/
│   └── convert_tflite.py         # Keras → float16 TFLite + inference verification
│
└── notebooks/
    └── thesis_analysis.ipynb     # All thesis figures in one notebook
```

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

Requirements: Python 3.9+, TensorFlow 2.13–2.15, NumPy, pandas, scikit-learn, scipy, matplotlib, seaborn, jupyter.

---

## Usage

### Step 1 — Generate synthetic training data

```bash
python data/synthetic_generator.py
# → data/generated/<exercise>/sequences.npy  (5 000, 90, 9)
# → data/generated/<exercise>/scores.npy     (5 000, 6)
```

Generates 25 000 samples total (5 exercises × 5 000) across four quality tiers:
excellent (25%) · good (35%) · acceptable (25%) · poor (15%).

### Step 2 — Train (Phase 1: pre-train on synthetic data)

```bash
python models/train.py --exercise all --phase 1
# or single exercise:
python models/train.py --exercise squat --phase 1 --epochs 50
```

Target: val MAE < 15.  Checkpoints saved to `models/saved/`.

### Step 3 — Fine-tune on real data  *(optional — after data collection)*

```bash
# First, annotate real reps:
python data/annotation_tool.py --exercise squat

# Then fine-tune:
python models/train.py --exercise squat --phase 2   # freeze Conv1D, lr=1e-4
python models/train.py --exercise squat --phase 3   # full fine-tune, lr=1e-5
```

Target after Phase 3: MAE < 8 · Pearson > 0.85.

### Step 4 — Evaluate

```bash
python evaluation/evaluate_model.py --phase 3
python evaluation/compare_approaches.py --phase 3
python evaluation/visualize_predictions.py --phase 3
# → evaluation/results_evaluation.csv
# → evaluation/results_comparison.csv
# → evaluation/figures/*.png
```

### Step 5 — Export TFLite

```bash
python export/convert_tflite.py --all --phase 3
# → export/tflite/form_scorer_<exercise>_v1.tflite  (~500 KB each)
```

Copy the `.tflite` files to the Flutter project:
```
ai_fitness_trainer/assets/models/
```

### Thesis notebook

```bash
jupyter notebook notebooks/thesis_analysis.ipynb
```

---

## Training Phases

| Phase | Data | Frozen layers | LR | Epochs | Target MAE |
|---|---|---|---|---|---|
| 1 — Pre-train | Synthetic (25 000) | none | 1e-3 | 50 | < 15 |
| 2 — Fine-tune | Real annotated | Conv1D | 1e-4 | 30 | < 10 |
| 3 — Full fine-tune | Synthetic + Real | none | 1e-5 | 20 | **< 8** |

---

## Real Data Collection

To collect real training data from the Flutter app:

1. Enable `PoseLog` recording in the app settings.
2. Export the pose log CSV from the app.
3. Run the annotation tool:
   ```bash
   python data/annotation_tool.py --exercise squat --input path/to/session.csv
   ```
4. Score each rep on all 6 dimensions using the sliders.
5. Annotated data is saved to `data/real/<exercise>/`.

Target: 10–20 participants × 50 reps = 500–1 000 labeled reps per exercise.

---

## Relationship to Flutter App

```
ai_fitness_trainer_ml/          ← this repo (Python, training only)
    export/tflite/*.tflite
          │
          │  copy to
          ▼
ai_fitness_trainer/             ← Flutter app repo
    assets/models/*.tflite      ← loaded by DLFormScorer.dart
```

The Flutter app runs in **rule-only mode** until the `.tflite` files are present.
Once copied, `DLFormScorer` activates and the hybrid blend (30% rule + 70% DL) takes effect.

---

## Evaluation Targets

| Metric | Target |
|---|---|
| MAE (overall dimension) | < 8 points |
| Pearson correlation | > 0.85 |
| ICC(2,1) | > 0.80 |
| TFLite model size | ~500 KB per exercise |
| Inference latency | ~3 ms on mid-range Android |
