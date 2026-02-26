"""Shared constants across the entire training pipeline."""

# ── Sequence dimensions ───────────────────────────────────────────────────────
N_FRAMES   = 90   # fixed sequence length fed to the model
N_ANGLES   = 9    # joint angle channels
N_FEATURES = 18   # angles (9) + frame-to-frame velocities (9)
N_OUTPUTS  = 6    # scored dimensions

# ── Exercises ─────────────────────────────────────────────────────────────────
EXERCISES = ['squat', 'pushup', 'plank', 'bicep_curl', 'lunge']

# ── Score dimensions (order matches model output and RepScore in Dart) ────────
SCORE_DIMS = ['depth', 'stability', 'symmetry', 'tempo', 'alignment', 'overall']

# ── Angle channel names (must match FeatureExtractor.dart ordering) ───────────
ANGLE_NAMES = [
    'leftKnee',       # 0
    'rightKnee',      # 1
    'leftHip',        # 2
    'rightHip',       # 3
    'leftElbow',      # 4
    'rightElbow',     # 5
    'leftShoulder',   # 6
    'rightShoulder',  # 7
    'spineAngle',     # 8
]

# ── Angle channel indices ─────────────────────────────────────────────────────
IDX_LEFT_KNEE      = 0
IDX_RIGHT_KNEE     = 1
IDX_LEFT_HIP       = 2
IDX_RIGHT_HIP      = 3
IDX_LEFT_ELBOW     = 4
IDX_RIGHT_ELBOW    = 5
IDX_LEFT_SHOULDER  = 6
IDX_RIGHT_SHOULDER = 7
IDX_SPINE          = 8

# Bilateral pairs used for symmetry scoring: (left_idx, right_idx)
BILATERAL_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]

# ── Feature normalisation constants (must match FeatureExtractor.dart) ────────
ANGLE_NORM  = 180.0  # angles / 180  → [0, 1]
VEL_NORM    = 30.0   # velocity / 30 → [-1, 1]

# ── RuleBasedScorer thresholds (must match rule_based_scorer.dart) ────────────
STABILITY_DENOM   = 25.0   # velocityVariance / 25
SYMMETRY_DENOM    = 20.0   # avgBilateralDiff / 20
TEMPO_GOOD_VEL    = 5.0    # ≤ 5 °/frame → tempo = 100
TEMPO_BAD_VEL     = 25.0   # ≥ 25 °/frame → tempo = 0
