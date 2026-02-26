"""
Synthetic angle sequence generator for exercise form scoring.

Generates 5 000 labeled samples per exercise (25 000 total).
Each sample:
  sequence : np.ndarray (90, 9)  — joint angles in degrees over 90 frames
  scores   : np.ndarray (6,)     — [depth, stability, symmetry, tempo,
                                     alignment, overall]  all in [0, 100]

Score formulas mirror RuleBasedScorer.dart exactly so the DL model learns
the same scoring logic with added generalisation from the Conv1D + BiLSTM.

Usage:
    python data/synthetic_generator.py                  # all exercises
    python data/synthetic_generator.py --exercise squat
    python data/synthetic_generator.py --n-samples 2000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    N_FRAMES, N_ANGLES, EXERCISES, BILATERAL_PAIRS,
    IDX_LEFT_KNEE, IDX_LEFT_ELBOW, IDX_SPINE,
    IDX_LEFT_HIP, IDX_RIGHT_HIP,
    STABILITY_DENOM, SYMMETRY_DENOM, TEMPO_GOOD_VEL, TEMPO_BAD_VEL,
)

# ── Output directory ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / 'generated'

# ── Exercise profiles ─────────────────────────────────────────────────────────
# neutral_angles : resting / start-of-rep position (degrees)
# peak_angles    : deepest / most contracted position (degrees)
# ideal_rom      : expected range-of-motion for primary joint (degrees)
# primary_joint  : channel index used for depth calculation

EXERCISE_PROFILES = {
    'squat': {
        'neutral':      [170, 170, 170, 170, 160, 160,  85,  85, 10],
        'peak':         [ 85,  85,  75,  75, 160, 160,  85,  85, 20],
        'ideal_rom':    85,
        'primary_joint': IDX_LEFT_KNEE,
        'is_isometric': False,
    },
    'pushup': {
        'neutral':      [170, 170, 170, 170, 160, 160,  90,  90,  5],
        'peak':         [170, 170, 170, 170,  80,  80,  90,  90,  5],
        'ideal_rom':    80,
        'primary_joint': IDX_LEFT_ELBOW,
        'is_isometric': False,
    },
    'plank': {
        'neutral':      [170, 170, 170, 170,  90,  90,  90,  90,  2],
        'peak':         [170, 170, 170, 170,  90,  90,  90,  90,  2],
        'ideal_rom':    5,
        'primary_joint': IDX_SPINE,
        'is_isometric': True,
    },
    'bicep_curl': {
        'neutral':      [170, 170, 170, 170, 160, 160, 170, 170,  5],
        'peak':         [170, 170, 170, 170,  40,  40, 170, 170,  5],
        'ideal_rom':    120,
        'primary_joint': IDX_LEFT_ELBOW,
        'is_isometric': False,
    },
    'lunge': {
        'neutral':      [170, 170, 170, 170, 160, 160,  85,  85, 10],
        'peak':         [ 90, 120, 100, 115, 160, 160,  85,  85, 12],
        'ideal_rom':    80,
        'primary_joint': IDX_LEFT_KNEE,
        'is_isometric': False,
    },
}

# ── Quality tiers ─────────────────────────────────────────────────────────────
# depth_ratio        : fraction of ideal ROM actually achieved
# jitter_std         : std of Gaussian noise added to angles (degrees)
# bilateral_diff_std : std of L-R asymmetry added per frame (degrees)
# descent_frames     : number of frames for the descent phase
#                      (controls peak velocity → tempo score)
# align_error_frac   : fraction of frames with injected posture errors

QUALITY_TIERS = {
    'excellent': {
        'weight':              0.25,
        'depth_ratio':         (0.90, 1.00),
        'jitter_std':          (0.1,  0.5),
        'bilateral_diff_std':  (0.5,  2.0),
        'descent_frames':      (22,   32),
        'align_error_frac':    (0.00, 0.05),
    },
    'good': {
        'weight':              0.35,
        'depth_ratio':         (0.70, 0.90),
        'jitter_std':          (0.5,  1.5),
        'bilateral_diff_std':  (2.0,  5.0),
        'descent_frames':      (15,   25),
        'align_error_frac':    (0.05, 0.20),
    },
    'acceptable': {
        'weight':              0.25,
        'depth_ratio':         (0.50, 0.70),
        'jitter_std':          (1.5,  3.0),
        'bilateral_diff_std':  (5.0, 10.0),
        'descent_frames':      (10,   18),
        'align_error_frac':    (0.20, 0.40),
    },
    'poor': {
        'weight':              0.15,
        'depth_ratio':         (0.20, 0.50),
        'jitter_std':          (3.0,  6.0),
        'bilateral_diff_std':  (10.0, 20.0),
        'descent_frames':      (5,    12),
        'align_error_frac':    (0.40, 0.80),
    },
}

_TIER_NAMES   = list(QUALITY_TIERS.keys())
_TIER_WEIGHTS = [QUALITY_TIERS[t]['weight'] for t in _TIER_NAMES]


# ── Trajectory generation ─────────────────────────────────────────────────────

def _make_trajectory(
    neutral: np.ndarray,
    peak: np.ndarray,
    n_frames: int = N_FRAMES,
    descent_frames: int = 25,
    hold_frames: int = 12,
    ascent_frames: int = 25,
) -> np.ndarray:
    """
    Build a smooth angle sequence: neutral → peak → neutral.

    Uses a cosine ease-in/ease-out curve for natural movement.
    Remaining frames are held at neutral (rest).
    """
    traj = np.zeros((n_frames, N_ANGLES), dtype=np.float32)
    frame = 0

    # Descent
    for i in range(descent_frames):
        alpha = 0.5 * (1.0 - np.cos(np.pi * i / max(descent_frames - 1, 1)))
        traj[frame] = neutral + alpha * (peak - neutral)
        frame += 1
        if frame >= n_frames:
            break

    # Hold at peak
    for _ in range(hold_frames):
        if frame >= n_frames:
            break
        traj[frame] = peak
        frame += 1

    # Ascent
    for i in range(ascent_frames):
        if frame >= n_frames:
            break
        alpha = 0.5 * (1.0 - np.cos(np.pi * i / max(ascent_frames - 1, 1)))
        traj[frame] = peak + alpha * (neutral - peak)
        frame += 1

    # Rest at neutral
    while frame < n_frames:
        traj[frame] = neutral
        frame += 1

    return traj


def _make_isometric_trajectory(
    hold_angles: np.ndarray,
    n_frames: int = N_FRAMES,
) -> np.ndarray:
    """Plank: held position with minimal base movement."""
    traj = np.tile(hold_angles, (n_frames, 1)).astype(np.float32)
    return traj


# ── Score computation (mirrors RuleBasedScorer.dart) ─────────────────────────

def compute_scores(
    sequence: np.ndarray,
    exercise: str,
    depth_ratio: float,
    align_error_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute 6 GT scores from the final (post-distortion) sequence.

    Uses the exact same formulas as rule_based_scorer.dart so that the
    DL model learns a generalisation of the rule-based scorer.

    Args:
        sequence         : (90, 9) angle array in degrees
        exercise         : exercise name string
        depth_ratio      : fraction of ideal ROM achieved (used directly
                           for depth — consistent with Dart formula)
        align_error_mask : boolean array (90,) — True where posture error
                           was injected (used for alignment score)

    Returns:
        scores : (6,) float32 array in [0, 100]
    """
    profile = EXERCISE_PROFILES[exercise]

    # 1. Depth — actualROM / idealROM × 100
    depth = float(np.clip(depth_ratio * 100.0, 0.0, 100.0))

    # 2. Stability — 1 − velocityVariance / STABILITY_DENOM
    velocities = np.diff(sequence, axis=0)            # (89, 9)
    vel_var    = float(np.var(np.abs(velocities)))
    stability  = float(np.clip((1.0 - vel_var / STABILITY_DENOM) * 100.0, 0.0, 100.0))

    # 3. Symmetry — 1 − avgBilateralDiff / SYMMETRY_DENOM
    diffs = np.concatenate([
        np.abs(sequence[:, l] - sequence[:, r]) for l, r in BILATERAL_PAIRS
    ])
    avg_bilateral = float(diffs.mean())
    symmetry = float(np.clip((1.0 - avg_bilateral / SYMMETRY_DENOM) * 100.0, 0.0, 100.0))

    # 4. Tempo — linear scale on peak velocity
    peak_vel = float(np.abs(velocities).max())
    if peak_vel <= TEMPO_GOOD_VEL:
        tempo = 100.0
    elif peak_vel >= TEMPO_BAD_VEL:
        tempo = 0.0
    else:
        tempo = float((1.0 - (peak_vel - TEMPO_GOOD_VEL) /
                       (TEMPO_BAD_VEL - TEMPO_GOOD_VEL)) * 100.0)

    # 5. Alignment — 1 − errorFrameCount / totalFrames
    error_count = int(align_error_mask.sum())
    alignment   = float(np.clip((1.0 - error_count / N_FRAMES) * 100.0, 0.0, 100.0))

    # 6. Overall — mean of all five
    overall = (depth + stability + symmetry + tempo + alignment) / 5.0

    return np.array([depth, stability, symmetry, tempo, alignment, overall],
                    dtype=np.float32)


# ── Sample generation ─────────────────────────────────────────────────────────

def _inject_alignment_errors(
    sequence: np.ndarray,
    exercise: str,
    error_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add posture distortions on frames flagged by error_mask."""
    seq = sequence.copy()
    n_err = int(error_mask.sum())
    if n_err == 0:
        return seq

    if exercise in ('squat', 'lunge', 'bicep_curl'):
        # Forward trunk lean: spine angle increases
        seq[error_mask, IDX_SPINE] += rng.uniform(15, 40, n_err)
    elif exercise == 'pushup':
        # Hip sag: hip angles decrease (body bends)
        sag = rng.uniform(10, 30, n_err)
        seq[error_mask, IDX_LEFT_HIP]  -= sag
        seq[error_mask, IDX_RIGHT_HIP] -= sag
    else:  # plank
        # Hip sag or pike
        sag = rng.uniform(10, 25, n_err)
        seq[error_mask, IDX_LEFT_HIP]  -= sag
        seq[error_mask, IDX_RIGHT_HIP] -= sag

    return seq


def generate_sample(
    exercise: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate one synthetic rep sequence with ground-truth scores.

    Returns:
        sequence : (90, 9) float32 array of joint angles in degrees
        scores   : (6,) float32 array in [0, 100]
    """
    profile = EXERCISE_PROFILES[exercise]
    neutral = np.array(profile['neutral'], dtype=np.float32)
    peak    = np.array(profile['peak'],    dtype=np.float32)

    # Sample quality tier
    tier_name = rng.choice(_TIER_NAMES, p=_TIER_WEIGHTS)
    tier      = QUALITY_TIERS[tier_name]

    def sample_range(key):
        lo, hi = tier[key]
        return float(rng.uniform(lo, hi))

    depth_ratio        = sample_range('depth_ratio')
    jitter_std         = sample_range('jitter_std')
    bilateral_diff_std = sample_range('bilateral_diff_std')
    descent_frames     = int(sample_range('descent_frames'))
    align_error_frac   = sample_range('align_error_frac')

    # Scale peak toward neutral to represent the achieved ROM
    actual_peak = neutral + depth_ratio * (peak - neutral)

    # ── Build base trajectory ──────────────────────────────────────────────
    if profile['is_isometric']:
        sequence = _make_isometric_trajectory(actual_peak)
    else:
        hold_frames   = max(5, N_FRAMES - 2 * descent_frames - 10)
        sequence = _make_trajectory(
            neutral, actual_peak,
            descent_frames=descent_frames,
            hold_frames=hold_frames,
            ascent_frames=descent_frames,
        )

    # ── Add Gaussian jitter ────────────────────────────────────────────────
    sequence += rng.normal(0.0, jitter_std, sequence.shape).astype(np.float32)

    # ── Add bilateral asymmetry ────────────────────────────────────────────
    for l_idx, r_idx in BILATERAL_PAIRS:
        asym = rng.normal(0.0, bilateral_diff_std, N_FRAMES).astype(np.float32)
        sequence[:, l_idx] += asym
        sequence[:, r_idx] -= asym * 0.5   # not perfectly mirrored

    # ── Inject alignment errors ────────────────────────────────────────────
    error_mask = rng.random(N_FRAMES) < align_error_frac
    sequence   = _inject_alignment_errors(sequence, exercise, error_mask, rng)

    # ── Clip to valid angle range ──────────────────────────────────────────
    sequence = np.clip(sequence, 0.0, 180.0)

    # ── Compute GT scores from the final sequence ──────────────────────────
    scores = compute_scores(sequence, exercise, depth_ratio, error_mask)

    return sequence, scores


# ── Batch generation ──────────────────────────────────────────────────────────

def generate_exercise(
    exercise: str,
    n_samples: int = 5000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate n_samples synthetic reps for one exercise.

    Returns:
        sequences : (n_samples, 90, 9)  float32
        scores    : (n_samples, 6)      float32 in [0, 100]
    """
    rng       = np.random.default_rng(seed)
    sequences = np.zeros((n_samples, N_FRAMES, N_ANGLES), dtype=np.float32)
    scores    = np.zeros((n_samples, 6),                  dtype=np.float32)

    for i in tqdm(range(n_samples), desc=f'  {exercise}', unit='sample'):
        sequences[i], scores[i] = generate_sample(exercise, rng)

    return sequences, scores


def generate_all(n_samples: int = 5000, seed: int = 42) -> None:
    """Generate and save datasets for all exercises."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    total = len(EXERCISES) * n_samples
    print(f'Generating {total:,} synthetic samples '
          f'({n_samples:,} × {len(EXERCISES)} exercises) …\n')

    for ex in EXERCISES:
        out_dir = DATA_DIR / ex
        out_dir.mkdir(exist_ok=True)

        seqs, sc = generate_exercise(ex, n_samples=n_samples, seed=seed)

        np.save(out_dir / 'sequences.npy', seqs)
        np.save(out_dir / 'scores.npy',    sc)

        print(f'  [{ex}] saved  sequences {seqs.shape}  scores {sc.shape}')
        print(f'         score ranges: '
              + '  '.join(f'{d}={sc[:,i].mean():.1f}±{sc[:,i].std():.1f}'
                          for i, d in enumerate(
                              ['depth','stability','symmetry','tempo','align','overall'])))
        print()

    print(f'Done. Files saved to {DATA_DIR}')


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic training data.')
    parser.add_argument('--exercise', choices=EXERCISES + ['all'], default='all')
    parser.add_argument('--n-samples', type=int, default=5000)
    parser.add_argument('--seed',      type=int, default=42)
    args = parser.parse_args()

    if args.exercise == 'all':
        generate_all(n_samples=args.n_samples, seed=args.seed)
    else:
        out_dir = DATA_DIR / args.exercise
        out_dir.mkdir(parents=True, exist_ok=True)
        seqs, sc = generate_exercise(args.exercise,
                                     n_samples=args.n_samples,
                                     seed=args.seed)
        np.save(out_dir / 'sequences.npy', seqs)
        np.save(out_dir / 'scores.npy',    sc)
        print(f'Saved to {out_dir}')
