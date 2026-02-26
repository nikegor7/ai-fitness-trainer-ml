"""
Expert annotation GUI for labeling real exercise rep sequences.

Loads CSV pose logs exported from the Flutter app (or any CSV with
one row per frame and columns matching ANGLE_NAMES), displays
the joint angle sequence as a line plot, and captures expert
scores (0–100) for each of the 6 scoring dimensions.

Annotations are saved to:
  data/real/<exercise>/sequences.npy   — (N, 90, 9)
  data/real/<exercise>/scores.npy      — (N, 6) in [0, 100]
  data/real/<exercise>/annotations.csv — human-readable record

Expected CSV format (one row per frame):
  rep_id, leftKnee, rightKnee, leftHip, rightHip,
  leftElbow, rightElbow, leftShoulder, rightShoulder, spineAngle
  (rep_id column groups frames into individual reps)

Usage:
    python data/annotation_tool.py
    python data/annotation_tool.py --exercise squat
    python data/annotation_tool.py --input data/raw/squat_session.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EXERCISES, SCORE_DIMS, ANGLE_NAMES, N_FRAMES

# tkinter and matplotlib imports are deferred to runtime so the module
# can be imported without a display (e.g. in tests or CI).

REAL_DIR = Path(__file__).parent / 'real'


# ── CSV parser ────────────────────────────────────────────────────────────────

def parse_pose_log(df) -> list[np.ndarray]:
    """
    Parse a pose-log DataFrame into a list of (90, 9) sequences.

    Each unique rep_id becomes one entry.  Variable-length reps are
    resampled to exactly N_FRAMES=90 via linear interpolation.

    Args:
        df : pandas DataFrame with angle columns matching ANGLE_NAMES

    Returns:
        List of (90, 9) float32 arrays.
    """
    sequences = []

    # Verify columns
    missing = [c for c in ANGLE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns in CSV: {missing}')

    def resample(frames: np.ndarray) -> np.ndarray:
        """Resample variable-length sequence to exactly N_FRAMES."""
        n = len(frames)
        if n < 2:
            return None
        x_orig = np.linspace(0.0, 1.0, n)
        x_new  = np.linspace(0.0, 1.0, N_FRAMES)
        seq    = np.zeros((N_FRAMES, len(ANGLE_NAMES)), dtype=np.float32)
        for j in range(len(ANGLE_NAMES)):
            fn       = interp1d(x_orig, frames[:, j], kind='linear')
            seq[:, j] = np.clip(fn(x_new), 0.0, 180.0)
        return seq

    if 'rep_id' in df.columns:
        for _, group in df.groupby('rep_id', sort=True):
            frames = group[ANGLE_NAMES].values.astype(np.float32)
            seq    = resample(frames)
            if seq is not None:
                sequences.append(seq)
    else:
        # Treat the whole file as one rep
        frames = df[ANGLE_NAMES].values.astype(np.float32)
        seq    = resample(frames)
        if seq is not None:
            sequences.append(seq)

    return sequences


# ── Annotation GUI ────────────────────────────────────────────────────────────

class AnnotationTool:
    """Tkinter GUI for scoring exercise rep sequences."""

    def __init__(self, root, exercise: str = 'squat'):
        import tkinter as tk
        from tkinter import ttk

        self.tk   = tk
        self.ttk  = ttk
        self.root = root
        self.root.title('Exercise Form Annotation Tool')
        self.root.geometry('1120x760')
        self.root.resizable(True, True)

        self.sequences:   list[np.ndarray] = []
        self.annotations: list[dict]       = []
        self.current_idx: int              = 0

        self.exercise = tk.StringVar(value=exercise)

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        import tkinter as tk
        from tkinter import ttk
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self._plt = plt

        # Top toolbar
        top = ttk.Frame(self.root, padding=(8, 6))
        top.pack(side='top', fill='x')

        ttk.Label(top, text='Exercise:').pack(side='left')
        ttk.Combobox(
            top, textvariable=self.exercise, width=14,
            values=EXERCISES, state='readonly',
        ).pack(side='left', padx=4)

        ttk.Button(top, text='Load CSV …',        command=self._load_csv).pack(side='left', padx=6)
        ttk.Button(top, text='Save annotations',  command=self._save).pack(side='left')

        self.status_var = tk.StringVar(value='No file loaded — use "Load CSV …"')
        ttk.Label(top, textvariable=self.status_var,
                  foreground='gray').pack(side='right')

        # Centre: plot + scoring panel
        center = ttk.Frame(self.root)
        center.pack(fill='both', expand=True, padx=8, pady=4)

        # Matplotlib canvas (left)
        self.fig, self.ax = plt.subplots(figsize=(7.5, 4.2))
        self.canvas = FigureCanvasTkAgg(self.fig, master=center)
        self.canvas.get_tk_widget().pack(side='left', fill='both', expand=True)

        # Scoring panel (right)
        score_frame = ttk.LabelFrame(center, text='Scores (0 – 100)', padding=10)
        score_frame.pack(side='right', fill='y', padx=(8, 4))

        self.score_vars: dict[str, tk.IntVar] = {}
        for dim in SCORE_DIMS:
            row = ttk.Frame(score_frame)
            row.pack(fill='x', pady=4)

            ttk.Label(row, text=f'{dim.capitalize():<11}',
                      width=11).pack(side='left')

            var   = tk.IntVar(value=50)
            scale = ttk.Scale(row, variable=var, from_=0, to=100,
                              orient='horizontal', length=150)
            scale.pack(side='left')

            # Numeric entry so the annotator can type exact values
            entry = ttk.Entry(row, textvariable=var, width=5)
            entry.pack(side='left', padx=(4, 0))

            self.score_vars[dim] = var

        ttk.Separator(score_frame).pack(fill='x', pady=10)

        nav = ttk.Frame(score_frame)
        nav.pack(fill='x')
        ttk.Button(nav, text='← Prev',        command=self._prev).pack(side='left', expand=True, fill='x')
        ttk.Button(nav, text='Save & Next →',  command=self._save_and_next).pack(side='left', expand=True, fill='x')

        self.rep_label = ttk.Label(score_frame, text='Rep — / —',
                                   font=('', 10, 'bold'))
        self.rep_label.pack(pady=(10, 0))

        self.annotated_label = ttk.Label(score_frame, text='Annotated: 0',
                                         foreground='green')
        self.annotated_label.pack()

    # ── Event handlers ────────────────────────────────────────────────────

    def _load_csv(self):
        from tkinter import filedialog, messagebox
        import pandas as pd

        path = filedialog.askopenfilename(
            title='Select pose log CSV',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
        )
        if not path:
            return
        try:
            df             = pd.read_csv(path)
            self.sequences = parse_pose_log(df)
            self.current_idx = 0
            self.status_var.set(
                f'Loaded {len(self.sequences)} reps  ←  {Path(path).name}'
            )
            self._show_current()
        except Exception as exc:
            messagebox.showerror('Load error', str(exc))

    def _show_current(self):
        if not self.sequences:
            return

        seq = self.sequences[self.current_idx]
        colors = [
            '#e41a1c', '#e41a1c',  # knees (red, dashed)
            '#377eb8', '#377eb8',  # hips (blue, dashed)
            '#4daf4a', '#4daf4a',  # elbows (green, dashed)
            '#984ea3', '#984ea3',  # shoulders (purple, dashed)
            '#ff7f00',             # spine (orange)
        ]
        styles = ['-', '--'] * 4 + ['-']

        self.ax.clear()
        for j, (name, color, style) in enumerate(
            zip(ANGLE_NAMES, colors, styles)
        ):
            self.ax.plot(seq[:, j], label=name, color=color,
                         linestyle=style, linewidth=1.3)

        self.ax.set_xlabel('Frame', fontsize=9)
        self.ax.set_ylabel('Angle (°)', fontsize=9)
        self.ax.set_title(
            f'Rep {self.current_idx + 1}  —  {self.exercise.get()}',
            fontsize=11, fontweight='bold',
        )
        self.ax.legend(loc='upper right', fontsize=7, ncol=2)
        self.ax.set_xlim(0, 89)
        self.ax.set_ylim(0, 185)
        self.fig.tight_layout()
        self.canvas.draw()

        n_total  = len(self.sequences)
        n_ann    = len(self.annotations)
        self.rep_label.config(
            text=f'Rep {self.current_idx + 1} / {n_total}')
        self.annotated_label.config(text=f'Annotated: {n_ann} / {n_total}')

        # Pre-fill scores if this rep was already annotated
        existing = next(
            (a for a in self.annotations if a['rep_idx'] == self.current_idx),
            None,
        )
        if existing:
            for dim in SCORE_DIMS:
                self.score_vars[dim].set(int(existing.get(dim, 50)))

    def _save_and_next(self):
        from tkinter import messagebox

        if not self.sequences:
            return

        scores = {dim: int(self.score_vars[dim].get()) for dim in SCORE_DIMS}
        scores['rep_idx']  = self.current_idx
        scores['exercise'] = self.exercise.get()

        # Update existing annotation or append
        existing_idx = next(
            (i for i, a in enumerate(self.annotations)
             if a['rep_idx'] == self.current_idx),
            None,
        )
        if existing_idx is not None:
            self.annotations[existing_idx] = scores
        else:
            self.annotations.append(scores)

        if self.current_idx < len(self.sequences) - 1:
            self.current_idx += 1
            self._show_current()
        else:
            messagebox.showinfo(
                'All reps annotated',
                f'All {len(self.sequences)} reps annotated.\n'
                'Click "Save annotations" to export.',
            )

    def _prev(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self._show_current()

    def _save(self):
        import pandas as pd
        from tkinter import messagebox

        if not self.annotations:
            messagebox.showwarning('Nothing to save', 'Annotate at least one rep first.')
            return

        exercise = self.exercise.get()
        out_dir  = REAL_DIR / exercise
        out_dir.mkdir(parents=True, exist_ok=True)

        # Sort annotations by rep_idx
        ann_sorted = sorted(self.annotations, key=lambda a: a['rep_idx'])
        df_ann     = import_pandas().DataFrame(ann_sorted)
        df_ann.to_csv(out_dir / 'annotations.csv', index=False)

        # Save numpy arrays
        indices = [a['rep_idx'] for a in ann_sorted]
        seqs    = np.stack([self.sequences[i] for i in indices])
        scores  = df_ann[SCORE_DIMS].values.astype(np.float32)

        np.save(out_dir / 'sequences.npy', seqs)
        np.save(out_dir / 'scores.npy',    scores)

        msg = (f'Saved {len(self.annotations)} annotations to:\n{out_dir}')
        messagebox.showinfo('Saved', msg)
        self.status_var.set(
            f'Saved {len(self.annotations)} annotations → {out_dir}')


def import_pandas():
    import pandas as pd
    return pd


# ── Entry point ───────────────────────────────────────────────────────────────

def main(exercise: str = 'squat', input_csv: str | None = None):
    import tkinter as tk

    root = tk.Tk()
    app  = AnnotationTool(root, exercise=exercise)

    if input_csv:
        # Auto-load a CSV if provided on the command line
        import pandas as pd
        from tkinter import messagebox
        try:
            df             = pd.read_csv(input_csv)
            app.sequences  = parse_pose_log(df)
            app.status_var.set(
                f'Loaded {len(app.sequences)} reps  ←  {Path(input_csv).name}'
            )
            app._show_current()
        except Exception as exc:
            messagebox.showerror('Auto-load error', str(exc))

    root.mainloop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exercise form annotation tool.')
    parser.add_argument('--exercise', choices=EXERCISES, default='squat')
    parser.add_argument('--input',    type=str, default=None,
                        help='Path to CSV pose log to load on startup.')
    args = parser.parse_args()
    main(exercise=args.exercise, input_csv=args.input)
