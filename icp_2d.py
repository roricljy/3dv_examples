"""
2D ICP (Iterative Closest Point) Visualizer
Educational tool for computer vision / image geometry courses.

Usage:
    python icp_2d.py

Dependencies:
    pip install numpy matplotlib scipy
"""

import os
import tkinter as tk
from tkinter import messagebox
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_POINTS = 20           # number of random target points
POINT_RANGE = (-3.0, 3.0)   # uniform sampling range for x and y
MAX_ITERATIONS = 100    # safety cap to prevent infinite loops
ANIMATION_DELAY_MS = 300    # milliseconds between ICP steps

gscale = 1.8 if "ANDROID_STORAGE" in os.environ else 1


# ---------------------------------------------------------------------------
# ICP Algorithm
# ---------------------------------------------------------------------------
class ICP2D:
    """Pure 2-D ICP algorithm: no GUI dependencies."""

    def __init__(self):
        self.target = None          # (N, 2) reference point set (fixed)
        self._prev_indices = None   # previous nearest-neighbor assignment

    # ------------------------------------------------------------------
    # Point generation
    # ------------------------------------------------------------------
    def generate_target(self, n: int = N_POINTS) -> np.ndarray:
        """Generate n random 2-D points within POINT_RANGE."""
        return np.random.uniform(POINT_RANGE[0], POINT_RANGE[1], (n, 2))

    def generate_source(
        self,
        target: np.ndarray,
        angle_deg: float,
        tx: float,
        ty: float,
    ) -> np.ndarray:
        """
        Apply a 2-D rigid transform to target and return the result as source.
        source = R * target + t
        """
        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        t = np.array([tx, ty])
        return (R @ target.T).T + t

    # ------------------------------------------------------------------
    # Core ICP components
    # ------------------------------------------------------------------
    def find_nearest_neighbors(
        self, source: np.ndarray, target: np.ndarray
    ):
        """
        For each point in source, find the index of the closest point in target.

        Returns:
            indices   : (N,) int array — target index for each source point
            distances : (N,) float array — corresponding distances
        """
        # dist_matrix[i, j] = distance from source[i] to target[j]
        dist_matrix = cdist(source, target)
        indices = np.argmin(dist_matrix, axis=1)
        distances = dist_matrix[np.arange(len(source)), indices]
        return indices, distances

    def compute_transform(
        self, source: np.ndarray, target_matched: np.ndarray
    ):
        """
        Compute the optimal rotation R and translation t that minimises
        sum of squared distances between source[i] and target_matched[i].

        Uses the SVD-based closed-form solution.
        Handles the reflection (det = -1) edge case explicitly.

        Returns:
            R : (2, 2) rotation matrix
            t : (2,)  translation vector
        """
        mu_s = source.mean(axis=0)
        mu_t = target_matched.mean(axis=0)

        # Center both sets
        S = source - mu_s
        T = target_matched - mu_t

        # Cross-covariance matrix
        H = S.T @ T                         # (2, 2)

        U, _, Vt = np.linalg.svd(H)

        # Reflection correction: ensure det(R) == +1
        d = np.linalg.det(Vt.T @ U.T)
        D = np.diag([1.0, d])

        R = Vt.T @ D @ U.T
        t = mu_t - R @ mu_s
        return R, t

    @staticmethod
    def apply_transform(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Apply rigid transform to a point set."""
        return (R @ points.T).T + t

    # ------------------------------------------------------------------
    # Single ICP iteration
    # ------------------------------------------------------------------
    def reset(self):
        """Reset internal state before a new ICP run."""
        self._prev_indices = None

    def step(self, source_current: np.ndarray):
        """
        Execute one ICP iteration.

        Returns:
            new_source  : (N, 2) transformed source points
            R           : (2, 2) rotation applied this step
            t           : (2,)  translation applied this step
            indices     : (N,)  nearest-neighbor assignment
            converged   : bool  True when assignment did not change
        """
        indices, _ = self.find_nearest_neighbors(source_current, self.target)

        # Convergence check: assignment unchanged from previous step
        if (
            self._prev_indices is not None
            and np.array_equal(indices, self._prev_indices)
        ):
            return source_current, np.eye(2), np.zeros(2), indices, True

        target_matched = self.target[indices]
        R, t = self.compute_transform(source_current, target_matched)
        new_source = self.apply_transform(source_current, R, t)

        self._prev_indices = indices.copy()
        return new_source, R, t, indices, False


# ---------------------------------------------------------------------------
# GUI Application
# ---------------------------------------------------------------------------
class ICP2DApp:
    """Tkinter + Matplotlib GUI for the 2-D ICP visualizer."""

    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("ICP 2D Visualizer")
        master.resizable(False, False)

        # Algorithm object
        self.icp = ICP2D()

        # State
        self.target: np.ndarray | None = None
        self.source_current: np.ndarray | None = None
        self.iteration = 0
        self.after_id = None            # pending after() handle

        # Build UI
        self._build_canvas_area()
        self._build_control_panel()

        # Handle window close via [X] button
        master.protocol("WM_DELETE_WINDOW", self.on_exit)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_canvas_area(self):
        """Create the matplotlib figure embedded in a tkinter frame."""
        self.fig = Figure(figsize=(6.4 * gscale, 4.8 * gscale), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect("equal")
        self.ax.grid(True, linestyle="--", alpha=0.4)
        self.ax.set_title("ICP 2D — Point Set Registration", fontsize=10 * gscale)
        self.ax.set_xlabel("x", fontsize=9 * gscale)
        self.ax.set_ylabel("y", fontsize=9 * gscale)
        self.ax.tick_params(labelsize=8 * gscale)

        # Placeholder scatter objects (updated on generate/step)
        self._scatter_target = self.ax.scatter(
            [], [], c="blue", s=60 * gscale ** 2, zorder=3
        )
        self._scatter_source = self.ax.scatter(
            [], [], c="red", s=60 * gscale ** 2, zorder=3
        )

        # Canvas widget
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _build_control_panel(self):
        """Build the bottom control panel with inputs and buttons."""
        panel = tk.Frame(self.master, relief=tk.GROOVE, bd=2, padx=8, pady=6)
        panel.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Row 0: parameter inputs ---
        row0 = tk.Frame(panel)
        row0.pack(fill=tk.X, pady=(0, 4))

        tk.Label(row0, text="Angle (deg):").pack(side=tk.LEFT)
        self.entry_angle = tk.Entry(row0, width=6)
        self.entry_angle.insert(0, "30")
        self.entry_angle.pack(side=tk.LEFT, padx=(2, 12))

        tk.Label(row0, text="Translate x:").pack(side=tk.LEFT)
        self.entry_tx = tk.Entry(row0, width=6)
        self.entry_tx.insert(0, "1.5")
        self.entry_tx.pack(side=tk.LEFT, padx=(2, 12))

        tk.Label(row0, text="Translate y:").pack(side=tk.LEFT)
        self.entry_ty = tk.Entry(row0, width=6)
        self.entry_ty.insert(0, "1.0")
        self.entry_ty.pack(side=tk.LEFT, padx=(2, 0))

        # --- Row 1: action buttons ---
        row1 = tk.Frame(panel)
        row1.pack(fill=tk.X, pady=(0, 4))

        self.btn_generate = tk.Button(
            row1, text="Generate", width=10, command=self.on_generate
        )
        self.btn_generate.pack(side=tk.LEFT, padx=4)

        self.btn_run = tk.Button(
            row1, text="Run", width=10, command=self.on_run, state=tk.DISABLED
        )
        self.btn_run.pack(side=tk.LEFT, padx=4)

        self.btn_exit = tk.Button(
            row1, text="Exit", width=10, command=self.on_exit
        )
        self.btn_exit.pack(side=tk.LEFT, padx=4)

        # --- Row 2: status label ---
        self.lbl_status = tk.Label(
            panel, text="Click Generate to create point sets.", anchor=tk.W, fg="gray"
        )
        self.lbl_status.pack(fill=tk.X)

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------
    def on_generate(self):
        """Generate target and source point sets, reset ICP state."""
        # Parse and validate inputs
        try:
            angle = float(self.entry_angle.get())
            tx = float(self.entry_tx.get())
            ty = float(self.entry_ty.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter numeric values for Angle, Translate x, and Translate y.")
            return

        # Cancel any running animation
        if self.after_id is not None:
            self.master.after_cancel(self.after_id)
            self.after_id = None

        # Generate points
        self.target = self.icp.generate_target(N_POINTS)
        source_init = self.icp.generate_source(self.target, angle, tx, ty)

        # Initialise ICP state
        self.icp.target = self.target
        self.icp.reset()
        self.source_current = source_init.copy()
        self.iteration = 0

        # Draw initial state
        self._update_plot(self.source_current, indices=None)
        self._set_status(f"Generated {N_POINTS} points  (angle={angle} deg, tx={tx}, ty={ty}). "
                         "Press Run to start ICP.")

        # Enable Run, allow re-generate
        self.btn_run.config(state=tk.NORMAL)
        self.btn_generate.config(state=tk.NORMAL)

    def on_run(self):
        """Start the ICP animation."""
        if self.source_current is None:
            messagebox.showinfo("Notice", "Please click Generate first to create point sets.")
            return

        # Disable buttons during animation
        self.btn_generate.config(state=tk.DISABLED)
        self.btn_run.config(state=tk.DISABLED)

        self._set_status("Running ICP...")
        self._run_step()

    def on_exit(self):
        """Cancel any pending animation and close the window."""
        if self.after_id is not None:
            self.master.after_cancel(self.after_id)
        self.master.destroy()

    # ------------------------------------------------------------------
    # ICP animation loop
    # ------------------------------------------------------------------
    def _run_step(self):
        """Execute one ICP iteration and schedule the next via after()."""
        if self.source_current is None:
            return

        self.iteration += 1

        new_source, R, t, indices, converged = self.icp.step(self.source_current)
        self.source_current = new_source

        self._update_plot(self.source_current, indices)

        if converged:
            self._set_status(
                f"Converged after {self.iteration - 1} iteration(s) — "
                "assignment unchanged."
            )
            self.btn_generate.config(state=tk.NORMAL)
            self.after_id = None
            return

        if self.iteration >= MAX_ITERATIONS:
            self._set_status(
                f"Reached maximum iterations ({MAX_ITERATIONS}) — did not converge."
            )
            self.btn_generate.config(state=tk.NORMAL)
            self.after_id = None
            return

        self._set_status(f"Iteration: {self.iteration}")
        self.after_id = self.master.after(ANIMATION_DELAY_MS, self._run_step)

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------
    def _update_plot(self, source_current: np.ndarray, indices):
        """
        Refresh the matplotlib axes with the current source positions
        and nearest-neighbour connecting lines.
        """
        ax = self.ax

        # Update source scatter (red) positions
        self._scatter_source.set_offsets(source_current)

        # Update target scatter (blue) — only needed on first draw
        if self.target is not None:
            self._scatter_target.set_offsets(self.target)

        # Remove previous connecting lines
        while ax.lines:
            ax.lines[0].remove()

        # Draw nearest-neighbour lines (grey dashed)
        if indices is not None and self.target is not None:
            for i, j in enumerate(indices):
                sx, sy = source_current[i]
                tx_, ty_ = self.target[j]
                ax.plot(
                    [sx, tx_], [sy, ty_],
                    color="gray", linestyle="--", linewidth=0.8 * gscale, alpha=0.5,
                    zorder=1,
                )

        # Adjust axes limits to contain both point sets with margin
        if self.target is not None:
            all_pts = np.vstack([self.target, source_current])
            x_min, y_min = all_pts.min(axis=0) - 1.0
            x_max, y_max = all_pts.max(axis=0) + 1.0
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        self.canvas.draw_idle()

    def _set_status(self, message: str):
        self.lbl_status.config(text=message, fg="black")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = ICP2DApp(root)
    root.mainloop()
