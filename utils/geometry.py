"""
Geometric utilities for putter tracking:
  - Arc fitting and classification
  - Angle calculations
  - Kalman filter for smooth tracking
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import splprep, splev
from typing import Optional


# ---------------------------------------------------------------------------
# Basic geometry
# ---------------------------------------------------------------------------

def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Signed angle from v1 to v2 in degrees (positive = CCW)."""
    angle = np.degrees(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0]))
    return (angle + 180) % 360 - 180


def line_angle_deg(p1: tuple, p2: tuple) -> float:
    """Angle of line from p1→p2 relative to horizontal, in [-90, 90]."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def point_to_line_distance(point: np.ndarray, line_pt: np.ndarray, line_dir: np.ndarray) -> float:
    """Perpendicular distance from point to an infinite line."""
    line_dir = line_dir / (np.linalg.norm(line_dir) + 1e-9)
    v = point - line_pt
    return float(np.linalg.norm(v - np.dot(v, line_dir) * line_dir))


def normalize_angle(deg: float) -> float:
    """Wrap angle to [-180, 180]."""
    return (deg + 180) % 360 - 180


# ---------------------------------------------------------------------------
# Arc fitting
# ---------------------------------------------------------------------------

def fit_arc(positions: list[tuple[float, float]], degree: int = 2) -> Optional[np.ndarray]:
    """
    Fit a polynomial arc to a sequence of (x, y) head positions.
    Returns polynomial coefficients (for np.poly1d) mapping x → y,
    or None if not enough points.
    """
    if len(positions) < degree + 1:
        return None
    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])
    try:
        coeffs = np.polyfit(xs, ys, degree)
        return coeffs
    except np.linalg.LinAlgError:
        return None


def evaluate_arc(coeffs: np.ndarray, x_range: tuple[float, float], n: int = 100) -> np.ndarray:
    """
    Evaluate fitted arc polynomial at n points in x_range.
    Returns shape (n, 2) array of (x, y) points.
    """
    xs = np.linspace(x_range[0], x_range[1], n)
    poly = np.poly1d(coeffs)
    ys = poly(xs)
    return np.stack([xs, ys], axis=1)


def arc_tangent_at(coeffs: np.ndarray, x: float) -> float:
    """
    Tangent angle (degrees from horizontal) of fitted arc at position x.
    """
    deriv = np.polyder(np.poly1d(coeffs))
    slope = deriv(x)
    return float(np.degrees(np.arctan(slope)))


def classify_path(path_angle_deg: float, tolerance: float = 1.5) -> str:
    """
    Classify club path as In-to-In, In-to-Out, or Out-to-In
    based on the path angle at impact vs. target line.
    Positive angle = path going right of target (in-to-out for right hander).
    """
    from models.shot_data import ClubPathType
    if abs(path_angle_deg) <= tolerance:
        return ClubPathType.IN_TO_IN
    elif path_angle_deg > 0:
        return ClubPathType.IN_TO_OUT
    else:
        return ClubPathType.OUT_TO_IN


def arc_length(positions: list[tuple[float, float]]) -> float:
    """Total arc length in pixels from a list of (x, y) positions."""
    total = 0.0
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i-1][0]
        dy = positions[i][1] - positions[i-1][1]
        total += np.hypot(dx, dy)
    return total


def smooth_trajectory(positions: list[tuple[float, float]], smoothing: float = 0.5) -> list[tuple[float, float]]:
    """Smooth a 2-D trajectory using cubic B-spline."""
    if len(positions) < 4:
        return positions
    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])
    try:
        tck, u = splprep([xs, ys], s=smoothing * len(positions))
        u_new = np.linspace(0, 1, len(positions))
        xs_s, ys_s = splev(u_new, tck)
        return list(zip(xs_s.tolist(), ys_s.tolist()))
    except Exception:
        return positions


# ---------------------------------------------------------------------------
# Kalman filter for 2-D tracking
# ---------------------------------------------------------------------------

class KalmanTracker2D:
    """
    Constant-velocity Kalman filter for 2-D position tracking.
    State vector: [x, y, vx, vy]
    Measurement:  [x, y]
    """

    def __init__(self, process_noise: float = 1.0, measurement_noise: float = 10.0):
        import cv2
        self.kf = cv2.KalmanFilter(4, 2)

        # Transition matrix (constant velocity)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Measurement matrix (observe x, y only)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32) * 100.0

        self._initialized = False

    def init(self, x: float, y: float):
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self._initialized = True

    def predict(self) -> tuple[float, float]:
        pred = self.kf.predict()
        return float(pred[0]), float(pred[1])

    def update(self, x: float, y: float) -> tuple[float, float]:
        if not self._initialized:
            self.init(x, y)
        meas = np.array([[x], [y]], dtype=np.float32)
        corrected = self.kf.correct(meas)
        return float(corrected[0]), float(corrected[1])

    @property
    def velocity(self) -> tuple[float, float]:
        state = self.kf.statePost
        return float(state[2]), float(state[3])
