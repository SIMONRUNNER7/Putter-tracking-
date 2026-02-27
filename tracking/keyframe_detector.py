"""
Automatic key-frame detection from a sequence of PutterPosition objects.

Key frames:
  1. ADDRESS      – putter nearly still, player about to start stroke
  2. BACKSWING PEAK – direction of motion reverses (velocity sign change)
  3. IMPACT       – putter reaches lowest y-coordinate (closest to ball),
                    maximum speed
  4. FOLLOW-THROUGH – motion slows again after impact
"""
from __future__ import annotations

import numpy as np
from typing import Optional
from models.shot_data import PutterPosition, KeyFrames


class KeyFrameDetector:

    # Minimum motion (pixels) needed to consider stroke "started"
    MOTION_THRESHOLD_PX = 5.0
    # Minimum frames in backswing to be valid
    MIN_BACKSWING_FRAMES = 5

    def detect(
        self,
        positions: list[PutterPosition],
        fps: float,
        min_confidence: float = 0.15,
    ) -> KeyFrames:
        kf = KeyFrames()
        n = len(positions)
        if n < 10:
            return kf

        # Filter to reasonably confident detections only
        good = [(p.frame_index, p.head_x, p.head_y)
                for p in positions if p.confidence >= min_confidence]
        if len(good) < 10:
            # Fall back to all positions
            good = [(p.frame_index, p.head_x, p.head_y) for p in positions]

        idxs = np.array([g[0] for g in good])
        xs   = np.array([g[1] for g in good], dtype=float)
        ys   = np.array([g[2] for g in good], dtype=float)

        # Smooth with a short rolling mean to reduce noise
        window = max(3, int(fps * 0.05))  # ~50 ms window
        xs_s = _rolling_mean(xs, window)
        ys_s = _rolling_mean(ys, window)

        # Velocities (pixels/frame)
        vx = np.gradient(xs_s)
        vy = np.gradient(ys_s)
        speed = np.hypot(vx, vy)

        # ---- ADDRESS: last frame before sustained motion starts
        first_motion = _first_sustained_motion(speed, self.MOTION_THRESHOLD_PX,
                                               min_frames=self.MIN_BACKSWING_FRAMES)
        if first_motion is not None and first_motion > 0:
            kf.address = int(idxs[max(0, first_motion - 1)])

        # ---- BACKSWING PEAK: velocity reversal (putter changes direction)
        # The putter typically moves in the x-direction during a putt.
        # We look for the point where vx changes sign after address.
        if first_motion is not None:
            sign_changes = _sign_change_indices(vx[first_motion:])
            if sign_changes:
                # First sign change after motion start = backswing peak
                peak_local = sign_changes[0]
                peak_global = first_motion + peak_local
                if peak_global < len(idxs):
                    kf.backswing_peak = int(idxs[peak_global])

        # ---- IMPACT: maximum speed after backswing peak
        if kf.backswing_peak is not None:
            bp_local = np.searchsorted(idxs, kf.backswing_peak)
            if bp_local < len(speed) - 3:
                after_peak_speed = speed[bp_local:]
                local_max = int(np.argmax(after_peak_speed))
                impact_local = bp_local + local_max
                kf.impact = int(idxs[min(impact_local, len(idxs) - 1)])

        # ---- FOLLOW-THROUGH: first speed drop below threshold after impact
        if kf.impact is not None:
            impact_local = np.searchsorted(idxs, kf.impact)
            if impact_local < len(speed) - 3:
                after_impact_speed = speed[impact_local:]
                peak_s = after_impact_speed[0] if len(after_impact_speed) > 0 else 1.0
                threshold = peak_s * 0.3
                below = np.where(after_impact_speed < threshold)[0]
                if len(below) > 0:
                    ft_local = impact_local + int(below[0])
                    kf.follow_through = int(idxs[min(ft_local, len(idxs) - 1)])
                else:
                    # Default: last 10% of frames
                    kf.follow_through = int(idxs[min(impact_local + 10, len(idxs) - 1)])

        return kf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple rolling mean with edge handling."""
    if window <= 1:
        return arr.copy()
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')


def _first_sustained_motion(
    speed: np.ndarray,
    threshold: float,
    min_frames: int,
) -> Optional[int]:
    """Return index of first frame where speed stays above threshold for min_frames."""
    n = len(speed)
    count = 0
    start = None
    for i in range(n):
        if speed[i] >= threshold:
            if count == 0:
                start = i
            count += 1
            if count >= min_frames:
                return start
        else:
            count = 0
            start = None
    return None


def _sign_change_indices(arr: np.ndarray) -> list[int]:
    """Return indices where arr changes sign (ignoring zeros)."""
    changes = []
    prev_sign = 0
    for i, v in enumerate(arr):
        if v > 0.1:
            s = 1
        elif v < -0.1:
            s = -1
        else:
            continue
        if prev_sign != 0 and s != prev_sign:
            changes.append(i)
        prev_sign = s
    return changes
