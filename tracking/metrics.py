"""
Golf metrics computation from tracked putter positions.

Computes:
  - Club path angle at impact (relative to target line)
  - Face angle at impact
  - Face-to-path deviation
  - Impact speed (in mph if calibrated)
  - Tempo ratio (backswing time / downswing time)
  - Estimated ball launch direction (D-plane simplified)
  - Estimated distance
  - Impact quality score
"""
from __future__ import annotations

import numpy as np
import math
from typing import Optional
from models.shot_data import (
    PutterPosition, ShotMetrics, KeyFrames, BallData,
    ClubPathType
)
from utils.geometry import fit_arc, arc_tangent_at, classify_path


# Physical calibration defaults
# These are overridden when the user performs a calibration step.
DEFAULT_PX_PER_METER = 300.0          # rough default
BALL_DIAMETER_M = 0.04267             # 42.67 mm (USGA)
TYPICAL_IMPACT_SPEED_MPH = 5.0        # rough default for conversion check


class MetricsComputer:
    """
    Compute all golf metrics from a sequence of PutterPositions.
    """

    def compute(
        self,
        positions: list[PutterPosition],
        kf: KeyFrames,
        fps: float,
        target_line_angle_deg: float = 0.0,
        px_per_cm: Optional[float] = None,
    ) -> ShotMetrics:
        metrics = ShotMetrics()

        if not positions:
            return metrics

        good = [p for p in positions if p.confidence > 0.2]
        if len(good) < 5:
            good = positions  # use all if we don't have enough

        xs = np.array([p.head_x for p in good])
        ys = np.array([p.head_y for p in good])
        ts = np.array([p.timestamp for p in good])

        # ---- Club path at impact --------------------------------------
        impact_pos = _position_at_keyframe(positions, kf.impact)
        if impact_pos is not None:
            # Fit arc to positions around impact (±20 frames)
            fi = kf.impact or 0
            window = [p for p in good
                      if abs(p.frame_index - fi) <= 20]
            if len(window) >= 3:
                arc_pts = [(p.head_x, p.head_y) for p in window]
                coeffs = fit_arc(arc_pts, degree=2)
                if coeffs is not None:
                    metrics.arc_coefficients = coeffs
                    # Tangent at impact x position
                    tang_deg = arc_tangent_at(coeffs, impact_pos.head_x)
                    # Club path relative to target line
                    metrics.club_path_deg = round(tang_deg - target_line_angle_deg, 2)
                    metrics.club_path_type = classify_path(metrics.club_path_deg)

        # ---- Face angle at impact -------------------------------------
        if impact_pos is not None and impact_pos.face_angle is not None:
            metrics.face_angle_deg = round(impact_pos.face_angle, 2)

        # ---- Face-to-path --------------------------------------------
        if metrics.club_path_deg is not None and metrics.face_angle_deg is not None:
            metrics.face_to_path_deg = round(
                metrics.face_angle_deg - metrics.club_path_deg, 2
            )

        # ---- Impact speed --------------------------------------------
        if kf.impact is not None:
            fi = kf.impact
            # Use a small window around impact (±3 frames)
            around = [p for p in good
                      if abs(p.frame_index - fi) <= 3]
            if len(around) >= 2:
                dx = around[-1].head_x - around[0].head_x
                dy = around[-1].head_y - around[0].head_y
                dt = around[-1].timestamp - around[0].timestamp
                if dt > 0:
                    speed_px_s = math.hypot(dx, dy) / dt
                    if px_per_cm is not None:
                        speed_cm_s = speed_px_s / px_per_cm
                        speed_mph = speed_cm_s * 0.0224   # cm/s → mph
                        metrics.impact_speed_mph = round(speed_mph, 1)
                    else:
                        # Normalised estimate (not physically accurate)
                        metrics.impact_speed_mph = round(speed_px_s / 40.0, 1)

        # ---- Tempo ---------------------------------------------------
        if kf.address is not None and kf.backswing_peak is not None and kf.impact is not None:
            addr_pos = _position_at_keyframe(positions, kf.address)
            peak_pos = _position_at_keyframe(positions, kf.backswing_peak)
            imp_pos  = _position_at_keyframe(positions, kf.impact)
            if addr_pos and peak_pos and imp_pos:
                back_time = peak_pos.timestamp - addr_pos.timestamp
                down_time = imp_pos.timestamp  - peak_pos.timestamp
                if down_time > 0.001:
                    metrics.backswing_time_s = round(back_time, 3)
                    metrics.downswing_time_s = round(down_time, 3)
                    metrics.tempo_ratio      = round(back_time / down_time, 2)

        # ---- Arc length (full stroke) ---------------------------------
        if len(good) >= 2:
            dists = np.hypot(np.diff(xs), np.diff(ys))
            metrics.arc_length_px = float(np.sum(dists))

        # ---- Ball data (estimated, D-plane simplified) ----------------
        ball = BallData()
        if metrics.face_angle_deg is not None and metrics.club_path_deg is not None:
            # Ball starts ~75% toward face angle (D-plane rule of thumb)
            ball.launch_direction_deg = round(
                0.75 * metrics.face_angle_deg + 0.25 * metrics.club_path_deg, 2
            )
        if metrics.impact_speed_mph is not None:
            # Typical putter gear effect: distance ≈ speed² / constant
            # For a 5 mph putt ≈ 10 ft (rough estimate)
            ball.estimated_distance_ft = round(
                (metrics.impact_speed_mph ** 1.8) * 0.55, 1
            )
            ball.estimated_speed_mph = round(metrics.impact_speed_mph * 0.85, 1)
        metrics.ball = ball

        # ---- Impact quality score ------------------------------------
        metrics.impact_score = _compute_impact_score(metrics)

        return metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _position_at_keyframe(
    positions: list[PutterPosition],
    frame_idx: Optional[int],
) -> Optional[PutterPosition]:
    if frame_idx is None:
        return None
    for p in positions:
        if p.frame_index == frame_idx:
            return p
    # Find closest
    closest = min(positions, key=lambda p: abs(p.frame_index - frame_idx), default=None)
    return closest


def _compute_impact_score(metrics: ShotMetrics) -> float:
    """
    Score from 0-100 based on:
      - Face angle closeness to square (0°) – 40 pts
      - Path type (In-to-In best)           – 30 pts
      - Face-to-path closeness to 0°        – 30 pts
    """
    score = 50.0   # Start at 50 if no data

    if metrics.face_angle_deg is not None:
        # Max 40 points for face angle ≤ 0.5°
        fa = abs(metrics.face_angle_deg)
        face_pts = max(0.0, 40.0 - fa * 12.0)
        score = face_pts
    else:
        score = 20.0

    if metrics.club_path_type is not None:
        path_pts = {
            ClubPathType.IN_TO_IN:  30.0,
            ClubPathType.IN_TO_OUT: 15.0,
            ClubPathType.OUT_TO_IN: 15.0,
        }.get(metrics.club_path_type, 15.0)
        score += path_pts
    else:
        score += 15.0

    if metrics.face_to_path_deg is not None:
        ftp = abs(metrics.face_to_path_deg)
        ftp_pts = max(0.0, 30.0 - ftp * 8.0)
        score += ftp_pts
    else:
        score += 15.0

    return round(min(100.0, max(0.0, score)), 1)
