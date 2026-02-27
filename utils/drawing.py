"""
Drawing utilities: overlay tracking annotations on video frames.
All drawing uses OpenCV on numpy arrays.
"""
from __future__ import annotations
import cv2
import numpy as np
import math
from typing import Optional
from models.shot_data import PutterPosition, ShotMetrics, KeyFrames, SwingPhase


# ---------------------------------------------------------------------------
# Color palette (BGR for OpenCV)
# ---------------------------------------------------------------------------
C_SHAFT        = (0, 220, 255)      # cyan-yellow
C_HEAD_BOX     = (0, 200, 60)       # green
C_FACE_ANGLE   = (0, 100, 255)      # orange-red
C_ARC          = (200, 80, 255)     # purple
C_TARGET_LINE  = (100, 255, 100)    # light green
C_KEYFRAME     = (0, 165, 255)      # orange
C_IMPACT       = (0, 0, 255)        # red
C_TEXT_BG      = (20, 20, 20)
C_TEXT_FG      = (230, 230, 230)
C_BALL         = (255, 255, 255)
C_BALL_DIR     = (0, 200, 255)


def draw_putter(
    frame: np.ndarray,
    pos: PutterPosition,
    draw_shaft: bool = True,
    draw_face: bool = True,
    alpha: float = 0.85,
) -> np.ndarray:
    """Draw putter overlay for one frame position."""
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # --- Head bounding circle
    cx, cy = int(pos.head_x), int(pos.head_y)
    cv2.circle(overlay, (cx, cy), 8, C_HEAD_BOX, 2)
    cv2.circle(overlay, (cx, cy), 2, C_HEAD_BOX, -1)

    # --- Shaft line
    if draw_shaft and pos.shaft_start and pos.shaft_end:
        p1 = (int(pos.shaft_start[0]), int(pos.shaft_start[1]))
        p2 = (int(pos.shaft_end[0]),   int(pos.shaft_end[1]))
        cv2.line(overlay, p1, p2, C_SHAFT, 2, cv2.LINE_AA)

    # --- Face angle indicator (small perpendicular line at head)
    if draw_face and pos.face_angle is not None:
        angle_rad = math.radians(pos.face_angle)
        length = 30
        dx = int(length * math.cos(angle_rad))
        dy = int(length * math.sin(angle_rad))
        cv2.line(overlay, (cx - dx, cy - dy), (cx + dx, cy + dy),
                 C_FACE_ANGLE, 3, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_trajectory_arc(
    frame: np.ndarray,
    positions: list[tuple[float, float]],
    highlight_impact: Optional[int] = None,
    color: tuple = C_ARC,
    thickness: int = 2,
) -> np.ndarray:
    """Draw the arc traced by the putter head."""
    if len(positions) < 2:
        return frame

    pts = np.array([(int(p[0]), int(p[1])) for p in positions], dtype=np.int32)

    # Draw the arc as a polyline
    for i in range(1, len(pts)):
        # Fade older positions
        frac = i / len(pts)
        c = tuple(int(cv2.COLOR_BGR2RGB and ch * frac) for ch in color)
        fade_c = tuple(int(ch * (0.3 + 0.7 * frac)) for ch in color)
        cv2.line(frame, tuple(pts[i-1]), tuple(pts[i]), fade_c, thickness, cv2.LINE_AA)

    # Highlight impact position
    if highlight_impact is not None and 0 <= highlight_impact < len(pts):
        cv2.circle(frame, tuple(pts[highlight_impact]), 12, C_IMPACT, 2)
        cv2.circle(frame, tuple(pts[highlight_impact]), 4, C_IMPACT, -1)

    return frame


def draw_target_line(
    frame: np.ndarray,
    center: tuple[int, int],
    angle_deg: float = 0.0,
    length: int = 600,
    color: tuple = C_TARGET_LINE,
) -> np.ndarray:
    """Draw the calibrated target line through center."""
    cx, cy = center
    angle_rad = math.radians(angle_deg)
    dx = int(length / 2 * math.cos(angle_rad))
    dy = int(length / 2 * math.sin(angle_rad))
    cv2.line(frame, (cx - dx, cy - dy), (cx + dx, cy + dy), color, 1,
             cv2.LINE_AA | cv2.LINE_8)
    return frame


def draw_club_path_arrow(
    frame: np.ndarray,
    origin: tuple[int, int],
    path_angle_deg: float,
    length: int = 80,
    color: tuple = (0, 140, 255),
) -> np.ndarray:
    """Draw club path direction arrow at impact."""
    angle_rad = math.radians(path_angle_deg)
    ex = int(origin[0] + length * math.cos(angle_rad))
    ey = int(origin[1] + length * math.sin(angle_rad))
    cv2.arrowedLine(frame, origin, (ex, ey), color, 2, cv2.LINE_AA, tipLength=0.25)
    return frame


def draw_hud(
    frame: np.ndarray,
    metrics_dict: dict,
    phase_label: str = "",
    confidence: float = 1.0,
) -> np.ndarray:
    """Draw heads-up display with key metrics in a corner panel."""
    h, w = frame.shape[:2]
    panel_w, panel_h = 280, 220
    margin = 12

    # Semi-transparent background
    overlay = frame.copy()
    x0, y0 = margin, margin
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
                  C_TEXT_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Title
    cv2.putText(frame, "PUTTER ANALYSIS", (x0 + 8, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 120), 1, cv2.LINE_AA)

    # Phase indicator
    if phase_label:
        cv2.putText(frame, phase_label, (x0 + 8, y0 + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_KEYFRAME, 1, cv2.LINE_AA)

    # Metrics rows
    y = y0 + 58
    for label, value in list(metrics_dict.items())[:6]:
        cv2.putText(frame, f"{label}:", (x0 + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(frame, str(value), (x0 + 140, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_TEXT_FG, 1, cv2.LINE_AA)
        y += 22

    # Confidence bar
    bar_x, bar_y = x0 + 8, y0 + panel_h - 20
    bar_w = panel_w - 16
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8),
                  (50, 50, 50), -1)
    conf_color = (0, 200, 80) if confidence > 0.6 else (0, 165, 255) if confidence > 0.3 else (0, 0, 200)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + int(bar_w * confidence), bar_y + 8), conf_color, -1)
    cv2.putText(frame, f"Track: {int(confidence*100)}%",
                (bar_x, bar_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)

    return frame


def draw_keyframe_marker(
    frame: np.ndarray,
    label: str,
    position: tuple[int, int],
) -> np.ndarray:
    """Draw a labelled key-frame marker."""
    x, y = position
    cv2.circle(frame, (x, y), 15, C_KEYFRAME, 2)
    cv2.putText(frame, label, (x + 18, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_KEYFRAME, 1, cv2.LINE_AA)
    return frame


def preprocess_for_detection(
    frame: np.ndarray,
    clahe_clip: float = 2.0,
) -> np.ndarray:
    """
    Enhance frame for robust detection under varying lighting.
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to the luminance channel and returns a processed BGR frame.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    # Bilateral filter: reduce noise, preserve edges
    return cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)


def create_thumbnail(frame: np.ndarray, size: tuple[int, int] = (160, 90)) -> np.ndarray:
    """Resize frame to thumbnail."""
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
