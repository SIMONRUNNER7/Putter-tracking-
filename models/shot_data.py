"""
Data models for shot analysis results.
All measurements use consistent units:
  - Angles: degrees (positive = open/right for right-handed golfer)
  - Distances: pixels (converted to cm via calibration)
  - Speed: pixels/frame (converted to mph/m·s⁻¹ via calibration)
  - Time: seconds
"""
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np


class SwingPhase(Enum):
    IDLE       = "idle"
    ADDRESS    = "address"
    BACKSWING  = "backswing"
    TRANSITION = "transition"
    DOWNSWING  = "downswing"
    IMPACT     = "impact"
    FOLLOW_THROUGH = "follow_through"


class ClubPathType(Enum):
    IN_TO_IN   = "In-to-In"
    IN_TO_OUT  = "In-to-Out"       # Push
    OUT_TO_IN  = "Out-to-In"       # Pull / slice path


@dataclass
class PutterPosition:
    """Single-frame position of the putter."""
    frame_index: int
    timestamp: float                    # seconds from start

    # Head center in image coordinates (pixels)
    head_x: float
    head_y: float

    # Shaft: start (grip end) and end (head end) in image pixels
    shaft_start: Optional[tuple] = None  # (x, y)
    shaft_end:   Optional[tuple] = None  # (x, y)

    # Face angle relative to calibrated target line (degrees)
    face_angle: Optional[float] = None

    # Shaft angle relative to vertical (degrees)
    shaft_angle: Optional[float] = None

    # Detection confidence [0..1]
    confidence: float = 0.0

    # Which detection method succeeded
    detection_method: str = "none"

    # Swing phase at this frame
    phase: SwingPhase = SwingPhase.IDLE


@dataclass
class KeyFrames:
    """Indices of the automatically detected key frames."""
    address:        Optional[int] = None
    backswing_peak: Optional[int] = None
    impact:         Optional[int] = None
    follow_through: Optional[int] = None

    def as_dict(self) -> dict:
        return {
            "Address":         self.address,
            "Backswing peak":  self.backswing_peak,
            "Impact":          self.impact,
            "Follow-through":  self.follow_through,
        }


@dataclass
class BallData:
    """Ball tracking / estimated metrics at impact."""
    launch_direction_deg: Optional[float] = None   # relative to target
    estimated_speed_mph: Optional[float] = None
    estimated_distance_ft: Optional[float] = None


@dataclass
class ShotMetrics:
    """
    Full analytics for one putt.
    All angular metrics are in degrees; positive = right-of-target for a
    right-handed golfer.
    """
    # ---------- Club path --------------------------------------------------
    # Angle of the club path at impact vs. target line
    club_path_deg: Optional[float]     = None
    club_path_type: Optional[ClubPathType] = None

    # Face angle at impact vs. target line
    face_angle_deg: Optional[float]    = None

    # Face-to-path difference (determines spin / curve)
    face_to_path_deg: Optional[float]  = None

    # ---------- Speed & tempo ----------------------------------------------
    impact_speed_mph: Optional[float]  = None
    backswing_time_s: Optional[float]  = None
    downswing_time_s: Optional[float]  = None
    tempo_ratio: Optional[float]       = None   # backswing / downswing

    # ---------- Arc geometry -----------------------------------------------
    # Polynomial coefficients fitting the putter arc (degree-2 poly in pixels)
    arc_coefficients: Optional[np.ndarray] = None
    arc_length_px: Optional[float]     = None

    # ---------- Ball -------------------------------------------------------
    ball: BallData = field(default_factory=BallData)

    # ---------- Impact quality score [0..100] ------------------------------
    impact_score: Optional[float]      = None

    def to_display_dict(self) -> dict:
        """Human-readable strings for all metrics."""
        def _fmt_angle(v):
            if v is None:
                return "—"
            sign = "+" if v > 0 else ""
            return f"{sign}{v:.1f}°"

        def _fmt_float(v, unit="", decimals=1):
            if v is None:
                return "—"
            return f"{v:.{decimals}f}{unit}"

        return {
            "Club Path":      f"{_fmt_angle(self.club_path_deg)}  ({self.club_path_type.value if self.club_path_type else '—'})",
            "Face Angle":     _fmt_angle(self.face_angle_deg),
            "Face-to-Path":   _fmt_angle(self.face_to_path_deg),
            "Impact Speed":   _fmt_float(self.impact_speed_mph, " mph"),
            "Tempo":          _fmt_float(self.tempo_ratio, ":1", 2) if self.tempo_ratio else "—",
            "Distance":       _fmt_float(self.ball.estimated_distance_ft, " ft"),
            "Ball Direction": _fmt_angle(self.ball.launch_direction_deg),
            "Impact Score":   _fmt_float(self.impact_score, "/100", 0),
        }


@dataclass
class ShotRecord:
    """One complete putting stroke session."""
    shot_id: str
    video_path: str
    fps: float
    frame_count: int
    thumbnail_path: Optional[str] = None

    positions: list[PutterPosition] = field(default_factory=list)
    keyframes: KeyFrames = field(default_factory=KeyFrames)
    metrics: ShotMetrics = field(default_factory=ShotMetrics)

    # Calibration: pixels per real-world unit (set by user)
    px_per_cm: Optional[float] = None
    target_line_angle_deg: float = 0.0   # screen angle of the target line
