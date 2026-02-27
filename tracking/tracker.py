"""
Full tracking pipeline: orchestrates per-frame detection and accumulates
a ShotRecord over the duration of a video.
"""
from __future__ import annotations

import cv2
import numpy as np
import time
from typing import Optional, Callable
from models.shot_data import (
    ShotRecord, PutterPosition, SwingPhase, ShotMetrics, KeyFrames
)
from tracking.putter_detector import PutterDetector, Detection
from tracking.metrics import MetricsComputer
from tracking.keyframe_detector import KeyFrameDetector


class PutterTracker:
    """
    Drives the full analysis pipeline for one video.

    Usage:
        tracker = PutterTracker()
        tracker.load_video(path)
        tracker.set_target_line(angle_deg)
        record = tracker.run(progress_cb=lambda p: ...)
    """

    # Detection warmup: analyse this many frames before expecting movement
    WARMUP_FRAMES = 30

    def __init__(self, use_yolo: bool = False, yolo_model_path: Optional[str] = None):
        self._cap: Optional[cv2.VideoCapture] = None
        self._video_path: str = ""
        self._fps: float = 30.0
        self._frame_count: int = 0
        self._target_line_angle: float = 0.0
        self._use_yolo = use_yolo
        self._yolo_model_path = yolo_model_path

        self._detector: Optional[PutterDetector] = None
        self._metrics_computer = MetricsComputer()
        self._kf_detector = KeyFrameDetector()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load_video(self, path: str) -> tuple[bool, str]:
        """Open the video file. Returns (success, error_message)."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False, f"Cannot open video: {path}"
        self._cap = cap
        self._video_path = path
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return True, ""

    def set_target_line(self, angle_deg: float) -> None:
        """Set the target line angle (screen angle in degrees, 0 = horizontal)."""
        self._target_line_angle = angle_deg

    def set_template_from_frame(self, frame_idx: int, bbox: tuple[int, int, int, int]) -> None:
        """Seed the detector template from a specific frame & bounding box."""
        if self._cap is None:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        if ret and self._detector is not None:
            self._detector.set_template(frame, bbox)

    # ------------------------------------------------------------------
    # Main analysis run
    # ------------------------------------------------------------------

    def run(
        self,
        progress_cb: Optional[Callable[[float, np.ndarray], None]] = None,
        stop_flag: Optional[Callable[[], bool]] = None,
    ) -> ShotRecord:
        """
        Process the entire video and return a fully populated ShotRecord.

        Args:
            progress_cb: called with (progress 0-1, annotated_frame) each frame.
            stop_flag:   callable that returns True when the user cancels.
        """
        assert self._cap is not None, "Call load_video() first."

        self._detector = PutterDetector(
            target_line_angle=self._target_line_angle,
            use_yolo=self._use_yolo,
            yolo_model_path=self._yolo_model_path,
        )

        # --- Warmup: feed background subtractor static frames --------
        warmup_frames = []
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(min(self.WARMUP_FRAMES, self._frame_count)):
            ret, f = self._cap.read()
            if ret:
                warmup_frames.append(f)
        if warmup_frames:
            self._detector.initialize_background(warmup_frames)

        # --- Main loop -----------------------------------------------
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        positions: list[PutterPosition] = []

        for frame_idx in range(self._frame_count):
            if stop_flag and stop_flag():
                break

            ret, frame = self._cap.read()
            if not ret:
                break

            timestamp = frame_idx / self._fps
            det = self._detector.detect(frame, frame_idx)

            pos = PutterPosition(
                frame_index=frame_idx,
                timestamp=timestamp,
                head_x=det.head_x,
                head_y=det.head_y,
                shaft_start=det.shaft_start,
                shaft_end=det.shaft_end,
                face_angle=det.face_angle,
                shaft_angle=det.shaft_angle,
                confidence=det.confidence,
                detection_method=det.method,
            )
            positions.append(pos)

            if progress_cb:
                progress_cb(frame_idx / max(1, self._frame_count - 1), frame)

        # --- Detect key frames ---------------------------------------
        kf = self._kf_detector.detect(positions, self._fps)

        # --- Label phases -------------------------------------------
        positions = _label_phases(positions, kf)

        # --- Compute metrics ----------------------------------------
        metrics = self._metrics_computer.compute(
            positions, kf, self._fps, self._target_line_angle
        )

        record = ShotRecord(
            shot_id=f"shot_{int(time.time())}",
            video_path=self._video_path,
            fps=self._fps,
            frame_count=len(positions),
            positions=positions,
            keyframes=kf,
            metrics=metrics,
            target_line_angle_deg=self._target_line_angle,
        )
        return record

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None


# ---------------------------------------------------------------------------
# Phase labelling
# ---------------------------------------------------------------------------

def _label_phases(
    positions: list[PutterPosition],
    kf: KeyFrames,
) -> list[PutterPosition]:
    """Assign SwingPhase to each position based on detected key frames."""
    n = len(positions)
    for i, pos in enumerate(positions):
        idx = pos.frame_index

        if kf.address is not None and idx < kf.address:
            pos.phase = SwingPhase.IDLE
        elif kf.address is not None and idx == kf.address:
            pos.phase = SwingPhase.ADDRESS
        elif kf.backswing_peak is not None and idx < kf.backswing_peak:
            pos.phase = SwingPhase.BACKSWING
        elif kf.backswing_peak is not None and idx == kf.backswing_peak:
            pos.phase = SwingPhase.TRANSITION
        elif kf.impact is not None and idx < kf.impact:
            pos.phase = SwingPhase.DOWNSWING
        elif kf.impact is not None and idx == kf.impact:
            pos.phase = SwingPhase.IMPACT
        else:
            pos.phase = SwingPhase.FOLLOW_THROUGH
    return positions
