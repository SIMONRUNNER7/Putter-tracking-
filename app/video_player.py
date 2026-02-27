"""
Video player widget with:
  - Frame-by-frame scrubbing
  - Slow motion (½x, ¼x)
  - Play / Pause
  - Key-frame markers on scrubber
  - Overlay rendering (arc, shaft, HUD)
  - Click-to-set-template functionality
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Callable
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QSizePolicy, QToolButton, QComboBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPoint, QRect
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QFont,
    QMouseEvent,
)

from models.shot_data import ShotRecord, PutterPosition, KeyFrames
from utils.drawing import (
    draw_putter, draw_trajectory_arc, draw_hud,
    draw_keyframe_marker,
)


# ---------------------------------------------------------------------------
# Frame scrubber with key-frame markers
# ---------------------------------------------------------------------------

class ScrubBar(QWidget):
    """
    Custom scrub bar that shows key-frame markers as coloured ticks.
    """

    position_changed = pyqtSignal(int)     # frame index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)
        self._total_frames = 1
        self._current = 0
        self._keyframes: dict[str, Optional[int]] = {}
        self._dragging = False
        self.setMouseTracking(True)
        self.setStyleSheet("background: #0e0e1e;")

    def set_total_frames(self, n: int) -> None:
        self._total_frames = max(1, n)
        self.update()

    def set_keyframes(self, kf: KeyFrames) -> None:
        self._keyframes = kf.as_dict()
        self.update()

    def set_position(self, frame: int) -> None:
        self._current = max(0, min(frame, self._total_frames - 1))
        self.update()

    # --- Paint
    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Background track
        track_y, track_h = h // 2 - 3, 6
        p.fillRect(QRect(0, track_y, w, track_h), QColor(30, 30, 60))

        # Played portion
        if self._total_frames > 1:
            played_w = int(w * self._current / (self._total_frames - 1))
            p.fillRect(QRect(0, track_y, played_w, track_h), QColor(70, 100, 200))

        # Key frame ticks
        kf_colors = {
            "Address":        QColor(100, 220, 100),
            "Backswing peak": QColor(255, 165, 0),
            "Impact":         QColor(255, 50, 50),
            "Follow-through": QColor(0, 180, 255),
        }
        p.setFont(QFont("Arial", 7))
        for name, fi in self._keyframes.items():
            if fi is None:
                continue
            tx = int(w * fi / max(1, self._total_frames - 1))
            col = kf_colors.get(name, QColor(200, 200, 200))
            p.setPen(QPen(col, 2))
            p.drawLine(tx, 2, tx, h - 2)
            p.setPen(QPen(col))
            p.drawText(QRect(tx + 2, 2, 60, 12), Qt.AlignmentFlag.AlignLeft,
                       name[:3])

        # Playhead
        px_x = int(w * self._current / max(1, self._total_frames - 1))
        p.setPen(QPen(QColor(220, 220, 220), 2))
        p.drawLine(px_x, 0, px_x, h)
        p.setBrush(QColor(220, 220, 220))
        p.drawEllipse(QPoint(px_x, h // 2), 5, 5)

        p.end()

    # --- Mouse interaction
    def mousePressEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._seek_to(e.position().x())

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self._dragging:
            self._seek_to(e.position().x())

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        self._dragging = False

    def _seek_to(self, screen_x: float) -> None:
        frac = max(0.0, min(1.0, screen_x / self.width()))
        frame = int(frac * (self._total_frames - 1))
        self._current = frame
        self.position_changed.emit(frame)
        self.update()


# ---------------------------------------------------------------------------
# Video display label (with rubber-band for template bbox)
# ---------------------------------------------------------------------------

class VideoLabel(QLabel):
    """QLabel subclass that handles rubber-band selection."""

    bbox_selected = pyqtSignal(tuple)   # (x, y, w, h) in original frame coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self._selecting = False
        self._select_start: Optional[QPoint] = None
        self._select_rect: Optional[QRect] = None
        self._orig_size: tuple[int, int] = (1, 1)   # original frame (w, h)

    def set_original_size(self, w: int, h: int) -> None:
        self._orig_size = (w, h)

    def enable_selection(self) -> None:
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._selecting = True

    def disable_selection(self) -> None:
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._selecting = False
        self._select_rect = None
        self.update()

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if self._selecting and e.button() == Qt.MouseButton.LeftButton:
            self._select_start = e.position().toPoint()
            self._select_rect = None

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self._selecting and self._select_start is not None:
            self._select_rect = QRect(self._select_start, e.position().toPoint()).normalized()
            self.update()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        if self._selecting and self._select_rect is not None:
            # Convert widget coords to original frame coords
            scale_x = self._orig_size[0] / self.width()
            scale_y = self._orig_size[1] / self.height()
            r = self._select_rect
            bbox = (
                int(r.x() * scale_x), int(r.y() * scale_y),
                int(r.width() * scale_x), int(r.height() * scale_y),
            )
            if bbox[2] > 5 and bbox[3] > 5:
                self.bbox_selected.emit(bbox)
            self.disable_selection()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._select_rect:
            p = QPainter(self)
            pen = QPen(QColor(0, 200, 255), 2, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawRect(self._select_rect)
            p.end()


# ---------------------------------------------------------------------------
# Main video player widget
# ---------------------------------------------------------------------------

class VideoPlayer(QWidget):
    """
    Full-featured video player for the analysis app.

    Public API:
      load_record(ShotRecord)       – load a completed analysis
      jump_to_frame(int)            – seek programmatically
      request_template_selection()  – start rubber-band selection mode
    """

    template_defined = pyqtSignal(int, tuple)   # (frame_idx, bbox)
    frame_changed    = pyqtSignal(int)           # current frame index

    # Playback speeds
    SPEEDS = {"¼x": 4, "½x": 2, "1x": 1}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: #0a0a18;")
        self._record: Optional[ShotRecord] = None
        self._frames:  list[np.ndarray] = []
        self._current_frame = 0
        self._playing = False
        self._speed_divisor = 1   # 1 = real speed, 2 = half speed, etc.
        self._show_overlay = True
        self._show_arc = True

        self._build_ui()
        self._timer = QTimer()
        self._timer.timeout.connect(self._advance_frame)

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        # Video display
        self._video_label = VideoLabel()
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setMinimumSize(480, 320)
        self._video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._video_label.setStyleSheet("background: #000;")
        self._video_label.bbox_selected.connect(self._on_bbox_selected)
        root.addWidget(self._video_label)

        # Scrub bar
        self._scrub = ScrubBar()
        self._scrub.position_changed.connect(self._on_scrub)
        root.addWidget(self._scrub)

        # Controls bar
        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(8, 0, 8, 4)
        ctrl.setSpacing(6)

        btn_style = """
            QPushButton {
                background: #1a1a3a;
                color: #ddd;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 12px;
            }
            QPushButton:hover { background: #2a2a5a; }
            QPushButton:pressed { background: #0a0a2a; }
        """
        self._btn_prev = QPushButton("◄")
        self._btn_play = QPushButton("▶")
        self._btn_next = QPushButton("►")
        self._btn_prev.clicked.connect(self._step_back)
        self._btn_play.clicked.connect(self._toggle_play)
        self._btn_next.clicked.connect(self._step_forward)
        for b in (self._btn_prev, self._btn_play, self._btn_next):
            b.setStyleSheet(btn_style)
            b.setFixedSize(38, 28)

        ctrl.addWidget(self._btn_prev)
        ctrl.addWidget(self._btn_play)
        ctrl.addWidget(self._btn_next)

        # Speed selector
        ctrl.addWidget(QLabel("Speed:"))
        self._speed_box = QComboBox()
        self._speed_box.addItems(["¼x", "½x", "1x"])
        self._speed_box.setCurrentText("1x")
        self._speed_box.currentTextChanged.connect(self._on_speed_change)
        self._speed_box.setStyleSheet("""
            QComboBox {
                background: #1a1a3a; color: #ddd;
                border: 1px solid #333; border-radius: 4px;
                padding: 2px 4px; font-size: 11px;
            }
        """)
        ctrl.addWidget(self._speed_box)

        # Frame counter
        self._frame_label = QLabel("0 / 0")
        self._frame_label.setStyleSheet("color: #888; font-size: 10px;")
        ctrl.addWidget(self._frame_label)

        ctrl.addStretch()

        # Overlay toggles
        self._btn_overlay = QPushButton("Overlay ON")
        self._btn_overlay.setCheckable(True)
        self._btn_overlay.setChecked(True)
        self._btn_overlay.clicked.connect(self._toggle_overlay)
        self._btn_overlay.setStyleSheet(btn_style)

        self._btn_template = QPushButton("Set Putter ROI")
        self._btn_template.clicked.connect(self._start_roi_selection)
        self._btn_template.setStyleSheet(btn_style)
        self._btn_template.setToolTip(
            "Draw a box around the putter head to help the tracker"
        )

        ctrl.addWidget(self._btn_overlay)
        ctrl.addWidget(self._btn_template)

        root.addLayout(ctrl)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_record(self, record: ShotRecord, raw_frames: list[np.ndarray]) -> None:
        """Load a ShotRecord + decoded raw frames."""
        self._record = record
        self._frames = raw_frames
        total = len(raw_frames)
        self._scrub.set_total_frames(total)
        self._scrub.set_keyframes(record.keyframes)
        if record.frame_count > 0:
            self._video_label.set_original_size(
                raw_frames[0].shape[1] if raw_frames else 640,
                raw_frames[0].shape[0] if raw_frames else 480,
            )
        self._current_frame = 0
        self._show_frame(0)

    def jump_to_frame(self, idx: int) -> None:
        idx = max(0, min(idx, len(self._frames) - 1))
        self._current_frame = idx
        self._scrub.set_position(idx)
        self._show_frame(idx)

    def request_template_selection(self) -> None:
        self._video_label.enable_selection()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _show_frame(self, idx: int) -> None:
        if not self._frames or idx >= len(self._frames):
            return
        frame = self._frames[idx].copy()

        if self._show_overlay and self._record:
            frame = self._render_overlay(frame, idx)

        self._display_frame(frame)
        self._frame_label.setText(f"{idx + 1} / {len(self._frames)}")
        self._scrub.set_position(idx)
        self.frame_changed.emit(idx)

    def _render_overlay(self, frame: np.ndarray, idx: int) -> np.ndarray:
        record = self._record
        pos_map = {p.frame_index: p for p in record.positions}
        pos = pos_map.get(idx)

        if pos and pos.confidence > 0.15:
            frame = draw_putter(frame, pos)

        # Arc (last N positions up to current frame)
        if self._show_arc:
            arc_pts = [
                (p.head_x, p.head_y)
                for p in record.positions
                if p.frame_index <= idx and p.confidence > 0.15
            ]
            if arc_pts:
                impact_local = None
                if record.keyframes.impact is not None:
                    # Find local index of impact in arc_pts sequence
                    impact_fi = record.keyframes.impact
                    impact_positions = [
                        i for i, p in enumerate(record.positions)
                        if p.frame_index == impact_fi
                    ]
                    if impact_positions:
                        ip_fi = impact_positions[0]
                        # How many of those are ≤ current?
                        visible_pts = [
                            p for p in record.positions
                            if p.frame_index <= idx and p.confidence > 0.15
                        ]
                        impact_local = next(
                            (i for i, p in enumerate(visible_pts)
                             if p.frame_index == impact_fi), None
                        )
                frame = draw_trajectory_arc(frame, arc_pts, impact_local)

        # HUD
        m = record.metrics
        md = m.to_display_dict() if m else {}
        phase = pos.phase.value.upper() if pos else ""
        conf = pos.confidence if pos else 0.0
        frame = draw_hud(frame, md, phase_label=phase, confidence=conf)

        # Key-frame labels
        kf_labels = {
            record.keyframes.address:        "A",
            record.keyframes.backswing_peak: "BS",
            record.keyframes.impact:         "IMP",
            record.keyframes.follow_through: "FT",
        }
        for kf_fi, label in kf_labels.items():
            if kf_fi == idx and kf_fi in pos_map:
                p = pos_map[kf_fi]
                frame = draw_keyframe_marker(
                    frame, label, (int(p.head_x), int(p.head_y))
                )

        return frame

    def _display_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self._video_label.setPixmap(
            pix.scaled(self._video_label.size(),
                       Qt.AspectRatioMode.KeepAspectRatio,
                       Qt.TransformationMode.SmoothTransformation)
        )

    # --- Playback
    def _toggle_play(self) -> None:
        if not self._frames:
            return
        self._playing = not self._playing
        self._btn_play.setText("⏸" if self._playing else "▶")
        if self._playing:
            fps = self._record.fps if self._record else 30.0
            interval = max(1, int(1000 / fps * self._speed_divisor))
            self._timer.start(interval)
        else:
            self._timer.stop()

    def _advance_frame(self) -> None:
        if self._current_frame >= len(self._frames) - 1:
            self._playing = False
            self._timer.stop()
            self._btn_play.setText("▶")
            return
        self._current_frame += 1
        self._show_frame(self._current_frame)

    def _step_back(self) -> None:
        self.jump_to_frame(self._current_frame - 1)

    def _step_forward(self) -> None:
        self.jump_to_frame(self._current_frame + 1)

    def _on_scrub(self, frame: int) -> None:
        self._playing = False
        self._timer.stop()
        self._btn_play.setText("▶")
        self._current_frame = frame
        self._show_frame(frame)

    def _on_speed_change(self, text: str) -> None:
        self._speed_divisor = self.SPEEDS.get(text, 1)

    def _toggle_overlay(self) -> None:
        self._show_overlay = self._btn_overlay.isChecked()
        self._btn_overlay.setText("Overlay ON" if self._show_overlay else "Overlay OFF")
        self._show_frame(self._current_frame)

    def _start_roi_selection(self) -> None:
        self.request_template_selection()

    def _on_bbox_selected(self, bbox: tuple) -> None:
        self.template_defined.emit(self._current_frame, bbox)
