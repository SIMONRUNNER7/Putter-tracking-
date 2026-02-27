"""
Right-side analysis panel widget.

Displays:
  - Key metrics (club path, face angle, speed, tempo, distance…)
  - Impact quality bar
  - D-plane explanation
  - Key-frame navigator buttons
"""
from __future__ import annotations

from typing import Optional, Callable
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QProgressBar, QPushButton, QScrollArea, QGridLayout,
    QGroupBox, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPalette

from models.shot_data import ShotMetrics, KeyFrames, ClubPathType


# ---------------------------------------------------------------------------
# Metric row widget
# ---------------------------------------------------------------------------

class MetricRow(QWidget):
    """One label + value row in the metrics panel."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        self._label = QLabel(label)
        self._label.setStyleSheet("color: #888; font-size: 11px;")
        self._label.setFixedWidth(110)

        self._value = QLabel("—")
        self._value.setStyleSheet(
            "color: #e8e8e8; font-size: 13px; font-weight: bold;"
        )
        self._value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(self._label)
        layout.addWidget(self._value, 1)

    def set_value(self, text: str, color: str = "#e8e8e8") -> None:
        self._value.setText(text)
        self._value.setStyleSheet(
            f"color: {color}; font-size: 13px; font-weight: bold;"
        )


# ---------------------------------------------------------------------------
# Impact quality bar
# ---------------------------------------------------------------------------

class ImpactBar(QWidget):
    """A colour-coded progress bar for impact quality (0–100)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        header = QLabel("Impact Quality")
        header.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(header)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setFixedHeight(18)
        self._bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #333;
                border-radius: 4px;
                background: #1a1a2e;
                color: white;
                font-size: 10px;
            }
            QProgressBar::chunk {
                border-radius: 3px;
            }
        """)
        layout.addWidget(self._bar)

    def set_score(self, score: Optional[float]) -> None:
        if score is None:
            self._bar.setValue(0)
            self._bar.setFormat("—")
            return
        v = int(score)
        self._bar.setValue(v)
        self._bar.setFormat(f"{v}/100")
        if v >= 75:
            color = "#00c853"   # green
        elif v >= 50:
            color = "#ff8f00"   # amber
        else:
            color = "#d32f2f"   # red
        self._bar.setStyleSheet(self._bar.styleSheet().replace(
            "QProgressBar::chunk {", f"QProgressBar::chunk {{ background: {color};"
        ))


# ---------------------------------------------------------------------------
# Key-frame navigator
# ---------------------------------------------------------------------------

class KeyFrameNav(QWidget):
    """Buttons to jump to each key frame in the video player."""

    frame_requested = pyqtSignal(int)   # emits the frame index to jump to

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self._btns: dict[str, QPushButton] = {}
        for name in ["Address", "Backswing", "Impact", "Follow-thru"]:
            btn = QPushButton(name)
            btn.setEnabled(False)
            btn.setFixedHeight(26)
            btn.setStyleSheet("""
                QPushButton {
                    background: #1e1e3a;
                    color: #aaa;
                    border: 1px solid #333;
                    border-radius: 4px;
                    font-size: 10px;
                    padding: 2px 4px;
                }
                QPushButton:enabled {
                    background: #2a2a5a;
                    color: #e8e8e8;
                    border-color: #5050a0;
                }
                QPushButton:enabled:hover {
                    background: #3a3a7a;
                }
            """)
            layout.addWidget(btn)
            self._btns[name] = btn

        self._frame_indices: dict[str, Optional[int]] = {k: None for k in self._btns}

    def update_keyframes(self, kf: KeyFrames) -> None:
        mapping = {
            "Address":      kf.address,
            "Backswing":    kf.backswing_peak,
            "Impact":       kf.impact,
            "Follow-thru":  kf.follow_through,
        }
        for name, fi in mapping.items():
            btn = self._btns[name]
            self._frame_indices[name] = fi
            btn.setEnabled(fi is not None)
            if fi is not None:
                # Disconnect old connections
                try:
                    btn.clicked.disconnect()
                except RuntimeError:
                    pass
                btn.clicked.connect(lambda _, f=fi: self.frame_requested.emit(f))


# ---------------------------------------------------------------------------
# Main analysis panel
# ---------------------------------------------------------------------------

class AnalysisPanel(QWidget):
    """
    Full right-side panel.
    Signals:
      keyframe_jump(int) — user clicked a key-frame button → jump to frame
    """

    keyframe_jump = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(270)
        self.setMaximumWidth(350)
        self.setStyleSheet("background-color: #12121e; color: #e8e8e8;")

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        # --- Title
        title = QLabel("SHOT ANALYSIS")
        title.setStyleSheet(
            "color: #80d080; font-size: 14px; font-weight: bold; letter-spacing: 2px;"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        # --- Path type badge
        self._path_badge = QLabel("—")
        self._path_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._path_badge.setFixedHeight(28)
        self._path_badge.setStyleSheet(
            "background: #1e1e3a; border-radius: 6px; color: #aaa; font-size: 12px;"
        )
        root.addWidget(self._path_badge)

        # --- Metrics grid
        metrics_group = QGroupBox("Metrics")
        metrics_group.setStyleSheet("""
            QGroupBox {
                color: #608060;
                font-size: 10px;
                border: 1px solid #2a2a4a;
                border-radius: 6px;
                margin-top: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """)
        mg_layout = QVBoxLayout(metrics_group)
        mg_layout.setSpacing(1)

        self._rows: dict[str, MetricRow] = {}
        for label in [
            "Club Path", "Face Angle", "Face-to-Path",
            "Impact Speed", "Tempo", "Distance",
            "Ball Direction",
        ]:
            row = MetricRow(label)
            mg_layout.addWidget(row)
            # Add a thin separator
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setStyleSheet("color: #1e1e38;")
            mg_layout.addWidget(sep)
            self._rows[label] = row

        root.addWidget(metrics_group)

        # --- Impact quality
        self._impact_bar = ImpactBar()
        root.addWidget(self._impact_bar)

        # --- D-plane hint
        self._dplane_label = QLabel("")
        self._dplane_label.setWordWrap(True)
        self._dplane_label.setStyleSheet("color: #6080a0; font-size: 10px; padding: 4px;")
        root.addWidget(self._dplane_label)

        # --- Key-frame navigator
        kf_group = QGroupBox("Jump to Key Frame")
        kf_group.setStyleSheet(metrics_group.styleSheet())
        kf_layout = QVBoxLayout(kf_group)
        self._kf_nav = KeyFrameNav()
        self._kf_nav.frame_requested.connect(self.keyframe_jump)
        kf_layout.addWidget(self._kf_nav)
        root.addWidget(kf_group)

        root.addStretch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_metrics(self, metrics: ShotMetrics, kf: KeyFrames) -> None:
        """Refresh all displayed values."""
        d = metrics.to_display_dict()

        def _color_angle(v):
            if v is None:
                return "#aaa"
            return "#00c853" if abs(v) < 1.0 else "#ff8f00" if abs(v) < 2.5 else "#d32f2f"

        self._rows["Club Path"].set_value(d["Club Path"])
        self._rows["Face Angle"].set_value(
            d["Face Angle"], _color_angle(metrics.face_angle_deg)
        )
        self._rows["Face-to-Path"].set_value(
            d["Face-to-Path"], _color_angle(metrics.face_to_path_deg)
        )
        self._rows["Impact Speed"].set_value(d["Impact Speed"])
        self._rows["Tempo"].set_value(d["Tempo"])
        self._rows["Distance"].set_value(d["Distance"])
        self._rows["Ball Direction"].set_value(
            d["Ball Direction"], _color_angle(metrics.ball.launch_direction_deg)
        )

        self._impact_bar.set_score(metrics.impact_score)
        self._kf_nav.update_keyframes(kf)

        # Path type badge
        if metrics.club_path_type is not None:
            type_name = metrics.club_path_type.value
            badge_color = {
                "In-to-In":  "#00c853",
                "In-to-Out": "#ff8f00",
                "Out-to-In": "#d32f2f",
            }.get(type_name, "#aaa")
            self._path_badge.setText(type_name)
            self._path_badge.setStyleSheet(
                f"background: #1e1e3a; border-radius: 6px; "
                f"color: {badge_color}; font-size: 12px; font-weight: bold;"
            )

        # D-plane hint
        self._update_dplane_hint(metrics)

    def clear(self) -> None:
        for row in self._rows.values():
            row.set_value("—")
        self._impact_bar.set_score(None)
        self._path_badge.setText("—")
        self._dplane_label.setText("")

    def _update_dplane_hint(self, metrics: ShotMetrics) -> None:
        fa = metrics.face_angle_deg
        cp = metrics.club_path_deg
        if fa is None or cp is None:
            self._dplane_label.setText("")
            return

        lines = []
        if abs(fa) < 0.5:
            lines.append("Square face — good start line.")
        elif fa > 0:
            lines.append(f"Face open {fa:.1f}° — ball will start right.")
        else:
            lines.append(f"Face closed {abs(fa):.1f}° — ball will start left.")

        ftp = fa - cp
        if abs(ftp) < 0.5:
            lines.append("No curve expected (straight putt).")
        elif ftp > 0:
            lines.append("Ball curves right (clockwise spin).")
        else:
            lines.append("Ball curves left (counter-clockwise spin).")

        self._dplane_label.setText("  ".join(lines))
