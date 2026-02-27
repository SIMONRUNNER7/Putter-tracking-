"""
Main application window.

Layout:
  ┌──────────────────────────────────────────────┬──────────────────┐
  │                                              │                  │
  │            VIDEO PLAYER                      │  ANALYSIS PANEL  │
  │       (with tracking overlay)                │  (metrics,       │
  │                                              │   arc view,      │
  │                                              │   key frames)    │
  │──────────────────────────────────────────────│                  │
  │   Scrub bar + controls                       │                  │
  ├──────────────────────────────────────────────┴──────────────────┤
  │   SHOT HISTORY (thumbnails row)                                  │
  └──────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import os
import cv2
import json
import threading
import numpy as np
import time
from typing import Optional
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QScrollArea, QLabel, QToolBar, QStatusBar,
    QFileDialog, QMessageBox, QProgressDialog, QFrame,
    QInputDialog, QPushButton, QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QSize
from PyQt6.QtGui import (
    QAction, QIcon, QPixmap, QImage, QKeySequence,
)

from app.video_player import VideoPlayer
from app.analysis_panel import AnalysisPanel
from app.trajectory_view import TrajectoryView
from tracking.tracker import PutterTracker
from models.shot_data import ShotRecord


# ---------------------------------------------------------------------------
# Background analysis worker
# ---------------------------------------------------------------------------

class AnalysisWorker(QObject):
    """Runs PutterTracker.run() in a background thread."""

    progress = pyqtSignal(float, object)   # (0-1, annotated frame or None)
    finished = pyqtSignal(object)          # ShotRecord
    error    = pyqtSignal(str)

    def __init__(self, tracker: PutterTracker):
        super().__init__()
        self._tracker = tracker
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            record = self._tracker.run(
                progress_cb=lambda p, f: self.progress.emit(p, f),
                stop_flag=lambda: self._stop,
            )
            if not self._stop:
                self.finished.emit(record)
        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Shot history thumbnail
# ---------------------------------------------------------------------------

class ShotThumbnail(QLabel):
    """Clickable thumbnail in the history bar."""

    clicked_record = pyqtSignal(object)    # ShotRecord

    def __init__(self, record: ShotRecord, thumb: Optional[np.ndarray], parent=None):
        super().__init__(parent)
        self._record = record
        self.setFixedSize(160, 90)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #2a2a5a;
                border-radius: 4px;
                background: #0a0a18;
            }
            QLabel:hover {
                border-color: #5050a0;
            }
        """)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if thumb is not None:
            self._set_thumb(thumb)
        else:
            self.setText("No preview")
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def _set_thumb(self, bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def mousePressEvent(self, _) -> None:
        self.clicked_record.emit(self._record)


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PutterTrack Pro")
        self.setMinimumSize(1100, 720)
        self.resize(1400, 860)

        self._tracker: Optional[PutterTracker] = None
        self._worker:  Optional[AnalysisWorker] = None
        self._thread:  Optional[QThread] = None
        self._current_record: Optional[ShotRecord] = None
        self._all_frames:  list[np.ndarray] = []  # decoded raw frames
        self._history:     list[ShotRecord] = []

        self._build_ui()
        self._build_menu()
        self._build_toolbar()
        self.statusBar().showMessage("Ready — open a video to start")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ---- Top: video + panels
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: video player
        self._player = VideoPlayer()
        self._player.template_defined.connect(self._on_template_defined)
        self._player.frame_changed.connect(self._on_frame_changed)
        top_splitter.addWidget(self._player)

        # Right: trajectory + analysis panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        self._traj_view = TrajectoryView()
        self._traj_view.setFixedHeight(260)
        right_layout.addWidget(self._traj_view)

        self._analysis = AnalysisPanel()
        self._analysis.keyframe_jump.connect(self._player.jump_to_frame)
        right_layout.addWidget(self._analysis, 1)

        top_splitter.addWidget(right_panel)
        top_splitter.setStretchFactor(0, 3)
        top_splitter.setStretchFactor(1, 1)
        top_splitter.setSizes([950, 380])

        root.addWidget(top_splitter, 1)

        # ---- Bottom: shot history bar
        hist_frame = QFrame()
        hist_frame.setFixedHeight(110)
        hist_frame.setStyleSheet("""
            QFrame {
                background: #0a0a18;
                border-top: 1px solid #1a1a3a;
            }
        """)
        hist_layout = QHBoxLayout(hist_frame)
        hist_layout.setContentsMargins(8, 4, 8, 4)
        hist_layout.setSpacing(8)
        hist_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        hist_header = QLabel("Shot History")
        hist_header.setStyleSheet("color: #608060; font-size: 10px;")
        hist_header.setFixedWidth(80)
        hist_layout.addWidget(hist_header, 0, Qt.AlignmentFlag.AlignTop)

        # Scrollable thumbnail area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll.setFixedHeight(100)

        self._hist_container = QWidget()
        self._hist_container.setStyleSheet("background: transparent;")
        self._hist_row = QHBoxLayout(self._hist_container)
        self._hist_row.setContentsMargins(0, 0, 0, 0)
        self._hist_row.setSpacing(8)
        self._hist_row.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._hist_row.addStretch()

        scroll.setWidget(self._hist_container)
        hist_layout.addWidget(scroll)

        root.addWidget(hist_frame, 0)

    def _build_menu(self) -> None:
        mb = self.menuBar()

        # File
        file_menu = mb.addMenu("&File")
        open_act = QAction("Open Video…", self, shortcut=QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._open_video)
        file_menu.addAction(open_act)
        file_menu.addSeparator()
        export_act = QAction("Export Analysis…", self)
        export_act.triggered.connect(self._export_analysis)
        file_menu.addAction(export_act)
        file_menu.addSeparator()
        quit_act = QAction("Quit", self, shortcut=QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # Analysis
        analysis_menu = mb.addMenu("&Analysis")
        analyse_act = QAction("Run Analysis", self, shortcut="F5")
        analyse_act.triggered.connect(self._run_analysis)
        analysis_menu.addAction(analyse_act)

        calib_act = QAction("Calibrate Target Line…", self)
        calib_act.triggered.connect(self._calibrate_target_line)
        analysis_menu.addAction(calib_act)

        analysis_menu.addSeparator()
        yolo_act = QAction("Use YOLO Model…", self)
        yolo_act.triggered.connect(self._load_yolo_model)
        analysis_menu.addAction(yolo_act)

        # Help
        help_menu = mb.addMenu("&Help")
        about_act = QAction("About PutterTrack Pro", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(20, 20))
        tb.setStyleSheet("""
            QToolBar {
                background: #0e0e22;
                border-bottom: 1px solid #1a1a3a;
                spacing: 4px;
            }
            QToolButton {
                background: #1a1a3a;
                color: #ddd;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 12px;
            }
            QToolButton:hover { background: #2a2a5a; }
        """)

        open_btn = QAction("📂 Open Video", self)
        open_btn.triggered.connect(self._open_video)
        tb.addAction(open_btn)

        tb.addSeparator()

        run_btn = QAction("▶ Run Analysis", self)
        run_btn.triggered.connect(self._run_analysis)
        tb.addAction(run_btn)

        tb.addSeparator()

        calib_btn = QAction("🎯 Set Target Line", self)
        calib_btn.triggered.connect(self._calibrate_target_line)
        tb.addAction(calib_btn)

        roi_btn = QAction("🔲 Set Putter ROI", self)
        roi_btn.triggered.connect(self._player.request_template_selection)
        tb.addAction(roi_btn)

        tb.addSeparator()
        export_btn = QAction("📤 Export", self)
        export_btn.triggered.connect(self._export_analysis)
        tb.addAction(export_btn)

        # Status / progress at far right
        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet("color: #888; font-size: 11px; padding: 0 8px;")
        tb.addWidget(self._progress_label)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Golf Video",
            os.path.expanduser("~"),
            "Video files (*.mp4 *.mov *.avi *.mkv *.m4v);;All files (*)",
        )
        if not path:
            return

        self.statusBar().showMessage(f"Loading {os.path.basename(path)}…")
        self._progress_label.setText("Loading…")

        # Decode all frames
        frames = _decode_video(path)
        if not frames:
            QMessageBox.warning(self, "Error", f"Could not read video:\n{path}")
            return

        self._all_frames = frames
        self._video_path = path

        # Create a minimal placeholder record for display
        placeholder = ShotRecord(
            shot_id="preview",
            video_path=path,
            fps=_get_fps(path),
            frame_count=len(frames),
        )
        self._player.load_record(placeholder, frames)
        self._analysis.clear()
        self._traj_view.clear()

        self.statusBar().showMessage(
            f"Loaded: {os.path.basename(path)}  —  "
            f"{len(frames)} frames @ {placeholder.fps:.1f} fps  —  "
            "Press F5 or click 'Run Analysis'"
        )
        self._progress_label.setText("")

    def _run_analysis(self) -> None:
        if not self._all_frames:
            QMessageBox.information(self, "No video", "Please open a video file first.")
            return

        # Stop any existing analysis
        self._stop_worker()

        target_angle = getattr(self, "_target_line_angle", 0.0)
        use_yolo = getattr(self, "_yolo_model_path", None) is not None
        yolo_path = getattr(self, "_yolo_model_path", None)

        self._tracker = PutterTracker(use_yolo=use_yolo, yolo_model_path=yolo_path)
        ok, err = self._tracker.load_video(self._video_path)
        if not ok:
            QMessageBox.warning(self, "Error", err)
            return
        self._tracker.set_target_line(target_angle)

        # Apply template if user set ROI
        if hasattr(self, "_pending_template"):
            fi, bbox = self._pending_template
            self._tracker.set_template_from_frame(fi, bbox)

        # Progress dialog
        self._progress_dlg = QProgressDialog(
            "Analysing putter motion…", "Cancel", 0, 100, self
        )
        self._progress_dlg.setWindowTitle("PutterTrack Analysis")
        self._progress_dlg.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress_dlg.setMinimumDuration(200)
        self._progress_dlg.canceled.connect(self._stop_worker)

        # Worker thread
        self._worker = AnalysisWorker(self._tracker)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_analysis_progress)
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.error.connect(self._on_analysis_error)
        self._thread.start()
        self.statusBar().showMessage("Running analysis…")

    def _on_analysis_progress(self, frac: float, frame: object) -> None:
        pct = int(frac * 100)
        if self._progress_dlg:
            self._progress_dlg.setValue(pct)
        self._progress_label.setText(f"Analysing {pct}%")

    def _on_analysis_done(self, record: ShotRecord) -> None:
        self._thread.quit()
        self._progress_dlg.close()
        self._progress_label.setText("")

        self._current_record = record
        self._history.append(record)

        # Update UI with results
        self._player.load_record(record, self._all_frames)
        self._analysis.update_metrics(record.metrics, record.keyframes)

        arc_pts = [
            (p.head_x, p.head_y)
            for p in record.positions
            if p.confidence > 0.2
        ]
        self._traj_view.update_data(record.metrics, arc_pts)

        # Add thumbnail to history
        thumb_frame = None
        if self._all_frames and record.keyframes.impact is not None:
            fi = min(record.keyframes.impact, len(self._all_frames) - 1)
            thumb_frame = cv2.resize(self._all_frames[fi], (160, 90))

        thumb = ShotThumbnail(record, thumb_frame)
        thumb.clicked_record.connect(self._load_history_shot)
        # Insert before stretch
        count = self._hist_row.count()
        self._hist_row.insertWidget(count - 1, thumb)

        self.statusBar().showMessage(
            f"Analysis complete — "
            f"Path: {record.metrics.club_path_deg or '—'}°  |  "
            f"Face: {record.metrics.face_angle_deg or '—'}°  |  "
            f"Speed: {record.metrics.impact_speed_mph or '—'} mph"
        )

    def _on_analysis_error(self, msg: str) -> None:
        self._thread.quit()
        if self._progress_dlg:
            self._progress_dlg.close()
        self._progress_label.setText("")
        QMessageBox.critical(self, "Analysis Error", f"An error occurred:\n\n{msg}")
        self.statusBar().showMessage("Analysis failed.")

    def _stop_worker(self) -> None:
        if self._worker:
            self._worker.stop()
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)

    def _load_history_shot(self, record: ShotRecord) -> None:
        self._current_record = record
        frames = _decode_video(record.video_path)
        if frames:
            self._all_frames = frames
            self._player.load_record(record, frames)
        self._analysis.update_metrics(record.metrics, record.keyframes)
        arc_pts = [
            (p.head_x, p.head_y)
            for p in record.positions
            if p.confidence > 0.2
        ]
        self._traj_view.update_data(record.metrics, arc_pts)

    def _on_template_defined(self, frame_idx: int, bbox: tuple) -> None:
        self._pending_template = (frame_idx, bbox)
        self.statusBar().showMessage(
            f"Putter ROI set (frame {frame_idx}, bbox {bbox}).  "
            "Run analysis to apply."
        )

    def _on_frame_changed(self, frame_idx: int) -> None:
        pass  # could update per-frame metrics here

    def _calibrate_target_line(self) -> None:
        angle, ok = QInputDialog.getDouble(
            self, "Target Line Angle",
            "Enter the screen angle of the target line\n"
            "(0° = horizontal, positive = tilted right):",
            value=getattr(self, "_target_line_angle", 0.0),
            min=-45.0, max=45.0, decimals=1,
        )
        if ok:
            self._target_line_angle = angle
            self.statusBar().showMessage(
                f"Target line set to {angle}°.  Re-run analysis to apply."
            )

    def _load_yolo_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load YOLO Model Weights",
            os.path.expanduser("~"),
            "Model files (*.pt *.onnx);;All files (*)",
        )
        if path:
            self._yolo_model_path = path
            self.statusBar().showMessage(
                f"YOLO model loaded: {os.path.basename(path)}"
            )

    def _export_analysis(self) -> None:
        if not self._current_record:
            QMessageBox.information(self, "No data", "Run analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Analysis", os.path.expanduser("~"),
            "JSON files (*.json)"
        )
        if not path:
            return
        record = self._current_record
        data = {
            "shot_id":    record.shot_id,
            "video_path": record.video_path,
            "fps":        record.fps,
            "metrics":    record.metrics.to_display_dict(),
            "keyframes":  record.keyframes.as_dict(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self.statusBar().showMessage(f"Exported to {path}")

    def _show_about(self) -> None:
        QMessageBox.about(
            self, "About PutterTrack Pro",
            "<h2>PutterTrack Pro</h2>"
            "<p>Golf putter motion analysis application.</p>"
            "<p><b>Features:</b><br>"
            "• Multi-method putter tracking (motion + Hough + optical flow + Kalman)<br>"
            "• Face angle, club path, tempo and speed analysis<br>"
            "• Frame-by-frame video replay with overlay<br>"
            "• Optional YOLO-based detection for maximum accuracy<br>"
            "• D-plane ball launch prediction</p>"
            "<p>Cross-platform: Windows &amp; macOS</p>"
        )

    def _open_video_path(self, path: str) -> None:
        """Programmatic video open (used by CLI argument)."""
        self._video_path = path
        frames = _decode_video(path)
        if not frames:
            return
        self._all_frames = frames
        from models.shot_data import ShotRecord
        placeholder = ShotRecord(
            shot_id="preview",
            video_path=path,
            fps=_get_fps(path),
            frame_count=len(frames),
        )
        self._player.load_record(placeholder, frames)
        self.statusBar().showMessage(
            f"Loaded: {os.path.basename(path)}  —  {len(frames)} frames"
        )

    def closeEvent(self, event) -> None:
        self._stop_worker()
        event.accept()


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def _decode_video(path: str) -> list[np.ndarray]:
    """Decode all frames from a video file into memory."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def _get_fps(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps
