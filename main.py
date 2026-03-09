#!/usr/bin/env python3
"""
PutterTrack Pro – Golf putter motion analysis application.

Tries to launch the PyQt6 GUI first; falls back to the OpenCV live-tracking
app (putter_live.py) if PyQt6 / a display server is not available.

Usage:
    python main.py [video_file]
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _launch_pyqt(video_path: str | None) -> None:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QFont
    from app.main_window import MainWindow

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("PutterTrack Pro")
    app.setApplicationDisplayName("PutterTrack Pro")
    app.setOrganizationName("PutterTrack")
    app.setApplicationVersion("1.0.0")
    app.setFont(QFont("Segoe UI", 10) if sys.platform == "win32"
                else QFont("Helvetica Neue", 10))

    qss_path = os.path.join(os.path.dirname(__file__), "assets", "styles", "dark_theme.qss")
    if os.path.exists(qss_path):
        with open(qss_path) as f:
            app.setStyleSheet(f.read())

    window = MainWindow()
    window.show()
    if video_path:
        window._open_video_path(video_path)

    sys.exit(app.exec())


def _launch_opencv(video_path: str | None) -> None:
    """Run putter_live directly (OpenCV-based), bypassing the PyQt6 launch menu."""
    from putter_live import PutterLive
    pl = PutterLive()
    if video_path:
        pl._load_video(video_path)
    pl.run()


def main() -> None:
    video_path = sys.argv[1] if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) else None

    try:
        _launch_pyqt(video_path)
    except (ImportError, Exception) as e:
        print(f"[PutterTrack] PyQt6 non disponible ({e}), démarrage en mode OpenCV...")
        _launch_opencv(video_path)


if __name__ == "__main__":
    main()
