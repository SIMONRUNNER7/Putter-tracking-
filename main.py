#!/usr/bin/env python3
"""
PutterTrack Pro – Golf putter motion analysis application.

Cross-platform (Windows / macOS / Linux).
Requires Python 3.10+ and the packages listed in requirements.txt.

Usage:
    python main.py [video_file]
"""
import sys
import os

# Ensure the project root is on the Python path when running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


def main() -> None:
    # High-DPI support (important for macOS Retina + Windows 4K)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("PutterTrack Pro")
    app.setApplicationDisplayName("PutterTrack Pro")
    app.setOrganizationName("PutterTrack")
    app.setApplicationVersion("1.0.0")

    # Default font
    app.setFont(QFont("Segoe UI", 10) if sys.platform == "win32"
                else QFont("Helvetica Neue", 10))

    # Load dark theme stylesheet
    qss_path = os.path.join(
        os.path.dirname(__file__), "assets", "styles", "dark_theme.qss"
    )
    if os.path.exists(qss_path):
        with open(qss_path, "r") as f:
            app.setStyleSheet(f.read())

    # Import here so Qt is initialized first
    from app.main_window import MainWindow

    window = MainWindow()
    window.show()

    # If a video path was provided on the command line, open it
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        window._video_path = sys.argv[1]
        window._open_video_path(sys.argv[1])

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
