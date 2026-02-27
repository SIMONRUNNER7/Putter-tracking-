"""
Trajectory visualization widget.

Draws a top-down schematic of:
  - The target line (horizontal)
  - The putter arc (fitted polynomial)
  - Club path arrow at impact
  - Ball launch direction arrow
  - Textual labels

This is a custom QWidget that paints with QPainter.
"""
from __future__ import annotations

import math
import numpy as np
from typing import Optional

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont,
    QPainterPath, QLinearGradient, QPolygonF,
)

from models.shot_data import ShotMetrics, ClubPathType


# Colors
COL_BG          = QColor(18, 18, 30)
COL_GRID        = QColor(40, 40, 60)
COL_TARGET      = QColor(80, 220, 100)
COL_ARC         = QColor(150, 80, 255)
COL_ARC_LIGHT   = QColor(200, 140, 255)
COL_PATH_ARROW  = QColor(255, 140, 0)
COL_FACE        = QColor(255, 70, 70)
COL_BALL_DIR    = QColor(0, 200, 255)
COL_TEXT        = QColor(220, 220, 220)
COL_TEXT_DIM    = QColor(120, 120, 140)
COL_PUTTER_HEAD = QColor(200, 200, 210)
COL_IMPACT_RING = QColor(255, 60, 60)


class TrajectoryView(QWidget):
    """
    Top-down schematic view of the putting stroke.
    Call update_data() to refresh with new analysis results.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 280)
        self.setStyleSheet("background-color: #12121e;")

        self._metrics: Optional[ShotMetrics] = None
        self._arc_pts: list[tuple[float, float]] = []   # normalized [0,1]
        self._has_data = False

    def update_data(
        self,
        metrics: ShotMetrics,
        raw_positions: list[tuple[float, float]],  # (head_x, head_y) sequence
    ) -> None:
        self._metrics = metrics
        self._arc_pts = _normalize_positions(raw_positions)
        self._has_data = bool(raw_positions)
        self.update()

    def clear(self) -> None:
        self._metrics = None
        self._arc_pts = []
        self._has_data = False
        self.update()

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2

        # Background
        painter.fillRect(self.rect(), COL_BG)

        # Grid
        self._draw_grid(painter, w, h)

        if not self._has_data or self._metrics is None:
            self._draw_placeholder(painter, w, h)
            painter.end()
            return

        # Target line
        self._draw_target_line(painter, w, h, cx, cy)

        # Arc
        self._draw_arc(painter, w, h, cx, cy)

        # Arrows at impact
        self._draw_arrows(painter, w, h, cx, cy)

        # Putter head schematic at impact
        self._draw_putter_head(painter, cx, cy)

        # Legend
        self._draw_legend(painter, w, h)

        painter.end()

    def _draw_grid(self, p: QPainter, w: int, h: int) -> None:
        pen = QPen(COL_GRID, 1, Qt.PenStyle.DotLine)
        p.setPen(pen)
        step = 40
        for x in range(0, w, step):
            p.drawLine(x, 0, x, h)
        for y in range(0, h, step):
            p.drawLine(0, y, w, y)

    def _draw_placeholder(self, p: QPainter, w: int, h: int) -> None:
        p.setPen(QPen(COL_TEXT_DIM))
        font = QFont("Arial", 11)
        p.setFont(font)
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                   "Load and analyse a video\nto see the club path")

    def _draw_target_line(self, p: QPainter, w: int, h: int, cx: float, cy: float) -> None:
        pen = QPen(COL_TARGET, 2, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawLine(0, int(cy), w, int(cy))
        p.setPen(QPen(COL_TEXT_DIM, 1))
        p.setFont(QFont("Arial", 8))
        p.drawText(QRectF(4, cy - 18, 80, 16), Qt.AlignmentFlag.AlignLeft, "Target line")

    def _draw_arc(self, p: QPainter, w: int, h: int, cx: float, cy: float) -> None:
        if len(self._arc_pts) < 2:
            return

        # Scale normalized positions to widget coordinates
        # Impact is centered; arc spans ±40% of widget width
        impact_norm = self._find_impact_norm()
        sx = w * 0.8   # horizontal scale
        sy = h * 0.4   # vertical scale

        pts_widget = []
        for (nx, ny) in self._arc_pts:
            wx = cx + (nx - impact_norm[0]) * sx
            wy = cy + (ny - impact_norm[1]) * sy
            pts_widget.append(QPointF(wx, wy))

        if len(pts_widget) < 2:
            return

        # Draw arc with gradient colour
        for i in range(1, len(pts_widget)):
            frac = i / len(pts_widget)
            c = _lerp_color(COL_ARC, COL_ARC_LIGHT, frac)
            pen = QPen(c, 2.5, Qt.PenStyle.SolidLine)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawLine(pts_widget[i-1], pts_widget[i])

        # Impact dot
        impact_idx = self._find_impact_idx()
        if 0 <= impact_idx < len(pts_widget):
            ip = pts_widget[impact_idx]
            p.setPen(QPen(COL_IMPACT_RING, 2))
            p.setBrush(QBrush(COL_IMPACT_RING.lighter(180)))
            p.drawEllipse(ip, 7.0, 7.0)

    def _draw_arrows(self, p: QPainter, w: int, h: int, cx: float, cy: float) -> None:
        m = self._metrics
        arrow_len = min(w, h) * 0.25

        # Club path arrow
        if m.club_path_deg is not None:
            angle_rad = math.radians(m.club_path_deg)  # relative to horizontal
            ex = cx + arrow_len * math.cos(angle_rad)
            ey = cy + arrow_len * math.sin(angle_rad)
            _draw_arrow(p, cx, cy, ex, ey, COL_PATH_ARROW, 2.5,
                        label=f"Path {_sign(m.club_path_deg)}{abs(m.club_path_deg):.1f}°")

        # Ball launch direction arrow
        if m.ball.launch_direction_deg is not None:
            angle_rad = math.radians(m.ball.launch_direction_deg)
            ex = cx + arrow_len * 1.2 * math.cos(angle_rad)
            ey = cy + arrow_len * 1.2 * math.sin(angle_rad)
            _draw_arrow(p, cx, cy, ex, ey, COL_BALL_DIR, 2,
                        label=f"Ball {_sign(m.ball.launch_direction_deg)}{abs(m.ball.launch_direction_deg):.1f}°",
                        dashed=True)

        # Face angle indicator (small perpendicular at impact)
        if m.face_angle_deg is not None:
            fa_rad = math.radians(m.face_angle_deg + 90)  # face is perpendicular to path
            fl = 35
            p.setPen(QPen(COL_FACE, 3, Qt.PenStyle.SolidLine))
            p.drawLine(
                QPointF(cx - fl * math.cos(fa_rad), cy - fl * math.sin(fa_rad)),
                QPointF(cx + fl * math.cos(fa_rad), cy + fl * math.sin(fa_rad)),
            )

    def _draw_putter_head(self, p: QPainter, cx: float, cy: float) -> None:
        p.setPen(QPen(COL_PUTTER_HEAD, 2))
        p.setBrush(QBrush(COL_PUTTER_HEAD.darker(200)))
        p.drawRect(QRectF(cx - 20, cy - 5, 40, 10))

    def _draw_legend(self, p: QPainter, w: int, h: int) -> None:
        items = [
            (COL_ARC,        "Club arc"),
            (COL_PATH_ARROW, "Club path"),
            (COL_BALL_DIR,   "Ball direction"),
            (COL_FACE,       "Face angle"),
            (COL_TARGET,     "Target line"),
        ]
        x, y = w - 130, 14
        p.setFont(QFont("Arial", 8))
        for color, label in items:
            p.setPen(QPen(color, 3))
            p.drawLine(x, y, x + 18, y)
            p.setPen(QPen(COL_TEXT_DIM))
            p.drawText(QRectF(x + 22, y - 7, 100, 14),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       label)
            y += 16

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_impact_norm(self) -> tuple[float, float]:
        """Normalized coordinates of the impact point (midpoint of arc)."""
        if not self._arc_pts:
            return (0.5, 0.5)
        mid = len(self._arc_pts) // 2
        return self._arc_pts[mid]

    def _find_impact_idx(self) -> int:
        return len(self._arc_pts) // 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_positions(
    positions: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Normalize raw pixel positions to [0, 1] range."""
    if not positions:
        return []
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = (x_max - x_min) or 1.0
    y_range = (y_max - y_min) or 1.0
    return [
        ((x - x_min) / x_range, (y - y_min) / y_range)
        for x, y in positions
    ]


def _lerp_color(c1: QColor, c2: QColor, t: float) -> QColor:
    r = int(c1.red()   + (c2.red()   - c1.red())   * t)
    g = int(c1.green() + (c2.green() - c1.green()) * t)
    b = int(c1.blue()  + (c2.blue()  - c1.blue())  * t)
    return QColor(r, g, b)


def _draw_arrow(
    p: QPainter,
    x1: float, y1: float,
    x2: float, y2: float,
    color: QColor,
    width: float = 2.0,
    label: str = "",
    dashed: bool = False,
) -> None:
    style = Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine
    pen = QPen(color, width, style)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    # Arrowhead
    angle = math.atan2(y2 - y1, x2 - x1)
    arrow_size = 10
    ax1 = x2 - arrow_size * math.cos(angle - math.pi/6)
    ay1 = y2 - arrow_size * math.sin(angle - math.pi/6)
    ax2 = x2 - arrow_size * math.cos(angle + math.pi/6)
    ay2 = y2 - arrow_size * math.sin(angle + math.pi/6)
    p.setBrush(QBrush(color))
    poly = QPolygonF([QPointF(x2, y2), QPointF(ax1, ay1), QPointF(ax2, ay2)])
    p.drawPolygon(poly)

    # Label
    if label:
        p.setPen(QPen(color))
        p.setFont(QFont("Arial", 8, QFont.Weight.Bold))
        lx = x2 + 6
        ly = y2 - 6
        p.drawText(QRectF(lx, ly, 100, 20), Qt.AlignmentFlag.AlignLeft, label)


def _sign(v: float) -> str:
    return "+" if v > 0 else ""
