"""
Outil de labeling manuel — bbox classique + OBB orientée (3 clics).

Usage:
    python3 training/label_tool.py --images data/raw_images --out data/putter_dataset

Modes:
    Mode bbox (défaut) : Click + glisse pour dessiner un rectangle droit
    Mode OBB  (Tab)    : 3 clics pour définir un rectangle orienté
                           Clic 1 = bas de la face
                           Clic 2 = haut de la face
                           Clic 3 = largeur du cadre

Raccourcis clavier:
    Tab             Basculer bbox ↔ OBB
    Click + glisse  Dessiner la bbox (mode bbox)
    Clic ×3         Définir l'OBB (mode OBB)
    Entrée / →      Sauvegarder + image suivante
    ←               Image précédente
    D               Effacer la bbox/OBB courante
    S               Passer sans annoter (pas de putter visible)
    Q               Quitter

Format sauvegardé:
    bbox : 0 cx cy w h              (5 valeurs, normalisées)
    OBB  : 0 x1 y1 x2 y2 x3 y3 x4 y4  (9 valeurs, 4 coins normalisés)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygon
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# ---------------------------------------------------------------------------
# Image canvas
# ---------------------------------------------------------------------------

class ImageCanvas(QWidget):
    bbox_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        self._pixmap: Optional[QPixmap] = None
        self._img_rect: QRect = QRect()

        # --- mode bbox (glisser) ---
        self._bbox: Optional[tuple] = None        # (x1,y1,x2,y2) pixels image
        self._drawing = False
        self._draw_start: Optional[QPoint] = None
        self._draw_current: Optional[QPoint] = None

        # --- mode OBB (3 clics) ---
        self._obb_mode = False
        self._obb_pts: list[tuple[int, int]] = []  # 0-3 points en pixels image
        self._obb_corners: Optional[list[tuple[int, int]]] = None  # 4 coins
        self._mouse_pos: Optional[QPoint] = None   # pour preview 3e clic

    # ---- API publique -------------------------------------------------------

    def set_image(self, bgr: np.ndarray) -> None:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, int(rgb.strides[0]), QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._reset_annotation()
        self._update_img_rect()
        self.update()

    def set_mode(self, obb: bool) -> None:
        self._obb_mode = obb
        self._reset_annotation()
        self.update()

    def is_obb_mode(self) -> bool:
        return self._obb_mode

    def obb_step(self) -> int:
        """0=attente P1, 1=attente P2, 2=attente P3, 3=complet."""
        if self._obb_corners is not None:
            return 3
        return len(self._obb_pts)

    def _reset_annotation(self):
        self._bbox = None
        self._drawing = False
        self._draw_start = None
        self._draw_current = None
        self._obb_pts = []
        self._obb_corners = None

    # --- bbox ---
    def set_bbox_yolo(self, cx: float, cy: float, bw: float, bh: float) -> None:
        if self._pixmap is None:
            return
        iw, ih = self._pixmap.width(), self._pixmap.height()
        x1 = int((cx - bw / 2) * iw)
        y1 = int((cy - bh / 2) * ih)
        x2 = int((cx + bw / 2) * iw)
        y2 = int((cy + bh / 2) * ih)
        self._bbox = (x1, y1, x2, y2)
        self._obb_corners = None
        self._obb_pts = []
        self.update()

    def get_bbox_yolo(self) -> Optional[tuple[float, float, float, float]]:
        if self._bbox is None or self._pixmap is None:
            return None
        x1, y1, x2, y2 = self._bbox
        iw, ih = self._pixmap.width(), self._pixmap.height()
        cx = (x1 + x2) / 2 / iw
        cy = (y1 + y2) / 2 / ih
        bw = abs(x2 - x1) / iw
        bh = abs(y2 - y1) / ih
        return (cx, cy, bw, bh)

    # --- OBB ---
    def set_obb_yolo(self, coords: list[float]) -> None:
        """coords = [x1,y1,x2,y2,x3,y3,x4,y4] normalisés."""
        if self._pixmap is None or len(coords) != 8:
            return
        iw, ih = self._pixmap.width(), self._pixmap.height()
        corners = [
            (int(coords[i] * iw), int(coords[i + 1] * ih))
            for i in range(0, 8, 2)
        ]
        self._obb_corners = corners
        self._obb_pts = []
        self._bbox = None
        self.update()

    def get_obb_yolo(self) -> Optional[list[float]]:
        if self._obb_corners is None or self._pixmap is None:
            return None
        iw, ih = self._pixmap.width(), self._pixmap.height()
        result = []
        for (x, y) in self._obb_corners:
            result.extend([x / iw, y / ih])
        return result

    def clear_bbox(self) -> None:
        self._reset_annotation()
        self.update()
        self.bbox_changed.emit()

    def has_annotation(self) -> bool:
        return self._bbox is not None or self._obb_corners is not None

    # ---- peinture -----------------------------------------------------------

    def resizeEvent(self, event):
        self._update_img_rect()
        super().resizeEvent(event)

    def _update_img_rect(self) -> None:
        if self._pixmap is None:
            self._img_rect = QRect()
            return
        cw, ch = self.width(), self.height()
        iw, ih = self._pixmap.width(), self._pixmap.height()
        scale = min(cw / iw, ch / ih)
        dw, dh = int(iw * scale), int(ih * scale)
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        self._img_rect = QRect(ox, oy, dw, dh)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(25, 25, 25))

        if self._pixmap is None:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Chargement…")
            return

        painter.drawPixmap(self._img_rect, self._pixmap)

        if self._obb_mode:
            self._paint_obb(painter)
        else:
            self._paint_bbox(painter)

    def _paint_bbox(self, painter: QPainter):
        if self._bbox:
            rect = self._img_to_canvas_rect(*self._bbox)
            painter.setPen(QPen(QColor(0, 230, 0), 2))
            painter.drawRect(rect)
        if self._drawing and self._draw_start and self._draw_current:
            rect = QRect(self._draw_start, self._draw_current).normalized()
            painter.setPen(QPen(QColor(255, 70, 70), 2, Qt.PenStyle.DashLine))
            painter.drawRect(rect)

    def _paint_obb(self, painter: QPainter):
        DOT_R = 6
        COLORS = [QColor(255, 140, 0), QColor(255, 220, 0), QColor(0, 200, 255)]
        LABELS = ["1 · bas face", "2 · haut face", "3 · largeur"]

        # Points placés
        for i, pt in enumerate(self._obb_pts):
            cp = self._img_to_canvas_pt(pt)
            painter.setPen(QPen(COLORS[i], 2))
            painter.setBrush(COLORS[i])
            painter.drawEllipse(cp, DOT_R, DOT_R)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawText(cp + QPoint(DOT_R + 4, 4), LABELS[i])

        # Ligne P1-P2 (axe de la face)
        if len(self._obb_pts) >= 2:
            p1c = self._img_to_canvas_pt(self._obb_pts[0])
            p2c = self._img_to_canvas_pt(self._obb_pts[1])
            painter.setPen(QPen(QColor(255, 180, 0), 2, Qt.PenStyle.DashLine))
            painter.drawLine(p1c, p2c)

        # Preview OBB live (souris en attente du 3e clic)
        if len(self._obb_pts) == 2 and self._mouse_pos and self._img_rect.contains(self._mouse_pos):
            preview = self._compute_corners(
                self._obb_pts[0],
                self._obb_pts[1],
                self._canvas_to_img_pt(self._mouse_pos),
            )
            if preview:
                self._draw_obb_polygon(painter, preview, QColor(0, 200, 255), alpha=100)

        # OBB finalisé
        if self._obb_corners:
            self._draw_obb_polygon(painter, self._obb_corners, QColor(0, 230, 180))
            # Flèche de direction (P1→P2 = axe de la face)
            c = self._obb_corners
            mid1 = ((c[0][0] + c[1][0]) // 2, (c[0][1] + c[1][1]) // 2)
            mid2 = ((c[2][0] + c[3][0]) // 2, (c[2][1] + c[3][1]) // 2)
            painter.setPen(QPen(QColor(0, 230, 180), 2))
            painter.drawLine(self._img_to_canvas_pt(mid1), self._img_to_canvas_pt(mid2))

    def _draw_obb_polygon(self, painter: QPainter, corners, color: QColor, alpha: int = 255):
        canvas_pts = [self._img_to_canvas_pt(c) for c in corners]
        poly = QPolygon([QPoint(*p) for p in [
            (cp.x(), cp.y()) for cp in canvas_pts
        ]])
        pen = QPen(color, 2)
        if alpha < 255:
            pen.setStyle(Qt.PenStyle.DotLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolygon(poly)

    # ---- souris -------------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if not self._img_rect.contains(event.pos()):
            return

        if self._obb_mode:
            if self._obb_corners is not None:
                # OBB déjà complet → recommencer
                self._obb_corners = None
                self._obb_pts = []
            if len(self._obb_pts) < 3:
                pt = self._canvas_to_img_pt(event.pos())
                self._obb_pts.append(pt)
                if len(self._obb_pts) == 3:
                    corners = self._compute_corners(*self._obb_pts)
                    if corners:
                        self._obb_corners = corners
                self.bbox_changed.emit()
                self.update()
        else:
            self._drawing = True
            self._draw_start = event.pos()
            self._draw_current = event.pos()

    def mouseMoveEvent(self, event):
        self._mouse_pos = event.pos()
        if self._drawing:
            self._draw_current = event.pos()
        if self._obb_mode and len(self._obb_pts) == 2:
            self.update()  # preview live
        elif self._drawing:
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drawing:
            self._drawing = False
            if self._draw_start and self._draw_current:
                r = QRect(self._draw_start, self._draw_current).normalized()
                if r.width() > 8 and r.height() > 8:
                    self._bbox = self._canvas_rect_to_img(r)
                    self.bbox_changed.emit()
            self.update()

    # ---- calcul OBB ---------------------------------------------------------

    @staticmethod
    def _compute_corners(
        p1: tuple, p2: tuple, p3: tuple
    ) -> Optional[list[tuple[int, int]]]:
        """
        P1/P2 = axe de la face (bas → haut).
        P3    = point latéral pour définir la demi-largeur.
        Retourne 4 coins dans l'ordre : c1, c2 (côté P1), c3, c4 (côté P2).
        """
        a = np.array(p1, dtype=float)
        b = np.array(p2, dtype=float)
        q = np.array(p3, dtype=float)

        axis = b - a
        length = np.linalg.norm(axis)
        if length < 4:
            return None

        u = axis / length                     # vecteur unitaire axe
        perp = np.array([-u[1], u[0]])        # perpendiculaire

        half_w = float(np.dot(q - a, perp))   # peut être négatif

        c1 = a + perp * half_w
        c2 = a - perp * half_w
        c3 = b - perp * half_w
        c4 = b + perp * half_w

        return [(int(c[0]), int(c[1])) for c in [c1, c2, c3, c4]]

    # ---- conversion coords --------------------------------------------------

    def _img_to_canvas_rect(self, x1, y1, x2, y2) -> QRect:
        ir = self._img_rect
        if self._pixmap is None:
            return QRect()
        iw, ih = self._pixmap.width(), self._pixmap.height()
        sx, sy = ir.width() / iw, ir.height() / ih
        return QRect(
            QPoint(ir.x() + int(x1 * sx), ir.y() + int(y1 * sy)),
            QPoint(ir.x() + int(x2 * sx), ir.y() + int(y2 * sy)),
        )

    def _img_to_canvas_pt(self, pt: tuple) -> QPoint:
        ir = self._img_rect
        if self._pixmap is None:
            return QPoint(0, 0)
        iw, ih = self._pixmap.width(), self._pixmap.height()
        sx, sy = ir.width() / iw, ir.height() / ih
        return QPoint(ir.x() + int(pt[0] * sx), ir.y() + int(pt[1] * sy))

    def _canvas_to_img_pt(self, qp: QPoint) -> tuple[int, int]:
        ir = self._img_rect
        if self._pixmap is None:
            return (0, 0)
        iw, ih = self._pixmap.width(), self._pixmap.height()
        sx, sy = iw / ir.width(), ih / ir.height()
        x = int((qp.x() - ir.x()) * sx)
        y = int((qp.y() - ir.y()) * sy)
        return (max(0, min(iw, x)), max(0, min(ih, y)))

    def _canvas_rect_to_img(self, r: QRect) -> tuple:
        ir = self._img_rect
        if self._pixmap is None:
            return (0, 0, 0, 0)
        iw, ih = self._pixmap.width(), self._pixmap.height()
        sx, sy = iw / ir.width(), ih / ir.height()
        x1 = int((r.left()   - ir.x()) * sx)
        y1 = int((r.top()    - ir.y()) * sy)
        x2 = int((r.right()  - ir.x()) * sx)
        y2 = int((r.bottom() - ir.y()) * sy)
        return (
            max(0, min(iw, x1)), max(0, min(ih, y1)),
            max(0, min(iw, x2)), max(0, min(ih, y2)),
        )


# ---------------------------------------------------------------------------
# Fenêtre principale
# ---------------------------------------------------------------------------

class LabelTool(QMainWindow):
    def __init__(self, images_dir: str, out_dir: str, labels_dir: str = None):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.out_dir    = Path(out_dir)
        self.out_labels = Path(labels_dir) if labels_dir else self.out_dir / "labels" / "all"
        self.out_images = self.out_dir / "images" / "all"
        self.out_labels.mkdir(parents=True, exist_ok=True)
        self.out_images.mkdir(parents=True, exist_ok=True)

        self.images = sorted([
            p for p in self.images_dir.iterdir()
            if p.suffix in SUPPORTED_EXTS
        ])
        self.idx = 0

        self._build_ui()
        if self.images:
            self._load_current()
        else:
            self.status_label.setText("Aucune image trouvée dans le dossier.")

    # ---- construction UI ----------------------------------------------------

    def _build_ui(self):
        self.setWindowTitle("Labeling putter")
        self.resize(1200, 860)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        self.progress = QProgressBar()
        self.progress.setMaximum(max(len(self.images), 1))
        self.progress.setFormat("%v / %m annotées")
        root.addWidget(self.progress)

        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        f = QFont(); f.setPointSize(12)
        self.status_label.setFont(f)
        root.addWidget(self.status_label)

        self.canvas = ImageCanvas()
        self.canvas.bbox_changed.connect(self._on_bbox_changed)
        root.addWidget(self.canvas, stretch=1)

        # Boutons
        btns = QHBoxLayout()

        self.btn_prev = QPushButton("◀  Précédent  (←)")
        self.btn_prev.clicked.connect(self._prev)
        btns.addWidget(self.btn_prev)

        self.btn_clear = QPushButton("🗑  Effacer  (D)")
        self.btn_clear.clicked.connect(self.canvas.clear_bbox)
        btns.addWidget(self.btn_clear)

        self.btn_mode = QPushButton("⬡  Mode OBB  (Tab)")
        self.btn_mode.setCheckable(True)
        self.btn_mode.clicked.connect(self._toggle_mode)
        btns.addWidget(self.btn_mode)

        self.btn_skip = QPushButton("⏭  Passer  (S)")
        self.btn_skip.clicked.connect(self._skip)
        btns.addWidget(self.btn_skip)

        self.btn_next = QPushButton("✔  Sauvegarder + Suivant  (→ / Entrée)")
        self.btn_next.setDefault(True)
        self.btn_next.setStyleSheet(
            "QPushButton { background: #2a7a2a; color: white; font-weight: bold; }"
        )
        self.btn_next.clicked.connect(self._save_and_next)
        btns.addWidget(self.btn_next)

        root.addLayout(btns)

        self.hint_label = QLabel()
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint_label.setStyleSheet("color: #666; font-size: 10px;")
        root.addWidget(self.hint_label)
        self._update_hint()

    def _update_hint(self):
        if self.canvas.is_obb_mode():
            step = self.canvas.obb_step()
            msgs = [
                "OBB — Clic 1 : bas de la face du putter",
                "OBB — Clic 2 : haut de la face",
                "OBB — Clic 3 : largeur du cadre  (déplace la souris pour prévisualiser)",
                "OBB — Entrée pour sauvegarder  |  clic pour recommencer  |  D = effacer",
            ]
            self.hint_label.setText(msgs[step])
            self.hint_label.setStyleSheet("color: #5af; font-size: 11px;")
        else:
            self.hint_label.setText(
                "BBOX — Click + glisse → rectangle  |  Tab = mode OBB  |  ← → Entrée D S Q"
            )
            self.hint_label.setStyleSheet("color: #666; font-size: 10px;")

    # ---- chargement ---------------------------------------------------------

    def _labeled_count(self) -> int:
        return sum(
            1 for p in self.images
            if (self.out_labels / (p.stem + ".txt")).exists()
        )

    def _load_current(self):
        img_path = self.images[self.idx]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            self._set_status(f"⚠ Impossible de lire {img_path.name}", error=True)
            return

        self.canvas.set_image(bgr)
        iw, ih = bgr.shape[1], bgr.shape[0]

        lbl_path = self.out_labels / (img_path.stem + ".txt")
        marker = ""
        if lbl_path.exists():
            lines = lbl_path.read_text().strip().splitlines()
            if lines:
                parts = lines[0].split()
                if len(parts) == 5:
                    # bbox classique
                    _, cx, cy, bw, bh = parts
                    self.canvas.set_bbox_yolo(float(cx), float(cy), float(bw), float(bh))
                    marker = "  ✓ bbox"
                elif len(parts) == 9:
                    # OBB : class x1 y1 x2 y2 x3 y3 x4 y4
                    coords = [float(p) for p in parts[1:]]
                    self.canvas.set_obb_yolo(coords)
                    marker = "  ✓ OBB"

        labeled = self._labeled_count()
        self.progress.setValue(labeled)
        self.setWindowTitle(
            f"[{self.idx + 1}/{len(self.images)}]  {img_path.name}{marker}  —  {labeled} annotées"
        )
        self._set_status(f"Image {self.idx + 1} / {len(self.images)}   {img_path.name}{marker}")
        self._update_hint()

    def _set_status(self, msg: str, error: bool = False):
        self.status_label.setText(msg)
        self.status_label.setStyleSheet(f"color: {'#cc4444' if error else '#cccccc'};")

    # ---- actions ------------------------------------------------------------

    def _toggle_mode(self):
        obb = self.btn_mode.isChecked()
        self.btn_mode.setText("⬡  Mode OBB (actif)  (Tab)" if obb else "⬡  Mode OBB  (Tab)")
        self.btn_mode.setStyleSheet(
            "QPushButton { background: #1a4a7a; color: white; }" if obb else ""
        )
        self.canvas.set_mode(obb)
        self._update_hint()

    def _on_bbox_changed(self):
        self._update_hint()
        if self.canvas.has_annotation():
            self._set_status(
                f"Image {self.idx + 1} / {len(self.images)}  —  annotation prête  →  Entrée pour sauvegarder"
            )

    def _save_current(self) -> bool:
        img_path = self.images[self.idx]
        lbl_path = self.out_labels / (img_path.stem + ".txt")

        # Priorité OBB
        obb = self.canvas.get_obb_yolo()
        if obb is not None:
            coords = " ".join(f"{v:.6f}" for v in obb)
            lbl_path.write_text(f"0 {coords}\n")
            self._copy_image(img_path)
            return True

        # Bbox classique
        bbox = self.canvas.get_bbox_yolo()
        if bbox is not None:
            cx, cy, bw, bh = bbox
            cx = max(0.001, min(0.999, cx))
            cy = max(0.001, min(0.999, cy))
            bw = max(0.001, min(1.0,   bw))
            bh = max(0.001, min(1.0,   bh))
            lbl_path.write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            self._copy_image(img_path)
            return True

        return False

    def _copy_image(self, img_path: Path):
        out_img = self.out_images / img_path.name
        if not out_img.exists():
            shutil.copy2(img_path, out_img)

    def _save_and_next(self):
        if not self._save_current():
            self._set_status("⚠  Dessine d'abord une annotation (bbox ou OBB) !", error=True)
            return
        self._advance()

    def _skip(self):
        self._advance()

    def _prev(self):
        if self.idx > 0:
            self.idx -= 1
            self._load_current()

    def _advance(self):
        if self.idx < len(self.images) - 1:
            self.idx += 1
            self._load_current()
        else:
            labeled = self._labeled_count()
            QMessageBox.information(
                self, "Terminé !",
                f"Toutes les images ont été parcourues.\n"
                f"{labeled} / {len(self.images)} images annotées.\n\n"
                f"Lance maintenant :\n"
                f"  python3 training/autolabel.py split --dataset data/putter_dataset\n"
                f"  python3 training/augment.py --dataset data/putter_dataset --factor 8",
            )

    # ---- raccourcis clavier -------------------------------------------------

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Right, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._save_and_next()
        elif key == Qt.Key.Key_Left:
            self._prev()
        elif key in (Qt.Key.Key_D, Qt.Key.Key_Delete):
            self.canvas.clear_bbox()
            self._update_hint()
        elif key == Qt.Key.Key_S:
            self._skip()
        elif key == Qt.Key.Key_Tab:
            self.btn_mode.setChecked(not self.btn_mode.isChecked())
            self._toggle_mode()
        elif key == Qt.Key.Key_Q:
            self.close()
        else:
            super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Outil de labeling manuel — putter bbox / OBB")
    parser.add_argument("--images", required=True, help="Dossier d'images")
    parser.add_argument("--out",    required=True, help="Dossier dataset (data/putter_dataset)")
    parser.add_argument("--labels", default=None,
                        help="Dossier labels existants (ex: data/putter_dataset/labels/train)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = LabelTool(args.images, args.out, args.labels)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
