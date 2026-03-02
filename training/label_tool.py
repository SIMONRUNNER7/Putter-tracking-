"""
Outil de labeling manuel — dessine les bounding boxes avec la souris.

Usage:
    python3 training/label_tool.py --images data/raw_images --out data/putter_dataset

Raccourcis clavier:
    Click + glisse  Dessiner la bbox
    Entrée / →      Sauvegarder + image suivante
    ←               Image précédente
    D               Effacer la bbox courante
    S               Passer sans annoter (pas de putter visible)
    Q               Quitter
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
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
# Image canvas — affiche l'image + permet de dessiner une bbox
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
        self._bbox: Optional[tuple] = None   # (x1, y1, x2, y2) en pixels image
        self._drawing = False
        self._draw_start: Optional[QPoint] = None
        self._draw_current: Optional[QPoint] = None

    # ---- public API -------------------------------------------------------

    def set_image(self, bgr: np.ndarray) -> None:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, int(rgb.strides[0]), QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._bbox = None
        self._drawing = False
        self._draw_start = None
        self._draw_current = None
        self._update_img_rect()
        self.update()

    def set_bbox_yolo(self, cx: float, cy: float, bw: float, bh: float) -> None:
        if self._pixmap is None:
            return
        iw = self._pixmap.width()
        ih = self._pixmap.height()
        x1 = int((cx - bw / 2) * iw)
        y1 = int((cy - bh / 2) * ih)
        x2 = int((cx + bw / 2) * iw)
        y2 = int((cy + bh / 2) * ih)
        self._bbox = (x1, y1, x2, y2)
        self.update()

    def get_bbox_yolo(self) -> Optional[tuple[float, float, float, float]]:
        if self._bbox is None or self._pixmap is None:
            return None
        x1, y1, x2, y2 = self._bbox
        iw = self._pixmap.width()
        ih = self._pixmap.height()
        cx = (x1 + x2) / 2 / iw
        cy = (y1 + y2) / 2 / ih
        bw = abs(x2 - x1) / iw
        bh = abs(y2 - y1) / ih
        return (cx, cy, bw, bh)

    def clear_bbox(self) -> None:
        self._bbox = None
        self.update()
        self.bbox_changed.emit()

    # ---- painting ---------------------------------------------------------

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
        dw = int(iw * scale)
        dh = int(ih * scale)
        ox = (cw - dw) // 2
        oy = (ch - dh) // 2
        self._img_rect = QRect(ox, oy, dw, dh)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(25, 25, 25))

        if self._pixmap is None:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Chargement…")
            return

        painter.drawPixmap(self._img_rect, self._pixmap)

        # Bbox validée (verte)
        if self._bbox:
            rect = self._img_to_canvas_rect(*self._bbox)
            pen = QPen(QColor(0, 230, 0), 2)
            painter.setPen(pen)
            painter.drawRect(rect)

        # Bbox en cours de dessin (rouge pointillé)
        if self._drawing and self._draw_start and self._draw_current:
            rect = QRect(self._draw_start, self._draw_current).normalized()
            pen = QPen(QColor(255, 70, 70), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(rect)

    # ---- souris -----------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._img_rect.contains(event.pos()):
                self._drawing = True
                self._draw_start = event.pos()
                self._draw_current = event.pos()

    def mouseMoveEvent(self, event):
        if self._drawing:
            self._draw_current = event.pos()
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

    # ---- conversion coordonnées -------------------------------------------

    def _img_to_canvas_rect(self, x1, y1, x2, y2) -> QRect:
        ir = self._img_rect
        if self._pixmap is None:
            return QRect()
        iw, ih = self._pixmap.width(), self._pixmap.height()
        sx = ir.width() / iw
        sy = ir.height() / ih
        return QRect(
            QPoint(ir.x() + int(x1 * sx), ir.y() + int(y1 * sy)),
            QPoint(ir.x() + int(x2 * sx), ir.y() + int(y2 * sy)),
        )

    def _canvas_rect_to_img(self, r: QRect) -> tuple:
        ir = self._img_rect
        if self._pixmap is None:
            return (0, 0, 0, 0)
        iw, ih = self._pixmap.width(), self._pixmap.height()
        sx = iw / ir.width()
        sy = ih / ir.height()
        x1 = int((r.left()   - ir.x()) * sx)
        y1 = int((r.top()    - ir.y()) * sy)
        x2 = int((r.right()  - ir.x()) * sx)
        y2 = int((r.bottom() - ir.y()) * sy)
        return (
            max(0, min(iw, x1)),
            max(0, min(ih, y1)),
            max(0, min(iw, x2)),
            max(0, min(ih, y2)),
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

    # ---- construction UI --------------------------------------------------

    def _build_ui(self):
        self.setWindowTitle("Labeling putter")
        self.resize(1100, 800)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        # Barre de progression
        self.progress = QProgressBar()
        self.progress.setMaximum(max(len(self.images), 1))
        self.progress.setFormat("%v / %m annotées")
        root.addWidget(self.progress)

        # Label de statut
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        f = QFont()
        f.setPointSize(12)
        self.status_label.setFont(f)
        root.addWidget(self.status_label)

        # Canvas
        self.canvas = ImageCanvas()
        self.canvas.bbox_changed.connect(self._on_bbox_changed)
        root.addWidget(self.canvas, stretch=1)

        # Boutons
        btns = QHBoxLayout()

        self.btn_prev = QPushButton("◀  Précédent  (←)")
        self.btn_prev.clicked.connect(self._prev)
        btns.addWidget(self.btn_prev)

        self.btn_clear = QPushButton("🗑  Effacer bbox  (D)")
        self.btn_clear.clicked.connect(self.canvas.clear_bbox)
        btns.addWidget(self.btn_clear)

        self.btn_skip = QPushButton("⏭  Passer / pas de putter  (S)")
        self.btn_skip.clicked.connect(self._skip)
        btns.addWidget(self.btn_skip)

        self.btn_next = QPushButton("✔  Sauvegarder + Suivant  (→ / Entrée)")
        self.btn_next.setDefault(True)
        self.btn_next.setStyleSheet("QPushButton { background: #2a7a2a; color: white; font-weight: bold; }")
        self.btn_next.clicked.connect(self._save_and_next)
        btns.addWidget(self.btn_next)

        root.addLayout(btns)

        hint = QLabel("Click + glisse → bbox  |  ← → ou Entrée pour naviguer  |  D = effacer  |  S = passer sans label  |  Q = quitter")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color: #666; font-size: 10px;")
        root.addWidget(hint)

    # ---- chargement -------------------------------------------------------

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

        lbl_path = self.out_labels / (img_path.stem + ".txt")
        marker = ""
        if lbl_path.exists():
            lines = lbl_path.read_text().strip().splitlines()
            if lines:
                parts = lines[0].split()
                if len(parts) >= 5:
                    _, cx, cy, bw, bh = parts[:5]
                    self.canvas.set_bbox_yolo(float(cx), float(cy), float(bw), float(bh))
            marker = "  ✓"

        labeled = self._labeled_count()
        self.progress.setValue(labeled)
        self.setWindowTitle(
            f"[{self.idx + 1}/{len(self.images)}]  {img_path.name}{marker}  —  {labeled} annotées"
        )
        self._set_status(f"Image {self.idx + 1} / {len(self.images)}   {img_path.name}{marker}")

    def _set_status(self, msg: str, error: bool = False):
        self.status_label.setText(msg)
        color = "#cc4444" if error else "#cccccc"
        self.status_label.setStyleSheet(f"color: {color};")

    # ---- actions ----------------------------------------------------------

    def _on_bbox_changed(self):
        if self.canvas.get_bbox_yolo():
            self._set_status(
                f"Image {self.idx + 1} / {len(self.images)}  —  bbox prête  →  appuie sur Entrée pour sauvegarder"
            )

    def _save_current(self) -> bool:
        bbox = self.canvas.get_bbox_yolo()
        if bbox is None:
            return False
        cx, cy, bw, bh = bbox
        cx = max(0.001, min(0.999, cx))
        cy = max(0.001, min(0.999, cy))
        bw = max(0.001, min(1.0,   bw))
        bh = max(0.001, min(1.0,   bh))

        img_path = self.images[self.idx]
        lbl_path = self.out_labels / (img_path.stem + ".txt")
        lbl_path.write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        out_img = self.out_images / img_path.name
        if not out_img.exists():
            shutil.copy2(img_path, out_img)
        return True

    def _save_and_next(self):
        if not self._save_current():
            self._set_status("⚠  Dessine d'abord une bbox autour du putter !", error=True)
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
                self,
                "Terminé !",
                f"Toutes les images ont été parcourues.\n"
                f"{labeled} / {len(self.images)} images annotées.\n\n"
                f"Lance maintenant :\n"
                f"  python3 training/autolabel.py split --dataset data/putter_dataset\n"
                f"  python3 training/augment.py --dataset data/putter_dataset --factor 8",
            )

    # ---- raccourcis clavier -----------------------------------------------

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Right, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._save_and_next()
        elif key == Qt.Key.Key_Left:
            self._prev()
        elif key in (Qt.Key.Key_D, Qt.Key.Key_Delete):
            self.canvas.clear_bbox()
        elif key == Qt.Key.Key_S:
            self._skip()
        elif key == Qt.Key.Key_Q:
            self.close()
        else:
            super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Outil de labeling manuel — putter bbox")
    parser.add_argument("--images", required=True, help="Dossier d'images brutes")
    parser.add_argument("--out",    required=True, help="Dossier dataset de sortie (data/putter_dataset)")
    parser.add_argument("--labels", default=None,  help="Dossier des labels existants (optionnel, ex: data/putter_dataset/labels/train)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = LabelTool(args.images, args.out, args.labels)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
