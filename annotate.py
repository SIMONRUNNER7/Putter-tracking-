#!/usr/bin/env python3
"""
Outil d'annotation rapide — tête de putter vue de dessus.

Usage:
    python3 annotate.py test-putt.mov          # depuis une vidéo
    python3 annotate.py real_frames/            # depuis un dossier d'images

Contrôles:
    Clic + glisser   → dessiner la bounding box
    Entrée / Espace  → valider et passer à la suivante
    S                → skip (pas de putter visible)
    Backspace        → effacer la box courante
    ←                → image précédente
    Q / Echap        → quitter et sauvegarder

Sortie (format YOLO):
    annotation_output/images/   → frames JPG
    annotation_output/labels/   → fichiers .txt YOLO
"""
import sys
import os
import cv2
import glob
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QProgressBar, QStatusBar, QSizePolicy
)
from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QKeySequence, QShortcut


OUTPUT_DIR   = "annotation_output"
IMAGES_DIR   = os.path.join(OUTPUT_DIR, "images")
LABELS_DIR   = os.path.join(OUTPUT_DIR, "labels")
CLASS_ID     = 0   # putter_head
FRAME_STEP   = 5   # extraire 1 frame sur N depuis la vidéo


# ---------------------------------------------------------------------------
# Canvas de dessin
# ---------------------------------------------------------------------------

class AnnotationCanvas(QLabel):
    box_drawn = pyqtSignal(QRect)   # émis quand l'utilisateur finit de dessiner

    def __init__(self):
        super().__init__()
        self._pixmap_orig: QPixmap | None = None
        self._box: QRect | None = None
        self._start: QPoint | None = None
        self._drawing = False
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background: #1a1a1a;")

    def set_image(self, pixmap: QPixmap) -> None:
        self._pixmap_orig = pixmap
        self._box = None
        self._redraw()

    def set_box(self, box: QRect | None) -> None:
        self._box = box
        self._redraw()

    def get_box(self) -> QRect | None:
        return self._box

    # ---- Mouse events -------------------------------------------------------

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._pixmap_orig:
            self._start   = e.pos()
            self._drawing = True
            self._box     = None

    def mouseMoveEvent(self, e):
        if self._drawing and self._start:
            self._box = QRect(self._start, e.pos()).normalized()
            self._redraw()

    def mouseReleaseEvent(self, e):
        if self._drawing:
            self._drawing = False
            if self._box and self._box.width() > 5 and self._box.height() > 5:
                self.box_drawn.emit(self._box)
            self._redraw()

    # ---- Drawing ------------------------------------------------------------

    def _redraw(self):
        if self._pixmap_orig is None:
            return
        scaled = self._pixmap_orig.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        # Compute offset (centered)
        ox = (self.width()  - scaled.width())  // 2
        oy = (self.height() - scaled.height()) // 2

        result = QPixmap(self.size())
        result.fill(QColor("#1a1a1a"))
        painter = QPainter(result)
        painter.drawPixmap(ox, oy, scaled)

        if self._box:
            pen = QPen(QColor("#00ff88"), 2)
            painter.setPen(pen)
            painter.drawRect(self._box)
            # Small label
            painter.setPen(QColor("#00ff88"))
            painter.drawText(self._box.topLeft() + QPoint(4, -6), "putter_head")

        painter.end()
        self.setPixmap(result)

    def resizeEvent(self, e):
        self._redraw()

    # ---- Convert box to image coordinates -----------------------------------

    def box_in_image_coords(self) -> tuple[float, float, float, float] | None:
        """Return (x_center, y_center, width, height) normalized [0-1]."""
        if self._box is None or self._pixmap_orig is None:
            return None

        scaled = self._pixmap_orig.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        ox = (self.width()  - scaled.width())  // 2
        oy = (self.height() - scaled.height()) // 2

        sx = self._pixmap_orig.width()  / scaled.width()
        sy = self._pixmap_orig.height() / scaled.height()

        bx = (self._box.x() - ox) * sx
        by = (self._box.y() - oy) * sy
        bw = self._box.width()  * sx
        bh = self._box.height() * sy

        iw = self._pixmap_orig.width()
        ih = self._pixmap_orig.height()

        cx = (bx + bw / 2) / iw
        cy = (by + bh / 2) / ih
        nw = bw / iw
        nh = bh / ih

        # Clamp
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        nw = max(0.0, min(1.0, nw))
        nh = max(0.0, min(1.0, nh))

        return cx, cy, nw, nh


# ---------------------------------------------------------------------------
# Fenêtre principale
# ---------------------------------------------------------------------------

class AnnotatorWindow(QMainWindow):
    def __init__(self, frames: list[tuple[str, str]]):
        """frames = list of (image_path, label_path)"""
        super().__init__()
        self.frames     = frames
        self.idx        = 0
        self.saved      = 0
        self.skipped    = 0

        os.makedirs(IMAGES_DIR, exist_ok=True)
        os.makedirs(LABELS_DIR, exist_ok=True)

        self._build_ui()
        self._load_frame()

    # ---- UI -----------------------------------------------------------------

    def _build_ui(self):
        self.setWindowTitle("Annotateur Putter — PutterTrack Pro")
        self.resize(1000, 750)
        self.setStyleSheet("background:#222; color:#eee; font-size:13px;")

        central = QWidget()
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(6)

        # Canvas
        self.canvas = AnnotationCanvas()
        self.canvas.box_drawn.connect(self._on_box_drawn)
        vbox.addWidget(self.canvas, stretch=1)

        # Progress
        self.progress = QProgressBar()
        self.progress.setMaximum(len(self.frames))
        self.progress.setTextVisible(True)
        self.progress.setStyleSheet(
            "QProgressBar{border:1px solid #555;border-radius:4px;height:18px;}"
            "QProgressBar::chunk{background:#00cc66;}"
        )
        vbox.addWidget(self.progress)

        # Buttons
        btn_row = QHBoxLayout()
        btn_style = (
            "QPushButton{background:#333;border:1px solid #555;border-radius:6px;"
            "padding:8px 20px;color:#eee;font-size:13px;}"
            "QPushButton:hover{background:#444;}"
            "QPushButton:pressed{background:#00cc66;color:#000;}"
        )

        self.btn_prev = QPushButton("← Précédent  [←]")
        self.btn_skip = QPushButton("Skip — pas de putter  [S]")
        self.btn_clear = QPushButton("Effacer box  [⌫]")
        self.btn_save = QPushButton("✓ Valider  [Entrée]")
        self.btn_save.setStyleSheet(btn_style +
            "QPushButton#save{background:#005533;}"
        )
        self.btn_save.setObjectName("save")

        for b in [self.btn_prev, self.btn_skip, self.btn_clear, self.btn_save]:
            b.setStyleSheet(btn_style)
            btn_row.addWidget(b)

        self.btn_prev.clicked.connect(self._prev)
        self.btn_skip.clicked.connect(self._skip)
        self.btn_clear.clicked.connect(self._clear_box)
        self.btn_save.clicked.connect(self._save_and_next)

        vbox.addLayout(btn_row)

        # Raccourcis clavier
        QShortcut(QKeySequence(Qt.Key.Key_Return),    self, self._save_and_next)
        QShortcut(QKeySequence(Qt.Key.Key_Space),     self, self._save_and_next)
        QShortcut(QKeySequence(Qt.Key.Key_S),         self, self._skip)
        QShortcut(QKeySequence(Qt.Key.Key_Backspace), self, self._clear_box)
        QShortcut(QKeySequence(Qt.Key.Key_Left),      self, self._prev)
        QShortcut(QKeySequence(Qt.Key.Key_Escape),    self, self.close)
        QShortcut(QKeySequence(Qt.Key.Key_Q),         self, self.close)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)

    # ---- Frame loading ------------------------------------------------------

    def _load_frame(self):
        if self.idx >= len(self.frames):
            self._finish()
            return

        img_path, _ = self.frames[self.idx]
        img = cv2.imread(img_path)
        if img is None:
            self._next_idx()
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.canvas.set_image(QPixmap.fromImage(qimg))

        # Reload existing annotation if any
        _, lbl_path = self.frames[self.idx]
        if os.path.isfile(lbl_path):
            # Show existing box (approximate)
            with open(lbl_path) as f:
                line = f.read().strip()
            if line:
                parts = line.split()
                if len(parts) == 5:
                    cx, cy, nw, nh = map(float, parts[1:])
                    # convert to canvas coords (rough)
                    self.status.showMessage(f"Annotation existante chargée")

        self.progress.setValue(self.idx)
        self._update_status()
        self.btn_prev.setEnabled(self.idx > 0)

    def _update_status(self):
        _, lbl_path = self.frames[self.idx]
        annotated = os.path.isfile(lbl_path)
        self.setWindowTitle(
            f"Annotateur — {self.idx + 1}/{len(self.frames)}  "
            f"({self.saved} sauvegardés, {self.skipped} skippés)"
        )
        hint = "✓ déjà annoté" if annotated else "Dessine la box autour de la TÊTE du putter"
        self.status.showMessage(
            f"Frame {self.idx + 1}/{len(self.frames)}  |  {hint}"
        )

    # ---- Actions ------------------------------------------------------------

    def _on_box_drawn(self, rect: QRect):
        self.status.showMessage("Box dessinée — Entrée pour valider, S pour skipper")

    def _save_and_next(self):
        coords = self.canvas.box_in_image_coords()
        if coords is None:
            self.status.showMessage("⚠ Dessine une box d'abord (ou appuie sur S pour skipper)")
            return

        img_src, lbl_path = self.frames[self.idx]

        # Copy image to output if not already there
        out_img = os.path.join(IMAGES_DIR, os.path.basename(img_src))
        if not os.path.isfile(out_img):
            import shutil
            shutil.copy2(img_src, out_img)

        # Save YOLO label
        cx, cy, nw, nh = coords
        with open(lbl_path, "w") as f:
            f.write(f"{CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        self.saved += 1
        self._next_idx()

    def _skip(self):
        """Mark frame as explicitly skipped (empty label = no object)."""
        _, lbl_path = self.frames[self.idx]
        open(lbl_path, "w").close()   # fichier vide = pas de détection
        self.skipped += 1
        self._next_idx()

    def _clear_box(self):
        self.canvas.set_box(None)
        self.status.showMessage("Box effacée")

    def _prev(self):
        if self.idx > 0:
            self.idx -= 1
            self._load_frame()

    def _next_idx(self):
        self.idx += 1
        if self.idx >= len(self.frames):
            self._finish()
        else:
            self._load_frame()

    def _finish(self):
        self.status.showMessage(
            f"✓ Terminé ! {self.saved} annotées, {self.skipped} skippées. "
            f"Dataset dans {OUTPUT_DIR}/"
        )
        self.setWindowTitle(f"Terminé — {self.saved} annotations sauvegardées")
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(
            self, "Annotation terminée",
            f"✓ {self.saved} images annotées\n"
            f"  {self.skipped} skippées\n\n"
            f"Dataset prêt dans : {os.path.abspath(OUTPUT_DIR)}/\n\n"
            f"Lance l'entraînement avec :\n"
            f"  python3 train_real.py"
        )


# ---------------------------------------------------------------------------
# Chargement des frames
# ---------------------------------------------------------------------------

def load_from_video(video_path: str) -> list[tuple[str, str]]:
    """Extraire les frames de la vidéo dans un dossier temporaire."""
    frames_dir = "annotation_frames"
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[annotate] Vidéo : {total} frames — extraction 1/{FRAME_STEP}...")

    i = 0
    paths = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % FRAME_STEP == 0:
            p = os.path.join(frames_dir, f"frame_{i:05d}.jpg")
            if not os.path.isfile(p):
                cv2.imwrite(p, frame)
            lbl = os.path.join(LABELS_DIR, f"frame_{i:05d}.txt")
            paths.append((p, lbl))
        i += 1

    cap.release()
    print(f"[annotate] {len(paths)} frames prêtes")
    return paths


def load_from_folder(folder: str) -> list[tuple[str, str]]:
    exts = ["*.jpg", "*.jpeg", "*.png"]
    imgs = []
    for e in exts:
        imgs.extend(glob.glob(os.path.join(folder, e)))
    imgs = sorted(imgs)
    os.makedirs(LABELS_DIR, exist_ok=True)
    return [
        (p, os.path.join(LABELS_DIR, os.path.splitext(os.path.basename(p))[0] + ".txt"))
        for p in imgs
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 annotate.py <video.mov | dossier_images/>")
        sys.exit(1)

    source = sys.argv[1]
    os.makedirs(LABELS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    if os.path.isfile(source):
        frames = load_from_video(source)
    elif os.path.isdir(source):
        frames = load_from_folder(source)
    else:
        print(f"[ERROR] Fichier ou dossier introuvable : {source}")
        sys.exit(1)

    if not frames:
        print("[ERROR] Aucune image trouvée")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setApplicationName("Annotateur Putter")

    window = AnnotatorWindow(frames)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
