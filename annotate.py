#!/usr/bin/env python3
"""
annotate.py – Outil d'annotation de têtes de putter (boîtes orientées)
=======================================================================
Usage:
    python3 annotate.py <dossier_ou_image>

    Exemple:
        python3 annotate.py ~/Desktop/screenshots/
        python3 annotate.py ma_photo.png

Contrôles:
    Click-drag          → dessiner une boîte autour d'une tête
    ← → flèches         → faire pivoter la boîte sélectionnée (±5°)
    ↑ ↓ flèches         → changer de boîte sélectionnée
    Clic droit          → supprimer la boîte la plus proche
    Entrée / S          → sauvegarder et image suivante
    Backspace           → supprimer la dernière boîte
    N / →               → image suivante (sans sauvegarder)
    P / ←               → image précédente
    Q / Echap           → quitter

Format de sortie : YOLO OBB  →  annotation_output/labels/*.txt
                               annotation_output/images/*.jpg
"""

import cv2
import numpy as np
import os
import sys
import math
import glob
import shutil

OUTPUT_DIR = "annotation_output"
IMG_OUT    = os.path.join(OUTPUT_DIR, "images")
LBL_OUT    = os.path.join(OUTPUT_DIR, "labels")


# ─────────────────────────────────────────────────────────────────────────────
class Annotator:

    def __init__(self, image_paths: list[str]):
        self.paths    = image_paths
        self.idx      = 0

        # Current state
        self.boxes: list[list] = []    # each: [cx, cy, w, h, angle_deg]
        self.sel     = -1              # selected box index
        self.drag_s  = None            # drag start (x, y)
        self.drag_e  = None            # drag end   (x, y)
        self.drawing = False

        # Original image (for coordinate reference)
        self.img_orig: np.ndarray | None = None
        self.img_h = 0
        self.img_w = 0

        os.makedirs(IMG_OUT, exist_ok=True)
        os.makedirs(LBL_OUT, exist_ok=True)

        cv2.namedWindow("Annotate", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Annotate", 1200, 750)
        cv2.setMouseCallback("Annotate", self._mouse)

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def _mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_s  = (x, y)
            self.drag_e  = (x, y)
            self.drawing = True

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.drag_e = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drag_e = (x, y)
            self.drawing = False
            if self.drag_s:
                x0, y0 = self.drag_s
                x1, y1 = self.drag_e
                bw, bh  = abs(x1 - x0), abs(y1 - y0)
                if bw > 6 and bh > 6:
                    cx = (x0 + x1) // 2
                    cy = (y0 + y1) // 2
                    self.boxes.append([cx, cy, bw, bh, 0.0])
                    self.sel = len(self.boxes) - 1
            self.drag_s = None
            self.drag_e = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Delete nearest box
            if self.boxes:
                dists   = [math.hypot(b[0] - x, b[1] - y) for b in self.boxes]
                nearest = int(np.argmin(dists))
                if dists[nearest] < 80:
                    self.boxes.pop(nearest)
                    self.sel = len(self.boxes) - 1

        elif event == cv2.EVENT_MOUSEWHEEL:
            # Scroll to rotate selected box (cross-platform fallback)
            if self.boxes and 0 <= self.sel < len(self.boxes):
                delta = 5.0 if flags > 0 else -5.0
                self.boxes[self.sel][4] = (self.boxes[self.sel][4] + delta) % 360

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _render(self) -> np.ndarray:
        if self.img_orig is None:
            return np.zeros((600, 1000, 3), dtype=np.uint8)

        disp = self.img_orig.copy()

        # In-progress drag preview
        if self.drawing and self.drag_s and self.drag_e:
            cv2.rectangle(disp, self.drag_s, self.drag_e, (200, 100, 0), 2)

        # All annotated boxes
        for i, box in enumerate(self.boxes):
            cx, cy, bw, bh, angle = box
            selected = (i == self.sel)
            self._draw_box(disp, cx, cy, bw, bh, angle, selected)

        # Arc through centers
        if len(self.boxes) >= 2:
            centers = np.array([(b[0], b[1]) for b in self.boxes], np.int32)
            cv2.polylines(disp, [centers.reshape(-1, 1, 2)],
                          False, (200, 100, 0), 3, cv2.LINE_AA)
            cv2.polylines(disp, [centers.reshape(-1, 1, 2)],
                          False, (255, 220, 80), 1, cv2.LINE_AA)

        # HUD
        fname = os.path.basename(self.paths[self.idx])
        lbl_saved = os.path.isfile(
            os.path.join(LBL_OUT,
                         os.path.splitext(fname)[0] + ".txt"))
        status_clr = (80, 200, 80) if lbl_saved else (180, 180, 180)
        status_txt = "✓ sauvegardé" if lbl_saved else "non sauvegardé"

        h, w = disp.shape[:2]

        # Dark strip at bottom
        cv2.rectangle(disp, (0, h - 55), (w, h), (20, 20, 20), -1)

        info = (f"{self.idx + 1}/{len(self.paths)}  {fname}  [{status_txt}]"
                f"   |   {len(self.boxes)} boîte(s)")
        cv2.putText(disp, info, (10, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_clr, 1, cv2.LINE_AA)

        hints = ("Click-drag: boîte   ←→: tourner   ↑↓: choisir   "
                 "Clic-droit: effacer   Entrée/S: sauv+suiv   N/P: nav   Q: quitter")
        cv2.putText(disp, hints, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 130, 130), 1, cv2.LINE_AA)

        if self.boxes and 0 <= self.sel < len(self.boxes):
            b = self.boxes[self.sel]
            sel_info = (f"Boîte {self.sel + 1}:  centre=({b[0]},{b[1]})  "
                        f"taille={b[2]}×{b[3]}  angle={b[4]:.1f}°")
            cv2.putText(disp, sel_info, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 220, 80), 1, cv2.LINE_AA)

        return disp

    def _draw_box(self, frame, cx, cy, bw, bh, angle, selected=False):
        rect  = ((float(cx), float(cy)), (float(bw), float(bh)), float(angle))
        pts   = cv2.boxPoints(rect).astype(np.int32)
        color = (255, 220, 80) if selected else (200, 100, 0)
        cv2.drawContours(frame, [pts], 0, (0, 0, 0),     4, cv2.LINE_AA)
        cv2.drawContours(frame, [pts], 0, color,          2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 6, (0, 0, 0),    -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)

    # ── Navigation & save ─────────────────────────────────────────────────────

    def _load(self):
        """Load current image and any existing annotation."""
        path = self.paths[self.idx]
        img  = cv2.imread(path)
        if img is None:
            print(f"[warn] impossible de lire {path}")
            self.img_orig = np.zeros((600, 1000, 3), dtype=np.uint8)
        else:
            self.img_orig = img
        self.img_h, self.img_w = self.img_orig.shape[:2]
        self.boxes   = []
        self.sel     = -1
        self.drawing = False
        self.drag_s  = None
        self.drag_e  = None
        self._load_existing_label(path)

    def _load_existing_label(self, img_path: str):
        """Reload YOLO OBB labels if they exist."""
        fname    = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(LBL_OUT, fname + ".txt")
        if not os.path.isfile(lbl_path):
            return
        W, H = self.img_w, self.img_h
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 9:   # YOLO OBB: class + 4 corners (x1y1…x4y4)
                    pts = np.array([float(v) for v in parts[1:]]).reshape(4, 2)
                    pts[:, 0] *= W
                    pts[:, 1] *= H
                    rect = cv2.minAreaRect(pts.astype(np.float32))
                    cx, cy = int(rect[0][0]), int(rect[0][1])
                    bw, bh = int(rect[1][0]), int(rect[1][1])
                    angle  = float(rect[2])
                    self.boxes.append([cx, cy, bw, bh, angle])
        if self.boxes:
            self.sel = len(self.boxes) - 1

    def _save(self):
        """Save YOLO OBB labels + copy image to output."""
        path  = self.paths[self.idx]
        fname = os.path.splitext(os.path.basename(path))[0]

        # Copy image
        dest_img = os.path.join(IMG_OUT, os.path.basename(path))
        if not os.path.isfile(dest_img):
            shutil.copy2(path, dest_img)

        # Write labels
        lbl_path = os.path.join(LBL_OUT, fname + ".txt")
        W, H = self.img_w, self.img_h
        with open(lbl_path, "w") as f:
            for box in self.boxes:
                cx, cy, bw, bh, angle = box
                rect = ((float(cx), float(cy)), (float(bw), float(bh)), float(angle))
                pts  = cv2.boxPoints(rect)
                pts_n = (pts / [W, H]).flatten().tolist()
                f.write("0 " + " ".join(f"{v:.6f}" for v in pts_n) + "\n")

        n = len(self.boxes)
        print(f"[saved] {lbl_path}  ({n} boîte{'s' if n != 1 else ''})")

    def _next(self):
        if self.idx < len(self.paths) - 1:
            self.idx += 1
            self._load()

    def _prev(self):
        if self.idx > 0:
            self.idx -= 1
            self._load()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        self._load()

        while True:
            cv2.imshow("Annotate", self._render())
            key = cv2.waitKey(30) & 0xFF

            if key in (ord('q'), ord('Q'), 27):     # Q / Esc → quit
                break

            elif key in (13, ord('s'), ord('S')):    # Entrée / S → save + next
                self._save()
                self._next()

            elif key in (ord('n'), ord('N')):        # N → next
                self._next()

            elif key in (ord('p'), ord('P')):        # P → prev
                self._prev()

            elif key == 81:                          # ← flèche gauche
                if self.boxes and 0 <= self.sel < len(self.boxes):
                    self.boxes[self.sel][4] = (self.boxes[self.sel][4] - 5) % 360

            elif key == 83:                          # → flèche droite
                if self.boxes and 0 <= self.sel < len(self.boxes):
                    self.boxes[self.sel][4] = (self.boxes[self.sel][4] + 5) % 360

            elif key == 82:                          # ↑ flèche haut → boîte précédente
                if self.boxes:
                    self.sel = (self.sel - 1) % len(self.boxes)

            elif key == 84:                          # ↓ flèche bas → boîte suivante
                if self.boxes:
                    self.sel = (self.sel + 1) % len(self.boxes)

            elif key in (8, 127):                    # Backspace / Delete
                if self.boxes:
                    if 0 <= self.sel < len(self.boxes):
                        self.boxes.pop(self.sel)
                    self.sel = len(self.boxes) - 1

        cv2.destroyAllWindows()
        total_lbl = len(glob.glob(os.path.join(LBL_OUT, "*.txt")))
        print(f"\nTerminé — {total_lbl} image(s) annotée(s) dans {OUTPUT_DIR}/")
        if total_lbl >= 5:
            print("Lance l'entraînement avec :  python3 train_annotated.py")


# ─────────────────────────────────────────────────────────────────────────────

def collect_images(source: str) -> list[str]:
    if os.path.isfile(source):
        return [source]
    if os.path.isdir(source):
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            paths.extend(glob.glob(os.path.join(source, ext)))
            paths.extend(glob.glob(os.path.join(source, ext.upper())))
        return sorted(set(paths))
    return []


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage: python3 annotate.py <dossier_screenshots/>")
        sys.exit(1)

    paths = collect_images(sys.argv[1])
    if not paths:
        print(f"[erreur] Aucune image trouvée dans : {sys.argv[1]}")
        sys.exit(1)

    print(f"[annotate] {len(paths)} image(s) chargée(s)")
    Annotator(paths).run()


if __name__ == "__main__":
    main()
