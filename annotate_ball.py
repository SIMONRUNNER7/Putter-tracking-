#!/usr/bin/env python3
"""
annotate_ball.py – Annotation balle de golf (1 clic ou bbox glissé)
====================================================================
Usage:
    python3 annotate_ball.py                    # scanne captures/raw/
    python3 annotate_ball.py <dossier_ou_image>

Contrôles:
    Clic gauche        → placer le centre de la balle (carré auto)
    Clic gauche + drag → dessiner une bbox libre autour de la balle
    Molette            → ajuster la taille du carré (mode clic simple)
    Clic droit         → supprimer l'annotation courante
    Entrée / S         → sauvegarder et image suivante
    N / →              → image suivante (sans sauvegarder)
    P / ←              → image précédente
    Q / Echap          → quitter

Sortie YOLO (classe 1 = balle) :
    annotation_output/images/
    annotation_output/labels/
"""

from __future__ import annotations
import cv2
import numpy as np
import os, sys, glob, shutil
from datetime import datetime

OUTPUT_DIR = "annotation_output"
IMG_OUT    = os.path.join(OUTPUT_DIR, "images")
LBL_OUT    = os.path.join(OUTPUT_DIR, "labels")

FONT       = cv2.FONT_HERSHEY_SIMPLEX
BALL_COLOR = (0, 200, 255)   # orange-jaune pour la balle
DIM_TEXT   = (130, 130, 130)
WHITE      = (255, 255, 255)
GREEN      = (60, 220, 80)
RED        = (60, 60, 220)

DEFAULT_BOX_PX = 32   # taille initiale du carré en pixels


class BallAnnotator:

    def __init__(self, paths: list[str]):
        self.paths   = paths
        self.idx     = 0

        # Annotation courante : (cx, cy, bw, bh) en pixels image, ou None
        self.bbox: tuple[int, int, int, int] | None = None

        # État drag
        self._drag_start: tuple[int, int] | None = None
        self._drag_cur:   tuple[int, int] | None = None
        self._dragging    = False

        # Taille du carré lors d'un simple clic
        self._box_size = DEFAULT_BOX_PX

        self.cursor       = (0, 0)
        self.img_orig: np.ndarray | None = None
        self.img_h = self.img_w = 0

        os.makedirs(IMG_OUT, exist_ok=True)
        os.makedirs(LBL_OUT, exist_ok=True)

        cv2.namedWindow("Annotate Ball", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Annotate Ball", 1280, 800)
        cv2.setMouseCallback("Annotate Ball", self._mouse)

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def _mouse(self, event, x, y, flags, param):
        self.cursor = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag_start = (x, y)
            self._drag_cur   = (x, y)
            self._dragging   = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._dragging:
                self._drag_cur = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self._dragging:
                self._dragging = False
                x0, y0 = self._drag_start
                dx = abs(x - x0)
                dy = abs(y - y0)
                if dx < 5 and dy < 5:
                    # simple clic → carré centré
                    h = self._box_size // 2
                    self.bbox = (x - h, y - h, self._box_size, self._box_size)
                else:
                    # drag → bbox libre
                    bx = min(x0, x);  by = min(y0, y)
                    bw = abs(x - x0); bh = abs(y - y0)
                    self.bbox = (bx, by, bw, bh)
                self._drag_start = self._drag_cur = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.bbox = None
            self._drag_start = self._drag_cur = None
            self._dragging   = False

        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = 4 if flags > 0 else -4
            self._box_size = max(8, min(200, self._box_size + delta))
            # Si une bbox simple clic est déjà posée, la redimensionner
            if self.bbox and not self._dragging:
                cx = self.bbox[0] + self.bbox[2] // 2
                cy = self.bbox[1] + self.bbox[3] // 2
                h = self._box_size // 2
                self.bbox = (cx - h, cy - h, self._box_size, self._box_size)

    # ── Rendu ─────────────────────────────────────────────────────────────────

    def _render(self) -> np.ndarray:
        if self.img_orig is None:
            return np.zeros((600, 1000, 3), dtype=np.uint8)

        disp = self.img_orig.copy()
        H, W = disp.shape[:2]

        # Aperçu drag en cours
        if self._dragging and self._drag_start and self._drag_cur:
            x0, y0 = self._drag_start
            x1, y1 = self._drag_cur
            dx, dy = abs(x1 - x0), abs(y1 - y0)
            if dx < 5 and dy < 5:
                # preview clic simple
                h = self._box_size // 2
                cx, cy = self.cursor
                cv2.rectangle(disp, (cx - h, cy - h), (cx + h, cy + h),
                              BALL_COLOR, 1, cv2.LINE_AA)
                cv2.circle(disp, (cx, cy), 3, BALL_COLOR, -1, cv2.LINE_AA)
            else:
                bx = min(x0, x1); by = min(y0, y1)
                bw = abs(x1 - x0); bh = abs(y1 - y0)
                cv2.rectangle(disp, (bx, by), (bx + bw, by + bh),
                              BALL_COLOR, 1, cv2.LINE_AA)
        elif not self.bbox:
            # Crosshair léger sur le curseur
            cx, cy = self.cursor
            h = self._box_size // 2
            cv2.rectangle(disp, (cx - h, cy - h), (cx + h, cy + h),
                          (80, 80, 80), 1, cv2.LINE_AA)
            cv2.line(disp, (cx - 12, cy), (cx + 12, cy), (80, 80, 80), 1)
            cv2.line(disp, (cx, cy - 12), (cx, cy + 12), (80, 80, 80), 1)

        # Bbox validée
        if self.bbox:
            bx, by, bw, bh = self.bbox
            # Clamp
            bx = max(0, min(W - 1, bx)); by = max(0, min(H - 1, by))
            bw = max(1, min(W - bx, bw)); bh = max(1, min(H - by, bh))
            cx_b, cy_b = bx + bw // 2, by + bh // 2
            cv2.rectangle(disp, (bx, by), (bx + bw, by + bh), (0, 0, 0), 3, cv2.LINE_AA)
            cv2.rectangle(disp, (bx, by), (bx + bw, by + bh), BALL_COLOR, 2, cv2.LINE_AA)
            cv2.circle(disp, (cx_b, cy_b), 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(disp, (cx_b, cy_b), 3, WHITE, -1, cv2.LINE_AA)
            _put(disp, f"balle  {bw}×{bh}px",
                 (bx + 4, by - 8 if by > 20 else by + bh + 18),
                 scale=0.45, color=BALL_COLOR)

        # HUD bas
        fname    = os.path.basename(self.paths[self.idx])
        lbl_done = os.path.isfile(
            os.path.join(LBL_OUT, os.path.splitext(fname)[0] + ".txt"))

        cv2.rectangle(disp, (0, H - 56), (W, H), (18, 18, 18), -1)
        info = (f"  {self.idx + 1}/{len(self.paths)}   {fname}"
                f"   {'✓ annotée' if lbl_done else '○ vide'}"
                f"   |   taille carré : {self._box_size}px")
        cv2.putText(disp, info, (8, H - 36), FONT, 0.50,
                    GREEN if lbl_done else WHITE, 1, cv2.LINE_AA)

        hint = ("Clic: pointer balle   Drag: bbox libre   Molette: taille   "
                "Clic-droit: effacer   Entrée/S: sauv+suiv   N/P: nav   Q: quitter")
        cv2.putText(disp, hint, (8, H - 12), FONT, 0.38, DIM_TEXT, 1, cv2.LINE_AA)

        return disp

    # ── I/O ───────────────────────────────────────────────────────────────────

    def _load(self):
        path = self.paths[self.idx]
        img  = cv2.imread(path)
        if img is None:
            print(f"[warn] impossible de lire {path}")
            self.img_orig = np.zeros((600, 1000, 3), dtype=np.uint8)
        else:
            self.img_orig = img
        self.img_h, self.img_w = self.img_orig.shape[:2]
        self.bbox        = None
        self._drag_start = None
        self._drag_cur   = None
        self._dragging   = False
        self._load_existing()

    def _load_existing(self):
        path  = self.paths[self.idx]
        fname = os.path.splitext(os.path.basename(path))[0]
        lpath = os.path.join(LBL_OUT, fname + ".txt")
        if not os.path.isfile(lpath):
            return
        W, H = self.img_w, self.img_h
        with open(lpath) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5 and parts[0] == "1":
                    _, cx_n, cy_n, bw_n, bh_n = parts
                    bw = int(float(bw_n) * W); bh = int(float(bh_n) * H)
                    bx = int(float(cx_n) * W - bw / 2)
                    by = int(float(cy_n) * H - bh / 2)
                    self.bbox = (bx, by, bw, bh)
                    break

    def _save(self):
        if self.bbox is None:
            print("[skip] pas d'annotation")
            return
        path  = self.paths[self.idx]
        fname = os.path.splitext(os.path.basename(path))[0]
        dest  = os.path.join(IMG_OUT, os.path.basename(path))
        if not os.path.isfile(dest):
            shutil.copy2(path, dest)

        W, H  = self.img_w, self.img_h
        bx, by, bw, bh = self.bbox
        bx = max(0, min(W - 1, bx)); by = max(0, min(H - 1, by))
        bw = max(1, min(W - bx, bw)); bh = max(1, min(H - by, bh))
        cx_n = (bx + bw / 2) / W
        cy_n = (by + bh / 2) / H
        bw_n = bw / W; bh_n = bh / H

        lpath = os.path.join(LBL_OUT, fname + ".txt")
        with open(lpath, "w") as f:
            f.write(f"1 {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}\n")
        print(f"[saved] {lpath}  bbox=({bx},{by},{bw},{bh})")

    def _next(self):
        if self.idx < len(self.paths) - 1:
            self.idx += 1;  self._load()

    def _prev(self):
        if self.idx > 0:
            self.idx -= 1;  self._load()

    # ── Boucle ────────────────────────────────────────────────────────────────

    def run(self):
        self._load()
        while True:
            cv2.imshow("Annotate Ball", self._render())
            raw = cv2.waitKey(20)
            if raw == -1:
                continue

            LEFT  = raw in (81, 2424832, 63234)
            RIGHT = raw in (83, 2555904, 63235)
            key   = raw & 0xFF

            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (13, ord('s'), ord('S')):     # Entrée/S → sauv+suiv
                self._save();  self._next()
            elif key in (ord('n'), ord('N')) or RIGHT:
                self._next()
            elif key in (ord('p'), ord('P')) or LEFT:
                self._prev()
            elif key in (8, 127):                     # Backspace/Del → effacer
                self.bbox = None

        cv2.destroyAllWindows()
        total = len(glob.glob(os.path.join(LBL_OUT, "*.txt")))
        print(f"\nTerminé — {total} image(s) annotée(s) dans {OUTPUT_DIR}/")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _put(img, text, pos, scale=0.45, color=WHITE):
    cv2.putText(img, text, (pos[0]+1, pos[1]+1),
                FONT, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, FONT, scale, color, 1, cv2.LINE_AA)


def collect_images(source: str) -> list[str]:
    if os.path.isfile(source):
        return [source]
    if os.path.isdir(source):
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
            paths.extend(glob.glob(os.path.join(source, ext)))
        return sorted(set(paths))
    return []


def main():
    # Source par défaut : frames raw impact+exit de captures/raw/
    if len(sys.argv) > 1:
        src = sys.argv[1]
        paths = collect_images(src)
    else:
        raw_dir = os.path.join("captures", "raw")
        if os.path.isdir(raw_dir):
            paths = sorted(
                glob.glob(os.path.join(raw_dir, "*_impact.jpg")) +
                glob.glob(os.path.join(raw_dir, "*_exit.jpg"))
            )
        else:
            # Fallback : tout le dossier captures
            paths = collect_images("captures")

    if not paths:
        print("[erreur] Aucune image trouvée.")
        print("  Lance : python3 annotate_ball.py <dossier>")
        sys.exit(1)

    print(f"[annotate_ball] {len(paths)} image(s)")
    print("  Clic = pointer la balle   Molette = taille du carré")
    BallAnnotator(paths).run()


if __name__ == "__main__":
    main()
