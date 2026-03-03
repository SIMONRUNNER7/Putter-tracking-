#!/usr/bin/env python3
"""
annotate.py – Annotation 3 clics pour têtes de putter (boîtes orientées)
=========================================================================
Usage:
    python3 annotate.py '/Users/simonlinot/Desktop/to train'
    python3 annotate.py image.png

Dessin d'une boîte en 3 clics :
    Clic 1  →  bas de la face du putter
    Clic 2  →  haut de la face  (donne l'orientation + côté gauche)
    Clic 3  →  largeur vers la droite

Autres contrôles :
    ← →  ou  A D     → tourner la boîte sélectionnée ±5°
    ↑ ↓  ou  W X     → changer de boîte sélectionnée
    Clic droit        → annuler le dessin en cours / supprimer boîte
    Entrée / S        → sauvegarder et image suivante
    Backspace         → supprimer la dernière boîte
    N                 → image suivante (sans sauvegarder)
    P                 → image précédente
    Q / Echap         → quitter

Sortie YOLO OBB :  annotation_output/images/  +  annotation_output/labels/
"""

import cv2
import numpy as np
import os, sys, math, glob, shutil

OUTPUT_DIR = "annotation_output"
IMG_OUT    = os.path.join(OUTPUT_DIR, "images")
LBL_OUT    = os.path.join(OUTPUT_DIR, "labels")

# ─────────────────────────────────────────────────────────────────────────────

class Annotator:

    def __init__(self, paths: list[str]):
        self.paths = paths
        self.idx   = 0

        self.boxes: list[list] = []   # [cx, cy, w, h, angle]
        self.sel   = -1

        # 3-click state
        self.click_state = 0          # 0=idle, 1=got p1, 2=got p1+p2
        self.p1 = None
        self.p2 = None
        self.cursor = (0, 0)

        self.img_orig: np.ndarray | None = None
        self.img_h = self.img_w = 0

        os.makedirs(IMG_OUT, exist_ok=True)
        os.makedirs(LBL_OUT, exist_ok=True)

        cv2.namedWindow("Annotate", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Annotate", 1280, 800)
        cv2.setMouseCallback("Annotate", self._mouse)

    # ── Mouse callback ────────────────────────────────────────────────────────

    def _mouse(self, event, x, y, flags, param):
        self.cursor = (x, y)

        if event == cv2.EVENT_MOUSEMOVE:
            return  # redraw handled in main loop

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.click_state == 0:
                self.p1          = (x, y)
                self.click_state = 1

            elif self.click_state == 1:
                self.p2          = (x, y)
                self.click_state = 2

            elif self.click_state == 2:
                self._finish_box(x, y)
                self.click_state = 0
                self.p1 = self.p2 = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.click_state > 0:
                # cancel current drawing
                self.click_state = 0
                self.p1 = self.p2 = None
            elif self.boxes:
                # delete nearest box
                dists   = [math.hypot(b[0] - x, b[1] - y) for b in self.boxes]
                nearest = int(np.argmin(dists))
                if dists[nearest] < 100:
                    self.boxes.pop(nearest)
                    self.sel = len(self.boxes) - 1

        elif event == cv2.EVENT_MOUSEWHEEL:
            if self.boxes and 0 <= self.sel < len(self.boxes):
                delta = 5.0 if flags > 0 else -5.0
                self.boxes[self.sel][4] = round(
                    (self.boxes[self.sel][4] + delta) % 360, 1)

    # ── 3-click box computation ───────────────────────────────────────────────

    def _box_corners(self, p3x: int, p3y: int):
        """
        Given p1 (bas face), p2 (haut face), p3 (largeur droite)
        → retourne les 4 coins du rectangle orienté.
        """
        p1 = np.array(self.p1, dtype=float)
        p2 = np.array(self.p2, dtype=float)
        p3 = np.array([p3x, p3y], dtype=float)

        edge     = p2 - p1
        edge_len = np.linalg.norm(edge)
        if edge_len < 2:
            return None

        edge_dir = edge / edge_len
        # Perpendiculaire vers la droite quand on regarde de p1 vers p2
        perp_dir = np.array([edge_dir[1], -edge_dir[0]])

        # Largeur = projection de (p3 - p1) sur la perp
        width = float(np.dot(p3 - p1, perp_dir))

        c1 = p1
        c2 = p2
        c3 = p2 + width * perp_dir
        c4 = p1 + width * perp_dir
        return np.array([c1, c2, c3, c4], dtype=np.float32)

    def _finish_box(self, p3x: int, p3y: int):
        corners = self._box_corners(p3x, p3y)
        if corners is None:
            return
        rect  = cv2.minAreaRect(corners)
        cx, cy = int(rect[0][0]), int(rect[0][1])
        bw, bh = int(rect[1][0]), int(rect[1][1])
        angle  = round(float(rect[2]), 1)
        self.boxes.append([cx, cy, bw, bh, angle])
        self.sel = len(self.boxes) - 1

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render(self) -> np.ndarray:
        if self.img_orig is None:
            return np.zeros((600, 1000, 3), dtype=np.uint8)
        disp = self.img_orig.copy()
        H, W = disp.shape[:2]

        # ── Preview selon l'état de clic ────────────────────────────────
        cx_cur, cy_cur = self.cursor

        if self.click_state >= 1 and self.p1:
            # Ligne p1 → curseur (ou p2)
            end = self.p2 if self.click_state == 2 else (cx_cur, cy_cur)
            cv2.line(disp, self.p1, end, (255, 220, 80), 2, cv2.LINE_AA)
            cv2.circle(disp, self.p1, 6, (255, 220, 80), -1)
            self._put(disp, "1", (self.p1[0] + 8, self.p1[1] - 8),
                      scale=0.5, color=(255, 220, 80))

        if self.click_state == 2 and self.p2:
            cv2.circle(disp, self.p2, 6, (255, 220, 80), -1)
            self._put(disp, "2", (self.p2[0] + 8, self.p2[1] - 8),
                      scale=0.5, color=(255, 220, 80))
            # Prévisualisation du rectangle
            corners = self._box_corners(cx_cur, cy_cur)
            if corners is not None:
                pts = corners.astype(np.int32)
                cv2.polylines(disp, [pts], True, (200, 100, 0), 2, cv2.LINE_AA)
                # Midpoint indicatif
                mid = corners.mean(axis=0).astype(int)
                cv2.circle(disp, tuple(mid), 4, (255, 255, 255), -1)

        # ── Boîtes validées ─────────────────────────────────────────────
        for i, box in enumerate(self.boxes):
            self._draw_box(disp, *box, selected=(i == self.sel))

        # Arc à travers les centres
        if len(self.boxes) >= 2:
            centers = np.array([(b[0], b[1]) for b in self.boxes], np.int32)
            cv2.polylines(disp, [centers.reshape(-1, 1, 2)],
                          False, (80, 80, 200), 3, cv2.LINE_AA)
            cv2.polylines(disp, [centers.reshape(-1, 1, 2)],
                          False, (200, 180, 255), 1, cv2.LINE_AA)

        # ── HUD ─────────────────────────────────────────────────────────
        fname    = os.path.basename(self.paths[self.idx])
        lbl_done = os.path.isfile(
            os.path.join(LBL_OUT,
                         os.path.splitext(fname)[0] + ".txt"))

        cv2.rectangle(disp, (0, H - 60), (W, H), (20, 20, 20), -1)
        info = (f"  {self.idx + 1}/{len(self.paths)}  {fname}"
                f"  {'✓' if lbl_done else '○'}   |   {len(self.boxes)} boîte(s)")
        cv2.putText(disp, info, (8, H - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (80, 200, 80) if lbl_done else (200, 200, 200), 1, cv2.LINE_AA)

        steps = ["Clic 1: bas face", "Clic 2: haut face", "Clic 3: largeur →"]
        step_txt = steps[self.click_state] if self.click_state < 3 else ""
        cv2.putText(disp, step_txt, (8, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 220, 80), 1, cv2.LINE_AA)

        hint = ("A/D ou ←→: tourner  |  W/X ou ↑↓: choisir boîte  |  "
                "Entrée/S: sauv+suiv  |  Clic-droit: annuler/supp  |  N/P: nav  |  Q: quitter")
        cv2.putText(disp, hint, (200, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA)

        if self.boxes and 0 <= self.sel < len(self.boxes):
            b = self.boxes[self.sel]
            sel_info = (f"  Boîte {self.sel + 1} :  "
                        f"centre ({b[0]}, {b[1]})  "
                        f"taille {b[2]}×{b[3]}  angle {b[4]:.1f}°")
            cv2.putText(disp, sel_info, (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 220, 80), 1, cv2.LINE_AA)

        return disp

    def _draw_box(self, frame, cx, cy, bw, bh, angle, selected=False):
        rect = ((float(cx), float(cy)), (float(bw), float(bh)), float(angle))
        pts  = cv2.boxPoints(rect).astype(np.int32)
        col  = (255, 220, 80) if selected else (200, 100, 0)
        cv2.drawContours(frame, [pts], 0, (0, 0, 0),  4, cv2.LINE_AA)
        cv2.drawContours(frame, [pts], 0, col,         2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 6, (0, 0, 0),     -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)

    def _put(self, frame, text, pos, scale=0.45, color=(255, 255, 255)):
        cv2.putText(frame, text, (pos[0]+1, pos[1]+1),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    # ── Navigation & persistence ──────────────────────────────────────────────

    def _load(self):
        path = self.paths[self.idx]
        img  = cv2.imread(path)
        if img is None:
            print(f"[warn] impossible de lire {path}")
            self.img_orig = np.zeros((600, 1000, 3), dtype=np.uint8)
        else:
            self.img_orig = img
        self.img_h, self.img_w = self.img_orig.shape[:2]
        self.boxes       = []
        self.sel         = -1
        self.click_state = 0
        self.p1 = self.p2 = None
        self._reload_labels(path)

    def _reload_labels(self, img_path: str):
        fname    = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(LBL_OUT, fname + ".txt")
        if not os.path.isfile(lbl_path):
            return
        W, H = self.img_w, self.img_h
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 9:
                    pts = np.array([float(v) for v in parts[1:]], dtype=np.float32).reshape(4, 2)
                    pts[:, 0] *= W;  pts[:, 1] *= H
                    rect  = cv2.minAreaRect(pts)
                    cx, cy = int(rect[0][0]), int(rect[0][1])
                    bw, bh = int(rect[1][0]), int(rect[1][1])
                    angle  = round(float(rect[2]), 1)
                    self.boxes.append([cx, cy, bw, bh, angle])
        if self.boxes:
            self.sel = len(self.boxes) - 1

    def _save(self):
        path  = self.paths[self.idx]
        fname = os.path.splitext(os.path.basename(path))[0]
        dest  = os.path.join(IMG_OUT, os.path.basename(path))
        if not os.path.isfile(dest):
            shutil.copy2(path, dest)
        W, H = self.img_w, self.img_h
        lbl_path = os.path.join(LBL_OUT, fname + ".txt")
        with open(lbl_path, "w") as f:
            for box in self.boxes:
                cx, cy, bw, bh, angle = box
                rect = ((float(cx), float(cy)), (float(bw), float(bh)), float(angle))
                pts  = cv2.boxPoints(rect)
                pts_n = (pts / [W, H]).flatten().tolist()
                f.write("0 " + " ".join(f"{v:.6f}" for v in pts_n) + "\n")
        print(f"[saved] {lbl_path}  ({len(self.boxes)} boîte(s))")

    def _next(self):
        if self.idx < len(self.paths) - 1:
            self.idx += 1;  self._load()

    def _prev(self):
        if self.idx > 0:
            self.idx -= 1;  self._load()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        self._load()
        while True:
            cv2.imshow("Annotate", self._render())

            raw = cv2.waitKey(30)
            if raw == -1:
                continue

            # Platform-agnostic arrow key detection (macOS returns large ints)
            LEFT  = raw in (81, 2424832, 63234)
            RIGHT = raw in (83, 2555904, 63235)
            UP    = raw in (82, 2490368, 63232)
            DOWN  = raw in (84, 2621440, 63233)
            key   = raw & 0xFF

            if key in (ord('q'), ord('Q'), 27):         # Q / Esc
                break

            elif key in (13, ord('s'), ord('S')):        # Entrée / S → save+next
                self._save();  self._next()

            elif key in (ord('n'), ord('N')) or RIGHT:   # N / → → next
                self._next()

            elif key in (ord('p'), ord('P')) or LEFT:    # P / ← → prev
                self._prev()

            elif key in (ord('a'), ord('A')):            # A → rotate −5°
                if self.boxes and 0 <= self.sel < len(self.boxes):
                    self.boxes[self.sel][4] = round(
                        (self.boxes[self.sel][4] - 5) % 360, 1)

            elif key in (ord('d'), ord('D')):            # D → rotate +5°
                if self.boxes and 0 <= self.sel < len(self.boxes):
                    self.boxes[self.sel][4] = round(
                        (self.boxes[self.sel][4] + 5) % 360, 1)

            elif key in (ord('w'), ord('W')) or UP:      # W / ↑ → boîte précédente
                if self.boxes:
                    self.sel = (self.sel - 1) % len(self.boxes)

            elif key in (ord('x'), ord('X')) or DOWN:    # X / ↓ → boîte suivante
                if self.boxes:
                    self.sel = (self.sel + 1) % len(self.boxes)

            elif key in (8, 127):                        # Backspace/Delete
                if self.click_state > 0:
                    self.click_state = max(0, self.click_state - 1)
                    if self.click_state == 0:
                        self.p1 = self.p2 = None
                elif self.boxes:
                    self.boxes.pop(self.sel if 0 <= self.sel < len(self.boxes) else -1)
                    self.sel = len(self.boxes) - 1

        cv2.destroyAllWindows()
        total = len(glob.glob(os.path.join(LBL_OUT, "*.txt")))
        print(f"\nTerminé — {total} image(s) annotée(s) dans {OUTPUT_DIR}/")
        if total >= 5:
            print("Lance l'entraînement :  python3 train_annotated.py")


# ─────────────────────────────────────────────────────────────────────────────

def collect_images(source: str) -> list[str]:
    if os.path.isfile(source):
        return [source]
    if os.path.isdir(source):
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG"):
            paths.extend(glob.glob(os.path.join(source, ext)))
        return sorted(set(paths))
    return []


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "."
    paths = collect_images(src)
    if not paths:
        print(f"[erreur] Aucune image dans : {src}")
        sys.exit(1)
    print(f"[annotate] {len(paths)} image(s) — 3 clics par boîte")
    Annotator(paths).run()


if __name__ == "__main__":
    main()
