#!/usr/bin/env python3
"""
annotate_unified.py – Annotation putter (OBB) + balle (bbox) en une passe
==========================================================================
Usage:
    python3 annotate_unified.py                  # scanne captures/raw/*_training.jpg
    python3 annotate_unified.py <dossier_ou_img>

Deux classes dans la même image :
    Classe 0 = putter  (OBB orientée, 3 clics)
    Classe 1 = balle   (bbox simple, 1 clic ou drag)

Contrôles :
    Tab               → basculer mode BALLE ↔ PUTTER
    --- Mode BALLE (défaut) ---
    Clic gauche       → poser la balle (carré centré)
    Clic + drag       → bbox libre
    Molette           → taille du carré
    --- Mode PUTTER ---
    Clic 1            → bas de la face
    Clic 2            → haut de la face
    Clic 3            → largeur
    Clic droit        → annuler dessin / supprimer OBB la plus proche
    Backspace         → annuler dernier clic OBB
    --- Commun ---
    Entrée / S        → sauvegarder et image suivante
    N / →             → suivant sans sauvegarder
    P / ←             → précédent
    Q / Echap         → quitter

Sortie YOLO dans annotation_output/
    Classe 0 (putter) : 0 x1 y1 x2 y2 x3 y3 x4 y4  (normalisé)
    Classe 1 (balle)  : 1 cx cy w h                  (normalisé)
"""

from __future__ import annotations
import cv2
import numpy as np
import os, sys, glob, shutil, math

OUTPUT_DIR = "annotation_output"
IMG_OUT    = os.path.join(OUTPUT_DIR, "images")
LBL_OUT    = os.path.join(OUTPUT_DIR, "labels")

FONT       = cv2.FONT_HERSHEY_SIMPLEX
BALL_COL   = (0, 200, 255)
PUTT_COL   = (255, 200, 50)
DIM_TEXT   = (110, 110, 110)
WHITE      = (255, 255, 255)
GREEN      = (50, 220, 80)
DEFAULT_SZ = 32


# ── Helpers ───────────────────────────────────────────────────────────────────

def _put(img, text, pos, scale=0.45, color=WHITE):
    cv2.putText(img, text, (pos[0]+1, pos[1]+1),
                FONT, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, FONT, scale, color, 1, cv2.LINE_AA)


def _box_corners(p1, p2, p3x, p3y):
    """3 clics → 4 coins du rectangle orienté."""
    a = np.array(p1, dtype=float)
    b = np.array(p2, dtype=float)
    q = np.array([p3x, p3y], dtype=float)
    axis = b - a
    L = np.linalg.norm(axis)
    if L < 2:
        return None
    u    = axis / L
    perp = np.array([-u[1], u[0]])
    hw   = float(np.dot(q - a, perp))
    c1 = a + perp * hw;  c2 = a - perp * hw
    c3 = b - perp * hw;  c4 = b + perp * hw
    return np.array([c1, c2, c3, c4], dtype=np.float32)


# ── Annotateur ────────────────────────────────────────────────────────────────

class UnifiedAnnotator:

    def __init__(self, paths: list[str]):
        self.paths = paths
        self.idx   = 0

        # Mode : 'ball' ou 'putter'
        self.mode = 'ball'

        # Balle : (bx, by, bw, bh) pixels ou None
        self.ball_bbox: tuple | None = None
        self._box_size = DEFAULT_SZ
        self._drag_start = None
        self._drag_cur   = None
        self._dragging   = False

        # Putter : liste de corner-arrays (4×2 float32), un par boîte
        self.putt_boxes: list[np.ndarray] = []
        self._p1 = self._p2 = None
        self._click_state = 0   # 0=idle 1=got p1 2=got p1+p2

        self.cursor = (0, 0)
        self.img_orig: np.ndarray | None = None
        self.img_h = self.img_w = 0

        os.makedirs(IMG_OUT, exist_ok=True)
        os.makedirs(LBL_OUT, exist_ok=True)

        cv2.namedWindow("Annotate", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Annotate", 1280, 800)
        cv2.setMouseCallback("Annotate", self._mouse)

    # ── Mouse ──────────────────────────────────────────────────────────────

    def _mouse(self, event, x, y, flags, param):
        self.cursor = (x, y)

        if self.mode == 'ball':
            self._mouse_ball(event, x, y, flags)
        else:
            self._mouse_putter(event, x, y, flags)

    def _mouse_ball(self, event, x, y, flags):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag_start = (x, y);  self._drag_cur = (x, y)
            self._dragging   = True
        elif event == cv2.EVENT_MOUSEMOVE and self._dragging:
            self._drag_cur = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self._dragging:
            self._dragging = False
            x0, y0 = self._drag_start
            if abs(x - x0) < 5 and abs(y - y0) < 5:
                h = self._box_size // 2
                self.ball_bbox = (x - h, y - h, self._box_size, self._box_size)
            else:
                bx = min(x0, x); by = min(y0, y)
                self.ball_bbox = (bx, by, abs(x - x0), abs(y - y0))
            self._drag_start = self._drag_cur = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.ball_bbox = None
            self._dragging = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            self._box_size = max(8, min(200, self._box_size + (4 if flags > 0 else -4)))
            if self.ball_bbox:
                cx = self.ball_bbox[0] + self.ball_bbox[2] // 2
                cy = self.ball_bbox[1] + self.ball_bbox[3] // 2
                h  = self._box_size // 2
                self.ball_bbox = (cx - h, cy - h, self._box_size, self._box_size)

    def _mouse_putter(self, event, x, y, flags):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self._click_state == 0:
                self._p1 = (x, y);  self._click_state = 1
            elif self._click_state == 1:
                self._p2 = (x, y);  self._click_state = 2
            elif self._click_state == 2:
                corners = _box_corners(self._p1, self._p2, x, y)
                if corners is not None:
                    self.putt_boxes.append(corners)
                self._click_state = 0;  self._p1 = self._p2 = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self._click_state > 0:
                self._click_state = 0;  self._p1 = self._p2 = None
            elif self.putt_boxes:
                dists   = [math.hypot(c[:, 0].mean() - x, c[:, 1].mean() - y)
                           for c in self.putt_boxes]
                nearest = int(np.argmin(dists))
                if dists[nearest] < 120:
                    self.putt_boxes.pop(nearest)

    # ── Rendu ──────────────────────────────────────────────────────────────

    def _render(self) -> np.ndarray:
        if self.img_orig is None:
            return np.zeros((600, 1000, 3), dtype=np.uint8)
        disp = self.img_orig.copy()
        H, W = disp.shape[:2]

        # — Balle —
        if self._dragging and self._drag_start and self._drag_cur:
            x0, y0 = self._drag_start; x1, y1 = self._drag_cur
            if abs(x1-x0) < 5 and abs(y1-y0) < 5:
                h = self._box_size // 2
                cv2.rectangle(disp, (x1-h, y1-h), (x1+h, y1+h), BALL_COL, 1)
            else:
                cv2.rectangle(disp, (min(x0,x1), min(y0,y1)),
                              (max(x0,x1), max(y0,y1)), BALL_COL, 1)
        elif not self.ball_bbox and self.mode == 'ball':
            cx, cy = self.cursor; h = self._box_size // 2
            cv2.rectangle(disp, (cx-h, cy-h), (cx+h, cy+h), (60, 60, 60), 1)

        if self.ball_bbox:
            bx, by, bw, bh = self.ball_bbox
            bx = max(0, min(W-1, bx)); by = max(0, min(H-1, by))
            bw = max(1, min(W-bx, bw)); bh = max(1, min(H-by, bh))
            cv2.rectangle(disp, (bx, by), (bx+bw, by+bh), (0, 0, 0), 3, cv2.LINE_AA)
            cv2.rectangle(disp, (bx, by), (bx+bw, by+bh), BALL_COL, 2, cv2.LINE_AA)
            cx_b, cy_b = bx+bw//2, by+bh//2
            cv2.circle(disp, (cx_b, cy_b), 4, (0, 0, 0), -1)
            cv2.circle(disp, (cx_b, cy_b), 3, WHITE, -1)
            _put(disp, f"balle {bw}x{bh}",
                 (bx+4, by-8 if by > 20 else by+bh+18), scale=0.4, color=BALL_COL)

        # — Putter : dessin en cours —
        if self._click_state >= 1 and self._p1:
            end = self._p2 if self._click_state == 2 else self.cursor
            cv2.line(disp, self._p1, end, PUTT_COL, 2, cv2.LINE_AA)
            cv2.circle(disp, self._p1, 6, PUTT_COL, -1)
            _put(disp, "1", (self._p1[0]+8, self._p1[1]-8), scale=0.45, color=PUTT_COL)
        if self._click_state == 2 and self._p2:
            cv2.circle(disp, self._p2, 6, PUTT_COL, -1)
            _put(disp, "2", (self._p2[0]+8, self._p2[1]-8), scale=0.45, color=PUTT_COL)
            preview = _box_corners(self._p1, self._p2, *self.cursor)
            if preview is not None:
                cv2.polylines(disp, [preview.astype(np.int32)], True,
                              PUTT_COL, 1, cv2.LINE_AA)

        # — Putter : boîtes validées —
        for corners in self.putt_boxes:
            pts = corners.astype(np.int32)
            cv2.drawContours(disp, [pts], 0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.drawContours(disp, [pts], 0, PUTT_COL, 2, cv2.LINE_AA)
            cx_p = int(corners[:, 0].mean()); cy_p = int(corners[:, 1].mean())
            cv2.circle(disp, (cx_p, cy_p), 4, (0, 0, 0), -1)
            cv2.circle(disp, (cx_p, cy_p), 3, WHITE, -1)

        if len(self.putt_boxes) >= 2:
            centers = np.array([(int(c[:, 0].mean()), int(c[:, 1].mean()))
                                for c in self.putt_boxes], np.int32)
            cv2.polylines(disp, [centers.reshape(-1, 1, 2)],
                          False, (80, 80, 200), 3, cv2.LINE_AA)

        # — HUD —
        fname    = os.path.basename(self.paths[self.idx])
        lbl_done = os.path.isfile(
            os.path.join(LBL_OUT, os.path.splitext(fname)[0] + ".txt"))

        cv2.rectangle(disp, (0, H-64), (W, H), (18, 18, 18), -1)

        mode_txt = (f"  MODE : {'BALLE' if self.mode == 'ball' else 'PUTTER'}"
                    f"   |   balle: {'✓' if self.ball_bbox else '○'}"
                    f"   putter: {len(self.putt_boxes)} boîte(s)")
        mode_col = BALL_COL if self.mode == 'ball' else PUTT_COL
        _put(disp, mode_txt, (8, H-44), scale=0.52, color=mode_col)

        info = (f"  {self.idx+1}/{len(self.paths)}   {fname}"
                f"   {'✓ sauvegardée' if lbl_done else '○'}")
        _put(disp, info, (8, H-22), scale=0.48,
             color=GREEN if lbl_done else WHITE)

        steps_ball = ["Clic: placer balle   Drag: bbox libre   Molette: taille"]
        steps_putt = ["Clic 1: bas face", "Clic 2: haut face", "Clic 3: largeur"]
        if self.mode == 'ball':
            hint_mode = steps_ball[0]
        else:
            hint_mode = steps_putt[min(self._click_state, 2)]
        hint = (f"{hint_mode}   |   Tab: changer mode   "
                f"Entrée/S: sauv+suiv   Clic-droit: effacer   Q: quitter")
        _put(disp, hint, (8, H-4), scale=0.36, color=DIM_TEXT)

        return disp

    # ── I/O ────────────────────────────────────────────────────────────────

    def _load(self):
        path = self.paths[self.idx]
        img  = cv2.imread(path)
        if img is None:
            print(f"[warn] impossible de lire {path}")
            self.img_orig = np.zeros((600, 1000, 3), dtype=np.uint8)
        else:
            self.img_orig = img
        self.img_h, self.img_w = self.img_orig.shape[:2]
        self.ball_bbox   = None
        self.putt_boxes  = []
        self._click_state = 0
        self._p1 = self._p2 = None
        self._drag_start = self._drag_cur = None
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
                if not parts:
                    continue
                cls = parts[0]
                if cls == "1" and len(parts) == 5:
                    _, cx_n, cy_n, bw_n, bh_n = parts
                    bw = int(float(bw_n)*W); bh = int(float(bh_n)*H)
                    bx = int(float(cx_n)*W - bw/2)
                    by = int(float(cy_n)*H - bh/2)
                    self.ball_bbox = (bx, by, bw, bh)
                elif cls == "0" and len(parts) == 9:
                    coords = [float(p) for p in parts[1:]]
                    pts = np.array(coords, dtype=np.float32).reshape(4, 2)
                    pts[:, 0] *= W;  pts[:, 1] *= H
                    self.putt_boxes.append(pts)

    def _save(self):
        if self.ball_bbox is None and not self.putt_boxes:
            print("[skip] aucune annotation")
            return
        path  = self.paths[self.idx]
        fname = os.path.splitext(os.path.basename(path))[0]
        dest  = os.path.join(IMG_OUT, os.path.basename(path))
        if not os.path.isfile(dest):
            shutil.copy2(path, dest)

        W, H  = self.img_w, self.img_h
        lpath = os.path.join(LBL_OUT, fname + ".txt")
        lines = []

        # Putter OBB
        for corners in self.putt_boxes:
            pts_n = (corners / [W, H]).flatten().tolist()
            lines.append("0 " + " ".join(f"{v:.6f}" for v in pts_n))

        # Balle bbox
        if self.ball_bbox:
            bx, by, bw, bh = self.ball_bbox
            bx = max(0, min(W-1, bx)); by = max(0, min(H-1, by))
            bw = max(1, min(W-bx, bw)); bh = max(1, min(H-by, bh))
            cx_n = (bx+bw/2)/W;  cy_n = (by+bh/2)/H
            lines.append(f"1 {cx_n:.6f} {cy_n:.6f} {bw/W:.6f} {bh/H:.6f}")

        with open(lpath, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[saved] {lpath}  ({len(self.putt_boxes)} putter, "
              f"{'balle ✓' if self.ball_bbox else 'pas de balle'})")

    def _next(self):
        if self.idx < len(self.paths) - 1:
            self.idx += 1;  self._load()

    def _prev(self):
        if self.idx > 0:
            self.idx -= 1;  self._load()

    # ── Boucle ─────────────────────────────────────────────────────────────

    def run(self):
        self._load()
        while True:
            cv2.imshow("Annotate", self._render())
            raw = cv2.waitKey(20)
            if raw == -1:
                continue

            LEFT  = raw in (81, 2424832, 63234)
            RIGHT = raw in (83, 2555904, 63235)
            key   = raw & 0xFF

            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (13, ord('s'), ord('S')):
                self._save();  self._next()
            elif key in (ord('n'), ord('N')) or RIGHT:
                self._next()
            elif key in (ord('p'), ord('P')) or LEFT:
                self._prev()
            elif key == 9:   # Tab → basculer mode
                self.mode = 'putter' if self.mode == 'ball' else 'ball'
                self._click_state = 0;  self._p1 = self._p2 = None
            elif key in (8, 127):  # Backspace
                if self.mode == 'putter' and self._click_state > 0:
                    self._click_state -= 1
                    if self._click_state == 0:
                        self._p1 = self._p2 = None
                    elif self._click_state == 1:
                        self._p2 = None
                elif self.mode == 'ball':
                    self.ball_bbox = None

        cv2.destroyAllWindows()
        total = len([f for f in os.listdir(LBL_OUT) if f.endswith(".txt")])
        print(f"\nTerminé — {total} image(s) annotée(s) dans {OUTPUT_DIR}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

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
    if len(sys.argv) > 1:
        paths = collect_images(sys.argv[1])
    else:
        raw_dir = os.path.join("captures", "raw")
        if os.path.isdir(raw_dir):
            paths = sorted(glob.glob(os.path.join(raw_dir, "*_training.jpg")))
        else:
            paths = collect_images("captures")

    if not paths:
        print("[erreur] Aucune image trouvée.")
        print("  Lance d'abord strobe_capture.py pour générer des _training.jpg")
        sys.exit(1)

    print(f"[annotate_unified] {len(paths)} image(s)")
    print("  Tab = basculer BALLE / PUTTER")
    UnifiedAnnotator(paths).run()


if __name__ == "__main__":
    main()
