#!/usr/bin/env python3
"""
Live putter tracking — Runner Arc Analysis.

Usage:
    python live_tracking.py
    python live_tracking.py --camera 1
    python live_tracking.py --video fichier.mp4
    python live_tracking.py --list

Touches : Q/Echap = quitter  |  R = recommencer
"""
import sys
import os
import argparse
import math
import cv2
import numpy as np
from typing import Optional, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tracking.putter_detector import PutterDetector

# --------------------------------------------------------------------------
# Modèle YOLO
# --------------------------------------------------------------------------
YOLO_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "runs", "detect", "runs", "putter", "putter_detector", "weights", "best.pt"
)

# --------------------------------------------------------------------------
# Paramètres
# --------------------------------------------------------------------------
WARMUP_FRAMES = 30
TOTAL_PUTTS   = 3
TRAIL_LENGTH  = 60

# Layout (pixels)
HEADER_H = 62
FOOTER_H = 58

# Ligne cible à 52 % de la hauteur totale du cadre
LINE_Y_FRAC = 0.52

# Colors (BGR)
C_RED   = (0,  30, 215)
C_WHITE = (255, 255, 255)
C_GRAY  = (170, 170, 170)
C_DARK  = (15,  15,  15)
C_TRAIL = (60,  60, 200)


# --------------------------------------------------------------------------
# Compteur de putts
# --------------------------------------------------------------------------
class PuttCounter:
    """Détecte les allers-retours du putter et compte les putts."""

    FWD_THRESH  = 45   # px de déplacement pour considérer le swing lancé
    BACK_THRESH = 22   # px pour considérer le retour terminé

    def __init__(self, total: int = TOTAL_PUTTS):
        self.total    = total
        self.count    = 0
        self._start_x: Optional[float] = None
        self._swinging = False

    def update(self, head_x: Optional[float]) -> bool:
        """Retourne True si un nouveau putt vient d'être comptabilisé."""
        if head_x is None or self.count >= self.total:
            return False
        if self._start_x is None:
            self._start_x = head_x
            return False
        dist = abs(head_x - self._start_x)
        if not self._swinging and dist > self.FWD_THRESH:
            self._swinging = True
        elif self._swinging and dist < self.BACK_THRESH:
            self._swinging = False
            self._start_x  = head_x
            self.count    += 1
            return True
        return False

    def reset(self):
        self.count     = 0
        self._start_x  = None
        self._swinging = False

    @property
    def done(self) -> bool:
        return self.count >= self.total


# --------------------------------------------------------------------------
# Dessin — Header
# --------------------------------------------------------------------------
def draw_header(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]

    # Fond sombre semi-transparent
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, HEADER_H), C_DARK, -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    # < Back
    cv2.putText(frame, "< Back", (18, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_WHITE, 1, cv2.LINE_AA)

    # Titre centré : "RUNNER" (blanc gras) + "  ARC ANALYSIS" (rouge)
    font_r = cv2.FONT_HERSHEY_DUPLEX
    font_a = cv2.FONT_HERSHEY_SIMPLEX
    (rw, _), _ = cv2.getTextSize("RUNNER",        font_r, 0.95, 2)
    (aw, _), _ = cv2.getTextSize("  ARC ANALYSIS", font_a, 0.70, 1)
    title_x = (w - rw - aw) // 2
    cv2.putText(frame, "RUNNER",         (title_x,       42), font_r, 0.95, C_WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, "  ARC ANALYSIS", (title_x + rw,  42), font_a, 0.70, C_RED,   1, cv2.LINE_AA)

    # Icône engrenage
    gx, gy = w - 36, HEADER_H // 2
    cv2.circle(frame, (gx, gy), 13, C_WHITE, 1, cv2.LINE_AA)
    cv2.circle(frame, (gx, gy),  5, C_WHITE, 1, cv2.LINE_AA)
    for ang in range(0, 360, 45):
        r = math.radians(ang)
        cv2.line(frame,
                 (int(gx + 9  * math.cos(r)), int(gy + 9  * math.sin(r))),
                 (int(gx + 14 * math.cos(r)), int(gy + 14 * math.sin(r))),
                 C_WHITE, 2, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Dessin — Footer
# --------------------------------------------------------------------------
def draw_footer(frame: np.ndarray, putt_count: int) -> None:
    h, w = frame.shape[:2]

    # Fond sombre semi-transparent
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - FOOTER_H), (w, h), C_DARK, -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    font = cv2.FONT_HERSHEY_DUPLEX
    msg  = "PUTT 3 TIMES NATURALLY"
    (tw, th), _ = cv2.getTextSize(msg, font, 0.72, 2)
    base_y = h - FOOTER_H + (FOOTER_H + th) // 2

    # Message centré
    cv2.putText(frame, msg, ((w - tw) // 2, base_y),
                font, 0.72, C_WHITE, 2, cv2.LINE_AA)

    # Compteur X / 3  (aligné à droite)
    s_n = str(putt_count)
    s_s = "/"
    s_t = str(TOTAL_PUTTS)
    (nw, _), _ = cv2.getTextSize(s_n, font, 1.20, 2)
    (sw, _), _ = cv2.getTextSize(s_s, font, 0.85, 1)
    (tw2,_), _ = cv2.getTextSize(s_t, font, 1.20, 2)
    rx = w - 24 - nw - sw - tw2
    cv2.putText(frame, s_n, (rx,            base_y), font, 1.20, C_WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, s_s, (rx + nw,       base_y - 3), font, 0.85, C_GRAY,  1, cv2.LINE_AA)
    cv2.putText(frame, s_t, (rx + nw + sw,  base_y), font, 1.20, C_WHITE, 2, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Dessin — Ligne cible pointillée
# --------------------------------------------------------------------------
def draw_aim_line(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    y = int(h * LINE_Y_FRAC)

    # Pointillés blancs
    dash, gap, x = 20, 12, 0
    while x < w:
        cv2.line(frame, (x, y), (min(x + dash, w), y), C_WHITE, 1, cv2.LINE_AA)
        x += dash + gap

    # Petite flèche gauche
    cv2.arrowedLine(frame, (60, y), (10, y), C_WHITE, 2, cv2.LINE_AA, tipLength=0.45)


# --------------------------------------------------------------------------
# Dessin — Arcs rouges du swing
# --------------------------------------------------------------------------
def draw_swing_arc(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    cy  = int(h * LINE_Y_FRAC)

    # Amplitude proportionnelle à la zone utile
    content_h = h - HEADER_H - FOOTER_H
    amp = int(content_h * 0.30)

    x0, x1 = int(w * 0.04), int(w * 0.96)
    n = 300

    pts_up, pts_dn = [], []
    for i in range(n):
        t   = i / (n - 1)
        x   = int(x0 + t * (x1 - x0))
        off = int(amp * 4 * t * (1 - t))
        pts_up.append((x, cy - off))
        pts_dn.append((x, cy + off))

    for i in range(1, n):
        cv2.line(frame, pts_up[i - 1], pts_up[i], C_RED, 2, cv2.LINE_AA)
        cv2.line(frame, pts_dn[i - 1], pts_dn[i], C_RED, 2, cv2.LINE_AA)

    # Marques verticales aux extrémités
    tk = 26
    cv2.line(frame, (x0, cy - tk), (x0, cy + tk), C_RED, 2, cv2.LINE_AA)
    cv2.line(frame, (x1, cy - tk), (x1, cy + tk), C_RED, 2, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Dessin — Cadre putter
# --------------------------------------------------------------------------
def draw_putter_box(frame: np.ndarray, bbox: tuple) -> None:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), C_RED, 2, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Caméra
# --------------------------------------------------------------------------
def list_cameras(max_index: int = 5) -> list:
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            found.append({"index": i, "width": w, "height": h, "fps": fps})
    return found


def pick_camera(forced_index: Optional[int]) -> int:
    if forced_index is not None:
        return forced_index
    cams = list_cameras()
    if not cams:
        print("[ERREUR] Aucune caméra détectée.")
        sys.exit(1)
    for c in cams:
        print(f"  index {c['index']} — {c['width']}x{c['height']} @ {c['fps']:.0f} fps")
    chosen = cams[-1]["index"] if len(cams) > 1 else cams[0]["index"]
    print(f"[INFO] Caméra choisie : {chosen}")
    return chosen


# --------------------------------------------------------------------------
# Boucle principale
# --------------------------------------------------------------------------
def run_live(source: Union[int, str]) -> None:
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir : {source}")
        sys.exit(1)

    # Chargement YOLO
    use_yolo = os.path.exists(YOLO_MODEL_PATH)
    if use_yolo:
        print(f"[INFO] YOLO chargé : {YOLO_MODEL_PATH}")
    else:
        print("[WARN] Modèle YOLO introuvable — détection par blob.")

    def make_detector():
        return PutterDetector(
            target_line_angle=0.0,
            use_yolo=use_yolo,
            yolo_model_path=YOLO_MODEL_PATH if use_yolo else None,
        )

    detector = make_detector()
    counter  = PuttCounter()
    trail: list = []

    # Warmup
    print(f"[INFO] Chauffe ({WARMUP_FRAMES} frames)…")
    buf = []
    for _ in range(WARMUP_FRAMES):
        ret, f = cap.read()
        if ret:
            buf.append(f)
    if buf:
        detector.initialize_background(buf)

    frame_idx = WARMUP_FRAMES
    print("[INFO] Tracking démarré — Q/Echap = quitter | R = reset")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Fin de la source.")
            break

        # --- Détection
        det = detector.detect(frame, frame_idx)
        head_x = det.head_x if det.confidence > 0.10 else None

        # --- Mise à jour compteur
        counter.update(head_x)

        # --- Trail
        if head_x is not None:
            trail.append((int(det.head_x), int(det.head_y)))
        if len(trail) > TRAIL_LENGTH:
            trail.pop(0)

        # --- Dessin trail (fondu)
        for i in range(1, len(trail)):
            frac  = i / len(trail)
            color = tuple(int(ch * frac) for ch in C_TRAIL)
            cv2.line(frame, trail[i - 1], trail[i], color, 2, cv2.LINE_AA)

        # --- Arc + ligne cible
        draw_swing_arc(frame)
        draw_aim_line(frame)

        # --- Cadre putter
        if det.bbox is not None and det.confidence > 0.10:
            draw_putter_box(frame, det.bbox)
        elif det.confidence > 0.05:
            cx, cy = int(det.head_x), int(det.head_y)
            cv2.drawMarker(frame, (cx, cy), C_RED,
                           cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)

        # --- Header & Footer
        draw_header(frame)
        draw_footer(frame, counter.count)

        # --- Message fin de session
        if counter.done:
            h, w = frame.shape[:2]
            msg = "Analysis complete!"
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
            cv2.putText(frame, msg, ((w - tw) // 2, h // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, C_WHITE, 2, cv2.LINE_AA)

        frame_idx += 1
        cv2.imshow("Runner Arc Analysis", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('r'), ord('R')):
            counter.reset()
            trail.clear()
            frame_idx = 0
            detector  = make_detector()
            buf = []
            for _ in range(WARMUP_FRAMES):
                r, f = cap.read()
                if r:
                    buf.append(f)
            if buf:
                detector.initialize_background(buf)
            print("[INFO] Reset.")

    cap.release()
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner Arc Analysis")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--camera", type=int, default=None)
    group.add_argument("--video",  type=str)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for c in list_cameras():
            print(f"  --camera {c['index']}  →  {c['width']}x{c['height']} @ {c['fps']:.0f} fps")
        sys.exit(0)

    source = args.video if args.video else pick_camera(args.camera)
    run_live(source)
