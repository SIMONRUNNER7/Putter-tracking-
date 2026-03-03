#!/usr/bin/env python3
"""
Live putter tracking — zones d'initialisation.

Usage:
    python live_tracking.py
    python live_tracking.py --camera 1
    python live_tracking.py --video fichier.mp4
    python live_tracking.py --list

Workflow :
  1. Phase SETUP  : place la balle dans la zone BALLE et le putter dans la
                    zone PUTTER (les cadres passent du rouge au vert).
  2. Phase TRACKING : le tracking démarre automatiquement.
  Touches : Q/Echap = quitter  |  R = recommencer
"""
import sys
import os
import argparse
import cv2
import numpy as np
from typing import Optional, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tracking.putter_detector import PutterDetector
from utils.drawing import draw_hud

# --------------------------------------------------------------------------
# Modèle YOLO entraîné
# --------------------------------------------------------------------------
YOLO_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "runs", "detect", "runs", "putter", "putter_detector", "weights", "best.pt"
)

# --------------------------------------------------------------------------
# Paramètres visuels
# --------------------------------------------------------------------------
WARMUP_FRAMES  = 30
TRAIL_LENGTH   = 80

# Couleurs BGR
C_BLUE     = (255, 80,   0)     # cadre putter
C_INACTIVE = (60,  60, 180)     # zone non remplie (rouge-brun)
C_ACTIVE   = (60, 200,  60)     # zone remplie (vert)
C_LINE     = (200, 200, 200)    # ligne horizontale
C_TRAIL    = (255, 100, 200)    # trail violet


# --------------------------------------------------------------------------
# Zones (fraction du cadre)
# --------------------------------------------------------------------------
# Ligne horizontale à 68 % de la hauteur
LINE_Y_FRAC = 0.68

# Zone balle : petite, à droite du centre, sous la ligne
BALL_ZONE = dict(cx=0.60, cy=0.81, hw=0.07, hh=0.09)  # centre + demi-largeur/hauteur

# Zone putter : plus grande, à gauche de la balle, sous la ligne
PUTT_ZONE = dict(cx=0.30, cy=0.81, hw=0.22, hh=0.12)


def zone_rect(frac: dict, w: int, h: int) -> tuple:
    """Retourne (x1, y1, x2, y2) en pixels à partir d'une définition fractionnelle."""
    cx = int(frac["cx"] * w)
    cy = int(frac["cy"] * h)
    hw = int(frac["hw"] * w)
    hh = int(frac["hh"] * h)
    return cx - hw, cy - hh, cx + hw, cy + hh


def rect_contains(rect: tuple, x: float, y: float) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def rect_overlap(r1: tuple, r2: tuple) -> bool:
    """True si les deux rectangles (x1,y1,x2,y2) se chevauchent."""
    return not (r2[0] > r1[2] or r2[2] < r1[0] or r2[1] > r1[3] or r2[3] < r1[1])


# --------------------------------------------------------------------------
# Détection balle de golf (cercle blanc)
# --------------------------------------------------------------------------
def detect_ball_in_zone(frame: np.ndarray, zone: tuple) -> Optional[tuple]:
    """
    Retourne (cx, cy, r) si une balle blanche est trouvée dans la zone,
    sinon None.
    """
    x1, y1, x2, y2 = zone
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Flou pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=20,
        minRadius=5,
        maxRadius=30,
    )

    if circles is None:
        return None

    # Vérifier que le cercle est bien blanc (lumineux)
    circles = np.uint16(np.around(circles[0]))
    for cx, cy, r in circles:
        # Masque circulaire sur le ROI
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (int(cx), int(cy)), int(r), 255, -1)
        mean_val = cv2.mean(gray, mask=mask)[0]
        if mean_val > 160:   # la balle de golf est très blanche
            return (int(cx) + x1, int(cy) + y1, int(r))

    return None


# --------------------------------------------------------------------------
# Dessin des zones et de la ligne
# --------------------------------------------------------------------------
def draw_setup_overlay(
    frame: np.ndarray,
    ball_zone: tuple,
    putt_zone: tuple,
    ball_ok: bool,
    putter_ok: bool,
) -> None:
    h, w = frame.shape[:2]
    line_y = int(LINE_Y_FRAC * h)

    # Ligne horizontale
    cv2.line(frame, (0, line_y), (w, line_y), C_LINE, 2, cv2.LINE_AA)

    alpha = 0.25
    overlay = frame.copy()

    for zone, ok, label in [
        (ball_zone, ball_ok,   "BALLE"),
        (putt_zone, putter_ok, "PUTTER"),
    ]:
        color = C_ACTIVE if ok else C_INACTIVE
        x1, y1, x2, y2 = zone
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # remplissage
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        overlay = frame.copy()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Label centré
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tx = x1 + (x2 - x1 - tw) // 2
        ty = y1 - 8
        cv2.putText(frame, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # Message d'état global
    if ball_ok and putter_ok:
        msg = "PRET — démarrage du tracking"
        color = C_ACTIVE
    else:
        parts = []
        if not ball_ok:
            parts.append("balle")
        if not putter_ok:
            parts.append("putter")
        msg = f"Positionne : {', '.join(parts)}"
        color = C_INACTIVE

    cv2.putText(frame, msg, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def draw_putter_bbox(frame: np.ndarray, bbox: tuple, conf: float) -> None:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), C_BLUE, 2, cv2.LINE_AA)
    label = f"putter {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), C_BLUE, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


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
    print("[INFO] Caméras :")
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
        print(f"[ERREUR] Impossible d'ouvrir la source : {source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Chargement YOLO
    use_yolo = os.path.exists(YOLO_MODEL_PATH)
    if use_yolo:
        print(f"[INFO] YOLO chargé : {YOLO_MODEL_PATH}")
    else:
        print("[WARN] Modèle YOLO introuvable — la zone PUTTER sera moins précise.")

    def make_detector():
        return PutterDetector(
            target_line_angle=0.0,
            use_yolo=use_yolo,
            yolo_model_path=YOLO_MODEL_PATH if use_yolo else None,
        )

    detector = make_detector()

    # Warmup arrière-plan
    print(f"[INFO] Chauffe arrière-plan ({WARMUP_FRAMES} frames)…")
    warmup_buf = []
    for _ in range(WARMUP_FRAMES):
        ret, f = cap.read()
        if ret:
            warmup_buf.append(f)
    if warmup_buf:
        detector.initialize_background(warmup_buf)

    STATE_SETUP    = "setup"
    STATE_TRACKING = "tracking"
    state = STATE_SETUP

    frame_idx = WARMUP_FRAMES
    trail: list = []

    # Compteur de frames consécutives avec balle+putter détectés
    ready_count    = 0
    READY_NEEDED   = 8   # 8 frames consécutives pour démarrer

    print("[INFO] Place la balle et le putter dans les zones.")
    print("[INFO] Q/Echap = quitter  |  R = recommencer")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Fin de la source.")
            break

        h, w = frame.shape[:2]
        ball_rect = zone_rect(BALL_ZONE, w, h)
        putt_rect = zone_rect(PUTT_ZONE, w, h)

        if state == STATE_SETUP:
            # --- Détection balle
            ball_det = detect_ball_in_zone(frame, ball_rect)
            ball_ok  = ball_det is not None

            # --- Détection putter dans sa zone (YOLO ou fallback blob)
            putter_ok = False
            putter_bbox_in_zone = None

            det = detector.detect(frame, frame_idx)

            if det.bbox is not None:
                # YOLO a trouvé quelque chose — est-ce dans la zone putter ?
                if rect_overlap(putt_rect, tuple(int(v) for v in det.bbox)):
                    putter_ok = True
                    putter_bbox_in_zone = det.bbox
            elif det.confidence > 0.15 and rect_contains(putt_rect, det.head_x, det.head_y):
                # fallback : centroid MOG2 dans la zone
                putter_ok = True

            # Dessiner les zones
            draw_setup_overlay(frame, ball_rect, putt_rect, ball_ok, putter_ok)

            # Afficher la balle détectée
            if ball_ok and ball_det:
                cv2.circle(frame, (ball_det[0], ball_det[1]), ball_det[2],
                           C_ACTIVE, 2, cv2.LINE_AA)

            # Afficher le putter détecté
            if putter_ok and putter_bbox_in_zone:
                draw_putter_bbox(frame, putter_bbox_in_zone, det.confidence)

            # Décompte pour lancer le tracking
            if ball_ok and putter_ok:
                ready_count += 1
            else:
                ready_count = 0

            if ready_count >= READY_NEEDED:
                state = STATE_TRACKING
                trail.clear()
                print("[INFO] Tracking démarré !")

        else:  # STATE_TRACKING
            det = detector.detect(frame, frame_idx)

            # Trail
            if det.confidence > 0.1:
                trail.append((det.head_x, det.head_y))
            if len(trail) > TRAIL_LENGTH:
                trail.pop(0)

            # Dessin trail
            for i in range(1, len(trail)):
                frac = i / len(trail)
                color = tuple(int(ch * frac) for ch in C_TRAIL)
                cv2.line(
                    frame,
                    (int(trail[i - 1][0]), int(trail[i - 1][1])),
                    (int(trail[i][0]),     int(trail[i][1])),
                    color, 2, cv2.LINE_AA,
                )

            # Cadre bleu autour de la tête
            if det.bbox is not None and det.confidence > 0.1:
                draw_putter_bbox(frame, det.bbox, det.confidence)
            elif det.confidence > 0.05:
                cx, cy = int(det.head_x), int(det.head_y)
                cv2.drawMarker(frame, (cx, cy), C_BLUE,
                               cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)

            # HUD
            metrics = {
                "Méthode":     det.method,
                "Face angle":  f"{det.face_angle:.1f}°" if det.face_angle else "—",
                "Shaft angle": f"{det.shaft_angle:.1f}°" if det.shaft_angle else "—",
            }
            draw_hud(frame, metrics, confidence=det.confidence)

        frame_idx += 1
        cv2.imshow("PutterTrack  |  Q=quitter  R=reset", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('r'), ord('R')):
            state = STATE_SETUP
            ready_count = 0
            trail.clear()
            frame_idx = 0
            detector = make_detector()
            buf = []
            for _ in range(WARMUP_FRAMES):
                r, f = cap.read()
                if r:
                    buf.append(f)
            if buf:
                detector.initialize_background(buf)
            print("[INFO] Reset — repositionne balle et putter.")

    cap.release()
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live putter tracking avec zones")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--camera", type=int, default=None)
    group.add_argument("--video", type=str)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        cams = list_cameras()
        for c in cams:
            print(f"  --camera {c['index']}  →  {c['width']}x{c['height']} @ {c['fps']:.0f} fps")
        sys.exit(0)

    source = args.video if args.video else pick_camera(args.camera)
    run_live(source)
