#!/usr/bin/env python3
"""
Live putter tracking via webcam — YOLO mode.

Usage:
    python live_tracking.py              # scan auto + caméra USB en priorité
    python live_tracking.py --camera 1  # forcer un index précis
    python live_tracking.py --list      # lister les caméras disponibles
    python live_tracking.py --video fichier.mp4  # vidéo fichier

Commandes pendant le live :
    Q / Echap  — quitter
    R          — reset du background model
"""
import sys
import os
import argparse
import cv2
import numpy as np
from typing import Optional, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tracking.putter_detector import PutterDetector
from models.shot_data import PutterPosition
from utils.drawing import draw_hud

# Chemin vers le modèle YOLO entraîné
YOLO_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "runs", "detect", "runs", "putter", "putter_detector", "weights", "best.pt"
)

WARMUP_FRAMES = 30
TRAIL_LENGTH  = 60   # frames de trajectoire affichées
BBOX_COLOR    = (255, 80, 0)   # bleu BGR
TRAIL_COLOR   = (255, 80, 200) # violet


def list_cameras(max_index: int = 5) -> list:
    """Scan les indices 0..max_index et retourne les caméras disponibles."""
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

    print("[INFO] Caméras détectées :")
    for c in cams:
        print(f"  index {c['index']} — {c['width']}x{c['height']} @ {c['fps']:.0f} fps")

    chosen = cams[-1]["index"] if len(cams) > 1 else cams[0]["index"]
    print(f"[INFO] Caméra sélectionnée : index {chosen} "
          f"({'USB/externe' if chosen > 0 else 'intégrée'})")
    return chosen


def draw_bbox(frame: np.ndarray, bbox: tuple, conf: float) -> None:
    """Cadre bleu autour de la tête du putter."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_COLOR, 2, cv2.LINE_AA)
    label = f"putter {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), BBOX_COLOR, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


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

    # --- Chargement YOLO --------------------------------------------------
    use_yolo = os.path.exists(YOLO_MODEL_PATH)
    if use_yolo:
        print(f"[INFO] Modèle YOLO chargé : {YOLO_MODEL_PATH}")
    else:
        print(f"[WARN] Modèle YOLO introuvable ({YOLO_MODEL_PATH}), fallback MOG2.")

    detector = PutterDetector(
        target_line_angle=0.0,
        use_yolo=use_yolo,
        yolo_model_path=YOLO_MODEL_PATH if use_yolo else None,
    )

    # --- Warmup -----------------------------------------------------------
    print(f"[INFO] Chauffe ({WARMUP_FRAMES} frames)...")
    warmup_frames = []
    for _ in range(WARMUP_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        warmup_frames.append(frame)
    if warmup_frames:
        detector.initialize_background(warmup_frames)
    print("[INFO] Prêt — Q/Echap : quitter  |  R : reset fond")

    frame_idx = WARMUP_FRAMES
    trail: list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Fin de la source.")
            break

        det = detector.detect(frame, frame_idx)
        frame_idx += 1

        # Trail
        if det.confidence > 0.1:
            trail.append((det.head_x, det.head_y))
        if len(trail) > TRAIL_LENGTH:
            trail.pop(0)

        # Dessin trail
        for i in range(1, len(trail)):
            frac = i / len(trail)
            color = tuple(int(ch * frac) for ch in TRAIL_COLOR)
            cv2.line(
                frame,
                (int(trail[i - 1][0]), int(trail[i - 1][1])),
                (int(trail[i][0]),     int(trail[i][1])),
                color, 2, cv2.LINE_AA,
            )

        # Cadre bleu YOLO ou point de fallback
        if det.bbox is not None and det.confidence > 0.1:
            draw_bbox(frame, det.bbox, det.confidence)
        elif det.confidence > 0.05:
            # Fallback : simple croix si pas de bbox YOLO
            cx, cy = int(det.head_x), int(det.head_y)
            cv2.drawMarker(frame, (cx, cy), BBOX_COLOR,
                           cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)

        # HUD
        metrics = {
            "Méthode":     det.method,
            "Face angle":  f"{det.face_angle:.1f}°" if det.face_angle else "—",
            "Shaft angle": f"{det.shaft_angle:.1f}°" if det.shaft_angle else "—",
        }
        draw_hud(frame, metrics, confidence=det.confidence)

        cv2.imshow("PutterTrack Live  |  Q=quitter  R=reset", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('r'), ord('R')):
            print("[INFO] Reset...")
            detector = PutterDetector(
                target_line_angle=0.0,
                use_yolo=use_yolo,
                yolo_model_path=YOLO_MODEL_PATH if use_yolo else None,
            )
            buf = []
            for _ in range(WARMUP_FRAMES):
                r, f = cap.read()
                if r:
                    buf.append(f)
            if buf:
                detector.initialize_background(buf)
            trail.clear()
            frame_idx = 0
            print("[INFO] Reset terminé.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live putter tracking")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--camera", type=int, default=None,
                       help="Index de la webcam (défaut: auto-détection USB)")
    group.add_argument("--video", type=str,
                       help="Chemin vers une vidéo fichier")
    parser.add_argument("--list", action="store_true",
                        help="Lister les caméras disponibles et quitter")
    args = parser.parse_args()

    if args.list:
        cams = list_cameras()
        if cams:
            print("Caméras disponibles :")
            for c in cams:
                print(f"  --camera {c['index']}  →  {c['width']}x{c['height']} @ {c['fps']:.0f} fps")
        else:
            print("Aucune caméra détectée.")
        sys.exit(0)

    source = args.video if args.video else pick_camera(args.camera)
    run_live(source)
