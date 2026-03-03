#!/usr/bin/env python3
"""
Live putter tracking via webcam.

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
from models.shot_data import PutterPosition, SwingPhase
from utils.drawing import draw_putter, draw_hud

WARMUP_FRAMES = 30
TRAIL_LENGTH = 60  # frames de trajectoire affichées


def list_cameras(max_index: int = 5) -> list:
    """Scan les indices 0..max_index et retourne les caméras disponibles."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            found.append({"index": i, "width": w, "height": h, "fps": fps})
    return found


def pick_camera(forced_index: Optional[int]) -> int:
    """
    Retourne l'index à utiliser.
    - Si forced_index est donné, l'utilise directement.
    - Sinon, préfère la caméra USB (index > 0) si disponible,
      sinon la caméra intégrée (index 0).
    """
    if forced_index is not None:
        return forced_index

    cams = list_cameras()
    if not cams:
        print("[ERREUR] Aucune caméra détectée.")
        sys.exit(1)

    print("[INFO] Caméras détectées :")
    for c in cams:
        print(f"  index {c['index']} — {c['width']}x{c['height']} @ {c['fps']:.0f} fps")

    # Priorité à la dernière caméra détectée (USB branchée après la built-in)
    chosen = cams[-1]["index"] if len(cams) > 1 else cams[0]["index"]
    print(f"[INFO] Caméra sélectionnée : index {chosen} "
          f"({'USB/externe' if chosen > 0 else 'intégrée'})")
    return chosen


def run_live(source: Union[int, str]) -> None:
    if isinstance(source, int):
        # Sur macOS : AVFoundation donne de meilleurs résultats
        cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir la source : {source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[INFO] Source ouverte — FPS : {fps:.1f}")
    print("[INFO] Chauffe du modèle de fond ({} frames)...".format(WARMUP_FRAMES))

    detector = PutterDetector(target_line_angle=0.0)

    # --- Warmup -----------------------------------------------------------
    warmup_frames = []
    for _ in range(WARMUP_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        warmup_frames.append(frame)
    if warmup_frames:
        detector.initialize_background(warmup_frames)
    print("[INFO] Prêt — appuie sur Q ou Echap pour quitter, R pour reset.")

    frame_idx = WARMUP_FRAMES
    trail: list[tuple[float, float]] = []  # historique positions tête

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Fin de la source.")
            break

        # Détection
        det = detector.detect(frame, frame_idx)
        frame_idx += 1

        # Mise à jour de la trajectoire
        if det.confidence > 0.1:
            trail.append((det.head_x, det.head_y))
        if len(trail) > TRAIL_LENGTH:
            trail.pop(0)

        # Construction d'un PutterPosition pour les helpers de dessin
        pos = PutterPosition(
            frame_index=frame_idx,
            timestamp=frame_idx / fps,
            head_x=det.head_x,
            head_y=det.head_y,
            shaft_start=det.shaft_start,
            shaft_end=det.shaft_end,
            face_angle=det.face_angle,
            shaft_angle=det.shaft_angle,
            confidence=det.confidence,
            detection_method=det.method,
        )

        # Dessin de la trajectoire (trail)
        for i in range(1, len(trail)):
            frac = i / len(trail)
            color = tuple(int(ch * frac) for ch in (200, 80, 255))
            cv2.line(
                frame,
                (int(trail[i - 1][0]), int(trail[i - 1][1])),
                (int(trail[i][0]), int(trail[i][1])),
                color, 2, cv2.LINE_AA,
            )

        # Dessin du putter
        if det.confidence > 0.05:
            draw_putter(frame, pos)

        # HUD
        metrics = {
            "Méthode": det.method,
            "Face angle": f"{det.face_angle:.1f}°" if det.face_angle else "—",
            "Shaft angle": f"{det.shaft_angle:.1f}°" if det.shaft_angle else "—",
            "Head X": f"{det.head_x:.0f} px",
            "Head Y": f"{det.head_y:.0f} px",
        }
        draw_hud(frame, metrics, confidence=det.confidence)

        cv2.imshow("PutterTrack Live  |  Q=quitter  R=reset", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # Q ou Echap
            break
        elif key in (ord('r'), ord('R')):
            print("[INFO] Reset du modèle de fond...")
            detector = PutterDetector(target_line_angle=0.0)
            warmup_buf = []
            for _ in range(WARMUP_FRAMES):
                r, f = cap.read()
                if r:
                    warmup_buf.append(f)
            if warmup_buf:
                detector.initialize_background(warmup_buf)
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

    if args.video:
        source = args.video
    else:
        source = pick_camera(args.camera)

    run_live(source)
