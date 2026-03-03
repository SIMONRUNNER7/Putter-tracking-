#!/usr/bin/env python3
"""
Live putter tracking via webcam.

Usage:
    python live_tracking.py              # webcam par défaut (index 0)
    python live_tracking.py --camera 1  # autre caméra
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tracking.putter_detector import PutterDetector
from models.shot_data import PutterPosition, SwingPhase
from utils.drawing import draw_putter, draw_hud

WARMUP_FRAMES = 30
TRAIL_LENGTH = 60  # frames de trajectoire affichées


def run_live(source: int | str) -> None:
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
    group.add_argument("--camera", type=int, default=0,
                       help="Index de la webcam (défaut: 0)")
    group.add_argument("--video", type=str,
                       help="Chemin vers une vidéo fichier")
    args = parser.parse_args()

    source = args.video if args.video else args.camera
    run_live(source)
