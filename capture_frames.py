"""
capture_frames.py  –  Outil de capture de frames réelles pour entraînement YOLO.

Usage:
    python3 capture_frames.py

Contrôles:
    ESPACE      – sauvegarder la frame courante
    F           – geler/dégeler l'image
    Q / ESC     – quitter

Objectif : capturer ~100-200 frames variées avec le putter visible,
sous des angles / éclairages / positions différents.
Les frames sont sauvegardées dans data/raw_real/ prêtes pour le label_tool.

Ensuite :
    python3 training/label_tool.py --images data/raw_real --out data/putter_dataset
    python3 training/train_detector.py --data data/putter_dataset/data.yaml --epochs 50
"""

import cv2
import os
import sys
import time
from datetime import datetime

OUT_DIR    = "data/raw_real"
TARGET_FPS = 240
FONT       = cv2.FONT_HERSHEY_SIMPLEX


def put_text(img, text, pos, scale=0.55, color=(220, 220, 220), bold=False):
    th = 2 if bold else 1
    cv2.putText(img, text, pos, FONT, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, FONT, scale, color,     th,     cv2.LINE_AA)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    backends = []
    if sys.platform == "darwin":
        avf = getattr(cv2, "CAP_AVFOUNDATION", None)
        if avf:
            backends.append(avf)
    backends.append(cv2.CAP_ANY)

    cap = None
    for be in backends:
        cap = cv2.VideoCapture(0, be)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            cap.set(cv2.CAP_PROP_EXPOSURE, -7)
            break

    if cap is None or not cap.isOpened():
        print("Impossible d'ouvrir la caméra.")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[cam] {W}x{H}")

    WIN = "Capture Frames – ESPACE=sauver  Q=quitter"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)

    saved   = 0
    frozen  = False
    frozen_frame = None
    flash_until  = 0.0
    t_last       = time.perf_counter()

    # Compter frames déjà existantes
    existing = len([f for f in os.listdir(OUT_DIR) if f.endswith('.jpg')])
    saved = existing

    print(f"[info] {existing} frames déjà dans {OUT_DIR}/")
    print("[info] Place le putter dans différentes positions et appuie sur ESPACE")

    while True:
        now = time.perf_counter()
        fps = 1.0 / max(1e-6, now - t_last)
        t_last = now

        if not frozen:
            ret, raw = cap.read()
            if not ret:
                break
            display = raw.copy()
        else:
            display = frozen_frame.copy()

        # Flash vert après sauvegarde
        if now < flash_until:
            cv2.rectangle(display, (0, 0), (W, H), (0, 200, 80), 8)

        # Grille de colonnes (7 cols)
        cw = W // 7
        for i in range(1, 7):
            cv2.line(display, (i * cw, 0), (i * cw, H), (55, 55, 55), 1)

        # Bandeau
        cv2.rectangle(display, (0, 0), (W, 36), (0, 0, 0), -1)
        state_txt = "GEL" if frozen else "LIVE"
        clr = (0, 200, 255) if not frozen else (30, 130, 255)
        put_text(display, f"  {state_txt}  |  ESPACE=sauver  F=geler  Q=quitter",
                 (8, 24), scale=0.55, color=(220, 220, 220))
        put_text(display, f"{fps:.0f}fps", (W - 70, 24), scale=0.45, color=(110, 110, 110))

        # Compteur
        put_text(display, f"{saved} frames", (W - 110, H - 12),
                 scale=0.55, color=(0, 200, 255), bold=True)

        # Conseil
        target = 150
        pct = min(1.0, saved / target)
        bar_w = 200
        bx = W // 2 - bar_w // 2
        cv2.rectangle(display, (bx, H - 18), (bx + bar_w, H - 8), (60, 60, 60), -1)
        cv2.rectangle(display, (bx, H - 18), (bx + int(bar_w * pct), H - 8),
                      (0, 200, 80) if pct >= 1.0 else (0, 130, 255), -1)
        put_text(display, f"Objectif: {target} frames", (bx, H - 22),
                 scale=0.38, color=(110, 110, 110))

        cv2.imshow(WIN, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('f'):
            if not frozen:
                ret2, raw2 = cap.read()
                frozen_frame = raw2.copy() if ret2 else raw.copy()
                frozen = True
            else:
                frozen = False
        elif key == ord(' '):
            tag  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = os.path.join(OUT_DIR, f"real_{tag}.jpg")
            frame_to_save = frozen_frame if frozen else raw
            cv2.imwrite(path, frame_to_save, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved    += 1
            flash_until = now + 0.15
            print(f"[save] #{saved:04d}  {path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[done] {saved} frames dans ./{OUT_DIR}/")
    print("\nÉtapes suivantes :")
    print(f"  1. python3 training/label_tool.py --images {OUT_DIR} --out data/putter_dataset")
    print("  2. python3 training/train_detector.py --data data/putter_dataset/data.yaml --epochs 50")


if __name__ == "__main__":
    main()
