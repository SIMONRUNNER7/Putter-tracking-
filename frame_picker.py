#!/usr/bin/env python3
"""
frame_picker.py — Sélection manuelle des meilleures frames par colonne.

Usage:
    python frame_picker.py                  # dernier shot dans captures/raw/
    python frame_picker.py captures/raw/shot_20240305_123456/

Touches:
    1-7       sélectionner la colonne
    ←  / ,    frame précédente
    →  / .    frame suivante
    S         sauvegarder et régénérer le composite
    Q / Esc   quitter
Souris:
    Clic gauche sur un thumbnail → sélectionner cette frame
"""

import cv2
import numpy as np
import os
import sys
import json
import glob
import argparse

N_COLS   = 7
BALL_CROP_R = 46      # rayon crop balle (même valeur que strobe_capture.py)
THUMB_H  = 190        # hauteur zone candidats
STATUS_H = 28
WIN_NAME = "Frame Picker"


# ── Lecture du shot ──────────────────────────────────────────────────────────

def find_latest_shot(base="captures/raw"):
    shots = sorted(glob.glob(os.path.join(base, "shot_*")))
    return shots[-1] if shots else None


def load_shot(shot_dir):
    meta_path = os.path.join(shot_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"[erreur] metadata.json introuvable dans {shot_dir}")
        sys.exit(1)
    with open(meta_path) as f:
        meta = json.load(f)

    candidates = []
    for col_idx in range(N_COLS):
        paths = sorted(glob.glob(
            os.path.join(shot_dir, f"col{col_idx + 1}", "frame_*.jpg")))
        frames = []
        for p in paths:
            img = cv2.imread(p)
            if img is not None:
                frames.append(img)
        candidates.append(frames)

    ball_rest_kf = None
    p = os.path.join(shot_dir, "ball_rest.jpg")
    if os.path.exists(p):
        ball_rest_kf = cv2.imread(p)

    ball_col1_kf = None
    p = os.path.join(shot_dir, "ball_col1.jpg")
    if os.path.exists(p):
        ball_col1_kf = cv2.imread(p)

    return meta, candidates, ball_rest_kf, ball_col1_kf


# ── Composite ────────────────────────────────────────────────────────────────

def _paste_ball(canvas, kf_halfres, pos_halfres, W, H):
    if kf_halfres is None or pos_halfres is None:
        return
    r  = BALL_CROP_R
    kh, kw = kf_halfres.shape[:2]
    frame  = cv2.resize(kf_halfres, (W, H))
    sx = W / kw;  sy = H / kh
    cx = int(pos_halfres[0] * sx)
    cy = int(pos_halfres[1] * sy)
    x0, x1 = max(0, cx - r), min(W, cx + r + 1)
    y0, y1 = max(0, cy - r), min(H, cy + r + 1)
    yy, xx  = np.mgrid[y0:y1, x0:x1]
    dist    = np.sqrt((xx - cx)**2 + (yy - cy)**2).astype(np.float32)
    alpha   = np.clip(1.0 - (dist / r)**2, 0.0, 1.0)[:, :, np.newaxis]
    roi_f   = frame[y0:y1, x0:x1].astype(np.float32)
    roi_c   = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = (roi_f * alpha + roi_c * (1 - alpha)).astype(np.uint8)


def build_composite(candidates, selections, meta,
                    ball_rest_kf, ball_col1_kf, active_col):
    W, H   = meta["W"], meta["H"]
    col_w  = W // N_COLS
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    for col_idx in range(N_COLS):
        x0 = col_idx * col_w
        x1 = W if col_idx == N_COLS - 1 else x0 + col_w
        frames = candidates[col_idx]
        sel    = selections[col_idx]
        if frames and 0 <= sel < len(frames):
            full = cv2.resize(frames[sel], (W, H))
            canvas[:, x0:x1] = full[:, x0:x1]

    # Lignes de grille
    for col_idx in range(1, N_COLS):
        cv2.line(canvas, (col_idx * col_w, 0), (col_idx * col_w, H),
                 (60, 60, 60), 1)
    # Numéros de colonnes
    for col_idx in range(N_COLS):
        cv2.putText(canvas, str(col_idx + 1),
                    (col_idx * col_w + 6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    # Surligner la colonne active
    ax0 = active_col * col_w
    ax1 = W if active_col == N_COLS - 1 else ax0 + col_w
    hl  = canvas.copy()
    cv2.rectangle(hl, (ax0, 0), (ax1, H), (0, 200, 255), -1)
    cv2.addWeighted(canvas, 0.82, hl, 0.18, 0, canvas)
    cv2.rectangle(canvas, (ax0, 0), (ax1, H), (0, 200, 255), 2)

    # Overlay balle repos (col 4)
    _paste_ball(canvas, ball_rest_kf, meta.get("ball_rest_pos"), W, H)

    # Overlay balle post-impact (col 1) si col 1 a une frame
    if candidates[0] and selections[0] < len(candidates[0]):
        _paste_ball(canvas, ball_col1_kf, meta.get("ball_col_pos_0"), W, H)

    return canvas


# ── Bande de thumbnails ───────────────────────────────────────────────────────

def build_strip(candidates, selections, active_col, W):
    frames = candidates[active_col]
    sel    = selections[active_col]

    if not frames:
        strip = np.full((THUMB_H, W, 3), 30, dtype=np.uint8)
        cv2.putText(strip, "Aucun candidat pour cette colonne",
                    (10, THUMB_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
        return strip

    n       = len(frames)
    tw      = min(W // n, 260)        # largeur d'un thumbnail
    total_w = tw * n

    strip = np.full((THUMB_H, max(total_w, W), 3), 20, dtype=np.uint8)

    for i, frame in enumerate(frames):
        th   = cv2.resize(frame, (tw, THUMB_H - 22))
        x0   = i * tw
        strip[0:THUMB_H - 22, x0:x0 + tw] = th

        # Numéro de frame
        cv2.putText(strip, str(i + 1),
                    (x0 + 4, THUMB_H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

        # Bordure verte = sélectionné
        color     = (0, 220, 0)   if i == sel else (50, 50, 50)
        thickness = 3             if i == sel else 1
        cv2.rectangle(strip, (x0, 0), (x0 + tw - 1, THUMB_H - 23),
                      color, thickness)

    return strip[:, :W]


# ── Barre de statut ───────────────────────────────────────────────────────────

def build_status(candidates, selections, active_col, dirty, W):
    n   = len(candidates[active_col])
    sel = selections[active_col] + 1 if n > 0 else 0
    msg = (f"  Col {active_col + 1}  |  Frame {sel}/{n}  |"
           f"  ← / → : nav   1-7 : colonne   S : sauvegarder   Q : quitter"
           + ("   [NON SAUVEGARDÉ]" if dirty else "   [sauvegardé]"))
    bar = np.full((STATUS_H, W, 3), 15, dtype=np.uint8)
    cv2.putText(bar, msg, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (200, 160, 60) if dirty else (140, 200, 140), 1)
    return bar


# ── Rendu complet ─────────────────────────────────────────────────────────────

def render(candidates, selections, meta, ball_rest_kf, ball_col1_kf,
           active_col, dirty):
    W = meta["W"]
    H = meta["H"]
    composite = build_composite(candidates, selections, meta,
                                ball_rest_kf, ball_col1_kf, active_col)
    strip  = cv2.resize(build_strip(candidates, selections, active_col, W),
                        (W, THUMB_H))
    status = build_status(candidates, selections, active_col, dirty, W)
    return np.vstack([composite, strip, status])


# ── Sauvegarde ────────────────────────────────────────────────────────────────

def save(shot_dir, candidates, selections, meta, ball_rest_kf, ball_col1_kf):
    W, H = meta["W"], meta["H"]

    # Mettre à jour le JSON
    meta["selected"] = {str(k): v for k, v in enumerate(selections)}
    with open(os.path.join(shot_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Régénérer le composite SANS surlignage colonne active
    col_w  = W // N_COLS
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for col_idx in range(N_COLS):
        x0 = col_idx * col_w
        x1 = W if col_idx == N_COLS - 1 else x0 + col_w
        frames = candidates[col_idx]
        sel    = selections[col_idx]
        if frames and 0 <= sel < len(frames):
            full = cv2.resize(frames[sel], (W, H))
            canvas[:, x0:x1] = full[:, x0:x1]
    for col_idx in range(1, N_COLS):
        cv2.line(canvas, (col_idx * col_w, 0), (col_idx * col_w, H),
                 (60, 60, 60), 1)
    for col_idx in range(N_COLS):
        cv2.putText(canvas, str(col_idx + 1),
                    (col_idx * col_w + 6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
    _paste_ball(canvas, ball_rest_kf, meta.get("ball_rest_pos"), W, H)
    if candidates[0] and selections[0] < len(candidates[0]):
        _paste_ball(canvas, ball_col1_kf, meta.get("ball_col_pos_0"), W, H)

    comp_path = os.path.join(shot_dir, "composite.jpg")
    cv2.imwrite(comp_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Copier aussi dans captures/ (même nom strobe_TAG.png)
    tag = meta.get("tag", "")
    if tag:
        out_path = os.path.join("captures", f"strobe_{tag}_pick.png")
        cv2.imwrite(out_path, canvas)
        print(f"[save] composite → {out_path}")

    print(f"[save] metadata → {os.path.join(shot_dir, 'metadata.json')}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sélection manuelle des frames")
    parser.add_argument("shot_dir", nargs="?",
                        help="Dossier shot (captures/raw/shot_...)")
    args = parser.parse_args()

    shot_dir = args.shot_dir or find_latest_shot()
    if not shot_dir or not os.path.isdir(shot_dir):
        print("Aucun shot trouvé. Lance d'abord strobe_capture.py.")
        sys.exit(1)

    print(f"[picker] Ouverture : {shot_dir}")
    meta, candidates, ball_rest_kf, ball_col1_kf = load_shot(shot_dir)
    W, H = meta["W"], meta["H"]

    # Initialiser les sélections depuis le JSON (ou défaut = 0)
    stored = meta.get("selected", {})
    selections = []
    for i in range(N_COLS):
        v = stored.get(str(i), 0)
        n = len(candidates[i])
        selections.append(min(v, n - 1) if n > 0 else 0)

    active_col = 3   # col 4 par défaut
    dirty      = False

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, W, H + THUMB_H + STATUS_H)

    # Callback souris : clic dans la bande de thumbnails → sélectionner
    def on_mouse(event, x, y, flags, param):
        nonlocal dirty
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if y < H:
            # Clic dans le composite : changer la colonne active
            col_w  = W // N_COLS
            clicked = min(x // col_w, N_COLS - 1)
            param["active_col"] = clicked
            return
        if y < H + THUMB_H:
            # Clic dans la bande de thumbnails
            frames = candidates[active_col]
            if not frames:
                return
            tw = min(W // len(frames), 260)
            idx = min(x // tw, len(frames) - 1)
            selections[active_col] = idx
            dirty = True

    cb_state = {"active_col": active_col}
    cv2.setMouseCallback(WIN_NAME, on_mouse, cb_state)

    while True:
        active_col = cb_state["active_col"]
        frame = render(candidates, selections, meta,
                       ball_rest_kf, ball_col1_kf, active_col, dirty)
        cv2.imshow(WIN_NAME, frame)
        key = cv2.waitKey(40) & 0xFF

        if key in (ord('q'), 27):           # Q / Esc
            break

        elif key == ord('s'):               # Sauvegarder
            save(shot_dir, candidates, selections, meta,
                 ball_rest_kf, ball_col1_kf)
            dirty = False

        elif key in range(ord('1'), ord('1') + N_COLS):   # 1-7 : changer col
            active_col = key - ord('1')
            cb_state["active_col"] = active_col

        elif key in (81, ord(','), ord('a')):    # ← : frame précédente
            n = len(candidates[active_col])
            if n > 0:
                selections[active_col] = max(0, selections[active_col] - 1)
                dirty = True

        elif key in (83, ord('.'), ord('d')):    # → : frame suivante
            n = len(candidates[active_col])
            if n > 0:
                selections[active_col] = min(n - 1, selections[active_col] + 1)
                dirty = True

    cv2.destroyAllWindows()
    if dirty:
        print("[picker] Modifications non sauvegardées. Lance 'S' avant de quitter.")


if __name__ == "__main__":
    main()
