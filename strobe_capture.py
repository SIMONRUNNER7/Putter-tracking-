"""
strobe_capture.py  –  Strobe capture pour données d'entraînement YOLO.

Contrôles:
    R        – recalibrer le fond (mat vide pendant 3s)
    Q / ESC  – quitter

Flux automatique:
  READY (3s calibration fond vide)
    → ARMED  : détecte balle + putter immobile 2s → affiche READY
               déclenche sur mouvement DROITE du putter → RECORDING
    → RECORDING (2.5s)
    → PREVIEW (3s)
    → ARMED  (retour, fond conservé)

Strobe putter  : cols 5-7 (backswing), col 4 (adresse = frame calme),
                 cols 2-3 (downswing), col 1 = balle seule post-impact.
                 Cols 1 et 7 masquées si tête n'atteint pas leur centre.
"""

from __future__ import annotations
import cv2
import numpy as np
import os
import sys
import time
from datetime import datetime
from enum import Enum

# ── Constantes ────────────────────────────────────────────────────────────────
TARGET_FPS       = 240
CALIB_SECS       = 3.0    # durée calibration fond vide
RECORD_SECS      = 2.5
PREVIEW_SECS     = 3.0
N_COLS           = 7
OUT_DIR          = "captures"
RAW_DIR          = os.path.join(OUT_DIR, "raw")
MOTION_THRESH    = 25
MOTION_AREA      = 80
SWING_PEAK_DELTA = 5
BALL_CROP_R      = 30

# Auto-déclenchement
PUTTER_STABLE_SEC  = 2.0   # secondes d'immobilité requises
PUTTER_ABSORB_SEC  = 1.0   # secondes d'absorption du putter dans le fond
PUTTER_STILL_PX    = 10    # pixels demi-res : seuil "immobile"
PUTTER_SWING_PX    = 18    # pixels demi-res : déplacement DROITE = swing
PUTTER_MIN_AREA    = 300   # aire min blob putter (demi-res px²)
BALL_BLOB_PX       = 40    # nb pixels min pour confirmer la balle au bon endroit

# Seule colonne où afficher la balle (col 1, index 0-based)
BALL_COLS = (0,)

# ── Palette ────────────────────────────────────────────────────────────────────
DIM_COLOR  = (40, 40, 40)
GRID_COLOR = (55, 55, 55)
TEXT_COLOR = (220, 220, 220)
DIM_TEXT   = (110, 110, 110)
ACCENT     = (0, 200, 255)
WHITE      = (255, 255, 255)
GREEN      = (0, 230, 80)
ORANGE     = (30, 130, 255)
PUTTER_CLR = (100, 180, 255)
FONT       = cv2.FONT_HERSHEY_SIMPLEX


class State(Enum):
    READY     = "READY"      # calibration fond (3s)
    ARMED     = "ARMED"      # attente balle + putter stable
    RECORDING = "RECORDING"
    PREVIEW   = "PREVIEW"


# ── Caméra ────────────────────────────────────────────────────────────────────
def open_camera():
    backends = []
    if sys.platform == "darwin":
        avf = getattr(cv2, "CAP_AVFOUNDATION", None)
        if avf:
            backends.append(avf)
    backends.append(cv2.CAP_ANY)
    for be in backends:
        cap = cv2.VideoCapture(0, be)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            cap.set(cv2.CAP_PROP_EXPOSURE, -7)
            W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"[cam] {W}×{H}  @{fps:.0f}fps")
            return cap, W, H, fps
    raise RuntimeError("Impossible d'ouvrir la caméra.")


# ── Utilitaires dessin ─────────────────────────────────────────────────────────
def put_text(img, text, pos, scale=0.55, color=TEXT_COLOR, thickness=1, bold=False):
    th = thickness + (1 if bold else 0)
    cv2.putText(img, text, pos, FONT, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, FONT, scale, color,     th,     cv2.LINE_AA)


def draw_pill(img, x, y, w, h, color, alpha=0.55):
    ov = img.copy()
    cv2.rectangle(ov, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def draw_column_grid(img, W, H):
    cw = W // N_COLS
    for i in range(1, N_COLS):
        cv2.line(img, (i * cw, 0), (i * cw, H), GRID_COLOR, 1)


def draw_guide_overlay(img, W, H, ball_pos_fullres=None):
    """
    Guidage permanent :
      - ligne horizontale blanche
      - cercle balle au BORD GAUCHE de col 4
      - zone putter collée à droite de la balle
    """
    cy  = H // 2
    cw  = W // N_COLS

    # Ligne de repère
    cv2.line(img, (0, cy), (W, cy), WHITE, 1, cv2.LINE_AA)

    # Cercle balle : centre à cw*3+18 → bord gauche collé au bord de col 4
    bx = ball_pos_fullres[0] if ball_pos_fullres else cw * 3 + 18
    by = ball_pos_fullres[1] if ball_pos_fullres else cy
    cv2.circle(img, (bx, by), 18, WHITE, 2, cv2.LINE_AA)
    put_text(img, "balle", (bx - 16, by + 32), scale=0.38, color=WHITE)

    # Zone putter : collée à droite de la balle, jusqu'à col 5
    px0 = bx + 22
    px1 = cw * 5
    py0 = cy - 55
    py1 = cy + 55
    ov  = img.copy()
    cv2.rectangle(ov, (px0, py0), (px1, py1), PUTTER_CLR, -1)
    cv2.addWeighted(ov, 0.12, img, 0.88, 0, img)
    cv2.rectangle(img, (px0, py0), (px1, py1), PUTTER_CLR, 1)
    put_text(img, "PUTTER", (px0 + 6, py0 - 8), scale=0.40, color=PUTTER_CLR)


# ── Composites ────────────────────────────────────────────────────────────────
def make_strobe(kf_frames, W, H, bg_frame=None):
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    col_w  = W // N_COLS
    bg_full = cv2.resize(bg_frame, (W, H)) if bg_frame is not None else None
    for i, kf in enumerate(kf_frames):
        x0 = i * col_w
        x1 = W if i == N_COLS - 1 else x0 + col_w
        if kf is None:
            canvas[:, x0:x1] = bg_full[:, x0:x1] if bg_full is not None else DIM_COLOR
        else:
            canvas[:, x0:x1] = cv2.resize(kf, (W, H))[:, x0:x1]
    draw_column_grid(canvas, W, H)
    for i in range(N_COLS):
        put_text(canvas, str(i + 1), (i * col_w + 6, 22), scale=0.5, color=DIM_TEXT)
    return canvas


def _paste_ball_circle(canvas, kf_halfres, pos_halfres, W, H):
    if kf_halfres is None or pos_halfres is None:
        return
    frame = cv2.resize(kf_halfres, (W, H))
    sx = W / (W // 2);  sy = H / (H // 2)
    cx = int(pos_halfres[0] * sx);  cy = int(pos_halfres[1] * sy)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), BALL_CROP_R, 255, -1)
    canvas[mask > 0] = frame[mask > 0]


def make_training_frame(kf_frames, ball_col_kfs, ball_col_poss,
                        W, H, bg_frame=None):
    canvas = make_strobe(kf_frames, W, H, bg_frame=bg_frame)
    for kf, pos in zip(ball_col_kfs or [], ball_col_poss or []):
        _paste_ball_circle(canvas, kf, pos, W, H)
    return canvas


# ── Sauvegarde ────────────────────────────────────────────────────────────────
def save_shot(kf_frames, composite, n,
              ball_rest=None, ball_col_kfs=None, ball_col_poss=None,
              W=1280, H=720):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    cv2.imwrite(os.path.join(OUT_DIR, f"strobe_{tag}.png"), composite)

    saved_raw = 0
    for i, kf in enumerate(kf_frames):
        if kf is None:
            continue
        cv2.imwrite(os.path.join(RAW_DIR, f"{tag}_col{i}.jpg"), kf,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_raw += 1

    if ball_rest is not None:
        kf_r, _ = ball_rest
        cv2.imwrite(os.path.join(RAW_DIR, f"{tag}_rest.jpg"),
                    cv2.resize(kf_r, (W, H)), [cv2.IMWRITE_JPEG_QUALITY, 95])

    labels  = ["ball_col3", "ball_col2", "ball_col1"]
    n_snaps = 0
    for i, kf in enumerate(ball_col_kfs or []):
        if kf is None:
            continue
        lbl = labels[i] if i < len(labels) else f"ball_{i}"
        cv2.imwrite(os.path.join(RAW_DIR, f"{tag}_{lbl}.jpg"),
                    cv2.resize(kf, (W, H)), [cv2.IMWRITE_JPEG_QUALITY, 95])
        n_snaps += 1

    cv2.imwrite(os.path.join(RAW_DIR, f"{tag}_training.jpg"), composite,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    comp_path = os.path.join(OUT_DIR, f"strobe_{tag}.png")
    print(f"[save] #{n:03d}  → {comp_path}  ({saved_raw} cols, {n_snaps} snaps balle)")
    return comp_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cap, W, H, cam_fps = open_camera()

    WIN = "Strobe Capture"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)

    state          = State.READY
    bg_model       = None
    bg_frame_color = None
    calib_start    = time.perf_counter()

    # Putter strobe
    kf_frames  = [None] * N_COLS
    kf_offs    = [float('inf')] * N_COLS
    kf_impact     = None           # col 4 : frame où putter est au plus proche de la balle
    kf_impact_off = float('inf')  # offset putter↔balle en x (demi-res)

    # Balle
    ball_rest      = None
    ball_col_kfs   = [None] * len(BALL_COLS)
    ball_col_poss  = [None] * len(BALL_COLS)
    ball_col_offs  = [float('inf')] * len(BALL_COLS)
    kf_rest        = None

    # Suivi directionnel putter (RECORDING)
    sw_cx_max  = -1
    sw_cx_min  = float('inf')
    sw_peaked  = False
    sw_done    = False
    sw_cx_prev = -1

    # Auto-déclenchement (ARMED)
    is_ready           = False   # balle + putter stables depuis PUTTER_STABLE_SEC
    absorb_start       = 0.0     # quand l'absorption du fond a commencé
    armed_for_swing    = False   # fond absorbé, prêt pour le swing
    putter_anchor      = None    # (cx, cy) demi-res quand putter stable
    putter_stable_since = 0.0

    rec_start  = 0.0
    preview_t  = 0.0
    shot_count = 0
    last_path  = ""
    composite  = None
    t_last     = time.perf_counter()

    while True:
        ret, raw = cap.read()
        if not ret:
            break
        now    = time.perf_counter()
        fps    = 1.0 / max(1e-6, now - t_last)
        t_last = now

        half   = cv2.resize(raw, (W // 2, H // 2))
        gray_h = cv2.GaussianBlur(
            cv2.cvtColor(half, cv2.COLOR_BGR2GRAY), (21, 21), 0)

        # ── Fond : mis à jour uniquement pendant la calibration ─────────────
        if bg_model is None:
            bg_model = gray_h.astype(np.float32)
        if state == State.READY:
            cv2.accumulateWeighted(gray_h.astype(np.float32), bg_model, 0.04)
            bg_frame_color = raw.copy()

        # ── Diff depuis le fond figé ────────────────────────────────────────
        diff    = np.abs(gray_h.astype(np.float32) - bg_model)
        _, th_img = cv2.threshold(diff.astype(np.uint8),
                                  MOTION_THRESH, 255, cv2.THRESH_BINARY)
        th_img  = cv2.dilate(th_img, None, iterations=3)
        cnts, _ = cv2.findContours(th_img, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        motion  = any(cv2.contourArea(c) >= MOTION_AREA for c in cnts)

        # ── Machine d'états ────────────────────────────────────────────────
        if state == State.READY:
            # Calibration automatique 3s → ARMED
            if now - calib_start >= CALIB_SECS:
                bg_model = gray_h.astype(np.float32)  # gèle le fond
                state    = State.ARMED
                is_ready = False
                putter_anchor       = None
                putter_stable_since = 0.0
                ball_rest           = None
                kf_impact           = None
                kf_impact_off       = float('inf')
                print("[calib] fond figé → ARMED")

        elif state == State.ARMED:
            kf_rest = half.copy()
            cw_h    = half.shape[1] // N_COLS
            cy_h    = half.shape[0] // 2

            # ── Détection balle : luminosité (balle = rond blanc, canaux > 200) ─
            ball_x_exp = cw_h * 3 + 15   # centre, bord gauche flush col 4
            ball_y_exp = cy_h
            br = 22
            region = half[max(0, ball_y_exp - br):ball_y_exp + br,
                          max(0, ball_x_exp - br):ball_x_exp + br]
            white_mask   = np.all(region > 200, axis=2)
            ball_present = int(np.sum(white_mask)) >= BALL_BLOB_PX
            if ball_present:
                ball_rest = (kf_rest, (ball_x_exp, ball_y_exp))
            else:
                ball_rest = None

            # ── Détection putter dans sa zone ────────────────────────────
            pz_x0 = cw_h * 3 + 11
            pz_x1 = min(cw_h * 6, half.shape[1])  # étendu à col 6 pour capturer le passage col4→5
            pz_y0 = max(cy_h - 55, 0)
            pz_y1 = min(cy_h + 55, half.shape[0])

            diff_pz = diff[pz_y0:pz_y1, pz_x0:pz_x1].astype(np.uint8)
            _, th_pz = cv2.threshold(diff_pz, MOTION_THRESH, 255,
                                     cv2.THRESH_BINARY)
            th_pz = cv2.dilate(th_pz, None, iterations=2)
            pz_cnts, _ = cv2.findContours(th_pz, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            pz_big = [c for c in pz_cnts
                      if cv2.contourArea(c) >= PUTTER_MIN_AREA]

            putter_found = False
            if pz_big:
                biggest = max(pz_big, key=cv2.contourArea)
                Mpz = cv2.moments(biggest)
                if Mpz["m00"] > 0:
                    pcx = int(Mpz["m10"] / Mpz["m00"]) + pz_x0
                    pcy = int(Mpz["m01"] / Mpz["m00"]) + pz_y0
                    putter_found = True

                    if not is_ready:
                        # ── Phase stabilisation du putter ────────────────
                        if putter_anchor is None:
                            putter_anchor       = (pcx, pcy)
                            putter_stable_since = now
                        elif abs(pcx - putter_anchor[0]) > PUTTER_STILL_PX:
                            putter_anchor       = (pcx, pcy)
                            putter_stable_since = now

                        if (now - putter_stable_since >= PUTTER_STABLE_SEC
                                and ball_rest is not None):
                            is_ready        = True
                            absorb_start    = now
                            armed_for_swing = False
                            print("[armed] putter stable → absorption fond")

                    elif not armed_for_swing:
                        # ── Phase absorption : fond absorbe le putter ────
                        # (alpha élevé = absorption rapide)
                        cv2.accumulateWeighted(
                            gray_h.astype(np.float32), bg_model, 0.25)
                        if now - absorb_start >= PUTTER_ABSORB_SEC:
                            armed_for_swing = True
                            print("[armed] READY – swing !")
                    else:
                        # ── Phase swing : putter franchit col4→col5, balle confirmée ──
                        if pcx >= cw_h * 4 and ball_rest is not None:
                            state         = State.RECORDING
                            rec_start     = now
                            kf_frames     = [None] * N_COLS
                            kf_offs       = [float('inf')] * N_COLS
                            kf_impact     = None
                            kf_impact_off = float('inf')
                            ball_col_kfs  = [None] * len(BALL_COLS)
                            ball_col_poss = [None] * len(BALL_COLS)
                            ball_col_offs = [float('inf')] * len(BALL_COLS)
                            sw_cx_max     = -1
                            sw_cx_min     = float('inf')
                            sw_peaked     = False
                            sw_done       = False
                            sw_cx_prev    = -1
                            is_ready      = False
                            armed_for_swing = False
                            putter_anchor = None
                            print(f"[rec] auto-trigger #{shot_count+1}"
                                  f"  balle={'oui' if ball_rest else 'non'}")

            if not putter_found and not is_ready:
                putter_anchor       = None
                putter_stable_since = 0.0

        elif state == State.RECORDING:
            sig_cnts = [c for c in cnts if cv2.contourArea(c) >= MOTION_AREA]
            cw_h     = half.shape[1] // N_COLS

            # — Putter : strobe colonne par colonne (downswing seul) ———————
            if sig_cnts:
                best = max(sig_cnts, key=cv2.contourArea)
                M = cv2.moments(best)
                if M["m00"] > 0:
                    cx_h = int(M["m10"] / M["m00"])
                    col  = min(cx_h // cw_h, N_COLS - 1)
                    off  = abs(cx_h - (col + 0.5) * cw_h)

                    if cx_h > sw_cx_max:
                        sw_cx_max = cx_h
                    if not sw_peaked:
                        if sw_cx_prev >= 0 and (sw_cx_max - cx_h) > SWING_PEAK_DELTA:
                            sw_peaked = True
                            print(f"[swing] pic cx={sw_cx_max}")
                    elif not sw_done:
                        sw_cx_min = min(sw_cx_min, cx_h)
                        if sw_cx_prev >= 0 and (cx_h - sw_cx_prev) > SWING_PEAK_DELTA:
                            sw_done = True
                    sw_cx_prev = cx_h

                    # Capture des colonnes pendant la descente uniquement
                    if sw_peaked and not sw_done:
                        # Col 4 (index 3) = impact : putter cx le plus proche du x balle
                        if ball_rest is not None:
                            impact_off = abs(cx_h - ball_rest[1][0])
                            if impact_off < kf_impact_off:
                                kf_impact_off = impact_off
                                kf_impact     = half.copy()
                        # Autres colonnes
                        if col != 3:
                            if off < kf_offs[col]:
                                kf_offs[col]   = off
                                kf_frames[col] = half.copy()

            # — Balle : scan luminosité dans col 1 après impact ———————————
            # La balle (ronde et blanche) laisse un patch très lumineux.
            if sw_peaked and ball_rest is not None:
                col0_y0 = max(0, cy_h - 30)
                col0_y1 = min(half.shape[0], cy_h + 30)
                col0_x1 = cw_h  # toute la largeur de col 1
                roi = half[col0_y0:col0_y1, 0:col0_x1]
                white_px = np.all(roi > 200, axis=2)
                n_white  = int(np.sum(white_px))
                if n_white >= 20:
                    ys, xs = np.where(white_px)
                    bx_b   = int(np.median(xs))
                    by_b   = int(np.median(ys)) + col0_y0
                    boff   = abs(bx_b - 0.5 * cw_h)
                    if boff < ball_col_offs[0]:
                        ball_col_offs[0] = boff
                        ball_col_kfs[0]  = half.copy()
                        ball_col_poss[0] = (bx_b, by_b)

            # — Fin d'enregistrement ————————————————————————————————————————
            if now - rec_start >= RECORD_SECS:
                # Col 4 = impact (frame où cx putter est le plus proche de la balle)
                kf_frames[3] = kf_impact
                # Cols 1 et 7 : masquées si tête n'a pas atteint leur centre
                if sw_cx_max < 6.5 * cw_h:
                    kf_frames[6] = None
                if sw_cx_min > 0.5 * cw_h:
                    kf_frames[0] = None

                shot_count += 1
                composite = make_training_frame(
                    kf_frames, ball_col_kfs, ball_col_poss,
                    W, H, bg_frame=bg_frame_color)
                last_path = save_shot(
                    kf_frames, composite, shot_count,
                    ball_rest=ball_rest,
                    ball_col_kfs=ball_col_kfs,
                    ball_col_poss=ball_col_poss, W=W, H=H)
                state     = State.PREVIEW
                preview_t = now

        elif state == State.PREVIEW:
            if now - preview_t >= PREVIEW_SECS:
                # Retour à ARMED (fond conservé, re-détecte balle + putter)
                state               = State.ARMED
                is_ready            = False
                armed_for_swing     = False
                putter_anchor       = None
                putter_stable_since = 0.0
                ball_rest           = None
                kf_impact           = None
                kf_impact_off       = float('inf')

        # ── Affichage ─────────────────────────────────────────────────────
        if state == State.PREVIEW and composite is not None:
            top_panel = composite
        else:
            live = cv2.resize(raw, (W, H))
            if state in (State.READY, State.ARMED):
                live = (live.astype(np.float32) * 0.75).astype(np.uint8)
            draw_column_grid(live, W, H)

            if state in (State.READY, State.ARMED):
                ball_full = None
                if state == State.ARMED and ball_rest is not None:
                    sx_d = W / (W // 2)
                    sy_d = H / (H // 2)
                    ball_full = (int(ball_rest[1][0] * sx_d),
                                 int(ball_rest[1][1] * sy_d))
                draw_guide_overlay(live, W, H, ball_pos_fullres=ball_full)

            # Cercle vert = balle confirmée
            if state == State.ARMED and ball_rest is not None:
                sx_d = W / (W // 2);  sy_d = H / (H // 2)
                pr   = (int(ball_rest[1][0] * sx_d),
                        int(ball_rest[1][1] * sy_d))
                cv2.circle(live, pr, BALL_CROP_R + 4, (0, 0, 0),       3, cv2.LINE_AA)
                cv2.circle(live, pr, BALL_CROP_R + 4, (255, 140, 0), 2, cv2.LINE_AA)  # bleu

            # Visualisation zone putter (barre de stabilité)
            if state == State.ARMED and putter_anchor is not None:
                sx_d = W / (W // 2);  sy_d = H / (H // 2)
                pax  = int(putter_anchor[0] * sx_d)
                pay  = int(putter_anchor[1] * sy_d)
                stab = min(1.0, (now - putter_stable_since) / PUTTER_STABLE_SEC)
                clr  = GREEN if is_ready else ORANGE
                cv2.circle(live, (pax, pay), 14, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(live, (pax, pay), 14, clr,       2, cv2.LINE_AA)
                # Petite barre de progression stabilité
                bw = 60
                bh = 6
                bx0 = pax - bw // 2;  by0 = pay + 20
                cv2.rectangle(live, (bx0, by0), (bx0 + bw, by0 + bh),
                              (60, 60, 60), -1)
                cv2.rectangle(live, (bx0, by0),
                              (bx0 + int(bw * stab), by0 + bh), clr, -1)

            # READY petit, blanc, coin bas-droit
            if state == State.ARMED and is_ready:
                put_text(live, "READY", (W - 88, H - 14),
                         scale=0.6, color=WHITE, bold=True)

            top_panel = live

        # Bandeau d'état
        if state == State.READY:
            rem = max(0.0, CALIB_SECS - (now - calib_start))
            bc  = (0, 100, 160)
            bt  = f"  Calibration…  {rem:.1f}s  (mat vide)   R=reset  Q=quitter  "
        elif state == State.ARMED:
            if is_ready:
                bc = (0, 140, 40)
                bt = "  READY – frappez !  "
            elif ball_rest and putter_anchor:
                stab_pct = int(min(1.0, (now - putter_stable_since)
                                   / PUTTER_STABLE_SEC) * 100)
                bc = (80, 80, 0)
                bt = f"  Posez le putter… stabilité {stab_pct}%  "
            elif ball_rest:
                bc = (80, 50, 0)
                bt = "  Balle ✓  –  posez le putter dans la zone  "
            else:
                bc = DIM_COLOR
                bt = "  Posez la balle (bord gauche col 4) et le putter  "
        elif state == State.RECORDING:
            rem  = max(0.0, RECORD_SECS - (now - rec_start))
            prog = 1.0 - rem / RECORD_SECS
            bc   = (0, 100, 200)
            bt   = f"  ENREGISTREMENT  {rem:.1f}s  "
            cv2.rectangle(top_panel, (0, H - 5), (int(W * prog), H),
                          (0, 160, 255), -1)
        else:  # PREVIEW
            rem_p = max(0.0, PREVIEW_SECS - (now - preview_t))
            bc    = (30, 110, 30)
            bt    = f"  SAUVEGARDÉ  –  {rem_p:.1f}s  "

        draw_pill(top_panel, 0, 0, W, 32, bc, alpha=0.7)
        put_text(top_panel, bt, (8, 22), scale=0.55, color=WHITE, bold=True)
        put_text(top_panel, f"#{shot_count:03d}", (W - 72, 22), scale=0.55, color=ACCENT)
        put_text(top_panel, f"{fps:.0f}fps",      (W - 72, 46), scale=0.45, color=DIM_TEXT)
        if last_path:
            put_text(top_panel, os.path.basename(last_path),
                     (8, H - 10), scale=0.40, color=DIM_TEXT)

        cv2.imshow(WIN, top_panel)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            bg_model    = None
            state       = State.READY
            calib_start = now
            ball_rest     = None
            kf_impact     = None
            kf_impact_off = float('inf')
            is_ready      = False
            putter_anchor       = None
            putter_stable_since = 0.0
            print("[bg] reset calibration")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[done] {shot_count} coups sauvegardés dans ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
