"""
strobe_capture.py  –  Strobe capture pour données d'entraînement YOLO.

Contrôles:
    R        – recalibrer le fond (mat vide pendant 3s)
    Q / ESC  – quitter

Flux automatique:
  READY (3s calibration fond vide)
    → ARMED  : balle en col 4 D'ABORD (prérequis)
               puis tête putter détectée (YOLO) et immobile 2s → READY
               déclenche sur mouvement du putter → RECORDING
    → RECORDING (2.5s)
    → PREVIEW (3s)
    → ARMED  (retour, fond conservé)

Strobe (7 cols) :
  Cols 1-3, 5-7 : frame où la TÊTE du putter est au centre de la colonne.
  Col 4 (impact) : frame où tête + balle sont visibles ensemble.
  Overlay col 4  : balle initiale (position de repos) cropée en cercle.
  Overlay col 1  : balle post-impact cropée en cercle.
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
CALIB_SECS       = 3.0
RECORD_SECS      = 2.5
PREVIEW_SECS     = 3.0
N_COLS           = 7
OUT_DIR          = "captures"
RAW_DIR          = os.path.join(OUT_DIR, "raw")
MOTION_THRESH    = 25
MOTION_AREA      = 80
SWING_PEAK_DELTA = 5
BALL_CROP_R      = 46

YOLO_MODEL_PATH  = "runs/detect/runs/putter/putter_detector/weights/best.pt"
YOLO_CONF_MIN    = 0.10   # confiance min pour valider une détection tête
YOLO_STILL_PX    = 12     # pixels demi-res : seuil "tête immobile"

# Auto-déclenchement
PUTTER_STABLE_SEC  = 2.0
PUTTER_ABSORB_SEC  = 1.0
BALL_BLOB_PX       = 40

# Colonne post-impact balle (col 1, index 0)
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
    READY     = "READY"
    ARMED     = "ARMED"
    RECORDING = "RECORDING"
    PREVIEW   = "PREVIEW"


# ── YOLO ──────────────────────────────────────────────────────────────────────
def load_yolo():
    try:
        from ultralytics import YOLO
        model = YOLO(YOLO_MODEL_PATH)
        print(f"[yolo] modèle chargé : {YOLO_MODEL_PATH}")
        return model
    except Exception as e:
        print(f"[yolo] ERREUR chargement : {e}  → fallback motion")
        return None


def yolo_detect_head(frame_bgr, model):
    """Retourne (cx, cy, conf) de la tête de putter, ou None."""
    if model is None:
        return None
    results = model(frame_bgr, conf=YOLO_CONF_MIN, verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return None
    boxes = results.boxes
    confs = boxes.conf.cpu().numpy()
    best  = int(confs.argmax())
    if confs[best] < YOLO_CONF_MIN:
        return None
    b  = boxes.xyxy[best].cpu().numpy()
    cx = int((b[0] + b[2]) / 2)
    cy = int((b[1] + b[3]) / 2)
    return (cx, cy, float(confs[best]))


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


def draw_guide_overlay(img, W, H, ball_pos_fullres=None, head_pos_fullres=None):
    cy  = H // 2
    cw  = W // N_COLS

    cv2.line(img, (0, cy), (W, cy), WHITE, 1, cv2.LINE_AA)

    bx = ball_pos_fullres[0] if ball_pos_fullres else cw * 3 + 18
    by = ball_pos_fullres[1] if ball_pos_fullres else cy
    cv2.circle(img, (bx, by), 18, WHITE, 2, cv2.LINE_AA)
    put_text(img, "balle", (bx - 16, by + 32), scale=0.38, color=WHITE)

    px0 = bx + 22
    px1 = cw * 5
    py0 = cy - 55
    py1 = cy + 55
    ov  = img.copy()
    cv2.rectangle(ov, (px0, py0), (px1, py1), PUTTER_CLR, -1)
    cv2.addWeighted(ov, 0.12, img, 0.88, 0, img)
    cv2.rectangle(img, (px0, py0), (px1, py1), PUTTER_CLR, 1)
    put_text(img, "PUTTER", (px0 + 6, py0 - 8), scale=0.40, color=PUTTER_CLR)

    # Croix cyan sur la tête YOLO
    if head_pos_fullres is not None:
        hx, hy = head_pos_fullres
        cv2.drawMarker(img, (hx, hy), ACCENT, cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)


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
    """Colle un crop circulaire de la balle avec dégradé (fondu aux bords)."""
    if kf_halfres is None or pos_halfres is None:
        return
    frame = cv2.resize(kf_halfres, (W, H))
    sx = W / (W // 2);  sy = H / (H // 2)
    cx = int(pos_halfres[0] * sx);  cy = int(pos_halfres[1] * sy)
    r  = BALL_CROP_R

    x0, x1 = max(0, cx - r), min(W, cx + r + 1)
    y0, y1 = max(0, cy - r), min(H, cy + r + 1)

    yy, xx  = np.mgrid[y0:y1, x0:x1]
    dist    = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
    # Dégradé quadratique : plein au centre, fondu à 0 sur le rayon
    alpha   = np.clip(1.0 - (dist / r) ** 2, 0.0, 1.0)[:, :, np.newaxis]

    roi_f = frame[y0:y1, x0:x1].astype(np.float32)
    roi_c = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = (roi_f * alpha + roi_c * (1.0 - alpha)).astype(np.uint8)


def make_training_frame(kf_frames, ball_rest, ball_col_kfs, ball_col_poss,
                        W, H, bg_frame=None):
    """
    Composite final :
    - Strobe 7 colonnes (tête au centre de chaque col ; col 4 = impact balle+tête)
    - Overlay cercle balle initiale en col 4 (position de repos)
    - Overlay cercle balle post-impact en col 1
    """
    canvas = make_strobe(kf_frames, W, H, bg_frame=bg_frame)

    # Balle initiale (col 4 : position de repos avant swing)
    if ball_rest is not None:
        kf_r, pos_r = ball_rest
        _paste_ball_circle(canvas, kf_r, pos_r, W, H)

    # Balle post-impact (col 1)
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
    yolo = load_yolo()

    WIN = "Strobe Capture"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)

    state          = State.READY
    bg_model       = None
    bg_frame_color = None
    calib_start    = time.perf_counter()

    # Strobe : une frame par colonne (tête au centre)
    kf_frames     = [None] * N_COLS
    kf_offs       = [float('inf')] * N_COLS   # dist tête ↔ centre col
    kf_impact     = None
    kf_impact_off = float('inf')

    # Balle
    ball_rest     = None
    ball_col_kfs  = [None] * len(BALL_COLS)
    ball_col_poss = [None] * len(BALL_COLS)
    ball_col_offs = [float('inf')] * len(BALL_COLS)
    kf_rest       = None

    # Suivi directionnel (RECORDING)
    sw_cx_max  = -1
    sw_cx_min  = float('inf')
    sw_peaked  = False
    sw_done    = False
    sw_cx_prev = -1

    # Auto-déclenchement (ARMED)
    is_ready            = False
    absorb_start        = 0.0
    armed_for_swing     = False
    putter_anchor       = None
    putter_stable_since = 0.0
    last_head_pos       = None   # (cx, cy) demi-res, dernière tête YOLO

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

        # ── Fond ─────────────────────────────────────────────────────────────
        if bg_model is None:
            bg_model = gray_h.astype(np.float32)
        if state == State.READY:
            cv2.accumulateWeighted(gray_h.astype(np.float32), bg_model, 0.04)
            bg_frame_color = raw.copy()

        # ── Diff (pour fallback motion en RECORDING) ──────────────────────
        diff    = np.abs(gray_h.astype(np.float32) - bg_model)
        _, th_img = cv2.threshold(diff.astype(np.uint8),
                                  MOTION_THRESH, 255, cv2.THRESH_BINARY)
        th_img  = cv2.dilate(th_img, None, iterations=3)
        cnts, _ = cv2.findContours(th_img, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

        # ── Machine d'états ──────────────────────────────────────────────────
        if state == State.READY:
            if now - calib_start >= CALIB_SECS:
                bg_model = gray_h.astype(np.float32)
                state    = State.ARMED
                is_ready = False
                putter_anchor       = None
                putter_stable_since = 0.0
                ball_rest           = None
                kf_impact           = None
                kf_impact_off       = float('inf')
                last_head_pos       = None
                print("[calib] fond figé → ARMED")

        elif state == State.ARMED:
            kf_rest = half.copy()
            cw_h    = half.shape[1] // N_COLS
            cy_h    = half.shape[0] // 2

            # ── Étape 1 : Balle en col 4 (prérequis absolu) ──────────────
            # Rien ne démarre tant que la balle n'est pas détectée ici.
            ball_x_exp = cw_h * 3 + 15
            ball_y_exp = cy_h
            br = 22
            region = half[max(0, ball_y_exp - br):ball_y_exp + br,
                          max(0, ball_x_exp - br):ball_x_exp + br]
            white_mask   = np.all(region > 200, axis=2)
            n_white      = int(np.sum(white_mask))
            if n_white >= BALL_BLOB_PX:
                # Vérification compacité : les pixels blancs doivent former un
                # blob compact (petit écart-type) – évite les faux positifs
                # (reflets, câbles dispersés, fond clair hors balle)
                wy, wx = np.where(white_mask)
                sx = float(np.std(wx)) if len(wx) > 1 else 0.0
                sy = float(np.std(wy)) if len(wy) > 1 else 0.0
                ball_present = (sx < 14) and (sy < 14)
            else:
                ball_present = False
            if ball_present:
                ball_rest = (kf_rest, (ball_x_exp, ball_y_exp))
            else:
                ball_rest = None
                if not is_ready:
                    putter_anchor       = None
                    putter_stable_since = 0.0

            # ── Étape 2 : Tête putter (seulement si balle confirmée) ─────
            # YOLO en priorité ; si échec → blob de mouvement dans la zone putter.
            putter_found = False
            if ball_present:
                head_det = yolo_detect_head(half, yolo)
                if head_det is None:
                    # Fallback 1 : diff vs fond dans la zone putter (cols 3-6)
                    pz_x0 = cw_h * 3
                    pz_x1 = min(cw_h * 6, half.shape[1])
                    pz_y0 = max(cy_h - 60, 0)
                    pz_y1 = min(cy_h + 60, half.shape[0])
                    diff_pz = diff[pz_y0:pz_y1, pz_x0:pz_x1].astype(np.uint8)
                    _, th_pz = cv2.threshold(diff_pz, MOTION_THRESH, 255,
                                             cv2.THRESH_BINARY)
                    th_pz = cv2.dilate(th_pz, None, iterations=2)
                    pz_cnts, _ = cv2.findContours(th_pz, cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)
                    pz_big = [c for c in pz_cnts if cv2.contourArea(c) >= 80]
                    if pz_big:
                        biggest = max(pz_big, key=cv2.contourArea)
                        Mpz = cv2.moments(biggest)
                        if Mpz["m00"] > 0:
                            fx = int(Mpz["m10"] / Mpz["m00"]) + pz_x0
                            fy = int(Mpz["m01"] / Mpz["m00"]) + pz_y0
                            head_det = (fx, fy, 0.0)   # conf 0 = fallback motion

                if head_det is None:
                    # Fallback 2 : détection directe par brillance (tête métal.
                    # blanc/argent très visible sur le mat sombre)
                    pz_x0b = cw_h * 3
                    pz_x1b = min(cw_h * 7, half.shape[1])
                    pz_y0b = max(cy_h - 60, 0)
                    pz_y1b = min(cy_h + 60, half.shape[0])
                    zone = half[pz_y0b:pz_y1b, pz_x0b:pz_x1b]
                    bright_mask = np.all(zone > 175, axis=2).astype(np.uint8) * 255
                    bright_mask = cv2.dilate(bright_mask, None, iterations=1)
                    b_cnts, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_SIMPLE)
                    ball_x_local = ball_x_exp - pz_x0b
                    b_candidates = []
                    for c in b_cnts:
                        if cv2.contourArea(c) < 60:
                            continue
                        Mb = cv2.moments(c)
                        if Mb["m00"] > 0:
                            bfx = int(Mb["m10"] / Mb["m00"])
                            if abs(bfx - ball_x_local) > 25:
                                b_candidates.append(
                                    (c, bfx + pz_x0b, int(Mb["m01"] / Mb["m00"]) + pz_y0b)
                                )
                    if b_candidates:
                        best_b = max(b_candidates, key=lambda t: cv2.contourArea(t[0]))
                        head_det = (best_b[1], best_b[2], 0.0)  # conf 0 = fallback brillance

                if head_det is not None:
                    pcx, pcy, conf = head_det
                    putter_found  = True
                    last_head_pos = (pcx, pcy)

                    if not is_ready:
                        # Stabilité de la tête YOLO
                        if putter_anchor is None:
                            putter_anchor       = (pcx, pcy)
                            putter_stable_since = now
                        elif abs(pcx - putter_anchor[0]) > YOLO_STILL_PX:
                            putter_anchor       = (pcx, pcy)
                            putter_stable_since = now

                        if now - putter_stable_since >= PUTTER_STABLE_SEC:
                            is_ready        = True
                            absorb_start    = now
                            armed_for_swing = False
                            print(f"[armed] balle ✓ + tête YOLO stable "
                                  f"(conf={conf:.2f}) → absorption fond")

                    elif not armed_for_swing:
                        # Absorption du putter dans le fond
                        cv2.accumulateWeighted(
                            gray_h.astype(np.float32), bg_model, 0.25)
                        if now - absorb_start >= PUTTER_ABSORB_SEC:
                            armed_for_swing = True
                            print("[armed] READY – swing !")
                    else:
                        # Déclenchement swing : tête en mouvement vers col ≥ 4
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
                            print(f"[rec] auto-trigger #{shot_count + 1}")

            if not putter_found and not is_ready:
                putter_anchor       = None
                putter_stable_since = 0.0

        elif state == State.RECORDING:
            cw_h = half.shape[1] // N_COLS
            cy_h = half.shape[0] // 2

            # ── Position tête : YOLO en priorité, motion en fallback ───────
            head_det = yolo_detect_head(half, yolo)
            if head_det is not None:
                cx_h = head_det[0]
            else:
                # Fallback : centroïde du plus grand blob de mouvement
                sig_cnts = [c for c in cnts if cv2.contourArea(c) >= MOTION_AREA]
                cx_h = None
                if sig_cnts:
                    best_c = max(sig_cnts, key=cv2.contourArea)
                    M = cv2.moments(best_c)
                    if M["m00"] > 0:
                        cx_h = int(M["m10"] / M["m00"])

            if cx_h is not None:
                col = min(cx_h // cw_h, N_COLS - 1)

                # Suivi directionnel (pic backswing → downswing)
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

                # ── Capture par colonne : tête la plus près du centre ─────
                if col == 3:
                    # Col 4 (impact) : DERNIÈRE frame où la balle est encore
                    # visible à sa position de repos pendant que le putter est
                    # dans la col 4 → c'est le moment exact du contact face/balle.
                    if ball_rest is not None:
                        bx_e = ball_rest[1][0]
                        by_e = ball_rest[1][1]
                        br_c = 20
                        ball_rgn = half[max(0, by_e - br_c):by_e + br_c,
                                        max(0, bx_e - br_c):bx_e + br_c]
                        ball_visible = (ball_rgn.size > 0 and
                                        int(np.sum(np.all(ball_rgn > 185, axis=2))) >= 12)
                        if ball_visible:
                            # Mise à jour continue → la DERNIÈRE frame valide = impact
                            kf_impact     = half.copy()
                            kf_impact_off = 0.0
                else:
                    # Cols 1-3 et 5-7 : meilleure frame = tête au centre de la col
                    col_center = (col + 0.5) * cw_h
                    off = abs(cx_h - col_center)
                    if off < kf_offs[col]:
                        kf_offs[col]   = off
                        kf_frames[col] = half.copy()

            # ── Balle post-impact : scan luminosité en col 1 ──────────────
            if sw_peaked and ball_rest is not None:
                col0_y0 = max(0, cy_h - 30)
                col0_y1 = min(half.shape[0], cy_h + 30)
                roi = half[col0_y0:col0_y1, 0:cw_h]
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

            # ── Fin d'enregistrement ────────────────────────────────────────
            if now - rec_start >= RECORD_SECS:
                kf_frames[3] = kf_impact
                # Masquer cols 1 et 7 si la tête n'y est pas passée
                if sw_cx_max < 6.5 * cw_h:
                    kf_frames[6] = None
                if sw_cx_min > 0.5 * cw_h:
                    kf_frames[0] = None

                shot_count += 1
                composite = make_training_frame(
                    kf_frames, ball_rest, ball_col_kfs, ball_col_poss,
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
                state               = State.ARMED
                is_ready            = False
                armed_for_swing     = False
                putter_anchor       = None
                putter_stable_since = 0.0
                ball_rest           = None
                kf_impact           = None
                kf_impact_off       = float('inf')

        # ── Affichage ─────────────────────────────────────────────────────────
        if state == State.PREVIEW and composite is not None:
            top_panel = composite
        else:
            live = cv2.resize(raw, (W, H))
            if state in (State.READY, State.ARMED):
                live = (live.astype(np.float32) * 0.75).astype(np.uint8)
            draw_column_grid(live, W, H)

            if state in (State.READY, State.ARMED):
                ball_full = None
                head_full = None
                if state == State.ARMED:
                    sx_d = W / (W // 2)
                    sy_d = H / (H // 2)
                    if ball_rest is not None:
                        ball_full = (int(ball_rest[1][0] * sx_d),
                                     int(ball_rest[1][1] * sy_d))
                    if last_head_pos is not None:
                        head_full = (int(last_head_pos[0] * sx_d),
                                     int(last_head_pos[1] * sy_d))
                draw_guide_overlay(live, W, H,
                                   ball_pos_fullres=ball_full,
                                   head_pos_fullres=head_full)

            # Cercle balle confirmée
            if state == State.ARMED and ball_rest is not None:
                sx_d = W / (W // 2);  sy_d = H / (H // 2)
                pr   = (int(ball_rest[1][0] * sx_d),
                        int(ball_rest[1][1] * sy_d))
                cv2.circle(live, pr, BALL_CROP_R + 4, (0, 0, 0),    3, cv2.LINE_AA)
                cv2.circle(live, pr, BALL_CROP_R + 4, (255, 140, 0), 2, cv2.LINE_AA)

            # Barre stabilité tête putter
            if state == State.ARMED and putter_anchor is not None:
                sx_d = W / (W // 2);  sy_d = H / (H // 2)
                pax  = int(putter_anchor[0] * sx_d)
                pay  = int(putter_anchor[1] * sy_d)
                stab = min(1.0, (now - putter_stable_since) / PUTTER_STABLE_SEC)
                clr  = GREEN if is_ready else ORANGE
                cv2.circle(live, (pax, pay), 14, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(live, (pax, pay), 14, clr,       2, cv2.LINE_AA)
                bw = 60;  bh = 6
                bx0 = pax - bw // 2;  by0 = pay + 20
                cv2.rectangle(live, (bx0, by0), (bx0 + bw, by0 + bh),
                              (60, 60, 60), -1)
                cv2.rectangle(live, (bx0, by0),
                              (bx0 + int(bw * stab), by0 + bh), clr, -1)

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
                bt = f"  Balle ✓  Tête YOLO détectée – stabilité {stab_pct}%  "
            elif ball_rest:
                bc = (80, 50, 0)
                bt = "  Balle ✓  –  posez le putter dans la zone  "
            else:
                bc = DIM_COLOR
                bt = "  Posez la balle (bord gauche col 4) puis le putter  "
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
            bg_model            = None
            state               = State.READY
            calib_start         = now
            ball_rest           = None
            kf_impact           = None
            kf_impact_off       = float('inf')
            is_ready            = False
            putter_anchor       = None
            putter_stable_since = 0.0
            last_head_pos       = None
            print("[bg] reset calibration")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[done] {shot_count} coups sauvegardés dans ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
