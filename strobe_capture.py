"""
strobe_capture.py  –  Strobe capture pour données d'entraînement YOLO.

Contrôles:
    SPACE    – compte à rebours 4s puis armement sur mouvement
    R        – reset du fond de référence
    Q / ESC  – quitter

Flux: READY → COUNTDOWN (4s) → ARMED (attend mouvement) → RECORDING → PREVIEW → READY

Strobe putter  : 7 colonnes, chaque colonne = frame où la tête est au centre
                 de la colonne (col 4 = tête sur la balle = impact exact).
Strobe balle   : colonnes 1, 2, 3 (gauche) — même logique centre de colonne.
                 La balle au repos (col 4) n'est PAS affichée (col 4 = impact pur).

Fichiers générés:
  captures/strobe_YYYYMMDD_HHMMSS.png
  captures/raw/…_col{0-6}.jpg      – keyframes putter
  captures/raw/…_ball_col{1-3}.jpg – keyframes balle
  captures/raw/…_rest.jpg          – balle au repos (référence)
  captures/raw/…_training.jpg      – composite complet
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
TARGET_FPS    = 240
PREP_SECS     = 4.0
RECORD_SECS   = 2.5
PREVIEW_SECS  = 3.0
N_COLS        = 7
OUT_DIR       = "captures"
RAW_DIR       = os.path.join(OUT_DIR, "raw")
MOTION_THRESH = 25
MOTION_AREA   = 80
SWING_PEAK_DELTA = 5
BALL_CROP_R      = 30
# Indices 0-based des colonnes où capturer la balle (cols 1, 2, 3 en 1-indexé)
BALL_COLS = (2, 1, 0)   # col 3 → col 2 → col 1 (de droite à gauche)

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
    COUNTDOWN = "COUNTDOWN"
    ARMED     = "ARMED"
    RECORDING = "RECORDING"
    PREVIEW   = "PREVIEW"


# ── Caméra ────────────────────────────────────────────────────────────────────
def open_camera() -> tuple[cv2.VideoCapture, int, int, float]:
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
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_column_grid(img, W, H):
    cw = W // N_COLS
    for i in range(1, N_COLS):
        cv2.line(img, (i * cw, 0), (i * cw, H), GRID_COLOR, 1)


def draw_guide_overlay(img, W, H, ball_pos_fullres=None):
    """
    Éléments de guidage permanents sur la vue de base :
      - ligne horizontale blanche (repère de hauteur)
      - cercle blanc pour poser la balle (col 4 centre)
      - zone putter (cols 5-7, côté droit)
    """
    cy = H // 2
    cw = W // N_COLS

    # Ligne de repère horizontale
    cv2.line(img, (0, cy), (W, cy), WHITE, 1, cv2.LINE_AA)

    # Cercle blanc : position de la balle
    bx = ball_pos_fullres[0] if ball_pos_fullres else cw * 3 + cw // 2
    by = ball_pos_fullres[1] if ball_pos_fullres else cy
    cv2.circle(img, (bx, by), 18, WHITE, 2, cv2.LINE_AA)
    put_text(img, "balle", (bx + 22, by + 5), scale=0.38, color=WHITE)

    # Zone putter (cols 5-7, côté droit)
    px0 = cw * 4
    px1 = W - 2
    py0 = cy - 55
    py1 = cy + 55
    ov = img.copy()
    cv2.rectangle(ov, (px0, py0), (px1, py1), PUTTER_CLR, -1)
    cv2.addWeighted(ov, 0.12, img, 0.88, 0, img)
    cv2.rectangle(img, (px0, py0), (px1, py1), PUTTER_CLR, 1)
    put_text(img, "PUTTER", (px0 + 6, py0 - 8), scale=0.40, color=PUTTER_CLR)


# ── Composites ────────────────────────────────────────────────────────────────
def make_strobe(kf_frames: list, W: int, H: int, bg_frame=None) -> np.ndarray:
    """
    Panneau strobe 7 colonnes (putter).
    bg_frame : frame couleur pleine-res pour les colonnes sans keyframe
               (jamais de colonne noire).
    """
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    col_w  = W // N_COLS
    bg_full = cv2.resize(bg_frame, (W, H)) if bg_frame is not None else None
    for i, kf in enumerate(kf_frames):
        x0 = i * col_w
        x1 = W if i == N_COLS - 1 else x0 + col_w
        if kf is None:
            if bg_full is not None:
                canvas[:, x0:x1] = bg_full[:, x0:x1]
            else:
                canvas[:, x0:x1] = DIM_COLOR
        else:
            full = cv2.resize(kf, (W, H))
            canvas[:, x0:x1] = full[:, x0:x1]
    draw_column_grid(canvas, W, H)
    for i in range(N_COLS):
        put_text(canvas, str(i + 1), (i * col_w + 6, 22), scale=0.5, color=DIM_TEXT)
    return canvas


def _paste_ball_circle(canvas, kf_halfres, pos_halfres, W, H):
    """Colle un crop circulaire centré sur la balle (pos en demi-res)."""
    if kf_halfres is None or pos_halfres is None:
        return
    frame = cv2.resize(kf_halfres, (W, H))
    sx = W / (W // 2);  sy = H / (H // 2)
    cx = int(pos_halfres[0] * sx);  cy = int(pos_halfres[1] * sy)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), BALL_CROP_R, 255, -1)
    canvas[mask > 0] = frame[mask > 0]


def make_training_frame(kf_frames, ball_col_kfs, ball_col_poss,
                        W: int, H: int, bg_frame=None) -> np.ndarray:
    """
    Image d'entraînement W×H :
      - strobe putter (7 cols, col 4 = impact pur sans balle)
      - crops circulaires balle dans cols 1, 2, 3 uniquement
    Pas de flèche.

    ball_col_kfs  : [kf_col3, kf_col2, kf_col1]  (halfres, ou None)
    ball_col_poss : [(cx,cy)×3]  (halfres)
    """
    canvas = make_strobe(kf_frames, W, H, bg_frame=bg_frame)
    for kf, pos in zip(ball_col_kfs or [], ball_col_poss or []):
        _paste_ball_circle(canvas, kf, pos, W, H)
    return canvas


# ── Sauvegarde ────────────────────────────────────────────────────────────────
def save_shot(kf_frames, composite, n,
              ball_rest=None, ball_col_kfs=None, ball_col_poss=None,
              W=1280, H=720) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    comp_path = os.path.join(OUT_DIR, f"strobe_{tag}.png")
    cv2.imwrite(comp_path, composite)

    saved_raw = 0
    for i, kf in enumerate(kf_frames):
        if kf is None:
            continue
        cv2.imwrite(os.path.join(RAW_DIR, f"{tag}_col{i}.jpg"), kf,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_raw += 1

    # Balle au repos (référence, non affichée sur le strobe)
    if ball_rest is not None:
        kf_r, _ = ball_rest
        cv2.imwrite(os.path.join(RAW_DIR, f"{tag}_rest.jpg"),
                    cv2.resize(kf_r, (W, H)), [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Snaps balle dans cols 3, 2, 1
    labels = ["ball_col3", "ball_col2", "ball_col1"]
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

    print(f"[save] #{n:03d}  → {comp_path}  "
          f"({saved_raw} cols putter, {n_snaps} snaps balle)")
    return comp_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cap, W, H, cam_fps = open_camera()

    WIN = "Strobe Capture"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)

    state          = State.READY
    bg_model       = None
    bg_frame_color = None   # dernière frame couleur pleine-res (fond propre)

    # Putter : keyframes par colonne
    kf_frames  = [None] * N_COLS
    kf_offs    = [float('inf')] * N_COLS

    # Balle : position de repos (ref) + keyframes colonnes 3, 2, 1
    ball_rest      = None               # (kf_halfres, (cx,cy)) depuis ARMED
    ball_col_kfs   = [None] * 3        # [kf_col3, kf_col2, kf_col1]
    ball_col_poss  = [None] * 3        # [(cx,cy), …] halfres
    ball_col_offs  = [float('inf')] * 3

    kf_rest    = None   # dernière frame demi-res calme (ARMED)

    # Suivi directionnel du putter
    sw_cx_max  = -1
    sw_cx_min  = float('inf')
    sw_peaked  = False
    sw_done    = False
    sw_cx_prev = -1

    cd_start   = 0.0
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

        # ── Fond ───────────────────────────────────────────────────────────
        if bg_model is None:
            bg_model = gray_h.astype(np.float32)
        elif state in (State.READY, State.COUNTDOWN):
            cv2.accumulateWeighted(gray_h.astype(np.float32), bg_model, 0.04)
            bg_frame_color = raw.copy()

        # ── Détection de mouvement ──────────────────────────────────────────
        diff = np.abs(gray_h.astype(np.float32) - bg_model)
        _, th_img = cv2.threshold(diff.astype(np.uint8),
                                  MOTION_THRESH, 255, cv2.THRESH_BINARY)
        th_img = cv2.dilate(th_img, None, iterations=3)
        cnts, _ = cv2.findContours(th_img, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        motion = any(cv2.contourArea(c) >= MOTION_AREA for c in cnts)

        # ── Machine d'états ────────────────────────────────────────────────
        if state == State.COUNTDOWN:
            if now - cd_start >= PREP_SECS:
                bg_model = gray_h.astype(np.float32)
                state    = State.ARMED

        elif state == State.ARMED:
            kf_rest = half.copy()

            gray_a  = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
            blur_a  = cv2.GaussianBlur(gray_a, (5, 5), 0)
            circles = cv2.HoughCircles(blur_a, cv2.HOUGH_GRADIENT, dp=1,
                                       minDist=30, param1=50, param2=12,
                                       minRadius=3, maxRadius=18)
            if circles is not None:
                circles = np.round(circles[0]).astype(int)
                Hh, Wh  = half.shape[:2]
                best = max(circles,
                           key=lambda c: int(gray_a[min(c[1], Hh-1),
                                                     min(c[0], Wh-1)]))
                ball_rest = (kf_rest, (int(best[0]), int(best[1])))

            if motion:
                state         = State.RECORDING
                rec_start     = now
                kf_frames     = [None] * N_COLS
                kf_offs       = [float('inf')] * N_COLS
                ball_col_kfs  = [None] * 3
                ball_col_poss = [None] * 3
                ball_col_offs = [float('inf')] * 3
                sw_cx_max     = -1
                sw_cx_min     = float('inf')
                sw_peaked     = False
                sw_done       = False
                sw_cx_prev    = -1
                print(f"[rec] coup #{shot_count + 1} déclenché"
                      f"  balle={'oui' if ball_rest else 'non détectée'}")

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

                    # Col 4 (index 3) = impact : offset par rapport à la balle
                    # Toutes les autres : offset par rapport au centre géométrique
                    if col == 3:
                        ball_rest_x = (ball_rest[1][0] if ball_rest
                                       else half.shape[1] // 2)
                        off = abs(cx_h - ball_rest_x)
                    else:
                        off = abs(cx_h - (col + 0.5) * cw_h)

                    # Suivi directionnel
                    if cx_h > sw_cx_max:
                        sw_cx_max = cx_h
                    if not sw_peaked:
                        if sw_cx_prev >= 0 and (sw_cx_max - cx_h) > SWING_PEAK_DELTA:
                            sw_peaked = True
                            print(f"[swing] pic à cx={sw_cx_max}, downswing")
                    elif not sw_done:
                        sw_cx_min = min(sw_cx_min, cx_h)
                        if sw_cx_prev >= 0 and (cx_h - sw_cx_prev) > SWING_PEAK_DELTA:
                            sw_done = True
                            print(f"[swing] follow-through terminé, cx={cx_h}")
                    sw_cx_prev = cx_h

                    if sw_peaked and not sw_done:
                        if off < kf_offs[col]:
                            kf_offs[col]   = off
                            kf_frames[col] = half.copy()

            # — Balle : même logique centre de colonne, cols 3→2→1 ————————
            # Filtre cône : blob doit être à GAUCHE de la balle au repos (~30°)
            if sw_peaked and sig_cnts and ball_rest is not None:
                rest_x, rest_y = ball_rest[1]
                small = [c for c in sig_cnts if cv2.contourArea(c) < 350]
                cands = small if small else sig_cnts
                for c in sorted(cands, key=cv2.contourArea):
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        bx = int(M["m10"] / M["m00"])
                        by = int(M["m01"] / M["m00"])
                        dx = rest_x - bx       # >0 = à gauche du repos
                        dy = abs(by - rest_y)
                        if dx < 5 or dy > dx * 0.65 + 12:
                            continue
                        # Affectation à la colonne balle correspondante
                        bcol = min(bx // cw_h, N_COLS - 1)
                        for bi, ci in enumerate(BALL_COLS):
                            if bcol == ci:
                                target_bx = (ci + 0.5) * cw_h
                                boff = abs(bx - target_bx)
                                if boff < ball_col_offs[bi]:
                                    ball_col_offs[bi]  = boff
                                    ball_col_kfs[bi]   = half.copy()
                                    ball_col_poss[bi]  = (bx, by)
                                break

            # — Fin d'enregistrement ————————————————————————————————————————
            if now - rec_start >= RECORD_SECS:
                # Cols 1 et 7 masquées si tête n'a pas atteint le centre
                if sw_cx_max < 6.5 * cw_h:
                    kf_frames[6] = None
                    print(f"[swing] col 7 masquée (peak={sw_cx_max:.0f})")
                if sw_cx_min > 0.5 * cw_h:
                    kf_frames[0] = None
                    print(f"[swing] col 1 masquée (min={sw_cx_min:.0f})")

                shot_count += 1
                composite = make_training_frame(
                    kf_frames, ball_col_kfs, ball_col_poss,
                    W, H, bg_frame=bg_frame_color)
                last_path = save_shot(
                    kf_frames, composite, shot_count,
                    ball_rest=ball_rest,
                    ball_col_kfs=ball_col_kfs,
                    ball_col_poss=ball_col_poss,
                    W=W, H=H)
                state     = State.PREVIEW
                preview_t = now

        elif state == State.PREVIEW:
            if now - preview_t >= PREVIEW_SECS:
                state    = State.READY
                bg_model = gray_h.astype(np.float32)

        # ── Affichage ─────────────────────────────────────────────────────
        if state == State.PREVIEW and composite is not None:
            top_panel = composite
        else:
            live = cv2.resize(raw, (W, H))
            if state in (State.READY, State.COUNTDOWN, State.ARMED):
                live = (live.astype(np.float32) * 0.75).astype(np.uint8)
            draw_column_grid(live, W, H)

            # Guide overlay : ligne + cercle balle + zone putter
            if state in (State.READY, State.COUNTDOWN, State.ARMED):
                ball_full = None
                if state == State.ARMED and ball_rest is not None:
                    sx_d = W / (W // 2);  sy_d = H / (H // 2)
                    ball_full = (int(ball_rest[1][0] * sx_d),
                                 int(ball_rest[1][1] * sy_d))
                draw_guide_overlay(live, W, H, ball_pos_fullres=ball_full)

            # ARMED : cercle vert de confirmation sur la balle détectée
            if state == State.ARMED and ball_rest is not None:
                sx_d = W / (W // 2);  sy_d = H / (H // 2)
                pr   = (int(ball_rest[1][0] * sx_d),
                        int(ball_rest[1][1] * sy_d))
                cv2.circle(live, pr, BALL_CROP_R + 4, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(live, pr, BALL_CROP_R + 4, GREEN,      2, cv2.LINE_AA)

            top_panel = live

        # Bandeau d'état
        if state == State.READY:
            bc = DIM_COLOR
            bt = "  SPACE = préparer   R = reset   Q = quitter  "
        elif state == State.COUNTDOWN:
            rem = max(0.0, PREP_SECS - (now - cd_start))
            bc  = (0, 130, 180)
            bt  = f"  Préparez-vous…  {rem:.1f}s  "
            n_s = str(int(rem) + 1)
            (tw, th_), _ = cv2.getTextSize(n_s, FONT, 6.0, 8)
            cv2.putText(top_panel, n_s,
                        ((W - tw) // 2, (H + th_) // 2),
                        FONT, 6.0, (0, 0, 0), 12, cv2.LINE_AA)
            cv2.putText(top_panel, n_s,
                        ((W - tw) // 2, (H + th_) // 2),
                        FONT, 6.0, WHITE, 8, cv2.LINE_AA)
        elif state == State.ARMED:
            bc = (0, 160, 50)
            bt = "  PRÊT – frappez quand vous voulez  "
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
        elif key == ord(' ') and state == State.READY:
            state         = State.COUNTDOWN
            cd_start      = now
            ball_rest     = None
            ball_col_kfs  = [None] * 3
            ball_col_poss = [None] * 3
            ball_col_offs = [float('inf')] * 3
            kf_rest       = None
        elif key == ord('r'):
            bg_model = None
            state    = State.READY
            print("[bg] background reset")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[done] {shot_count} coups sauvegardés dans ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
