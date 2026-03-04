"""
strobe_capture.py  –  Strobe capture pour données d'entraînement YOLO.

Contrôles:
    SPACE    – compte à rebours 4s puis armement sur mouvement
    R        – reset du fond de référence
    Q / ESC  – quitter

Flux: READY → COUNTDOWN (4s) → ARMED (attend mouvement) → RECORDING → PREVIEW → READY

Composite sauvegardé (W × 2H):
  ┌─────────────────────────────┐
  │  Strobe 7 colonnes (putter) │  ← H px
  ├─────────────────────────────┤
  │  Overlay balle (alpha 50%)  │  ← H px (pleine hauteur)
  │  impact──────────►sortie    │
  └─────────────────────────────┘

Fichiers générés:
  captures/strobe_YYYYMMDD_HHMMSS.png   – composite complet
  captures/ball_YYYYMMDD_HHMMSS.png     – overlay balle seul (pleine résolution)
  captures/raw/…_col{0-6}.jpg           – keyframes individuels putter
  captures/raw/…_impact.jpg             – frame à l'impact
  captures/raw/…_exit.jpg               – frame à la sortie (balle)
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
MOTION_AREA   = 80      # aire minimale pour détecter un mouvement
SWING_PEAK_DELTA = 5    # pixels demi-res pour confirmer le changement de direction
BALL_CROP_R      = 30   # rayon du crop circulaire balle (pixels pleine-res)
BALL_SNAP_DISTS  = (28, 62)  # demi-res : distances pour les 2 snaps d'exit après repos

# ── Palette ────────────────────────────────────────────────────────────────────
DIM_COLOR  = (40, 40, 40)
GRID_COLOR = (55, 55, 55)
TEXT_COLOR = (220, 220, 220)
DIM_TEXT   = (110, 110, 110)
ACCENT     = (0, 200, 255)
WHITE      = (255, 255, 255)
GREEN      = (0, 230, 80)
ORANGE     = (30, 130, 255)
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


# ── Composites ────────────────────────────────────────────────────────────────
def make_strobe(kf_frames: list, W: int, H: int) -> np.ndarray:
    """Panneau strobe 7 colonnes (putter)."""
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    col_w  = W // N_COLS
    for i, kf in enumerate(kf_frames):
        x0 = i * col_w
        x1 = W if i == N_COLS - 1 else x0 + col_w
        if kf is None:
            canvas[:, x0:x1] = DIM_COLOR
        else:
            full = cv2.resize(kf, (W, H))
            canvas[:, x0:x1] = full[:, x0:x1]
    draw_column_grid(canvas, W, H)
    for i in range(N_COLS):
        put_text(canvas, str(i + 1), (i * col_w + 6, 22), scale=0.5, color=DIM_TEXT)
    return canvas


def make_ball_panel(kf_impact, kf_exit, pos_impact, pos_exit, W: int, H: int) -> np.ndarray:
    """
    Panneau balle : superposition alpha 50% de la frame impact + frame sortie.
    Dessine l'impact (cercle vert), la sortie (cercle orange) et une flèche.

    kf_impact / kf_exit : frames en demi-résolution (W//2 × H//2)
    pos_impact / pos_exit : (cx, cy) en coordonnées demi-résolution
    Le panneau fait W × (H//2).
    """
    panel_h = H
    # Facteurs de mise à l'échelle demi-res → panneau
    sx = W / (W // 2)          # = 2.0
    sy = panel_h / (H // 2)   # = 2.0

    if kf_impact is None:
        panel = np.zeros((panel_h, W, 3), dtype=np.uint8)
        put_text(panel, "balle : en attente de données…",
                 (12, panel_h // 2 + 6), scale=0.5, color=DIM_TEXT)
        return panel

    fi   = cv2.resize(kf_impact, (W, panel_h))
    fe   = cv2.resize(kf_exit   if kf_exit is not None else kf_impact, (W, panel_h))
    panel = cv2.addWeighted(fi, 0.5, fe, 0.5, 0)

    def to_panel(pos):
        return (int(pos[0] * sx), int(pos[1] * sy))

    pa = to_panel(pos_impact) if pos_impact else None
    pb = to_panel(pos_exit)   if pos_exit   else None

    if pa:
        cv2.circle(panel, pa, 10, GREEN, 2, cv2.LINE_AA)
        put_text(panel, "impact",
                 (pa[0] + 13, pa[1] + 5), scale=0.45, color=GREEN)

    if pb and pa:
        cv2.circle(panel, pb, 10, ORANGE, 2, cv2.LINE_AA)
        put_text(panel, "sortie",
                 (pb[0] + 13, pb[1] + 5), scale=0.45, color=ORANGE)
        cv2.arrowedLine(panel, pa, pb, ACCENT, 2, cv2.LINE_AA, tipLength=0.08)

    # Séparateur en haut du panneau
    cv2.line(panel, (0, 0), (W, 0), GRID_COLOR, 1)
    put_text(panel, "TRAJECTOIRE BALLE",
             (8, 16), scale=0.45, color=DIM_TEXT)

    return panel


def _paste_ball_circle(canvas, kf_halfres, pos_halfres, W, H):
    """Colle un crop CIRCULAIRE serré centré sur la balle (pos en demi-res)."""
    if kf_halfres is None or pos_halfres is None:
        return
    frame = cv2.resize(kf_halfres, (W, H))
    sx = W / (W // 2);  sy = H / (H // 2)
    cx = int(pos_halfres[0] * sx);  cy = int(pos_halfres[1] * sy)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), BALL_CROP_R, 255, -1)
    canvas[mask > 0] = frame[mask > 0]


def make_training_frame(kf_frames, ball_snaps, W: int, H: int) -> np.ndarray:
    """
    Image d'entraînement W×H : strobe putter + 3 crops circulaires de la balle.
    ball_snaps : liste de (kf_halfres, pos_halfres) – [repos, mid, exit]
    Flèche depuis le premier snap (repos) vers le dernier (exit).
    """
    canvas = make_strobe(kf_frames, W, H)

    sx = W / (W // 2);  sy = H / (H // 2)
    for kf, pos in (ball_snaps or []):
        _paste_ball_circle(canvas, kf, pos, W, H)

    if ball_snaps and len(ball_snaps) >= 2:
        p0 = (int(ball_snaps[0][1][0] * sx), int(ball_snaps[0][1][1] * sy))
        p1 = (int(ball_snaps[-1][1][0] * sx), int(ball_snaps[-1][1][1] * sy))
        cv2.arrowedLine(canvas, p0, p1, (0, 0, 0), 4, cv2.LINE_AA, tipLength=0.10)
        cv2.arrowedLine(canvas, p0, p1, ACCENT,    2, cv2.LINE_AA, tipLength=0.10)

    return canvas


def make_composite(kf_frames, W, H,
                   kf_impact=None, kf_exit=None,
                   pos_impact=None, pos_exit=None) -> np.ndarray:
    """Composite complet : strobe putter (H) + overlay balle (H//2)."""
    strobe = make_strobe(kf_frames, W, H)
    ball   = make_ball_panel(kf_impact, kf_exit, pos_impact, pos_exit, W, H)
    return np.vstack([strobe, ball])


# ── Sauvegarde ────────────────────────────────────────────────────────────────
def save_shot(kf_frames, composite, n,
              ball_snaps=None, W=1280, H=720) -> str:
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

    for i, (kf, _) in enumerate(ball_snaps or []):
        names = ["rest", "mid", "exit"]
        label = names[i] if i < len(names) else f"ball{i}"
        cv2.imwrite(os.path.join(RAW_DIR, f"{tag}_{label}.jpg"),
                    cv2.resize(kf, (W, H)), [cv2.IMWRITE_JPEG_QUALITY, 95])

    cv2.imwrite(os.path.join(RAW_DIR, f"{tag}_training.jpg"), composite,
                [cv2.IMWRITE_JPEG_QUALITY, 95])

    n_snaps = len(ball_snaps) if ball_snaps else 0
    print(f"[save] #{n:03d}  → {comp_path}  ({saved_raw} cols putter, {n_snaps} snaps balle)")
    return comp_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cap, W, H, cam_fps = open_camera()

    WIN = "Strobe Capture"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)

    state      = State.READY
    bg_model   = None
    # Putter : keyframes par colonne
    kf_frames  = [None] * N_COLS
    kf_offs    = [float('inf')] * N_COLS
    # Balle : 2 keyframes
    ball_snaps = []     # [(kf_halfres, (cx,cy)), …]  repos / mid / exit
    kf_rest    = None   # dernière frame demi-res calme (ARMED)
    # Suivi directionnel du putter (downswing droite→gauche uniquement)
    sw_cx_max  = -1     # cx le plus à droite vu en demi-res
    sw_peaked  = False  # True une fois passé le pic backswing
    sw_done    = False  # True une fois reparti à droite (follow-through fini)
    sw_cx_prev = -1     # cx précédent putter

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

        # ── Fond : mise à jour uniquement hors enregistrement ──────────────
        if bg_model is None:
            bg_model = gray_h.astype(np.float32)
        elif state in (State.READY, State.COUNTDOWN):
            cv2.accumulateWeighted(gray_h.astype(np.float32), bg_model, 0.04)

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
                bg_model = gray_h.astype(np.float32)  # fige le fond
                state    = State.ARMED

        elif state == State.ARMED:
            # Mémorise la dernière frame calme (balle immobile)
            kf_rest = half.copy()

            # Hough circles : détecte la balle ronde sur le fond sombre
            gray_a  = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
            blur_a  = cv2.GaussianBlur(gray_a, (5, 5), 0)
            circles = cv2.HoughCircles(blur_a, cv2.HOUGH_GRADIENT, dp=1,
                                       minDist=30, param1=50, param2=12,
                                       minRadius=3, maxRadius=18)
            if circles is not None:
                circles = np.round(circles[0]).astype(int)
                Hh, Wh  = half.shape[:2]
                # Balle blanche = centre de cercle le plus brillant
                best = max(circles,
                           key=lambda c: int(gray_a[min(c[1], Hh-1),
                                                     min(c[0], Wh-1)]))
                ball_snaps = [(kf_rest, (int(best[0]), int(best[1])))]

            if motion:
                state      = State.RECORDING
                rec_start  = now
                kf_frames  = [None] * N_COLS
                kf_offs    = [float('inf')] * N_COLS
                sw_cx_max  = -1
                sw_peaked  = False
                sw_done    = False
                sw_cx_prev = -1
                print(f"[rec] coup #{shot_count + 1} déclenché"
                      f"  balle={'oui' if ball_snaps else 'non détectée'}")

        elif state == State.RECORDING:
            sig_cnts = [c for c in cnts if cv2.contourArea(c) >= MOTION_AREA]

            # — Strobe putter : keyframe par colonne (downswing droite→gauche seul) —
            if sig_cnts:
                best = max(sig_cnts, key=cv2.contourArea)
                M = cv2.moments(best)
                if M["m00"] > 0:
                    cx_h = int(M["m10"] / M["m00"])
                    cw_h = half.shape[1] // N_COLS
                    col  = min(cx_h // cw_h, N_COLS - 1)
                    off  = abs(cx_h - (col + 0.5) * cw_h)

                    # Suivi directionnel
                    if cx_h > sw_cx_max:
                        sw_cx_max = cx_h
                    if not sw_peaked:
                        # Pic détecté quand on revient de SWING_PEAK_DELTA px
                        if sw_cx_prev >= 0 and (sw_cx_max - cx_h) > SWING_PEAK_DELTA:
                            sw_peaked = True
                            print(f"[swing] pic à cx={sw_cx_max}, downswing")
                    elif not sw_done:
                        # Fin du downswing : repart à droite
                        if sw_cx_prev >= 0 and (cx_h - sw_cx_prev) > SWING_PEAK_DELTA:
                            sw_done = True
                            print(f"[swing] follow-through terminé, cx={cx_h}")
                    sw_cx_prev = cx_h

                    # N'enregistre que pendant le downswing
                    if sw_peaked and not sw_done:
                        if off < kf_offs[col]:
                            kf_offs[col]   = off
                            kf_frames[col] = half.copy()

            # — Balle : snaps d'exit dans le cône gauche —
            # La balle part toujours vers la GAUCHE (sens du coup).
            # Le putter fait son follow-through côté droit → on filtre.
            if sw_peaked and sig_cnts and len(ball_snaps) < 3:
                pos_ref  = ball_snaps[0][1] if ball_snaps else None
                snap_i   = len(ball_snaps)
                target_d = (BALL_SNAP_DISTS[snap_i - 1]
                            if snap_i > 0 and pos_ref else 0)
                small = [c for c in sig_cnts if cv2.contourArea(c) < 350]
                cands = small if small else sig_cnts
                for c in sorted(cands, key=cv2.contourArea):
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        bx = int(M["m10"] / M["m00"])
                        by = int(M["m01"] / M["m00"])
                        if pos_ref is not None:
                            dx = pos_ref[0] - bx      # >0 = blob est à gauche
                            dy = abs(by - pos_ref[1])
                            # Filtre cône : doit être À GAUCHE et dans ~30°
                            if dx < 5 or dy > dx * 0.65 + 12:
                                continue
                            dist = (dx ** 2 + dy ** 2) ** 0.5
                        else:
                            dist = 0
                        if dist >= target_d:
                            ball_snaps.append((half.copy(), (bx, by)))
                            if pos_ref is None:
                                pos_ref = (bx, by)
                            break

            # — Fin d'enregistrement —
            if now - rec_start >= RECORD_SECS:
                shot_count += 1
                composite  = make_training_frame(kf_frames, ball_snaps, W, H)
                last_path  = save_shot(kf_frames, composite, shot_count,
                                       ball_snaps, W, H)
                state      = State.PREVIEW
                preview_t  = now

        elif state == State.PREVIEW:
            if now - preview_t >= PREVIEW_SECS:
                state    = State.READY
                bg_model = gray_h.astype(np.float32)

        # ── Affichage (W×H unique — strobe + balle fusionnés) ─────────────
        if state == State.PREVIEW and composite is not None:
            top_panel = composite
        else:
            live = cv2.resize(raw, (W, H))
            if state in (State.READY, State.COUNTDOWN, State.ARMED):
                live = (live.astype(np.float32) * 0.75).astype(np.uint8)
            draw_column_grid(live, W, H)
            # Pendant ARMED : affiche le crop circulaire sur la balle détectée
            if state == State.ARMED and ball_snaps:
                sx_d = W / (W // 2);  sy_d = H / (H // 2)
                pr   = (int(ball_snaps[0][1][0] * sx_d),
                        int(ball_snaps[0][1][1] * sy_d))
                cv2.circle(live, pr, BALL_CROP_R + 4, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(live, pr, BALL_CROP_R + 4, GREEN,      2, cv2.LINE_AA)
                put_text(live, "balle", (pr[0] + BALL_CROP_R + 8, pr[1] + 5),
                         scale=0.45, color=GREEN)
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
            state      = State.COUNTDOWN
            cd_start   = now
            ball_snaps = []
            kf_rest    = None
        elif key == ord('r'):
            bg_model = None
            state    = State.READY
            print("[bg] background reset")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[done] {shot_count} coups sauvegardés dans ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
