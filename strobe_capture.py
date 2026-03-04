"""
strobe_capture.py  –  Automated strobe-frame capture for YOLO training data.

Usage:
    python3 strobe_capture.py

Controls:
    SPACE       – start 4-second preparation countdown
    R           – reset background reference
    Q / ESC     – quit

Flow:  READY → COUNTDOWN (4s) → ARMED (waits for motion) → RECORDING → PREVIEW → READY

Each completed shot auto-saves to  captures/
  • strobe_YYYYMMDD_HHMMSS.png          – 7-column composite (for review)
  • raw/YYYYMMDD_HHMMSS_col{0-6}.jpg    – individual keyframes (for labelling)
"""

from __future__ import annotations
import cv2
import numpy as np
import os
import sys
import time
from datetime import datetime
from enum import Enum

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_FPS    = 240
PREP_SECS     = 4.0       # preparation countdown before armed
RECORD_SECS   = 2.5       # recording window after motion trigger
PREVIEW_SECS  = 2.5       # show composite before auto-reset
N_COLS        = 7
OUT_DIR       = "captures"
RAW_DIR       = os.path.join(OUT_DIR, "raw")
MOTION_THRESH = 25
MOTION_AREA   = 80

# ── Palette ────────────────────────────────────────────────────────────────────
DIM_COLOR  = (40, 40, 40)
GRID_COLOR = (55, 55, 55)
TEXT_COLOR = (220, 220, 220)
DIM_TEXT   = (110, 110, 110)
ACCENT     = (0, 200, 255)
WHITE      = (255, 255, 255)
FONT       = cv2.FONT_HERSHEY_SIMPLEX


class State(Enum):
    READY     = "READY"
    COUNTDOWN = "COUNTDOWN"
    ARMED     = "ARMED"
    RECORDING = "RECORDING"
    PREVIEW   = "PREVIEW"


# ── Camera ────────────────────────────────────────────────────────────────────
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
    raise RuntimeError("Cannot open camera.")


# ── Helpers ───────────────────────────────────────────────────────────────────
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


def make_composite(kf_frames: list, W: int, H: int) -> np.ndarray:
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


def save_shot(kf_frames: list, composite: np.ndarray, n: int) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    comp_path = os.path.join(OUT_DIR, f"strobe_{tag}.png")
    cv2.imwrite(comp_path, composite)
    saved = sum(1 for i, kf in enumerate(kf_frames) if kf is not None and
                cv2.imwrite(os.path.join(RAW_DIR, f"{tag}_col{i}.jpg"), kf,
                            [cv2.IMWRITE_JPEG_QUALITY, 95]))
    print(f"[save] #{n:03d} → {comp_path}  ({saved} raw)")
    return comp_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cap, W, H, cam_fps = open_camera()

    WIN = "Strobe Capture"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)

    state      = State.READY
    bg_model   = None
    kf_frames  = [None] * N_COLS
    kf_offs    = [float('inf')] * N_COLS
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

        # Background: update while idle or counting down
        if bg_model is None:
            bg_model = gray_h.astype(np.float32)
        elif state in (State.READY, State.COUNTDOWN):
            cv2.accumulateWeighted(gray_h.astype(np.float32), bg_model, 0.04)

        # Motion detection (used in ARMED state)
        diff = np.abs(gray_h.astype(np.float32) - bg_model)
        _, th_img = cv2.threshold(diff.astype(np.uint8),
                                  MOTION_THRESH, 255, cv2.THRESH_BINARY)
        th_img = cv2.dilate(th_img, None, iterations=3)
        cnts, _ = cv2.findContours(th_img, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        motion = any(cv2.contourArea(c) >= MOTION_AREA for c in cnts)

        # ── State machine ──────────────────────────────────────────────────
        if state == State.COUNTDOWN:
            if now - cd_start >= PREP_SECS:
                # Freeze background right now (putter is placed, ready to swing)
                bg_model = gray_h.astype(np.float32)
                state    = State.ARMED

        elif state == State.ARMED:
            if motion:
                state     = State.RECORDING
                rec_start = now
                kf_frames = [None] * N_COLS
                kf_offs   = [float('inf')] * N_COLS
                print(f"[rec] shot #{shot_count + 1} triggered")

        elif state == State.RECORDING:
            # Per-column keyframe capture
            if cnts:
                best = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(best) >= MOTION_AREA:
                    M = cv2.moments(best)
                    if M["m00"] > 0:
                        cx_h = int(M["m10"] / M["m00"])
                        cw_h = half.shape[1] // N_COLS
                        col  = min(cx_h // cw_h, N_COLS - 1)
                        off  = abs(cx_h - (col + 0.5) * cw_h)
                        if off < kf_offs[col]:
                            kf_offs[col]   = off
                            kf_frames[col] = half.copy()

            if now - rec_start >= RECORD_SECS:
                shot_count += 1
                composite   = make_composite(kf_frames, W, H)
                last_path   = save_shot(kf_frames, composite, shot_count)
                state       = State.PREVIEW
                preview_t   = now

        elif state == State.PREVIEW:
            if now - preview_t >= PREVIEW_SECS:
                state    = State.READY
                bg_model = gray_h.astype(np.float32)

        # ── Display ────────────────────────────────────────────────────────
        if state == State.PREVIEW and composite is not None:
            frame = composite.copy()
        else:
            frame = cv2.resize(raw, (W, H))
            if state in (State.READY, State.COUNTDOWN, State.ARMED):
                frame = (frame.astype(np.float32) * 0.75).astype(np.uint8)
            draw_column_grid(frame, W, H)

        # Banner
        if state == State.READY:
            banner_c = DIM_COLOR
            banner_t = "  SPACE = préparer   R = reset   Q = quitter  "
        elif state == State.COUNTDOWN:
            rem_cd   = max(0.0, PREP_SECS - (now - cd_start))
            banner_c = (0, 130, 180)
            banner_t = f"  Préparez-vous…  {rem_cd:.1f}s  "
            # Big countdown number in center
            n_str = str(int(rem_cd) + 1)
            (tw, th_), _ = cv2.getTextSize(n_str, FONT, 6.0, 8)
            cv2.putText(frame, n_str,
                        ((W - tw) // 2, (H + th_) // 2),
                        FONT, 6.0, (0, 0, 0), 12, cv2.LINE_AA)
            cv2.putText(frame, n_str,
                        ((W - tw) // 2, (H + th_) // 2),
                        FONT, 6.0, WHITE, 8, cv2.LINE_AA)
        elif state == State.ARMED:
            banner_c = (0, 160, 50)
            banner_t = "  PRÊT – frappez quand vous voulez  "
        elif state == State.RECORDING:
            rem      = max(0.0, RECORD_SECS - (now - rec_start))
            prog     = 1.0 - rem / RECORD_SECS
            banner_c = (0, 100, 200)
            banner_t = f"  ENREGISTREMENT  {rem:.1f}s  "
            cv2.rectangle(frame, (0, H - 5), (int(W * prog), H),
                          (0, 160, 255), -1)
        else:  # PREVIEW
            rem_p    = max(0.0, PREVIEW_SECS - (now - preview_t))
            banner_c = (30, 110, 30)
            banner_t = f"  SAUVEGARDÉ  –  {rem_p:.1f}s  "

        draw_pill(frame, 0, 0, W, 32, banner_c, alpha=0.7)
        put_text(frame, banner_t, (8, 22), scale=0.55, color=WHITE, bold=True)

        put_text(frame, f"#{shot_count:03d}", (W - 72, 22), scale=0.55, color=ACCENT)
        put_text(frame, f"{fps:.0f}fps",      (W - 72, 46), scale=0.45, color=DIM_TEXT)
        if last_path:
            put_text(frame, os.path.basename(last_path),
                     (8, H - 10), scale=0.40, color=DIM_TEXT)

        cv2.imshow(WIN, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' ') and state == State.READY:
            state    = State.COUNTDOWN
            cd_start = now
        elif key == ord('r'):
            bg_model = None
            state    = State.READY
            print("[bg] background reset")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[done] {shot_count} shots saved to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
