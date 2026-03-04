"""
strobe_capture.py  –  Automated strobe-frame capture for YOLO training data.

Usage:
    python3 strobe_capture.py

Controls:
    SPACE       – arm / force-trigger recording
    R           – reset background reference
    Q / ESC     – quit

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
TARGET_FPS    = 240        # camera will cap at its own max
RECORD_SECS   = 2.5        # recording window after trigger
PREVIEW_SECS  = 2.5        # show composite before auto-reset
N_COLS        = 7
OUT_DIR       = "captures"
RAW_DIR       = os.path.join(OUT_DIR, "raw")
MOTION_THRESH = 25         # bg-subtraction pixel threshold
MOTION_AREA   = 80         # min contour area to trigger

# ── Palette (dark premium look) ────────────────────────────────────────────────
BG_COLOR   = (18, 18, 18)
DIM_COLOR  = (40, 40, 40)
GRID_COLOR = (55, 55, 55)
TEXT_COLOR = (220, 220, 220)
DIM_TEXT   = (110, 110, 110)
ACCENT     = (0, 200, 255)    # cyan-ish
RED        = (50, 50, 220)
GREEN      = (80, 200, 80)
WHITE      = (255, 255, 255)

FONT       = cv2.FONT_HERSHEY_SIMPLEX


class State(Enum):
    READY     = "READY"
    RECORDING = "RECORDING"
    PREVIEW   = "PREVIEW"


# ── Camera open ───────────────────────────────────────────────────────────────
def open_camera() -> tuple[cv2.VideoCapture, int, int, float]:
    backends: list = []
    if sys.platform == "darwin":
        avf = getattr(cv2, "CAP_AVFOUNDATION", None)
        if avf is not None:
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


# ── Drawing helpers ────────────────────────────────────────────────────────────
def put_text(img, text, pos, scale=0.55, color=TEXT_COLOR, thickness=1, bold=False):
    th = thickness + (1 if bold else 0)
    cv2.putText(img, text, pos, FONT, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, FONT, scale, color,     th,     cv2.LINE_AA)


def draw_pill(img, x, y, w, h, color, alpha=0.55):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_column_grid(img, W, H, n=N_COLS, col=GRID_COLOR):
    cw = W // n
    for i in range(1, n):
        cv2.line(img, (i * cw, 0), (i * cw, H), col, 1)


# ── Strobe composite ───────────────────────────────────────────────────────────
def make_composite(kf_frames: list, W: int, H: int) -> np.ndarray:
    """Stack the 7 keyframes side-by-side into a single image."""
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    col_w  = W // N_COLS
    # find a reference frame for background
    ref = next((f for f in kf_frames if f is not None), None)
    sf_h, sf_w = (ref.shape[0], ref.shape[1]) if ref is not None else (H // 2, W // 2)

    for i, kf in enumerate(kf_frames):
        x0 = i * col_w
        x1 = W if i == N_COLS - 1 else x0 + col_w
        if kf is None:
            canvas[:, x0:x1] = DIM_COLOR
            continue
        full = cv2.resize(kf, (W, H))
        canvas[:, x0:x1] = full[:, x0:x1]

    draw_column_grid(canvas, W, H)

    # Column index labels
    for i in range(N_COLS):
        x0 = i * col_w
        put_text(canvas, str(i + 1), (x0 + 6, 22), scale=0.5, color=DIM_TEXT)

    return canvas


# ── Auto-save ─────────────────────────────────────────────────────────────────
def save_shot(kf_frames: list, composite: np.ndarray, shot_n: int) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    comp_path = os.path.join(OUT_DIR, f"strobe_{tag}.png")
    cv2.imwrite(comp_path, composite)

    saved_raw = 0
    for i, kf in enumerate(kf_frames):
        if kf is None:
            continue
        raw_path = os.path.join(RAW_DIR, f"{tag}_col{i}.jpg")
        cv2.imwrite(raw_path, kf, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_raw += 1

    print(f"[save] shot #{shot_n:03d}  → {comp_path}  ({saved_raw} raw frames)")
    return comp_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    cap, W, H, cam_fps = open_camera()
    # Store ~60fps of content regardless of camera speed
    replay_sub = max(1, round(cam_fps / 60))

    WIN = "Strobe Capture"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)

    state      = State.READY
    bg_model   = None          # float32 half-res background
    kf_frames  = [None] * N_COLS
    kf_offs    = [float('inf')] * N_COLS
    rec_start  = 0.0
    preview_t  = 0.0
    rep_ctr    = 0
    shot_count = 0
    last_path  = ""
    composite  = None
    armed      = False         # True = will record on next motion
    motion_ok  = False

    t_last = time.perf_counter()

    while True:
        ret, raw = cap.read()
        if not ret:
            break
        now   = time.perf_counter()
        dt    = max(1e-6, now - t_last)
        t_last = now
        fps   = 1.0 / dt

        half = cv2.resize(raw, (W // 2, H // 2))
        gray_h = cv2.GaussianBlur(
            cv2.cvtColor(half, cv2.COLOR_BGR2GRAY), (21, 21), 0)

        # ── Build / update background ──────────────────────────────────────
        if bg_model is None:
            bg_model = gray_h.astype(np.float32)
        else:
            if state == State.READY:
                cv2.accumulateWeighted(gray_h.astype(np.float32),
                                        bg_model, 0.04)

        # ── Motion detection ───────────────────────────────────────────────
        diff = np.abs(gray_h.astype(np.float32) - bg_model)
        _, th = cv2.threshold(diff.astype(np.uint8), MOTION_THRESH, 255, cv2.THRESH_BINARY)
        th = cv2.dilate(th, None, iterations=3)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_ok = any(cv2.contourArea(c) >= MOTION_AREA for c in cnts)

        # ── State machine ──────────────────────────────────────────────────
        if state == State.READY:
            if armed and motion_ok:
                # Trigger!
                state      = State.RECORDING
                rec_start  = now
                kf_frames  = [None] * N_COLS
                kf_offs    = [float('inf')] * N_COLS
                rep_ctr    = 0
                armed      = False
                print(f"[rec]  shot #{shot_count + 1} started")

        elif state == State.RECORDING:
            rep_ctr += 1
            # Per-column keyframe capture
            if cnts:
                best_c = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(best_c) >= MOTION_AREA:
                    M = cv2.moments(best_c)
                    if M["m00"] > 0:
                        cx_h  = int(M["m10"] / M["m00"])
                        cw_h  = half.shape[1] // N_COLS
                        col   = min(cx_h // cw_h, N_COLS - 1)
                        off   = abs(cx_h - (col + 0.5) * cw_h)
                        if off < kf_offs[col]:
                            kf_offs[col]   = off
                            kf_frames[col] = half.copy()

            if now - rec_start >= RECORD_SECS:
                shot_count += 1
                composite   = make_composite(kf_frames, W, H)
                last_path   = save_shot(kf_frames, composite, shot_count)
                state       = State.PREVIEW
                preview_t   = now
                armed       = False   # wait for manual re-arm

        elif state == State.PREVIEW:
            if now - preview_t >= PREVIEW_SECS:
                state    = State.READY
                bg_model = gray_h.astype(np.float32)

        # ── Build display frame ────────────────────────────────────────────
        if state == State.PREVIEW and composite is not None:
            frame = composite.copy()
        else:
            frame = cv2.resize(raw, (W, H))
            if state == State.READY:
                # Dim slightly so overlay pops
                frame = (frame.astype(np.float32) * 0.7).astype(np.uint8)
            draw_column_grid(frame, W, H)

        # ── State banner ───────────────────────────────────────────────────
        if state == State.READY:
            if armed:
                banner_col = (0, 180, 60)
                banner_txt = "  ARMED – waiting for motion  "
            else:
                banner_col = DIM_COLOR
                banner_txt = "  SPACE = arm   R = reset bg   Q = quit  "
        elif state == State.RECORDING:
            rem = max(0.0, RECORD_SECS - (now - rec_start))
            prog = 1.0 - rem / RECORD_SECS
            banner_col = (0, 100, 200)
            banner_txt = f"  RECORDING  {rem:.1f}s  "
            cv2.rectangle(frame, (0, H - 5), (int(W * prog), H),
                          (0, 160, 255), -1)
        else:  # PREVIEW
            rem_p = max(0.0, PREVIEW_SECS - (now - preview_t))
            banner_col = (30, 100, 30)
            banner_txt = f"  SAVED  –  next in {rem_p:.1f}s  "

        draw_pill(frame, 0, 0, W, 32, banner_col, alpha=0.7)
        put_text(frame, banner_txt, (8, 22), scale=0.55,
                 color=WHITE, bold=True)

        # Shot counter + fps (top-right)
        put_text(frame, f"#{shot_count:03d}",
                 (W - 72, 22), scale=0.55, color=ACCENT)
        put_text(frame, f"{fps:.0f}fps",
                 (W - 72, 46), scale=0.45, color=DIM_TEXT)

        # Last saved path (bottom)
        if last_path:
            put_text(frame, os.path.basename(last_path),
                     (8, H - 10), scale=0.40, color=DIM_TEXT)

        cv2.imshow(WIN, frame)

        # ── Key handling ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            if state == State.READY:
                armed = not armed
            elif state == State.RECORDING:
                pass  # ignore during recording
        elif key == ord('r'):
            bg_model = None
            print("[bg] background reset")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[done] {shot_count} shots saved to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
