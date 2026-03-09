#!/usr/bin/env python3
"""
putter_live.py – Real-time golf putter stroke analyser
======================================================
Tracking:
  PRIMARY  – Zone-based motion detection (draw zone with Z)
  FALLBACK – ArUco markers / CSRT tracker + Hough edge angle

Controls:
  Z      – Draw detection zone (click-drag over putter head rest area)
  Space  – Start countdown manually
  C      – Calibrate: current face direction = 0° / target line
  D      – Toggle debug overlay
  R      – Reset session
  F      – Enter CSRT-ROI selection (click-drag over putter head)
  V      – Import video file (offline / hors live mode)
  T      – Enter annotation/training mode (from KEYFRAMES view)
  S      – Save annotation (in annotation mode)
  Esc    – Cancel annotation / Quit
  M      – Save printable ArUco marker PNGs to ./markers/
  Q/Esc  – Quit
"""
from __future__ import annotations

import cv2
import numpy as np
import json
import math
import os
import sys
import time
import subprocess

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "runs", "detect", "runs", "putter", "putter_detector", "weights", "best.pt"
)
from collections import deque
from scipy.interpolate import CubicSpline
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
RECORD_SECS        = 2.5    # stroke capture window (seconds)
COUNTDOWN_SECS     = 3      # countdown duration
SMOOTH_WIN         = 5      # moving-average window (frames)
REPLAY_FPS         = 12     # slow-motion replay speed
TARGET_FPS         = 240    # ask camera for max fps (will cap at what it supports)
PANEL_H            = 130    # height of metrics panel below video (px)
ARUCO_DICT_ID      = cv2.aruco.DICT_4X4_50 if hasattr(cv2, "aruco") else None

ARC_STRAIGHT_PX    = 12    # max deviation → "Straight"
ARC_SLIGHT_PX      = 35    # max deviation → "Slight Arc" (else "Strong Arc")

STILL_THRESHOLD_PX = 20    # px – max spread of positions to be "still"
STILL_SECS         = 1.5
ZONE_MIN_AREA      = 50    # px² – min contour area in zone to count as object

# ── Zones par défaut (caméra overhead 1280×720, ajustables avec Z/B) ──────────
# Exprimées en fraction du frame (x, y, w, h) pour s'adapter à toute résolution
PUTTER_ZONE_REL = (0.46, 0.37, 0.14, 0.27)   # tête de putter au repos (zone blanche)
BALL_ZONE_REL   = (0.32, 0.37, 0.13, 0.27)   # balle – collée à gauche de la zone putter
BALL_BRIGHT_THR = 185
BALL_MIN_PX     = 50

ANN_DIR            = "annotations"  # directory for YOLO OBB training data

# ──────────────────────────────────────────────────────────────────────────────
# Colours  (BGR)
# ──────────────────────────────────────────────────────────────────────────────
C = {
    "green":  (30, 220, 80),
    "red":    (30, 30, 220),
    "yellow": (20, 220, 220),
    "blue":   (220, 150, 30),
    "white":  (240, 240, 240),
    "gray":   (120, 120, 120),
    "dark":   (18, 18, 18),
    "orange": (30, 150, 255),
    "cyan":   (220, 220, 30),
}


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────
class AppState(Enum):
    READY      = "READY"
    ZONE_SETUP = "ZONE_SETUP"   # user drawing detection zone
    COUNTDOWN  = "COUNTDOWN"
    RECORDING  = "RECORDING"
    ANALYSIS   = "ANALYSIS"
    REPLAY     = "REPLAY"
    KEYFRAMES  = "KEYFRAMES"
    ANNOTATE   = "ANNOTATE"     # annotation / training tool


@dataclass
class FrameRec:
    ts:    float
    pos:   tuple          # (x, y) – smoothed head position
    angle: float          # face angle relative to target (degrees, signed)
    vel:   float          # pixels / second


@dataclass
class Result:
    positions:      list
    angles:         list
    timestamps:     list
    velocities:     list
    impact_idx:     int
    face_address:   float
    face_impact:    float
    face_min:       float
    face_max:       float
    face_avg:       float
    arc_devs:       list   # signed perpendicular distances (px)
    arc_max_px:     float
    arc_class:      str    # "Straight" / "Slight Arc" / "Strong Arc"
    launch_dir:     float  = 0.0   # path direction at impact (°, + = right)


# ──────────────────────────────────────────────────────────────────────────────
# ArUco compatibility layer
# ──────────────────────────────────────────────────────────────────────────────
def _build_aruco():
    if not hasattr(cv2, "aruco"):
        return None, None
    try:
        d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        p = cv2.aruco.DetectorParameters()
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = 53
        p.adaptiveThreshConstant   = 7
        p.minMarkerPerimeterRate   = 0.01
        p.errorCorrectionRate      = 0.8
        det = cv2.aruco.ArucoDetector(d, p)
        return det.detectMarkers, d
    except AttributeError:
        pass
    try:
        d = cv2.aruco.Dictionary_get(ARUCO_DICT_ID)
        p = cv2.aruco.DetectorParameters_create()
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = 53
        p.minMarkerPerimeterRate   = 0.01
        return lambda g: cv2.aruco.detectMarkers(g, d, parameters=p), d
    except Exception:
        return None, None


def _save_markers(aruco_dict, out_dir: str = "markers"):
    if aruco_dict is None:
        print("ArUco not available – cannot generate markers.")
        return
    os.makedirs(out_dir, exist_ok=True)
    for mid in [0, 1]:
        try:
            img = cv2.aruco.generateImageMarker(aruco_dict, mid, 400)
        except AttributeError:
            img = cv2.aruco.drawMarker(aruco_dict, mid, 400)
        path = os.path.join(out_dir, f"putter_marker_{mid}.png")
        cv2.imwrite(path, img)
    print(f"[markers] saved to ./{out_dir}/")


# ──────────────────────────────────────────────────────────────────────────────
# Main application
# ──────────────────────────────────────────────────────────────────────────────
class PutterLive:

    def __init__(self):
        self._replay_sub = 2   # overwritten by _open_camera based on actual fps
        self._video_mode = False   # True when reading from a file instead of camera
        self._video_path: Optional[str] = None
        self.cap = self._open_camera()
        self.W   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._aruco_fn, self._aruco_dict = _build_aruco()
        self.using_aruco = self._aruco_fn is not None

        # Fallback CSRT tracker
        self.tracker: Optional[object] = None
        self.roi:     Optional[tuple]  = None
        self._sel_mode  = False
        self._sel_start: Optional[tuple] = None
        self._sel_end:   Optional[tuple] = None

        self._track_mode = "none"

        # ── Zone-based detection ──────────────────────────────────────────
        # Zones pré-définies selon la résolution caméra
        self._zone_rect = (
            int(PUTTER_ZONE_REL[0] * self.W), int(PUTTER_ZONE_REL[1] * self.H),
            int(PUTTER_ZONE_REL[2] * self.W), int(PUTTER_ZONE_REL[3] * self.H),
        )
        self._ball_zone_rect = (
            int(BALL_ZONE_REL[0] * self.W), int(BALL_ZONE_REL[1] * self.H),
            int(BALL_ZONE_REL[2] * self.W), int(BALL_ZONE_REL[3] * self.H),
        )
        self._zone_mog2  = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=36, detectShadows=False)
        self._zone_pos_hist: deque = deque(maxlen=20)  # recent zone positions
        self._zone_still_since: Optional[float] = None
        self._both_since: Optional[float] = None       # présence simultanée putter+balle
        self._zone_last_pos: Optional[tuple] = None
        self._zone_setup_start: Optional[tuple] = None
        self._zone_setup_end:   Optional[tuple] = None

        # ── Live keyframe capture per column ─────────────────────────────
        self._kf_col_frames: list   = [None] * 7
        self._kf_col_offs:   list   = [float('inf')] * 7
        self._kf_col_centers: list  = [None] * 7   # (cx,cy) in _half px, motion centroid
        self._all_live_detections: list        = []    # [(cx_h, cy_h, half_frame), ...]
        self._address_x_half: Optional[float]  = None  # putter x at address (half-res)
        self._dyn_col_bounds_half: Optional[list] = None  # 8 x-values for 7 dynamic cols (half-res)
        self._rec_bg_live:   Optional[np.ndarray] = None
        self._ball_roi_ref:  Optional[np.ndarray] = None
        self._ball_moved:    bool = False
        self._half_buf: deque    = deque(maxlen=6)  # ring buffer for impact timing
        self._shot_count: int    = 0
        self._ball_init_pos: Optional[tuple] = None   # ball centre before stroke
        self._ball_last_pos: Optional[tuple] = None   # ball last tracked position after impact (half-res)
        self._setup_which    = "putter"                 # 'putter' ou 'ball'
        self._yolo_tick      = 0
        self._yolo_ready_box: Optional[tuple] = None   # (x1,y1,x2,y2) dernière détection YOLO
        self._last_yolo_half_pos: Optional[tuple] = None  # dernière position YOLO demi-res (tracking continuité)

        # ── KEYFRAMES post-processing ─────────────────────────────────────
        self._strobe_indices: list = []
        self._kf_init       = False
        self._kf_idx        = 0
        self._kf_last       = 0.0
        self._kf_done       = False
        self._kf_cur_col: Optional[int] = None
        self._kf_flash_col  = -1
        self._kf_flash_t    = 0.0
        self._kf_best_off: dict = {}
        self._kf_bg: Optional[np.ndarray] = None
        self._strobe_det: Optional[list] = None

        # ── YOLO model ────────────────────────────────────────────────────
        self._yolo = None
        if _YOLO_AVAILABLE and os.path.isfile(_MODEL_PATH):
            self._yolo = _YOLO(_MODEL_PATH)
            print(f"[yolo] model loaded: {_MODEL_PATH}")
        else:
            print("[yolo] model not found – YOLO disabled")

        # ── Calibration / target line ─────────────────────────────────────
        self.ref_angle = 0.0
        self.tgt_angle = 0.0

        # ── State machine ─────────────────────────────────────────────────
        self.state      = AppState.READY
        self._cd_start  = 0.0
        self._rec_start = 0.0

        # ── Recording ─────────────────────────────────────────────────────
        self.records: list[FrameRec] = []
        self._pos_buf = deque(maxlen=SMOOTH_WIN)
        self._ang_buf = deque(maxlen=SMOOTH_WIN)

        # ── Replay ────────────────────────────────────────────────────────
        self.result: Optional[Result]        = None
        self._rep_frames: list[np.ndarray]   = []
        self._rep_positions: list            = []   # (x,y) or None per stored frame
        self._rep_idx   = 0
        self._rep_last  = 0.0
        self._rep_ctr   = 0
        self._rep_loops = 0

        # ── Ghost / debug ─────────────────────────────────────────────────
        self._last_frame: Optional[np.ndarray] = None
        self._last_path: list = []
        self.debug     = False
        self._dbg_info: dict = {}

        # ── Annotation / training tool ────────────────────────────────────
        # Each box: [cx, cy, w, h, angle_deg]  (in display-image pixel coords)
        self._ann_boxes: list        = []
        self._ann_composite: Optional[np.ndarray] = None
        self._ann_drag_start: Optional[tuple] = None
        self._ann_drag_end:   Optional[tuple] = None
        self._ann_drawing:    bool   = False
        self._ann_sel_box:    int    = -1   # index of last/selected box

        # OpenCV window
        cv2.namedWindow("Putter Live", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Putter Live", self.W, self.H + PANEL_H)
        cv2.setMouseCallback("Putter Live", self._on_mouse)

    # ── Camera ────────────────────────────────────────────────────────────────

    def _open_camera(self) -> cv2.VideoCapture:
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
                actual_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                # store every Nth frame so we keep ~60 fps of content
                self._replay_sub = max(1, round(actual_fps / 60))
                print(f"[cam] fps={actual_fps:.0f}  replay_sub={self._replay_sub}  exposure={actual_exp}")
                return cap

        raise RuntimeError(
            "Cannot open camera.\n"
            "• Make sure no other app is using it.\n"
            "• macOS: grant camera permission in System Settings → Privacy."
        )

    def _load_video(self, path: str) -> bool:
        """Switch capture source to a video file. Returns True on success."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"[video] Impossible d'ouvrir : {path}")
            return False

        self.cap.release()
        self.cap = cap
        self.W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._replay_sub = max(1, round(vid_fps / 60))
        self._video_mode = True
        self._video_path = path

        self._zone_rect = (
            int(PUTTER_ZONE_REL[0] * self.W), int(PUTTER_ZONE_REL[1] * self.H),
            int(PUTTER_ZONE_REL[2] * self.W), int(PUTTER_ZONE_REL[3] * self.H),
        )
        self._ball_zone_rect = (
            int(BALL_ZONE_REL[0] * self.W), int(BALL_ZONE_REL[1] * self.H),
            int(BALL_ZONE_REL[2] * self.W), int(BALL_ZONE_REL[3] * self.H),
        )
        self._zone_mog2 = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=36, detectShadows=False)
        self._zone_pos_hist.clear()
        self._zone_still_since = None

        cv2.resizeWindow("Putter Live", self.W, self.H + PANEL_H)
        print(f"[video] chargé : {os.path.basename(path)}  fps={vid_fps:.0f}")
        return True

    def _open_video_file(self) -> None:
        """Open a file dialog (subprocess) then load the chosen video."""
        _dialog_script = (
            "import tkinter as tk; from tkinter import filedialog; "
            "root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True); "
            "p = filedialog.askopenfilename("
            "    title='Importer une video',"
            "    filetypes=[('Fichiers video', '*.mp4 *.mov *.avi *.mkv *.m4v'),"
            "               ('Tous les fichiers', '*.*')]);"
            "root.destroy(); print(p)"
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", _dialog_script],
                capture_output=True, text=True, timeout=120,
            )
            path = result.stdout.strip()
        except Exception as e:
            print(f"[video] Erreur boîte de dialogue : {e}")
            return
        if path:
            self._load_video(path)

    # ── Mouse callback ────────────────────────────────────────────────────────

    def _on_mouse(self, event, x, y, flags, param):
        # ── Zone setup ────────────────────────────────────────────────────
        if self.state == AppState.ZONE_SETUP:
            if event == cv2.EVENT_LBUTTONDOWN:
                self._zone_setup_start = (x, y)
                self._zone_setup_end   = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self._zone_setup_start:
                self._zone_setup_end = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and self._zone_setup_start:
                self._zone_setup_end = (x, y)
                x0, y0 = self._zone_setup_start
                x1, y1 = self._zone_setup_end
                zw, zh = abs(x1 - x0), abs(y1 - y0)
                if zw > 20 and zh > 20:
                    if self._setup_which == "ball":
                        self._ball_zone_rect = (min(x0, x1), min(y0, y1), zw, zh)
                        print(f"[ball_zone] set to {self._ball_zone_rect}")
                    else:
                        self._zone_rect = (min(x0, x1), min(y0, y1), zw, zh)
                        # Reset MOG2 so it re-learns the background
                        self._zone_mog2 = cv2.createBackgroundSubtractorMOG2(
                            history=120, varThreshold=36, detectShadows=False)
                        self._zone_pos_hist.clear()
                        self._zone_still_since = None
                        print(f"[zone] set to {self._zone_rect}")
                self._zone_setup_start = None
                self._zone_setup_end   = None
                self.state = AppState.READY
            return

        # ── Annotation mode ───────────────────────────────────────────────
        if self.state == AppState.ANNOTATE:
            if event == cv2.EVENT_LBUTTONDOWN:
                self._ann_drag_start = (x, y)
                self._ann_drag_end   = (x, y)
                self._ann_drawing    = True
            elif event == cv2.EVENT_MOUSEMOVE and self._ann_drawing:
                self._ann_drag_end = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and self._ann_drawing:
                self._ann_drag_end = (x, y)
                self._ann_drawing  = False
                if self._ann_drag_start:
                    x0, y0 = self._ann_drag_start
                    x1, y1 = self._ann_drag_end
                    bw, bh = abs(x1 - x0), abs(y1 - y0)
                    if bw > 8 and bh > 8:
                        cx = (x0 + x1) // 2
                        cy = (y0 + y1) // 2
                        # Angle from drag direction (wider axis)
                        angle = math.degrees(math.atan2(-(y1 - y0), x1 - x0)) if bw >= bh else 90.0
                        self._ann_boxes.append([cx, cy, bw, bh, round(angle, 1)])
                        self._ann_sel_box = len(self._ann_boxes) - 1
                self._ann_drag_start = None
                self._ann_drag_end   = None
            elif event == cv2.EVENT_MOUSEWHEEL:
                # Rotate selected box ±5° per notch
                if self._ann_boxes:
                    idx = self._ann_sel_box if 0 <= self._ann_sel_box < len(self._ann_boxes) else len(self._ann_boxes) - 1
                    delta = 5.0 if flags > 0 else -5.0
                    self._ann_boxes[idx][4] = round((self._ann_boxes[idx][4] + delta) % 360, 1)
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Delete nearest box
                if self._ann_boxes:
                    dists = [math.hypot(b[0] - x, b[1] - y) for b in self._ann_boxes]
                    nearest = int(np.argmin(dists))
                    if dists[nearest] < 70:
                        self._ann_boxes.pop(nearest)
                        self._ann_sel_box = len(self._ann_boxes) - 1
            return

        # ── CSRT ROI selection ────────────────────────────────────────────
        if not self._sel_mode:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self._sel_start = (x, y)
            self._sel_end   = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._sel_start:
            self._sel_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self._sel_start:
            self._sel_end = (x, y)
            x0, y0 = self._sel_start
            x1, y1 = self._sel_end
            roi = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            if roi[2] > 10 and roi[3] > 10:
                self.roi = roi
            self._sel_mode  = False
            self._sel_start = None
            self._sel_end   = None
            if self.roi:
                self._init_fallback()

    def _init_fallback(self):
        frame = self._last_frame
        if frame is None or self.roi is None:
            return
        try:
            self.tracker = cv2.TrackerCSRT_create()
        except AttributeError:
            self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, self.roi)
        self.using_aruco = False
        print("[fallback] CSRT tracker initialised.")

    # ── Zone-based motion detection ───────────────────────────────────────────

    def _detect_in_zone(self, frame: np.ndarray) -> Optional[tuple]:
        """
        Apply MOG2 background subtraction inside the drawn zone.
        Returns (cx, cy) of the largest moving blob, or None.
        """
        if self._zone_rect is None:
            return None
        zx, zy, zw, zh = self._zone_rect
        zx = max(0, zx); zy = max(0, zy)
        zx2 = min(self.W, zx + zw); zy2 = min(self.H, zy + zh)
        crop = frame[zy:zy2, zx:zx2]
        if crop.size == 0:
            return None

        fg = self._zone_mog2.apply(crop)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.dilate(fg, kernel, iterations=2)

        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < ZONE_MIN_AREA:
            return None
        M = cv2.moments(c)
        if M["m00"] <= 0:
            return None
        cx = int(M["m10"] / M["m00"]) + zx
        cy = int(M["m01"] / M["m00"]) + zy
        return (cx, cy)

    # ── ArUco / CSRT detection ────────────────────────────────────────────────

    def _detect(self, frame: np.ndarray) -> tuple[Optional[tuple], Optional[float]]:
        self._dbg_info = {}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._aruco_fn is not None:
            corners, ids, _ = self._aruco_fn(gray)
            if ids is not None and len(ids) > 0:
                self.using_aruco = True
                self._track_mode = "aruco"
                c = corners[0][0]
                center = tuple(c.mean(axis=0).astype(int))
                dx = float(c[1][0] - c[0][0])
                dy = float(c[1][1] - c[0][1])
                raw = math.degrees(math.atan2(-dy, dx))
                self._dbg_info = {"corners": c, "marker_id": int(ids[0][0])}
                return center, raw

        if self.tracker is not None:
            ok, bbox = self.tracker.update(frame)
            if ok:
                x, y, w, h = (int(v) for v in bbox)
                center = (x + w // 2, y + h // 2)
                roi_gray = gray[y:y + h, x:x + w]
                edges = cv2.Canny(roi_gray, 50, 150)
                lines = cv2.HoughLinesP(edges, 1, math.pi / 180, 15,
                                        minLineLength=8, maxLineGap=5)
                raw = None
                if lines is not None:
                    angs = [
                        math.degrees(math.atan2(-(l[0][3] - l[0][1]),
                                                  l[0][2] - l[0][0]))
                        for l in lines
                    ]
                    raw = float(np.median(angs)) if angs else None
                self.using_aruco = False
                self._track_mode = "csrt"
                self._dbg_info = {"roi": (x, y, w, h), "fallback": True}
                return center, raw

        # ── YOLO fallback ─────────────────────────────────────────────────
        if self._yolo is not None:
            try:
                _half = cv2.resize(frame, (self.W // 2, self.H // 2))
                _preds = self._yolo.predict(_half, conf=0.25, verbose=False)
                _boxes = _preds[0].boxes
                if _boxes is not None and len(_boxes) > 0:
                    _confs = _boxes.conf.tolist()
                    _bi = int(max(range(len(_confs)), key=lambda k: _confs[k]))
                    _x1, _y1, _x2, _y2 = [int(v) for v in _boxes.xyxy[_bi].tolist()]
                    center = ((_x1 + _x2), (_y1 + _y2))  # half-res × 2 = full-res
                    # Angle from box orientation (width vs height)
                    bw, bh = _x2 - _x1, _y2 - _y1
                    raw = 0.0 if bw >= bh else 90.0
                    self._track_mode = "yolo"
                    self._yolo_ready_box = (_x1 * 2, _y1 * 2, _x2 * 2, _y2 * 2)
                    return center, raw
            except Exception:
                pass

        self._track_mode = "none"
        return None, None

    # ── Smoothing ─────────────────────────────────────────────────────────────

    def _smooth_pos(self, pos: Optional[tuple]) -> Optional[tuple]:
        if pos is None:
            return None
        self._pos_buf.append(pos)
        return (
            int(np.mean([p[0] for p in self._pos_buf])),
            int(np.mean([p[1] for p in self._pos_buf])),
        )

    def _smooth_angle(self, angle: Optional[float]) -> Optional[float]:
        if angle is None:
            return None
        self._ang_buf.append(angle)
        return float(np.mean(self._ang_buf))

    def _velocity(self, pos: Optional[tuple], now: float) -> float:
        if pos is None or not self.records:
            return 0.0
        prev = self.records[-1]
        dt = now - prev.ts
        if dt <= 0:
            return 0.0
        return math.hypot(pos[0] - prev.pos[0], pos[1] - prev.pos[1]) / dt

    # ── Analysis ──────────────────────────────────────────────────────────────

    def _find_impact(self, recs: list[FrameRec]) -> int:
        if len(recs) < 3:
            return 0
        vels = np.array([r.vel for r in recs])
        k = min(7, len(vels))
        if k > 1:
            vels = np.convolve(vels, np.ones(k) / k, mode="same")
        return int(np.argmax(vels))

    def _arc_metrics(self, recs: list[FrameRec]):
        if len(recs) < 2:
            return [], 0.0, "Straight"
        pts = np.array([r.pos for r in recs], dtype=float)
        rad  = math.radians(self.tgt_angle)
        d    = np.array([math.cos(rad), -math.sin(rad)])
        perp = np.array([-d[1], d[0]])
        origin = pts[0]
        devs = [float((p - origin) @ perp) for p in pts]
        mx = max(abs(v) for v in devs) if devs else 0.0
        cls = (
            "Straight"   if mx < ARC_STRAIGHT_PX else
            "Slight Arc" if mx < ARC_SLIGHT_PX   else
            "Strong Arc"
        )
        return devs, mx, cls

    def _analyze(self) -> Optional[Result]:
        recs = self.records
        if len(recs) < 3:
            return None
        imp  = self._find_impact(recs)
        devs, mx, cls = self._arc_metrics(recs)
        angles = [r.angle for r in recs]

        launch_dir = 0.0
        if 0 < imp < len(recs) - 1:
            p1 = np.array(recs[imp - 1].pos, dtype=float)
            p2 = np.array(recs[imp + 1].pos, dtype=float)
            dx, dy = p2 - p1
            if dx != 0 or dy != 0:
                path_abs = math.degrees(math.atan2(-dy, dx))
                raw = path_abs - self.tgt_angle
                launch_dir = (raw + 180) % 360 - 180

        return Result(
            positions   = [r.pos   for r in recs],
            angles      = angles,
            timestamps  = [r.ts    for r in recs],
            velocities  = [r.vel   for r in recs],
            impact_idx  = imp,
            face_address= recs[0].angle,
            face_impact = recs[imp].angle,
            face_min    = float(min(angles)),
            face_max    = float(max(angles)),
            face_avg    = float(np.mean(angles)),
            arc_devs    = devs,
            arc_max_px  = mx,
            arc_class   = cls,
            launch_dir  = launch_dir,
        )

    def _save_json(self, r: Result):
        ts = time.strftime("%Y%m%d_%H%M%S")
        data = {
            "timestamp":  ts,
            "positions":  [[p[0], p[1]] for p in r.positions],
            "face_angles":  r.angles,
            "timestamps":   r.timestamps,
            "velocities":   r.velocities,
            "impact_index": r.impact_idx,
            "metrics": {
                "face_at_address": round(r.face_address, 3),
                "face_at_impact":  round(r.face_impact,  3),
                "face_min":        round(r.face_min,     3),
                "face_max":        round(r.face_max,     3),
                "face_avg":        round(r.face_avg,     3),
                "arc_max_px":      round(r.arc_max_px,   1),
                "arc_class":       r.arc_class,
            },
            "arc_perpendicular": [round(d, 2) for d in r.arc_devs],
        }
        fname = f"session_{ts}.json"
        with open(fname, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[saved] {fname}")

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _put(self, frame, text, pos, scale=0.55, color=None, thick=1):
        if color is None:
            color = C["white"]
        cv2.putText(frame, text, (pos[0] + 1, pos[1] + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, C["dark"], thick + 1, cv2.LINE_AA)
        cv2.putText(frame, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    def _draw_target_line(self, frame):
        cx, cy = self.W // 2, self.H // 2
        rad  = math.radians(self.tgt_angle)
        half = self.W // 2
        dx = int(math.cos(rad) * half)
        dy = int(math.sin(rad) * half)
        cv2.line(frame, (cx - dx, cy + dy), (cx + dx, cy - dy),
                 C["red"], 1, cv2.LINE_AA)
        self._put(frame, "TARGET", (cx + dx - 65, cy - dy - 8),
                  scale=0.4, color=C["red"])

    def _draw_dashed(self, frame, p1, p2, color, dash=12):
        x1, y1 = p1; x2, y2 = p2
        dist = math.hypot(x2 - x1, y2 - y1)
        n = max(1, int(dist / dash))
        for i in range(0, n, 2):
            ax = int(x1 + (x2 - x1) * i / n);       ay = int(y1 + (y2 - y1) * i / n)
            bx = int(x1 + (x2 - x1) * min(i+1, n) / n)
            by = int(y1 + (y2 - y1) * min(i+1, n) / n)
            cv2.line(frame, (ax, ay), (bx, by), color, 1, cv2.LINE_AA)

    def _draw_path(self, frame, positions, color=None, width=2, centerline=True):
        if len(positions) < 2:
            return
        if color is None:
            color = C["green"]
        pts = np.array(positions, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], False, color, width, cv2.LINE_AA)
        if centerline and len(positions) >= 4:
            xs = np.array([p[0] for p in positions], float)
            ys = np.array([p[1] for p in positions], float)
            cf = np.polyfit(xs, ys, 1)
            x0, x1 = int(xs.min()), int(xs.max())
            lighter = tuple(min(v + 80, 255) for v in color)
            cv2.line(frame,
                     (x0, int(np.polyval(cf, x0))),
                     (x1, int(np.polyval(cf, x1))),
                     lighter, 1, cv2.LINE_AA)

    def _draw_hud(self, frame, fps: float, extra: list = None):
        track_label = {"aruco": "ArUco", "csrt": "CSRT",
                       "zone": "Zone", "none": "None ✗"}.get(self._track_mode, "?")
        source_label = (
            f"VIDEO : {os.path.basename(self._video_path)}" if self._video_mode
            else "SOURCE: Camera"
        )
        lines = [
            f"STATE : {self.state.value}",
            f"FPS   : {fps:>4.0f}",
            f"TRACK : {track_label}",
            source_label,
        ]
        if extra:
            lines.extend(extra)
        y = 30
        for ln in lines:
            self._put(frame, ln, (14, y))
            y += 22

    def _draw_guidance(self, frame):
        cx, cy = self.W // 2, self.H // 2
        w, h   = 220, 90
        corners = [(cx - w, cy - h), (cx + w, cy - h),
                   (cx + w, cy + h), (cx - w, cy + h)]
        for i in range(4):
            self._draw_dashed(frame, corners[i], corners[(i + 1) % 4], C["gray"])
        pz_ok = self._zone_rect is not None
        bz_ok = self._ball_zone_rect is not None
        if not pz_ok and not bz_ok:
            zone_hint = "Z = zone putter   B = zone balle   (dessiner les deux)"
        elif not pz_ok:
            zone_hint = "Z = zone putter  [balle OK]"
        elif not bz_ok:
            zone_hint = "B = zone balle  [putter OK]"
        else:
            zone_hint = "Zones actives – putter en zone putter pour demarrer"
        tips = [
            zone_hint,
            "C = calibrate target   |   F = CSRT ROI   |   M = print markers",
            "V = importer video (hors live)   |   SPACE = lancer compte a rebours",
        ]
        for i, t in enumerate(tips):
            sz = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.47, 1)[0]
            self._put(frame, t, (cx - sz[0] // 2, cy + h + 28 + i * 21),
                      scale=0.47, color=C["gray"])

    def _draw_countdown(self, frame, remaining: float):
        n    = int(math.ceil(remaining))
        text = str(n) if n > 0 else "GO!"
        clr  = C["green"] if n <= 0 else C["white"]
        sc, th = 3.5, 5
        sz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sc, th)[0]
        px = (self.W - sz[0]) // 2
        py = (self.H + sz[1]) // 2
        cv2.putText(frame, text, (px + 3, py + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, C["dark"], th + 3, cv2.LINE_AA)
        cv2.putText(frame, text, (px, py),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, clr, th, cv2.LINE_AA)

    def _draw_face_arrow(self, frame, pos, angle_deg):
        rad = math.radians(angle_deg + self.tgt_angle)
        ex  = int(pos[0] + math.cos(rad) * 50)
        ey  = int(pos[1] - math.sin(rad) * 50)
        cv2.arrowedLine(frame, pos, (ex, ey),
                        C["cyan"], 2, cv2.LINE_AA, tipLength=0.3)

    def _draw_rounded_box(self, frame, pts: np.ndarray, color, thickness=1, radius=6):
        """Contour d'un rectangle orienté avec coins arrondis (Bézier quadratique)."""
        fpts = pts.astype(float)
        n = len(fpts)
        for i in range(n):
            p1 = fpts[i];  p2 = fpts[(i + 1) % n]
            v = p2 - p1;   L = float(np.linalg.norm(v))
            if L < 1:
                continue
            u = v / L;  r = min(radius, L / 2 - 1)
            a = (p1 + u * r).astype(int)
            b = (p2 - u * r).astype(int)
            cv2.line(frame, tuple(a), tuple(b), color, thickness, cv2.LINE_AA)
        for i in range(n):
            pv = fpts[(i - 1) % n];  pc = fpts[i];  pn = fpts[(i + 1) % n]
            v1 = pv - pc;  v2 = pn - pc
            L1 = float(np.linalg.norm(v1));  L2 = float(np.linalg.norm(v2))
            if L1 < 1 or L2 < 1:
                continue
            r = min(radius, L1 / 2 - 1, L2 / 2 - 1)
            a = pc + v1 / L1 * r;  b = pc + v2 / L2 * r
            arc = np.array(
                [(1 - t)**2 * a + 2 * (1 - t) * t * pc + t**2 * b
                 for t in np.linspace(0, 1, 8)],
                dtype=np.int32,
            )
            cv2.polylines(frame, [arc.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)

    def _draw_tapered_arc(self, frame, smooth_pts, shadow_color, main_color, max_thick=7):
        """Arc effilé : épais au centre, pointu aux deux extrémités."""
        n = len(smooth_pts)
        if n < 2:
            return
        for pass_color, pass_extra in [(shadow_color, 3), (main_color, 0)]:
            for i in range(n - 1):
                t = i / max(1, n - 2)
                thick = max(1, int(round(1 + (max_thick - 1) * math.sin(t * math.pi))))
                cv2.line(frame,
                         tuple(smooth_pts[i][0]),
                         tuple(smooth_pts[i + 1][0]),
                         pass_color, thick + pass_extra, cv2.LINE_AA)

    def _draw_results(self, frame, r: Result):
        ph = 90
        y0 = self.H - ph
        ov = frame.copy()
        cv2.rectangle(ov, (0, y0), (self.W, self.H), (18, 18, 18), -1)
        cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)

        self._put(frame, "FACE ANGLE", (14, y0 + 17), scale=0.43, color=C["gray"])
        vals = [
            f"Addr {r.face_address:+.1f}",
            f"Imp  {r.face_impact:+.1f}",
            f"Min  {r.face_min:+.1f}",
            f"Max  {r.face_max:+.1f}",
            f"Avg  {r.face_avg:+.1f}",
        ]
        x, y = 14, y0 + 36
        for i, v in enumerate(vals):
            self._put(frame, v + "°", (x, y), scale=0.46)
            x += 145
            if i == 1:
                x, y = 14, y0 + 58

        arc_clr = (C["green"]  if r.arc_class == "Straight" else
                   C["yellow"] if r.arc_class == "Slight Arc" else
                   C["orange"])
        arc_txt = f"{r.arc_class}  ({r.arc_max_px:.0f} px)"
        sz = cv2.getTextSize(arc_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)[0]
        self._put(frame, arc_txt,
                  (self.W - sz[0] - 14, y0 + 55),
                  scale=0.72, color=arc_clr, thick=2)

        self._put(frame, "SPACE: start countdown   R: reset   Q: quit",
                  (14, self.H - 7), scale=0.38, color=C["gray"])

    def _draw_sel_rect(self, frame):
        if self._sel_start and self._sel_end:
            cv2.rectangle(frame, self._sel_start, self._sel_end, C["yellow"], 1)
        self._put(frame, "DRAG to select putter head ROI",
                  (14, self.H // 2), scale=0.6, color=C["yellow"])

    def _draw_debug(self, frame):
        di = self._dbg_info
        if "corners" in di:
            c = di["corners"].astype(int)
            for i in range(4):
                cv2.line(frame, tuple(c[i]), tuple(c[(i + 1) % 4]), C["cyan"], 1)
                cv2.circle(frame, tuple(c[i]), 3, C["green"], -1)
            self._put(frame, f"ID:{di.get('marker_id', '?')}",
                      (int(c[0][0]), int(c[0][1]) - 6),
                      scale=0.4, color=C["cyan"])
        if "roi" in di:
            x, y, w, h = di["roi"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), C["blue"], 1)
            self._put(frame, "CSRT", (x, y - 4), scale=0.38, color=C["blue"])

    # ── Annotation tool helpers ───────────────────────────────────────────────

    def _draw_ann_box(self, frame, cx: int, cy: int, bw: int, bh: int,
                      angle: float, selected: bool = False):
        """Draw a rotated bounding box (blue) with center dot."""
        rect = ((float(cx), float(cy)), (float(bw), float(bh)), float(angle))
        pts  = cv2.boxPoints(rect).astype(np.int32)
        color = (255, 220, 80) if selected else (200, 100, 0)   # bright/dim blue
        cv2.drawContours(frame, [pts], 0, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 5, color, 1, cv2.LINE_AA)

    def _draw_ann_arc(self, frame, centers: list):
        """Draw smooth arc through the annotation box centres."""
        if len(centers) < 2:
            return
        pts = np.array(centers, np.int32)
        cv2.polylines(frame, [pts.reshape(-1, 1, 2)], False,
                      (200, 100, 0), 3, cv2.LINE_AA)
        cv2.polylines(frame, [pts.reshape(-1, 1, 2)], False,
                      (255, 220, 80), 1, cv2.LINE_AA)

    def _save_annotation(self):
        """
        Save current annotation as:
          annotations/ann_<ts>.jpg   – composite image (video area only)
          annotations/ann_<ts>.txt   – YOLO OBB labels (class + 4 corner points)
        """
        if not self._ann_boxes or self._ann_composite is None:
            print("[annotate] Nothing to save.")
            return
        os.makedirs(ANN_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(ANN_DIR, f"ann_{ts}.jpg")
        lbl_path = os.path.join(ANN_DIR, f"ann_{ts}.txt")

        # Save only the video portion (crop panel)
        img = self._ann_composite[:self.H].copy()
        cv2.imwrite(img_path, img)

        # YOLO OBB format: 0 x1 y1 x2 y2 x3 y3 x4 y4  (normalised to img size)
        with open(lbl_path, "w") as f:
            for box in self._ann_boxes:
                cx, cy, bw, bh, angle = box
                rect = ((float(cx), float(cy)), (float(bw), float(bh)), float(angle))
                pts  = cv2.boxPoints(rect)           # 4 corners (float)
                pts_n = (pts / [self.W, self.H]).flatten().tolist()
                f.write("0 " + " ".join(f"{v:.6f}" for v in pts_n) + "\n")

        n = len(self._ann_boxes)
        print(f"[annotate] saved {img_path}  ({n} box{'es' if n != 1 else ''})")

        # Count total annotations saved
        total = len([x for x in os.listdir(ANN_DIR) if x.endswith(".txt")])
        print(f"[annotate] total sessions in {ANN_DIR}/: {total}")
        if total >= 10:
            print("[annotate] ≥ 10 sessions ready – you can now run training:")
            print(f"  python3 train_annotated.py  (creates YOLO dataset from {ANN_DIR}/)")

    # ── Keyframe detection helper ─────────────────────────────────────────────

    def _detect_putter_col(self, idx: int):
        if self._kf_bg is None or idx == 0:
            return None
        f    = self._rep_frames[idx]
        gray = cv2.GaussianBlur(
            cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (21, 21), 0
        ).astype(np.float32)
        diff  = np.abs(gray - self._kf_bg)
        _, th = cv2.threshold(diff.astype(np.uint8), 8, 255, cv2.THRESH_BINARY)
        th    = cv2.dilate(th, None, iterations=3)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 60:
            return None
        M = cv2.moments(c)
        if M["m00"] <= 0:
            return None
        cx    = int(M["m10"] / M["m00"])
        sf_w  = self._rep_frames[0].shape[1]
        col_w = sf_w // 7
        col   = min(cx // col_w, 6)
        offset = abs(cx - (col + 0.5) * col_w)
        return col, offset

    # ── Dynamic column computation ────────────────────────────────────────────

    def _compute_dynamic_columns(self) -> None:
        """Compute dynamic column boundaries from the actual stroke range,
        then assign the best half-res keyframe to each of the 7 columns.

        Column layout (0-indexed):
          Col 0-2 : backswing  (left of address, 3 equal slices)
          Col 3   : address/impact (centred on ball zone)
          Col 4-6 : follow-through (right of address, 3 equal slices)

        Col 3 is preserved from ball-impact detection done during RECORDING.
        """
        N = 7
        detections = self._all_live_detections   # [(cx_h, cy_h, frame), ...]
        if not detections:
            return

        # Preserve the impact frame set during RECORDING (ball-moved detection)
        impact_frame  = self._kf_col_frames[3]
        impact_center = self._kf_col_centers[3]
        impact_offs   = self._kf_col_offs[3]

        # Address x (half-res) – prefer ball zone centre as stable reference
        if self._ball_zone_rect is not None:
            _bzx, _bzy, _bzw, _bzh = self._ball_zone_rect
            x_addr_h = float(_bzx + _bzw // 2) / 2.0   # full-res → half-res
        elif self._address_x_half is not None:
            x_addr_h = self._address_x_half
        else:
            x_addr_h = float(detections[0][0])

        xs          = [d[0] for d in detections]
        x_min_h     = min(xs)
        x_max_h     = max(xs)
        left_range  = max(x_addr_h - x_min_h, 30.0)
        right_range = max(x_max_h - x_addr_h, 30.0)

        # Skip if stroke range is too small to be meaningful
        if left_range + right_range < 60:
            return

        lcw = left_range  / 3.0    # backswing column width (half-res px)
        rcw = right_range / 3.0    # follow-through column width
        acw = (lcw + rcw) / 2.0    # address column width

        # 8 boundaries for 7 columns:
        # Col i occupies [b[i], b[i+1]]
        b    = [0.0] * 8
        b[3] = x_addr_h - acw / 2.0   # left edge of address col
        b[4] = x_addr_h + acw / 2.0   # right edge of address col
        b[2] = b[3] - lcw
        b[1] = b[2] - lcw
        b[0] = b[1] - lcw
        b[5] = b[4] + rcw
        b[6] = b[5] + rcw
        b[7] = b[6] + rcw

        self._dyn_col_bounds_half = b

        # Reset column data
        self._kf_col_frames  = [None] * N
        self._kf_col_offs    = [float('inf')] * N
        self._kf_col_centers = [None] * N

        # Restore impact frame for col 3
        if impact_frame is not None:
            self._kf_col_frames[3]  = impact_frame
            self._kf_col_offs[3]    = impact_offs if impact_offs != float('inf') else 0.0
            self._kf_col_centers[3] = impact_center

        # Assign each detection to its dynamic column
        for cx_h, cy_h, frame_h in detections:
            assigned = False
            for col in range(N):
                if b[col] <= cx_h < b[col + 1]:
                    if col == 3:
                        assigned = True
                        break   # preserve impact col
                    col_cx = (b[col] + b[col + 1]) / 2.0
                    offset  = abs(cx_h - col_cx)
                    if offset < self._kf_col_offs[col]:
                        self._kf_col_offs[col]    = offset
                        self._kf_col_frames[col]  = frame_h
                        self._kf_col_centers[col] = (cx_h, cy_h)
                    assigned = True
                    break
            if not assigned:
                # Outside [b[0], b[7]] – clamp to col 0 or col 6
                col = 0 if cx_h < b[0] else N - 1
                if col != 3:
                    col_cx = (b[col] + b[col + 1]) / 2.0
                    offset  = abs(cx_h - col_cx)
                    if offset < self._kf_col_offs[col]:
                        self._kf_col_offs[col]    = offset
                        self._kf_col_frames[col]  = frame_h
                        self._kf_col_centers[col] = (cx_h, cy_h)

    # ── Strobe composite ──────────────────────────────────────────────────────

    def _run_yolo_strobe(self, composite: np.ndarray) -> list:
        """Lance YOLO sur l'image composite (résolution display) et retourne
        une détection par colonne en coordonnées display."""
        N_COLS = 7
        result = [None] * N_COLS
        if self._yolo is None:
            print("[yolo] model is None – skipping detection")
            return result

        h, w = composite.shape[:2]
        col_w = w // N_COLS

        preds = self._yolo.predict(composite, conf=0.15, verbose=False)
        boxes = preds[0].boxes
        if boxes is None or len(boxes) == 0:
            print("[yolo] composite: no detection")
            return result

        xywh  = boxes.xywh.tolist()
        confs = boxes.conf.tolist()

        # Associer chaque détection à sa colonne (la plus haute conf gagne).
        # Rejeter les détections dont le centre est trop loin du centre attendu
        # de la colonne (évite les faux positifs aux frontières).
        best_conf = [-1.0] * N_COLS
        for j in range(len(xywh)):
            cx, cy, bw, bh = xywh[j][:4]
            conf = confs[j]
            col = min(int(cx // col_w), N_COLS - 1)
            expected_cx = (col + 0.5) * col_w
            if abs(cx - expected_cx) > col_w * 0.65:
                continue   # trop loin du centre de colonne → ignorer
            if conf > best_conf[col]:
                best_conf[col] = conf
                result[col] = (cx, cy, bw, bh, 0.0)
            print(f"[yolo] composite col={col} cx={cx:.0f} cy={cy:.0f} conf={conf:.2f}")

        return result

    def _draw_strobe_composite(self) -> np.ndarray:  # noqa: C901
        N_COLS = 7
        HDR_H  = 38   # dark header height (matches reference image)
        # BGR colour palette (reference = dark bg, gold text, red arc, cyan ball)
        GOLD  = (0,  180, 220)    # amber-gold
        WHITE = (240, 240, 240)
        RED   = (30,   30, 220)   # pure red in BGR
        CYAN  = (220, 200,  30)   # cyan-ish in BGR

        out = np.zeros((self.H + PANEL_H, self.W, 3), dtype=np.uint8)

        # ── 1. Strobe mosaic ──────────────────────────────────────────────
        src_cols = self._kf_col_frames
        if not any(f is not None for f in src_cols):
            frames = self._rep_frames
            if not frames:
                return out
            src_cols = [
                frames[idx] if idx is not None and idx < len(frames) else None
                for idx in (list(self._strobe_indices[:N_COLS]) + [None] * N_COLS)[:N_COLS]
            ]

        ref_frame = next((f for f in src_cols if f is not None), None)
        if ref_frame is None:
            return out

        sf_h, sf_w = ref_frame.shape[:2]
        col_w_src  = sf_w // N_COLS
        frames     = self._rep_frames
        composite  = (frames[0].copy() if frames
                      else np.zeros((sf_h, sf_w, 3), dtype=np.uint8))

        # Dynamic column bounds (half-res coords), or None → fallback to uniform
        dyn_b = self._dyn_col_bounds_half

        for i, kf in enumerate(src_cols[:N_COLS]):
            if kf is None:
                continue
            if dyn_b is not None:
                # Extract the actual stroke-range slice and stretch to column slot
                src_x0 = max(0, int(round(dyn_b[i])))
                src_x1 = min(sf_w, int(round(dyn_b[i + 1])))
                if src_x0 >= src_x1:
                    src_x0 = max(0, int(dyn_b[i]))
                    src_x1 = min(sf_w, src_x0 + max(1, int(dyn_b[i + 1] - dyn_b[i])))
                dst_x0 = i * col_w_src
                dst_x1 = sf_w if i == N_COLS - 1 else dst_x0 + col_w_src
                composite[:, dst_x0:dst_x1] = cv2.resize(
                    kf[:, src_x0:src_x1], (dst_x1 - dst_x0, sf_h))
            else:
                x0 = i * col_w_src
                x1 = sf_w if i == N_COLS - 1 else x0 + col_w_src
                composite[:, x0:x1] = kf[:, x0:x1]

        out[:self.H] = cv2.resize(composite, (self.W, self.H))
        out[self.H:] = (out[self.H:].astype(np.int32) * 55 // 100).astype(np.uint8)

        # ── 2. Détection des têtes ─────────────────────────────────────────
        col_w_disp = self.W  // N_COLS
        centers_disp: list = [None] * N_COLS

        # Taille de boîte par défaut en pixels display
        _DEF_BOX_W = int(self.W / N_COLS * 0.72)
        _DEF_BOX_H = int(self.H * 0.35)

        if self._strobe_det is None:
            # Priorité : réutiliser les centres YOLO capturés pendant l'enregistrement.
            # Avec colonnes dynamiques, mapper half-res x via le remap de colonne.
            det_from_rec = [None] * N_COLS
            for _ci, _ctr in enumerate(self._kf_col_centers):
                if _ctr is not None:
                    if dyn_b is not None:
                        # Map half-res x through dynamic-slice → display coordinates
                        _src_x0_h = max(0.0, dyn_b[_ci])
                        _src_x1_h = min(float(sf_w), dyn_b[_ci + 1])
                        _col_range_h = _src_x1_h - _src_x0_h
                        if _col_range_h > 0:
                            _frac = (_ctr[0] - _src_x0_h) / _col_range_h
                            _disp_x0_c = _ci * col_w_disp
                            _disp_x1_c = (col_w_disp * (_ci + 1)
                                          if _ci < N_COLS - 1 else self.W)
                            _dcx = int(_disp_x0_c + _frac * (_disp_x1_c - _disp_x0_c))
                        else:
                            _dcx = _ci * col_w_disp + col_w_disp // 2
                        _dcy = int(_ctr[1] * self.H / sf_h)
                    else:
                        _dcx = int(_ctr[0] * 2)
                        _dcy = int(_ctr[1] * 2)
                    det_from_rec[_ci] = (
                        _dcx, _dcy,
                        float(_DEF_BOX_W), float(_DEF_BOX_H), 0.0,
                    )
            if any(d is not None for d in det_from_rec):
                self._strobe_det = det_from_rec
            else:
                # Fallback : YOLO sur l'image composite assemblée
                self._strobe_det = self._run_yolo_strobe(out[:self.H].copy())

        for i, box in enumerate(self._strobe_det):
            if box is None:
                continue
            cx, cy, bw, bh, angle_deg = box
            dcx = int(cx)
            dcy = int(cy)
            dw  = max(float(bw), _DEF_BOX_W * 0.5)
            dh  = max(float(bh), _DEF_BOX_H * 0.5)
            pts = cv2.boxPoints(((float(dcx), float(dcy)), (dw, dh), angle_deg))
            self._draw_rounded_box(out, pts, WHITE, thickness=2, radius=8)
            centers_disp[i] = (dcx, dcy)

        # ── 3. Putter arc: thick red spline through all column centres ─────
        def _fill_col_gaps(pts, n_cols, cw):
            """Interpolate missing columns + reject y-outliers.

            Pass 1 – fit a polynomial to valid y values and reject points
                      that deviate more than 75 px from the trend.
            Pass 2 – fill remaining None slots with linear y-interpolation
                      so the arc never breaks.  The arc can only move in
                      one direction (column order = left → right).
            """
            result = list(pts)
            # Pass 1: outlier rejection via polynomial fit on column index
            valid = [(i, p) for i, p in enumerate(result) if p is not None]
            if len(valid) >= 3:
                vi = np.array([i for i, _ in valid], float)
                vy = np.array([p[1] for _, p in valid], float)
                degree = min(2, len(valid) - 1)
                coeffs = np.polyfit(vi, vy, degree)
                for i, p in list(valid):
                    if abs(p[1] - np.polyval(coeffs, i)) > 75:
                        result[i] = None   # outlier – drop it
            # Pass 2: gap fill by linear interpolation in column order
            valid_idx = [i for i, p in enumerate(result) if p is not None]
            if len(valid_idx) < 2:
                return result
            first, last = valid_idx[0], valid_idx[-1]
            for i in range(first, last + 1):
                if result[i] is not None:
                    continue
                prev_i = max(j for j in valid_idx if j < i)
                next_i = min(j for j in valid_idx if j > i)
                t = (i - prev_i) / (next_i - prev_i)
                _, py = result[prev_i]
                _, ny = result[next_i]
                expected_x = int((i + 0.5) * cw)   # column centre x
                result[i] = (expected_x, int(py + t * (ny - py)))
            return result

        def _normalize_x(pts, col_w):
            """Rescale x-coordinates so the leftmost point → col1 centre,
            rightmost point → col7 centre.  Y-coordinates are unchanged."""
            valid_xs = [p[0] for p in pts if p is not None]
            if len(valid_xs) < 2:
                return pts
            x_min, x_max = min(valid_xs), max(valid_xs)
            if x_max <= x_min:
                return pts
            col1_cx = 0.5 * col_w         # col 1 centre  (0-indexed col 0)
            col7_cx = 6.5 * col_w         # col 7 centre  (0-indexed col 6)
            result = []
            for p in pts:
                if p is None:
                    result.append(None)
                else:
                    new_x = col1_cx + (p[0] - x_min) / (x_max - x_min) * (col7_cx - col1_cx)
                    result.append((int(round(new_x)), p[1]))
            return result

        def _spline_through(pts_list, n_fine=500, col_w=None):
            """Cubic spline with horizontal tangents (dy/dt = 0) at both extremes
            and at the impact column (col 4, 0-indexed col 3).

            The arc is split into two segments at the impact point so that:
              • backswing extreme  → horizontal tangent (arc is "taut" to the left)
              • col-4 impact point → horizontal tangent (arc tangent to baseline)
              • follow-through extreme → horizontal tangent (arc is "taut" to the right)
            """
            valid = [(i, p) for i, p in enumerate(pts_list) if p is not None]
            if len(valid) < 2:
                return None
            xs = np.array([p[0] for _, p in valid], float)
            ys = np.array([p[1] for _, p in valid], float)
            # Enforce strictly increasing x so the arc never reverses direction
            for j in range(1, len(xs)):
                if xs[j] <= xs[j - 1]:
                    xs[j] = xs[j - 1] + 1.0

            n = len(xs)

            # Find impact split point (x closest to col-4 centre = 3.5 * col_w)
            imp_idx = None
            if col_w is not None and n >= 3:
                imp_x = 3.5 * col_w
                imp_idx = int(np.argmin(np.abs(xs - imp_x)))
                if imp_idx == 0 or imp_idx == n - 1:
                    imp_idx = None   # degenerate – fall back to single segment

            if imp_idx is None:
                # Single segment: clamp y-derivative to 0 at both endpoints
                t_k = np.linspace(0.0, 1.0, n)
                try:
                    cs_x = CubicSpline(t_k, xs)
                    cs_y = CubicSpline(t_k, ys, bc_type=((1, 0.0), (1, 0.0)))
                except Exception:
                    return None
                t_f = np.linspace(0.0, 1.0, n_fine)
                return np.stack([cs_x(t_f), cs_y(t_f)], axis=1).astype(np.int32).reshape(-1, 1, 2)

            # Two segments: left (0..imp_idx) and right (imp_idx..n-1)
            xs_L, ys_L = xs[:imp_idx + 1], ys[:imp_idx + 1]
            xs_R, ys_R = xs[imp_idx:],     ys[imp_idx:]
            t_L = np.linspace(0.0, 1.0, len(xs_L))
            t_R = np.linspace(0.0, 1.0, len(xs_R))
            n_L = n_fine // 2
            n_R = n_fine - n_L
            try:
                cs_xL = CubicSpline(t_L, xs_L)
                cs_yL = CubicSpline(t_L, ys_L, bc_type=((1, 0.0), (1, 0.0)))
                cs_xR = CubicSpline(t_R, xs_R)
                cs_yR = CubicSpline(t_R, ys_R, bc_type=((1, 0.0), (1, 0.0)))
            except Exception:
                return None
            t_fL = np.linspace(0.0, 1.0, n_L)
            t_fR = np.linspace(0.0, 1.0, n_R)
            pts_L = np.stack([cs_xL(t_fL), cs_yL(t_fL)], axis=1)
            pts_R = np.stack([cs_xR(t_fR), cs_yR(t_fR)], axis=1)
            all_pts = np.vstack([pts_L, pts_R[1:]])   # skip duplicate junction point
            return all_pts.astype(np.int32).reshape(-1, 1, 2)

        centers_filled   = _fill_col_gaps(centers_disp, N_COLS, col_w_disp)
        centers_filled   = _normalize_x(centers_filled, col_w_disp)
        smooth = _spline_through(centers_filled, col_w=col_w_disp)
        if smooth is not None:
            self._draw_tapered_arc(out, smooth,
                                   shadow_color=(0, 0, 50),
                                   main_color=RED, max_thick=7)

        # ── 4. Ball trajectory (impact col 4 → dernière pos balle col 1) ───
        ball_init = self._ball_init_pos
        if ball_init is None and self._ball_zone_rect is not None:
            _bx, _by, _bw, _bh = self._ball_zone_rect
            ball_init = (_bx + _bw // 2, _by + _bh // 2)

        if ball_init is not None:
            bix, biy = int(ball_init[0]), int(ball_init[1])

            # Dernière position réelle de la balle après impact (half-res × 2 → display)
            ball_last = self._ball_last_pos
            if ball_last is not None:
                tgt_x = int(ball_last[0] * 2)
                tgt_y = int(ball_last[1] * 2)
            else:
                # Fallback : centre de la col 1 à la même hauteur que l'impact
                tgt_x = col_w_disp // 2
                tgt_y = biy

            # Ligne cyan : position impact → dernière position balle
            cv2.line(out, (bix, biy), (tgt_x, tgt_y), (200, 80, 0), 2, cv2.LINE_AA)

            # Grand cercle cyan à la dernière position balle (direction de la balle)
            cv2.circle(out, (tgt_x, tgt_y), 22, (0, 0, 0),  -1, cv2.LINE_AA)
            cv2.circle(out, (tgt_x, tgt_y), 20, CYAN,         2, cv2.LINE_AA)

            # Petit cercle cyan à la position d'impact (col 4)
            cv2.circle(out, (bix, biy), 14, (0, 0, 0),  -1, cv2.LINE_AA)
            cv2.circle(out, (bix, biy), 12, CYAN,         2, cv2.LINE_AA)

        # ── 5. Putter-head centre markers (gold dot + white ring) ─────────
        # Draw actual detections (solid); interpolated/filtered positions
        # (from centers_filled) are shown with a dashed ring so the user
        # can distinguish measured vs. estimated columns.
        actual_set = {i for i, p in enumerate(centers_disp) if p is not None}
        for i, pt in enumerate(centers_filled):
            if pt is None:
                continue
            if i in actual_set:
                # Actual detection – solid ring
                cv2.circle(out, pt,  9, (0,   0,   0),   -1, cv2.LINE_AA)
                cv2.circle(out, pt,  7, (30, 160, 220),   -1, cv2.LINE_AA)
                cv2.circle(out, pt,  9, (200, 200, 200),   1, cv2.LINE_AA)
            else:
                # Interpolated / filtered – smaller dashed ring
                cv2.circle(out, pt,  6, (0,   0,   0),   -1, cv2.LINE_AA)
                cv2.circle(out, pt,  5, (30, 100, 160),   -1, cv2.LINE_AA)
                cv2.circle(out, pt,  6, (120, 120, 120),   1, cv2.LINE_AA)

        # ── 6. Column dividers ─────────────────────────────────────────────
        for i in range(1, N_COLS):
            cv2.line(out, (i * col_w_disp, 0),
                     (i * col_w_disp, self.H + PANEL_H), (55, 55, 55), 1)

        # ── 7. Impact column: red vertical line at column 4 ───────────────
        imp_cx = 3 * col_w_disp + col_w_disp // 2
        cv2.line(out, (imp_cx, 0), (imp_cx, self.H), (0, 0, 200), 2, cv2.LINE_AA)

        # ── 8. Horizontal baseline through putter-head average y ──────────
        valid_c = [pt for pt in centers_disp if pt is not None]
        if valid_c:
            avg_y = int(np.mean([pt[1] for pt in valid_c]))
            cv2.line(out, (0, avg_y), (self.W, avg_y), (100, 120, 180), 1, cv2.LINE_AA)

        # ── 9. Dark header bar (reference UI style) ────────────────────────
        cv2.rectangle(out, (0, 0), (self.W, HDR_H), (25, 40, 65), -1)
        cv2.line(out, (0, HDR_H), (self.W, HDR_H), (0, 110, 180), 1)

        def _angle_str(v):
            if v is None:
                return "--"
            side = "R" if v > 0.05 else ("L" if v < -0.05 else "")
            return f"{side}{abs(v):.1f}°"

        r_res = self.result
        face_str = _angle_str(r_res.face_impact if r_res else None)
        path_str = _angle_str(r_res.launch_dir  if r_res else None)
        arc_str  = r_res.arc_class if r_res else "--"
        arc_clr  = ((30, 200,  60) if r_res and r_res.arc_class == "Straight" else
                    (30, 200, 200) if r_res and r_res.arc_class == "Slight Arc" else
                    (0, 140, 255))

        # Left: title
        cv2.putText(out, "ANALYSE  SWING", (12, HDR_H - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, GOLD, 1, cv2.LINE_AA)

        # Centre: metrics
        mx = 205
        for txt, clr in [(f"Face {face_str}", WHITE),
                         (f"Path {path_str}", WHITE),
                         (arc_str,            arc_clr)]:
            (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.putText(out, txt, (mx, HDR_H - 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, txt, (mx, HDR_H - 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, clr, 1, cv2.LINE_AA)
            mx += tw + 22

        # Right: shot counter
        shot_str = f"#{self._shot_count:03d}"
        (sw, _), _ = cv2.getTextSize(shot_str, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.putText(out, shot_str, (self.W - sw - 12, HDR_H - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, GOLD, 1, cv2.LINE_AA)

        # ── 10. Bottom hint ────────────────────────────────────────────────
        hint = "L: retour live   SPACE: repositionner + nouveau coup   T: annoter   Q: quitter"
        (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        cv2.putText(out, hint, (self.W - hw - 8, self.H + PANEL_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 100, 100), 1, cv2.LINE_AA)

        return out

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        fps_buf = deque(maxlen=30)
        t_last  = time.time()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                if self._video_mode:
                    # Loop video back to the beginning
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        print("[video] Impossible de relire la vidéo.")
                        break
                else:
                    print("[error] Camera read failed.")
                    break
            self._last_frame = frame.copy()

            now = time.time()
            fps_buf.append(1.0 / max(1e-6, now - t_last))
            t_last = now
            fps    = float(np.mean(fps_buf))

            # Detect & smooth (ArUco / CSRT)
            raw_pos, raw_angle = self._detect(frame)
            pos   = self._smooth_pos(raw_pos)
            angle = self._smooth_angle(
                (raw_angle - self.ref_angle) if raw_angle is not None else None
            )
            vel = self._velocity(pos, now)

            # ── READY ─────────────────────────────────────────────────────
            if self.state == AppState.READY:
                self._draw_target_line(frame)
                self._draw_guidance(frame)

                # Ghost path
                if self._last_path:
                    self._draw_path(frame, self._last_path,
                                    color=(35, 80, 35), width=1, centerline=False)

                # ── YOLO putter (toutes les 4 frames) ──────────────────────
                self._yolo_tick += 1
                if self._yolo is not None and self._yolo_tick % 4 == 0:
                    try:
                        _half_r = cv2.resize(frame, (self.W // 2, self.H // 2))
                        _preds  = self._yolo.predict(_half_r, conf=0.12, verbose=False)
                        _boxes  = _preds[0].boxes
                        if _boxes is not None and len(_boxes) > 0:
                            _confs = _boxes.conf.tolist()
                            _bi    = int(max(range(len(_confs)), key=lambda k: _confs[k]))
                            _x1h, _y1h, _x2h, _y2h = [int(v) for v in
                                                        _boxes.xyxy[_bi].tolist()]
                            # Remettre en coordonnées pleine résolution
                            self._yolo_ready_box = (_x1h*2, _y1h*2, _x2h*2, _y2h*2)
                        else:
                            self._yolo_ready_box = None
                    except Exception:
                        self._yolo_ready_box = None

                # ── Dessiner zone putter (blanc) + détection par luminosité ─
                putter_in_zone = False
                if self._zone_rect is not None:
                    zx, zy, zw, zh = self._zone_rect
                    _proi = frame[zy:zy + zh, zx:zx + zw]
                    if _proi.size > 0:
                        _pg = cv2.cvtColor(_proi, cv2.COLOR_BGR2GRAY)
                        putter_in_zone = int(np.sum(_pg > BALL_BRIGHT_THR)) > BALL_MIN_PX * 4
                    p_clr = C["white"] if putter_in_zone else (120, 120, 120)
                    cv2.rectangle(frame, (zx, zy), (zx + zw, zy + zh),
                                  p_clr, 2, cv2.LINE_AA)
                    if self._yolo_ready_box is not None:
                        _px1, _py1, _px2, _py2 = self._yolo_ready_box
                        cv2.rectangle(frame, (_px1, _py1), (_px2, _py2),
                                      C["cyan"], 2, cv2.LINE_AA)
                        self._track_mode = "yolo"

                # ── Dessiner zone balle (verte) + détection balle blanche ───
                ball_in_zone = False
                if self._ball_zone_rect is not None:
                    _bx, _by, _bw, _bh = self._ball_zone_rect
                    _ball_roi = frame[_by:_by + _bh, _bx:_bx + _bw]
                    if _ball_roi.size > 0:
                        _gb = cv2.cvtColor(_ball_roi, cv2.COLOR_BGR2GRAY)
                        _, _br2 = cv2.threshold(_gb, BALL_BRIGHT_THR, 255, cv2.THRESH_BINARY)
                        ball_in_zone = cv2.countNonZero(_br2) > BALL_MIN_PX
                    ball_clr = C["green"] if ball_in_zone else (80, 180, 80)
                    cv2.rectangle(frame, (_bx, _by), (_bx + _bw, _by + _bh),
                                  ball_clr, 2, cv2.LINE_AA)

                # ── Auto-start : putter + balle présents simultanément ──────
                both = putter_in_zone and ball_in_zone
                if both:
                    if self._both_since is None:
                        self._both_since = now
                    elapsed = now - self._both_since
                    prog    = min(1.0, elapsed / STILL_SECS)
                    # Anneau de progression sur la zone putter
                    if self._zone_rect is not None:
                        _zc = (zx + zw // 2, zy + zh // 2)
                        cv2.ellipse(frame, _zc, (30, 30), -90, 0, int(360 * prog),
                                    C["green"], 3, cv2.LINE_AA)
                        self._put(frame, f"{elapsed:.1f}s / {STILL_SECS:.0f}s",
                                  (_zc[0] + 35, _zc[1]), scale=0.45, color=C["green"])
                    if elapsed >= STILL_SECS:
                        self._both_since = None
                        self.state       = AppState.COUNTDOWN
                        self._cd_start   = now
                else:
                    self._both_since = None
                    if pos:
                        cv2.circle(frame, pos, 7, C["green"], -1)

                extra = [f"Face : {angle:+.1f}°"] if angle is not None else []
                self._draw_hud(frame, fps, extra)

            # ── ZONE_SETUP ────────────────────────────────────────────────
            elif self.state == AppState.ZONE_SETUP:
                if self._zone_setup_start and self._zone_setup_end:
                    cv2.rectangle(frame,
                                  self._zone_setup_start, self._zone_setup_end,
                                  C["white"], 2, cv2.LINE_AA)
                _setup_msg = (
                    "Glisser pour definir la ZONE BALLE  (autour de la balle au depart)"
                    if self._setup_which == "ball" else
                    "Glisser pour definir la ZONE PUTTER  (autour de la tete au repos)"
                )
                self._put(frame, _setup_msg, (14, 55), scale=0.65, color=C["white"])
                self._draw_hud(frame, fps)

            # ── COUNTDOWN ─────────────────────────────────────────────────
            elif self.state == AppState.COUNTDOWN:
                rem = COUNTDOWN_SECS - (now - self._cd_start)
                self._draw_target_line(frame)
                if pos:
                    cv2.circle(frame, pos, 7, C["white"], -1)
                # Also show zones during countdown
                if self._ball_zone_rect is not None:
                    bx, by, bw, bh = self._ball_zone_rect
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh),
                                  C["green"], 1)
                if self._zone_rect is not None:
                    zx, zy, zw, zh = self._zone_rect
                    cv2.rectangle(frame, (zx, zy), (zx + zw, zy + zh),
                                  C["white"], 1)
                self._draw_countdown(frame, rem)
                self._draw_hud(frame, fps)
                if rem <= 0:
                    self.state          = AppState.RECORDING
                    self._rec_start     = now
                    self.records        = []
                    self._rep_frames    = []
                    self._rep_positions = []
                    self._rep_ctr       = 0
                    self._pos_buf.clear()
                    self._ang_buf.clear()
                    self._kf_col_frames   = [None] * 7
                    self._kf_col_offs     = [float('inf')] * 7
                    self._kf_col_centers  = [None] * 7
                    self._rec_bg_live     = None
                    self._ball_roi_ref    = None
                    self._ball_moved      = False
                    self._half_buf.clear()
                    self._ball_init_pos   = None
                    self._ball_last_pos   = None
                    # Initialiser le tracking YOLO depuis le centre de la zone putter
                    # pour éviter de partir sur un putter du rack
                    self._last_yolo_half_pos = None
                    if self._zone_rect is not None:
                        _zx, _zy, _zw, _zh = self._zone_rect
                        self._last_yolo_half_pos = (
                            (_zx + _zw // 2) // 2,
                            (_zy + _zh // 2) // 2,
                        )
                    elif self._yolo_ready_box is not None:
                        _rx1, _ry1, _rx2, _ry2 = self._yolo_ready_box
                        self._last_yolo_half_pos = (
                            int((_rx1 + _rx2) / 4),
                            int((_ry1 + _ry2) / 4),
                        )

            # ── RECORDING ─────────────────────────────────────────────────
            elif self.state == AppState.RECORDING:
                rem = RECORD_SECS - (now - self._rec_start)

                # Record tracked position (ArUco / CSRT)
                if pos is not None:
                    self.records.append(FrameRec(
                        ts=now, pos=pos, angle=angle or 0.0, vel=vel
                    ))

                # Zone-position also feeds column capture
                zone_pos_rec = self._detect_in_zone(frame)
                track_pos = zone_pos_rec or pos   # prefer zone centroid

                # Per-column live capture using YOLO detection
                _half  = cv2.resize(frame, (self.W // 2, self.H // 2))
                _yolo_cx_h: Optional[int] = None
                _yolo_cy_h: Optional[int] = None

                # ── YOLO tracking toutes les 2 frames sur _half ─────────────
                if self._yolo is not None and self._rep_ctr % 2 == 0:
                    try:
                        _pr = self._yolo.predict(_half, conf=0.08, verbose=False)
                        _br = _pr[0].boxes
                        if _br is not None and len(_br) > 0:
                            _xywh_list = _br.xywh.tolist()
                            _conf_list  = _br.conf.tolist()
                            if self._last_yolo_half_pos is not None:
                                # Continuité : prendre la détection la plus proche
                                # de la dernière position connue (évite de sauter
                                # sur un putter du rack après l'impact)
                                _lx, _ly = self._last_yolo_half_pos
                                _ci = min(range(len(_xywh_list)),
                                          key=lambda k: math.hypot(
                                              _xywh_list[k][0] - _lx,
                                              _xywh_list[k][1] - _ly))
                            else:
                                # Première détection : plus haute confiance
                                _ci = int(max(range(len(_conf_list)),
                                              key=lambda k: _conf_list[k]))
                            # Coordonnées en demi-résolution
                            _yolo_cx_h = int(_xywh_list[_ci][0])
                            _yolo_cy_h = int(_xywh_list[_ci][1])
                            self._last_yolo_half_pos = (_yolo_cx_h, _yolo_cy_h)
                            # Pleine résolution pour track_pos et records
                            _ycx = _yolo_cx_h * 2
                            _ycy = _yolo_cy_h * 2
                            track_pos = (_ycx, _ycy)   # meilleure source pour la colonne
                            if pos is None:             # ajouter aux records si pas d'ArUco
                                _rv = (
                                    math.hypot(_ycx - self.records[-1].pos[0],
                                               _ycy - self.records[-1].pos[1])
                                    / max(1e-6, now - self.records[-1].ts)
                                    if self.records else 0.0
                                )
                                self.records.append(
                                    FrameRec(ts=now, pos=(_ycx, _ycy), angle=0.0, vel=_rv))
                    except Exception:
                        pass

                # ── Collecte détections YOLO pour colonnes dynamiques ─────────
                if _yolo_cx_h is not None:
                    if self._address_x_half is None:
                        self._address_x_half = float(_yolo_cx_h)
                    self._all_live_detections.append(
                        (_yolo_cx_h, _yolo_cy_h, _half.copy()))

                _gray_h = cv2.GaussianBlur(
                    cv2.cvtColor(_half, cv2.COLOR_BGR2GRAY), (21, 21), 0)
                if self._rec_bg_live is None:
                    self._rec_bg_live = _gray_h.astype(np.float32)
                    _bh, _bw = _half.shape[:2]
                    if self._ball_zone_rect is not None:
                        _bzx, _bzy, _bzw, _bzh = self._ball_zone_rect
                        _bry1 = max(0, _bzy // 2);  _bry2 = min(_bh, (_bzy + _bzh) // 2)
                        _brx1 = max(0, _bzx // 2);  _brx2 = min(_bw, (_bzx + _bzw) // 2)
                    else:
                        _bry1, _bry2 = _bh // 2 - 20, _bh // 2 + 20
                        _brx1, _brx2 = _bw // 2 - 40, _bw // 2 + 40
                    _roi_slice = _gray_h[_bry1:_bry2, _brx1:_brx2]
                    self._ball_roi_ref = _roi_slice.astype(np.float32) if _roi_slice.size > 0 else None
                    # Store full-res ball center for trajectory display
                    self._ball_init_pos = (
                        (_brx1 + _brx2) * 1,   # already half-res x; scale ×2 below
                        (_bry1 + _bry2) * 1,
                    )
                    if self._ball_zone_rect is not None:
                        _bzx, _bzy, _bzw, _bzh = self._ball_zone_rect
                        self._ball_init_pos = (_bzx + _bzw // 2, _bzy + _bzh // 2)
                else:
                    # ── Background subtraction → colonne par mouvement ──────
                    _diff_h = np.abs(_gray_h.astype(np.float32) - self._rec_bg_live)
                    _, _th_h = cv2.threshold(
                        _diff_h.astype(np.uint8), 8, 255, cv2.THRESH_BINARY)
                    _th_h = cv2.dilate(_th_h, None, iterations=3)
                    _cnts_h, _ = cv2.findContours(
                        _th_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if _cnts_h:
                        _c_h = max(_cnts_h, key=cv2.contourArea)
                        if cv2.contourArea(_c_h) >= 60:
                            _M_h = cv2.moments(_c_h)
                            if _M_h["m00"] > 0:
                                _cx_h   = int(_M_h["m10"] / _M_h["m00"])
                                _cy_h   = int(_M_h["m01"] / _M_h["m00"])
                                # Si YOLO a détecté cette frame, affiner le centre
                                if _yolo_cx_h is not None:
                                    _cx_h = _yolo_cx_h
                                    _cy_h = _yolo_cy_h
                                # Collecte pour colonnes dynamiques (bg-sub si YOLO absent)
                                if _yolo_cx_h is None:
                                    if self._address_x_half is None:
                                        self._address_x_half = float(_cx_h)
                                    self._all_live_detections.append(
                                        (_cx_h, _cy_h, _half.copy()))
                    # Ball impact detection
                    if (not self._ball_moved
                            and bool(self._all_live_detections)
                            and self._ball_roi_ref is not None):
                        _bh2, _bw2 = _half.shape[:2]
                        if self._ball_zone_rect is not None:
                            _bzx, _bzy, _bzw, _bzh = self._ball_zone_rect
                            _bry1 = max(0, _bzy // 2);  _bry2 = min(_bh2, (_bzy + _bzh) // 2)
                            _brx1 = max(0, _bzx // 2);  _brx2 = min(_bw2, (_bzx + _bzw) // 2)
                        else:
                            _bry1, _bry2 = _bh2 // 2 - 20, _bh2 // 2 + 20
                            _brx1, _brx2 = _bw2 // 2 - 40, _bw2 // 2 + 40
                        _roi_now = _gray_h[_bry1:_bry2, _brx1:_brx2].astype(np.float32)
                        if np.mean(np.abs(_roi_now - self._ball_roi_ref)) > 15.0:
                            self._ball_moved = True
                            # Use the frame from ~3 frames ago to catch putter AT impact,
                            # not already past the ball.
                            early = (self._half_buf[0]
                                     if len(self._half_buf) >= 3
                                     else _half)
                            self._kf_col_frames[3]  = early
                            self._kf_col_offs[3]    = 0.0
                            # centroid = ball zone centre (putter is at ball)
                            if self._ball_zone_rect is not None:
                                _bzx, _bzy, _bzw, _bzh = self._ball_zone_rect
                                self._kf_col_centers[3] = (
                                    (_bzx + _bzw // 2) // 2,  # half-res
                                    (_bzy + _bzh // 2) // 2,
                                )

                # ── Suivi de la balle après l'impact ─────────────────────────
                # Cherche le blob blanc (balle) dans la moitié du frame
                # correspondant à la direction du coup (côté opposé à l'adresse)
                if self._ball_moved:
                    _bh2, _bw2 = _half.shape[:2]
                    # Zone de recherche : toute la largeur, bande horizontale de la balle
                    if self._ball_zone_rect is not None:
                        _bzx2, _bzy2, _bzw2, _bzh2 = self._ball_zone_rect
                        _sry1 = max(0, _bzy2 // 2 - 20)
                        _sry2 = min(_bh2, (_bzy2 + _bzh2) // 2 + 20)
                    else:
                        _sry1, _sry2 = _bh2 // 2 - 30, _bh2 // 2 + 30
                    _band = _gray_h[_sry1:_sry2, :]
                    # Seuil luminosité élevée → balle blanche
                    _, _ball_mask = cv2.threshold(_band, BALL_BRIGHT_THR, 255, cv2.THRESH_BINARY)
                    _ball_cnts, _ = cv2.findContours(
                        _ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if _ball_cnts:
                        _bc = max(_ball_cnts, key=cv2.contourArea)
                        if cv2.contourArea(_bc) >= BALL_MIN_PX:
                            _bM = cv2.moments(_bc)
                            if _bM["m00"] > 0:
                                _bcx = int(_bM["m10"] / _bM["m00"])
                                _bcy = int(_bM["m01"] / _bM["m00"]) + _sry1
                                # Stocker en coordonnées half-res
                                self._ball_last_pos = (_bcx, _bcy)

                # Ring buffer for impact-timing look-back
                self._half_buf.append(_half.copy())
                self._rep_ctr += 1
                if self._rep_ctr % self._replay_sub == 0:
                    self._rep_frames.append(_half.copy())
                    self._rep_positions.append(track_pos)

                self._draw_target_line(frame)
                if self.records:
                    self._draw_path(frame, [r.pos for r in self.records])
                if track_pos:
                    cv2.circle(frame, track_pos, 8, C["green"], -1)

                prog = max(0.0, 1.0 - rem / RECORD_SECS)
                cv2.rectangle(frame, (0, self.H - 4),
                              (int(self.W * prog), self.H), C["green"], -1)

                extra = [f"REC {rem:.1f}s"]
                if angle is not None:
                    extra.append(f"Face : {angle:+.1f}°")
                self._draw_hud(frame, fps, extra)

                if rem <= 0:
                    self.state = AppState.ANALYSIS

            # ── ANALYSIS ──────────────────────────────────────────────────
            elif self.state == AppState.ANALYSIS:
                self.result = self._analyze()
                if self.result:
                    self._save_json(self.result)
                    self._last_path = list(self.result.positions)
                self._compute_dynamic_columns()   # dynamic column boundaries from stroke
                self._kf_init    = False
                self._strobe_det = None
                self._kf_done   = False
                self._shot_count += 1
                self.state      = AppState.KEYFRAMES

            # ── REPLAY ────────────────────────────────────────────────────
            elif self.state == AppState.REPLAY:
                r = self.result
                if self._rep_frames:
                    if now - self._rep_last >= 1.0 / REPLAY_FPS:
                        next_idx = (self._rep_idx + 1) % len(self._rep_frames)
                        if next_idx == 0:
                            self._rep_loops += 1
                        self._rep_idx  = next_idx
                        self._rep_last = now
                        if self._rep_loops >= 1:
                            self.state = AppState.KEYFRAMES
                    frame = cv2.resize(self._rep_frames[self._rep_idx],
                                       (self.W, self.H))
                self._draw_target_line(frame)
                if r:
                    self._draw_path(frame, r.positions)
                    if 0 <= r.impact_idx < len(r.positions):
                        imp_pos = r.positions[r.impact_idx]
                        cv2.circle(frame, imp_pos, 14, C["yellow"], 2)
                        self._draw_face_arrow(frame, imp_pos, r.face_impact)
                    self._draw_results(frame, r)
                hud_extra = ["REPLAY"]
                if not r:
                    hud_extra.append("No tracking — draw zone (Z) or press F")
                self._draw_hud(frame, fps, hud_extra)

            # ── KEYFRAMES ─────────────────────────────────────────────────
            elif self.state == AppState.KEYFRAMES:
                if not self._kf_init:
                    self._kf_init      = True
                    self._kf_idx       = 0
                    self._kf_last      = now
                    self._kf_done      = False
                    self._kf_cur_col   = None
                    self._kf_flash_col = -1
                    self._kf_flash_t   = 0.0
                    self._kf_best_off  = {}
                    self._strobe_indices = [None] * 7
                    if self._rep_frames:
                        bg_raw      = cv2.cvtColor(self._rep_frames[0], cv2.COLOR_BGR2GRAY)
                        self._kf_bg = cv2.GaussianBlur(bg_raw, (21, 21), 0).astype(np.float32)

                KF_FPS  = 6
                N_COLS  = 7
                vid_h   = self.H
                col_w_d = self.W // N_COLS

                if not self._kf_done and self._rep_frames:
                    if now - self._kf_last >= 1.0 / KF_FPS:
                        self._kf_last = now
                        det = self._detect_putter_col(self._kf_idx)
                        if det is not None:
                            col, offset = det
                            self._kf_cur_col = col
                            best = self._kf_best_off.get(col, float('inf'))
                            if offset < best:
                                self._kf_best_off[col]    = offset
                                self._strobe_indices[col] = self._kf_idx
                                self._kf_flash_col        = col
                                self._kf_flash_t          = now
                        else:
                            self._kf_cur_col = None
                        self._kf_idx += 1
                        if self._kf_idx >= len(self._rep_frames):
                            self._kf_done = True

                if not self._kf_done:
                    f_idx  = min(self._kf_idx, len(self._rep_frames) - 1)
                    cur_f  = self._rep_frames[f_idx]
                    sf_h2, sf_w2 = cur_f.shape[:2]
                    cw_src = sf_w2 // N_COLS

                    # Gel progressif : colonnes scannées = frame figée, autres = frame courante
                    composite2 = cur_f.copy()
                    for i in range(N_COLS):
                        src_f = None
                        # Priorité : frame capturée en live (RECORDING)
                        if self._kf_col_frames[i] is not None:
                            src_f = self._kf_col_frames[i]
                        # Sinon : meilleur index replay trouvé jusqu'ici
                        elif (self._strobe_indices[i] is not None
                              and self._strobe_indices[i] < len(self._rep_frames)):
                            src_f = self._rep_frames[self._strobe_indices[i]]
                        if src_f is not None:
                            _dyn_b2 = self._dyn_col_bounds_half
                            if _dyn_b2 is not None:
                                _sx0 = max(0, int(round(_dyn_b2[i])))
                                _sx1 = min(sf_w2, int(round(_dyn_b2[i + 1])))
                                _dx0 = i * cw_src
                                _dx1 = sf_w2 if i == N_COLS - 1 else _dx0 + cw_src
                                if _sx0 < _sx1:
                                    composite2[:, _dx0:_dx1] = cv2.resize(
                                        src_f[:, _sx0:_sx1], (_dx1 - _dx0, sf_h2))
                                else:
                                    composite2[:, _dx0:_dx1] = src_f[:, _dx0:_dx1]
                            else:
                                x0 = i * cw_src
                                x1 = sf_w2 if i == N_COLS - 1 else x0 + cw_src
                                composite2[:, x0:x1] = src_f[:, x0:x1]

                    frame = np.zeros((self.H + PANEL_H, self.W, 3), dtype=np.uint8)
                    frame[:vid_h] = cv2.resize(composite2, (self.W, vid_h))
                    for i in range(1, N_COLS):
                        cv2.line(frame, (i * col_w_d, 0),
                                 (i * col_w_d, vid_h), (80, 80, 80), 1)
                else:
                    frame = self._draw_strobe_composite()

            # ── ANNOTATE ──────────────────────────────────────────────────
            elif self.state == AppState.ANNOTATE:
                if self._ann_composite is not None:
                    frame = self._ann_composite.copy()
                else:
                    frame = np.zeros((self.H + PANEL_H, self.W, 3), dtype=np.uint8)

                # In-progress box preview
                if self._ann_drawing and self._ann_drag_start and self._ann_drag_end:
                    cv2.rectangle(frame, self._ann_drag_start, self._ann_drag_end,
                                  (200, 100, 0), 2)

                # All annotated boxes
                for i, box in enumerate(self._ann_boxes):
                    cx, cy, bw, bh, angle = box
                    self._draw_ann_box(frame, cx, cy, bw, bh, angle,
                                       selected=(i == self._ann_sel_box))

                # Arc through box centers
                centers = [(b[0], b[1]) for b in self._ann_boxes]
                self._draw_ann_arc(frame, centers)

                # Instructions panel
                hints = [
                    "Click-drag: draw box   Scroll-wheel: rotate   Right-click: delete",
                    f"S = save  ({len(self._ann_boxes)} box{'es' if len(self._ann_boxes) != 1 else ''})   Esc = back",
                ]
                for i, h in enumerate(hints):
                    self._put(frame, h, (14, self.H + 30 + i * 28),
                              scale=0.48, color=C["gray"])

            # ── ROI selection overlay ──────────────────────────────────────
            if self._sel_mode:
                self._draw_sel_rect(frame)

            # ── Debug overlay ──────────────────────────────────────────────
            if self.debug:
                self._draw_debug(frame)

            # Pad to full window height
            if frame.shape[0] == self.H:
                padded = np.zeros((self.H + PANEL_H, self.W, 3), dtype=np.uint8)
                padded[:self.H] = frame
                frame = padded
            cv2.imshow("Putter Live", frame)

            # ── Key handling ──────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (27,):   # Esc
                if self.state == AppState.ANNOTATE:
                    self.state = AppState.KEYFRAMES
                else:
                    break

            elif key in (ord('q'), ord('Q')):
                break

            elif key == ord(' '):
                if self.state == AppState.READY:
                    # READY → démarrer le compte à rebours
                    self.state     = AppState.COUNTDOWN
                    self._cd_start = now
                elif self.state in (AppState.REPLAY, AppState.KEYFRAMES):
                    # Après un coup : revenir en READY pour repositionner
                    self.state          = AppState.READY
                    self.records        = []
                    self._rep_frames    = []
                    self._rep_positions = []
                    self.result         = None
                    self._pos_buf.clear()
                    self._ang_buf.clear()
                    self._kf_init        = False
                    self._strobe_indices = []
                    self._strobe_det     = None
                    self._kf_col_frames   = [None] * 7
                    self._kf_col_offs     = [float('inf')] * 7
                    self._kf_col_centers  = [None] * 7
                    self._all_live_detections = []
                    self._address_x_half      = None
                    self._dyn_col_bounds_half = None
                    self._rec_bg_live     = None
                    self._ball_roi_ref    = None
                    self._ball_moved      = False
                    self._half_buf.clear()
                    self._ball_init_pos  = None
                    self._ball_last_pos  = None
                    self._last_yolo_half_pos = None
                    print("[reset] READY – repositionnez balle et putter, puis SPACE pour démarrer")

            elif key in (ord('z'), ord('Z')):
                if self.state == AppState.READY:
                    self._setup_which = "putter"
                    self.state = AppState.ZONE_SETUP
                    print("[zone] Drag to draw putter detection zone.")

            elif key in (ord('b'), ord('B')):
                if self.state == AppState.READY:
                    self._setup_which = "ball"
                    self.state = AppState.ZONE_SETUP
                    print("[ball_zone] Drag to draw ball zone.")

            elif key in (ord('t'), ord('T')):
                if self.state == AppState.KEYFRAMES:
                    self._ann_composite = self._draw_strobe_composite()
                    self._ann_boxes     = []
                    self._ann_sel_box   = -1
                    self._ann_drawing   = False
                    self.state          = AppState.ANNOTATE
                    print("[annotate] Draw blue boxes around putter heads.")

            elif key in (ord('l'), ord('L')):
                if self.state == AppState.KEYFRAMES:
                    # Return to live view without full reset
                    self.state = AppState.READY
                    print("[live] Back to READY.")

            elif key in (ord('s'), ord('S')):
                if self.state == AppState.ANNOTATE:
                    self._save_annotation()

            elif key in (ord('c'), ord('C')):
                if self.state == AppState.READY and raw_angle is not None:
                    self.ref_angle = raw_angle
                    self.tgt_angle = raw_angle
                    self._ang_buf.clear()
                    print(f"[cal] ref_angle={self.ref_angle:.1f}°")

            elif key in (ord('d'), ord('D')):
                self.debug = not self.debug

            elif key in (ord('r'), ord('R')):
                self.state          = AppState.READY
                self.records        = []
                self._rep_frames    = []
                self._rep_positions = []
                self.result         = None
                self._pos_buf.clear()
                self._ang_buf.clear()
                self._kf_init        = False
                self._strobe_indices = []
                self._strobe_det     = None
                self._kf_col_frames   = [None] * 7
                self._kf_col_offs     = [float('inf')] * 7
                self._kf_col_centers  = [None] * 7
                self._rec_bg_live     = None
                self._ball_roi_ref    = None
                self._ball_moved      = False
                self._half_buf.clear()
                self._ball_init_pos  = None
                self._ball_last_pos  = None
                self._last_yolo_half_pos = None
                print("[reset]")

            elif key in (ord('f'), ord('F')):
                if self.state == AppState.READY:
                    self._sel_mode = True
                    print("[fallback] Drag to select putter head ROI.")

            elif key in (ord('m'), ord('M')):
                _save_markers(self._aruco_dict)

            elif key in (ord('v'), ord('V')):
                if self.state == AppState.READY:
                    self._open_video_file()

        self.cap.release()
        cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────────
# Launch menu
# ──────────────────────────────────────────────────────────────────────────────

def _show_launch_menu() -> Optional[str]:
    """
    Show a PyQt6 launch menu with two buttons.
    Returns:
      'camera'     → use live camera
      '<filepath>' → use that video file
      None         → user closed the window (quit)
    """
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QFileDialog, QFrame,
    )
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QFont, QColor

    app = QApplication.instance() or QApplication(sys.argv)

    choice: list[Optional[str]] = [None]

    win = QWidget()
    win.setWindowTitle("PutterTrack Pro")
    win.setFixedSize(480, 340)
    win.setStyleSheet("background: #0e0e22;")

    root_layout = QVBoxLayout(win)
    root_layout.setContentsMargins(40, 36, 40, 36)
    root_layout.setSpacing(0)

    # ── Title ────────────────────────────────────────────────────────────────
    title = QLabel("PutterTrack Pro")
    title.setAlignment(Qt.AlignmentFlag.AlignCenter)
    title.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
    title.setStyleSheet("color: #60c060; background: transparent;")
    root_layout.addWidget(title)

    sub = QLabel("Choisissez votre mode d'analyse")
    sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
    sub.setStyleSheet("color: #666; font-size: 12px; background: transparent;")
    root_layout.addWidget(sub)

    root_layout.addSpacing(28)

    # ── Separator ────────────────────────────────────────────────────────────
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setStyleSheet("color: #1a1a3a;")
    root_layout.addWidget(sep)

    root_layout.addSpacing(28)

    # ── Button style ─────────────────────────────────────────────────────────
    BTN = """
        QPushButton {
            background: #14143a;
            color: #ddd;
            border: 2px solid #2a2a6a;
            border-radius: 10px;
            padding: 18px 12px;
            font-size: 15px;
            text-align: left;
        }
        QPushButton:hover  { background: #1e1e52; border-color: #5050b0; color: #fff; }
        QPushButton:pressed{ background: #282870; }
    """

    # ── Camera button ────────────────────────────────────────────────────────
    btn_live = QPushButton("  \U0001f4f7   Caméra en direct")
    btn_live.setStyleSheet(BTN)
    btn_live.setFixedHeight(70)
    btn_live.setCursor(Qt.CursorShape.PointingHandCursor)

    def _pick_camera():
        choice[0] = "camera"
        win.close()

    btn_live.clicked.connect(_pick_camera)
    root_layout.addWidget(btn_live)

    root_layout.addSpacing(16)

    # ── Import video button ───────────────────────────────────────────────────
    btn_video = QPushButton("  \U0001f4c1   Importer une vidéo  (hors live)")
    btn_video.setStyleSheet(BTN)
    btn_video.setFixedHeight(70)
    btn_video.setCursor(Qt.CursorShape.PointingHandCursor)

    def _pick_video():
        path, _ = QFileDialog.getOpenFileName(
            win,
            "Importer une vidéo",
            os.path.expanduser("~"),
            "Fichiers vidéo (*.mp4 *.mov *.avi *.mkv *.m4v);;Tous les fichiers (*.*)",
        )
        if path:
            choice[0] = path
            win.close()

    btn_video.clicked.connect(_pick_video)
    root_layout.addWidget(btn_video)

    root_layout.addSpacing(24)

    # ── Quit hint ────────────────────────────────────────────────────────────
    hint = QLabel("Fermer cette fenêtre pour quitter")
    hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
    hint.setStyleSheet("color: #444; font-size: 10px; background: transparent;")
    root_layout.addWidget(hint)

    win.show()
    app.exec()

    return choice[0]


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    selection = _show_launch_menu()
    if selection is None:
        sys.exit(0)

    try:
        pl = PutterLive()
        if selection != "camera":
            pl._load_video(selection)
        pl.run()
    except RuntimeError as e:
        print(f"\n[error] {e}\n")
        sys.exit(1)
