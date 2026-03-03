#!/usr/bin/env python3
"""
putter_live.py – Real-time golf putter stroke analyser
======================================================
Tracking:
  PRIMARY  – ArUco markers (opencv-contrib required)
  FALLBACK – CSRT / KCF tracker + Hough edge angle

Controls:
  Space  – Start countdown (3-2-1-GO) then record stroke
  C      – Calibrate: current face direction = 0° / target line
  D      – Toggle debug overlay
  R      – Reset session
  F      – Enter fallback-ROI selection (click-drag over putter head)
  M      – Save printable ArUco marker PNGs to ./markers/
  Esc/Q  – Quit
"""
from __future__ import annotations

import cv2
import numpy as np
import json
import math
import os
import sys
import time

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
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
RECORD_SECS      = 2.5    # stroke capture window (seconds)
COUNTDOWN_SECS   = 3      # countdown duration
SMOOTH_WIN       = 5      # moving-average window (frames)
REPLAY_FPS       = 12     # slow-motion replay speed
REPLAY_SUB       = 2      # store every Nth frame for replay (memory saving)
TARGET_FPS       = 60
PANEL_H          = 130    # height of metrics panel below video (px)
ARUCO_DICT_ID    = cv2.aruco.DICT_4X4_50 if hasattr(cv2, "aruco") else None

ARC_STRAIGHT_PX  = 12    # max deviation → "Straight"
ARC_SLIGHT_PX    = 35    # max deviation → "Slight Arc" (else "Strong Arc")

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
    COUNTDOWN  = "COUNTDOWN"
    RECORDING  = "RECORDING"
    ANALYSIS   = "ANALYSIS"
    REPLAY     = "REPLAY"
    KEYFRAMES  = "KEYFRAMES"


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
    """
    Return (detect_fn, dict_obj) where detect_fn(gray) → (corners, ids, _).
    Returns (None, None) if ArUco is unavailable.
    """
    if not hasattr(cv2, "aruco"):
        return None, None
    try:
        d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        p = cv2.aruco.DetectorParameters()
        # More permissive params: detect small/distant markers
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
    """Write two printable ArUco marker PNGs (IDs 0 and 1)."""
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
    print(f"[markers] saved to ./{out_dir}/  (print at 30–50 mm, laminate if possible)")


# ──────────────────────────────────────────────────────────────────────────────
# Main application
# ──────────────────────────────────────────────────────────────────────────────
class PutterLive:

    def __init__(self):
        self.cap = self._open_camera()
        self.W   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._aruco_fn, self._aruco_dict = _build_aruco()
        self.using_aruco = self._aruco_fn is not None

        # Fallback tracker
        self.tracker: Optional[object] = None
        self.roi:     Optional[tuple]  = None

        self._track_mode = "none"   # "aruco" | "csrt" | "none"

        # Strobe composite: one frame index per column (None = not yet captured)
        self._strobe_indices: list = []

        # KEYFRAMES state
        self._kf_init       = False
        self._kf_idx        = 0
        self._kf_last       = 0.0
        self._kf_done       = False
        self._kf_cur_col: Optional[int]  = None  # column where putter is now
        self._kf_flash_col  = -1                 # column to flash (new best)
        self._kf_flash_t    = 0.0                # timestamp of last flash
        self._kf_best_off: dict = {}             # col → best centroid offset
        self._kf_bg: Optional[np.ndarray] = None
        # YOLO detections cache: list of (x1,y1,x2,y2) per column in stored-frame
        # space, or None if no detection. Computed once in phase 2.
        self._strobe_det: Optional[list] = None

        # Load trained YOLOv8 putter detector
        self._yolo = None
        if _YOLO_AVAILABLE and os.path.isfile(_MODEL_PATH):
            self._yolo = _YOLO(_MODEL_PATH)
            print(f"[yolo] model loaded: {_MODEL_PATH}")
        else:
            print("[yolo] model not found or ultralytics missing – boxes disabled")

        self._sel_mode  = False
        self._sel_start: Optional[tuple] = None
        self._sel_end:   Optional[tuple] = None

        # Calibration / target line
        self.ref_angle = 0.0   # raw face angle at calibration (absolute)
        self.tgt_angle = 0.0   # target-line angle on screen (degrees, CCW from +x)

        # State machine
        self.state      = AppState.READY
        self._cd_start  = 0.0
        self._rec_start = 0.0

        # Live recordings
        self.records: list[FrameRec] = []

        # Smoothing buffers
        self._pos_buf = deque(maxlen=SMOOTH_WIN)
        self._ang_buf = deque(maxlen=SMOOTH_WIN)

        # Replay
        self.result: Optional[Result]        = None
        self._rep_frames: list[np.ndarray]   = []
        self._rep_idx  = 0
        self._rep_last = 0.0
        self._rep_ctr  = 0       # subsampling counter
        self._rep_loops = 0      # full replay loops completed

        # Last captured frame (used for CSRT init without timing mismatch)
        self._last_frame: Optional[np.ndarray] = None

        # Ghost path from previous session
        self._last_path: list = []

        # Debug
        self.debug     = False
        self._dbg_info: dict = {}

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
                # Short exposure to freeze motion (reduce blur)
                # V4L2: 1=manual, 3=auto  |  value: log₂ seconds (−7 ≈ 1/128 s)
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                cap.set(cv2.CAP_PROP_EXPOSURE, -7)
                actual = cap.get(cv2.CAP_PROP_EXPOSURE)
                print(f"[cam] exposure={actual}  (increase toward 0 if too dark)")
                return cap

        raise RuntimeError(
            "Cannot open camera.\n"
            "• Make sure no other app is using it.\n"
            "• macOS: grant camera permission in System Settings → Privacy."
        )

    # ── Mouse (ROI selection for fallback) ────────────────────────────────────

    def _on_mouse(self, event, x, y, flags, param):
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
        # Use the last displayed frame so the ROI matches exactly what the user drew on
        frame = self._last_frame
        if frame is None or self.roi is None:
            return
        try:
            self.tracker = cv2.TrackerCSRT_create()
        except AttributeError:
            self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, self.roi)
        self.using_aruco = False
        print("[fallback] CSRT tracker initialised on stored frame.")

    # ── Detection ─────────────────────────────────────────────────────────────

    def _detect(self, frame: np.ndarray) -> tuple[Optional[tuple], Optional[float]]:
        """
        Returns (head_pos, raw_face_angle_deg).
        raw_face_angle is the absolute angle before ref subtraction.
        """
        self._dbg_info = {}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── ArUco primary ──────────────────────────────────────────────────
        if self._aruco_fn is not None:
            corners, ids, _ = self._aruco_fn(gray)
            if ids is not None and len(ids) > 0:
                self.using_aruco  = True
                self._track_mode  = "aruco"
                c = corners[0][0]                         # shape (4, 2)
                center = tuple(c.mean(axis=0).astype(int))
                dx = float(c[1][0] - c[0][0])
                dy = float(c[1][1] - c[0][1])
                raw = math.degrees(math.atan2(-dy, dx))   # screen → math angle
                self._dbg_info = {"corners": c, "marker_id": int(ids[0][0])}
                return center, raw

        # ── CSRT / KCF fallback ───────────────────────────────────────────
        if self.tracker is not None:
            ok, bbox = self.tracker.update(frame)
            if ok:
                x, y, w, h = (int(v) for v in bbox)
                center = (x + w // 2, y + h // 2)
                roi_gray = gray[y:y + h, x:x + w]
                edges = cv2.Canny(roi_gray, 50, 150)
                lines = cv2.HoughLinesP(
                    edges, 1, math.pi / 180, 15,
                    minLineLength=8, maxLineGap=5,
                )
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

    # ── Velocity ──────────────────────────────────────────────────────────────

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
        """Frame where smoothed velocity peaks (= impact moment)."""
        if len(recs) < 3:
            return 0
        vels = np.array([r.vel for r in recs])
        k = min(7, len(vels))
        if k > 1:
            vels = np.convolve(vels, np.ones(k) / k, mode="same")
        return int(np.argmax(vels))

    def _arc_metrics(self, recs: list[FrameRec]):
        """
        Compute signed perpendicular deviations of the head path
        from the target line. Returns (deviations, max_dev, classification).
        """
        if len(recs) < 2:
            return [], 0.0, "Straight"
        pts = np.array([r.pos for r in recs], dtype=float)
        rad = math.radians(self.tgt_angle)
        # Direction along target line
        d    = np.array([math.cos(rad), -math.sin(rad)])
        perp = np.array([-d[1], d[0]])    # perpendicular (signed inside/out)
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
        imp     = self._find_impact(recs)
        devs, mx, cls = self._arc_metrics(recs)
        angles  = [r.angle for r in recs]

        # Path direction at impact: angle of velocity vector vs target line
        launch_dir = 0.0
        if 0 < imp < len(recs) - 1:
            p1 = np.array(recs[imp - 1].pos, dtype=float)
            p2 = np.array(recs[imp + 1].pos, dtype=float)
            dx, dy = p2 - p1
            if dx != 0 or dy != 0:
                path_abs = math.degrees(math.atan2(-dy, dx))
                raw = path_abs - self.tgt_angle
                # normalise to (-180, 180]
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
        ts   = time.strftime("%Y%m%d_%H%M%S")
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
        """Text with drop shadow."""
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
        x1, y1 = p1
        x2, y2 = p2
        dist = math.hypot(x2 - x1, y2 - y1)
        n = max(1, int(dist / dash))
        for i in range(0, n, 2):
            ax = int(x1 + (x2 - x1) * i / n)
            ay = int(y1 + (y2 - y1) * i / n)
            bx = int(x1 + (x2 - x1) * min(i + 1, n) / n)
            by = int(y1 + (y2 - y1) * min(i + 1, n) / n)
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
                       "motion": "Motion", "none": "None ✗"}.get(self._track_mode, "?")
        lines = [
            f"STATE : {self.state.value}",
            f"FPS   : {fps:>4.0f}",
            f"TRACK : {track_label}",
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
        tips = [
            "Place putter head in frame & align face to target line",
            "C = calibrate target   |   F = fallback ROI   |   M = print markers",
            "SPACE = start",
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

    def _draw_results(self, frame, r: Result):
        ph = 90
        y0 = self.H - ph
        ov = frame.copy()
        cv2.rectangle(ov, (0, y0), (self.W, self.H), (18, 18, 18), -1)
        cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)

        # Face angle block
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

        # Arc classification (right side, prominent)
        arc_clr = (C["green"]  if r.arc_class == "Straight" else
                   C["yellow"] if r.arc_class == "Slight Arc" else
                   C["orange"])
        arc_txt = f"{r.arc_class}  ({r.arc_max_px:.0f} px)"
        sz = cv2.getTextSize(arc_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)[0]
        self._put(frame, arc_txt,
                  (self.W - sz[0] - 14, y0 + 55),
                  scale=0.72, color=arc_clr, thick=2)

        # Hint
        self._put(frame, "SPACE: new shot   R: reset   Q: quit",
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

    # ── Keyframe detection helper ─────────────────────────────────────────────

    def _detect_putter_col(self, idx: int):
        """
        Background-subtract frame idx against frame 0.
        Returns (col 0-6, offset_from_center_px) or None.
        offset is how far the centroid is from the column's x-midpoint —
        lower = putter more centred in that column = better keyframe.
        """
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
        cx     = int(M["m10"] / M["m00"])
        sf_w   = self._rep_frames[0].shape[1]
        col_w  = sf_w // 7
        col    = min(cx // col_w, 6)
        offset = abs(cx - (col + 0.5) * col_w)   # distance to column centre
        return col, offset

    # ── Strobe composite ──────────────────────────────────────────────────────

    def _run_yolo_strobe(self) -> list:
        """
        Run YOLOv8 on each of the 7 keyframes and return a list of
        (x1, y1, x2, y2) in stored-frame pixel space, or None per column.
        Only considers boxes whose centroid falls inside the column's x range.
        """
        N_COLS = 7
        frames = self._rep_frames
        if not frames or self._yolo is None:
            return [None] * N_COLS

        sf_w   = frames[0].shape[1]
        col_w  = sf_w // N_COLS
        result = []

        for i, fidx in enumerate(self._strobe_indices[:N_COLS]):
            if fidx is None:
                result.append(None)
                continue

            x0_col = i * col_w
            x1_col = sf_w if i == N_COLS - 1 else x0_col + col_w

            preds = self._yolo.predict(frames[fidx], conf=0.20, verbose=False)
            best  = None        # (x1, y1, x2, y2, conf)
            for box in preds[0].boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                bcx = (bx1 + bx2) / 2
                conf = float(box.conf[0])
                if x0_col <= bcx < x1_col:
                    if best is None or conf > best[4]:
                        best = (bx1, by1, bx2, by2, conf)

            if best:
                result.append(best[:4])
                print(f"[yolo] col {i} → box {[int(v) for v in best[:4]]} "
                      f"conf={best[4]:.2f}")
            else:
                result.append(None)

        return result

    def _draw_strobe_composite(self) -> np.ndarray:
        """
        Stroboscopic composite:
          - Background = first recorded frame (full view, no crop)
          - For each of 7 columns: replace that column's pixels with the
            strip taken from the frame where the putter was at the center
            of that column → putter appears at 7 positions on one image.
          - Bottom panel: Face Angle + Launch Direction metrics.
        """
        frames  = self._rep_frames
        N_COLS  = 7
        n_frames = len(frames)

        vid_h   = self.H
        panel_h = PANEL_H

        out = np.zeros((self.H + PANEL_H, self.W, 3), dtype=np.uint8)

        if n_frames == 0:
            return out

        # ── Work in stored-frame space (W//2 × H//2) ─────────────────────
        sf_w = frames[0].shape[1]   # e.g. 640
        sf_h = frames[0].shape[0]   # e.g. 360
        col_w = sf_w // N_COLS      # column width in stored-frame space

        # Start with first frame as background
        composite = frames[0].copy()

        # Lay in captured columns; uncaptured columns keep the background
        for i, frame_idx in enumerate(self._strobe_indices[:N_COLS]):
            if frame_idx is None:
                continue                   # not yet captured — keep bg
            x0 = i * col_w
            x1 = sf_w if i == N_COLS - 1 else x0 + col_w
            composite[:, x0:x1] = frames[frame_idx][:, x0:x1]

        # Resize composite to display area (no distortion — same aspect ratio)
        out[:vid_h] = cv2.resize(composite, (self.W, vid_h))

        # ── YOLO boxes + arc ──────────────────────────────────────────────
        if self._strobe_det is None:
            self._strobe_det = self._run_yolo_strobe()

        scale_x = self.W  / sf_w
        scale_y = vid_h   / sf_h
        centers_disp = []   # (dx, dy) for valid detections, in display space

        for i, box in enumerate(self._strobe_det):
            if box is None:
                centers_disp.append(None)
                continue
            bx1, by1, bx2, by2 = box
            dx1 = int(bx1 * scale_x);  dy1 = int(by1 * scale_y)
            dx2 = int(bx2 * scale_x);  dy2 = int(by2 * scale_y)
            dcx = (dx1 + dx2) // 2;    dcy = (dy1 + dy2) // 2
            # Bounding box
            cv2.rectangle(out, (dx1, dy1), (dx2, dy2), (0, 255, 100), 2)
            # Centre dot
            cv2.circle(out, (dcx, dcy), 4, (0, 255, 100), -1)
            centers_disp.append((dcx, dcy))

        # Arc through centres (parabolic fit on display coords)
        valid_pts = [(cx, cy) for cx, cy in centers_disp if cx is not None]
        if len(valid_pts) >= 3:
            xs = np.array([p[0] for p in valid_pts], dtype=np.float64)
            ys = np.array([p[1] for p in valid_pts], dtype=np.float64)
            coeffs = np.polyfit(xs, ys, 2)
            x_lo, x_hi = int(xs.min()), int(xs.max())
            arc_xs = np.linspace(x_lo, x_hi, 300)
            arc_ys = np.polyval(coeffs, arc_xs)
            arc_pts = np.array(
                [[int(x), int(y)] for x, y in zip(arc_xs, arc_ys)
                 if 0 <= y < vid_h],
                dtype=np.int32
            )
            if len(arc_pts) >= 2:
                cv2.polylines(out, [arc_pts.reshape(-1, 1, 2)],
                              False, (0, 180, 255), 3, cv2.LINE_AA)

        # Column dividers
        col_w_disp = self.W // N_COLS
        for i in range(1, N_COLS):
            cv2.line(out, (i * col_w_disp, 0), (i * col_w_disp, vid_h),
                     (60, 60, 60), 1)

        # thin separator line
        cv2.line(out, (0, vid_h), (self.W, vid_h), (60, 60, 60), 1)

        # ── metrics panel ─────────────────────────────────────────────────
        r = self.result

        def _angle_str(v: Optional[float]) -> str:
            if v is None:
                return "--"
            side = "R" if v > 0.05 else ("L" if v < -0.05 else "")
            return f"{side}{abs(v):.1f}°"

        face_str = _angle_str(r.face_impact if r else None)
        path_str = _angle_str(r.launch_dir  if r else None)

        col_l = self.W // 4          # centre of left column
        col_r = 3 * self.W // 4     # centre of right column
        lbl_y = vid_h + int(panel_h * 0.30)
        val_y = vid_h + int(panel_h * 0.80)

        for label, value, cx in [("Face Angle", face_str, col_l),
                                   ("Launch Direction", path_str, col_r)]:
            # label (small, grey)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)
            cv2.putText(out, label, (cx - tw // 2, lbl_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (160, 160, 160), 1, cv2.LINE_AA)
            # value (large, white)
            scale = 2.8
            thick = 4
            (vw, vh), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            cv2.putText(out, value, (cx - vw // 2, val_y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)

        return out

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        fps_buf = deque(maxlen=30)
        t_last  = time.time()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[error] Camera read failed – check connection.")
                break
            self._last_frame = frame.copy()   # keep for CSRT init

            now = time.time()
            fps_buf.append(1.0 / max(1e-6, now - t_last))
            t_last = now
            fps    = float(np.mean(fps_buf))

            # ── Detect & smooth ───────────────────────────────────────────
            raw_pos, raw_angle = self._detect(frame)
            pos   = self._smooth_pos(raw_pos)
            angle = self._smooth_angle(
                (raw_angle - self.ref_angle) if raw_angle is not None else None
            )
            vel = self._velocity(pos, now)

            # ── State machine ─────────────────────────────────────────────

            # ── READY ─────────────────────────────────────────────────────
            if self.state == AppState.READY:
                self._draw_target_line(frame)
                self._draw_guidance(frame)
                if self._last_path:
                    self._draw_path(frame, self._last_path,
                                    color=(35, 80, 35), width=1, centerline=False)
                if pos:
                    cv2.circle(frame, pos, 7, C["green"], -1)
                extra = [f"Face : {angle:+.1f}°"] if angle is not None else []
                self._draw_hud(frame, fps, extra)

            # ── COUNTDOWN ─────────────────────────────────────────────────
            elif self.state == AppState.COUNTDOWN:
                rem = COUNTDOWN_SECS - (now - self._cd_start)
                self._draw_target_line(frame)
                if pos:
                    cv2.circle(frame, pos, 7, C["white"], -1)
                self._draw_countdown(frame, rem)
                self._draw_hud(frame, fps)
                if rem <= 0:
                    self.state      = AppState.RECORDING
                    self._rec_start = now
                    self.records    = []
                    self._rep_frames = []
                    self._rep_ctr   = 0
                    self._pos_buf.clear()
                    self._ang_buf.clear()

            # ── RECORDING ─────────────────────────────────────────────────
            elif self.state == AppState.RECORDING:
                rem = RECORD_SECS - (now - self._rec_start)
                if pos is not None:
                    self.records.append(FrameRec(
                        ts=now, pos=pos, angle=angle or 0.0, vel=vel
                    ))
                self._rep_ctr += 1
                if self._rep_ctr % REPLAY_SUB == 0:
                    # Store raw frame BEFORE drawing overlays
                    self._rep_frames.append(cv2.resize(frame, (self.W // 2, self.H // 2)))

                self._draw_target_line(frame)
                if self.records:
                    self._draw_path(frame, [r.pos for r in self.records])
                if pos:
                    cv2.circle(frame, pos, 7, C["green"], -1)

                # Progress bar
                prog = max(0.0, 1.0 - rem / RECORD_SECS)
                cv2.rectangle(frame, (0, self.H - 4),
                              (int(self.W * prog), self.H), C["green"], -1)

                extra = [f"REC {rem:.1f}s"]
                if angle is not None:
                    extra.append(f"Face : {angle:+.1f}°")
                self._draw_hud(frame, fps, extra)

                if rem <= 0:
                    self.state = AppState.ANALYSIS

            # ── ANALYSIS (instant, one frame) ─────────────────────────────
            elif self.state == AppState.ANALYSIS:
                self.result = self._analyze()
                if self.result:
                    self._save_json(self.result)
                    self._last_path = list(self.result.positions)
                self._kf_init     = False   # will be initialized on KEYFRAMES entry
                self._strobe_det  = None
                self._rep_idx   = 0
                self._rep_loops = 0
                self._rep_last  = now
                self.state      = AppState.REPLAY

            # ── REPLAY ────────────────────────────────────────────────────
            elif self.state == AppState.REPLAY:
                r = self.result
                # Always show captured frames (independent of tracking result)
                if self._rep_frames:
                    if now - self._rep_last >= 1.0 / REPLAY_FPS:
                        next_idx = (self._rep_idx + 1) % len(self._rep_frames)
                        if next_idx == 0:           # completed one full loop
                            self._rep_loops += 1
                        self._rep_idx  = next_idx
                        self._rep_last = now
                        if self._rep_loops >= 1:    # after one pass → key frames
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
                    hud_extra.append("No tracking — add ArUco or press F")
                self._draw_hud(frame, fps, hud_extra)

            # ── KEYFRAMES ─────────────────────────────────────────────────
            elif self.state == AppState.KEYFRAMES:
                # ── Init on first entry ────────────────────────────────────
                if not self._kf_init:
                    self._kf_init        = True
                    self._kf_idx         = 0
                    self._kf_last        = now
                    self._kf_done        = False
                    self._kf_cur_col     = None
                    self._kf_flash_col   = -1
                    self._kf_flash_t     = 0.0
                    self._kf_best_off    = {}
                    self._strobe_indices = [None] * 7
                    if self._rep_frames:
                        bg_raw         = cv2.cvtColor(self._rep_frames[0],
                                                      cv2.COLOR_BGR2GRAY)
                        self._kf_bg    = cv2.GaussianBlur(
                            bg_raw, (21, 21), 0).astype(np.float32)

                KF_FPS  = 6          # ~0.4× of 15 fps stored
                N_COLS  = 7
                vid_h   = self.H
                col_w_d = self.W // N_COLS

                # ── Advance one frame at 0.4× speed ───────────────────────
                if not self._kf_done and self._rep_frames:
                    if now - self._kf_last >= 1.0 / KF_FPS:
                        self._kf_last = now
                        det = self._detect_putter_col(self._kf_idx)
                        if det is not None:
                            col, offset = det
                            self._kf_cur_col = col
                            # Keep the frame where putter is most centred
                            # in this column (beats backswing frames too)
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
                    # ── PHASE 1: show current frame + column highlights ────
                    f_idx = min(self._kf_idx, len(self._rep_frames) - 1)
                    disp  = cv2.resize(self._rep_frames[f_idx], (self.W, self.H))
                    frame = np.zeros((self.H + PANEL_H, self.W, 3), dtype=np.uint8)
                    frame[:self.H] = disp

                    # Soft tint on column where putter currently is
                    if self._kf_cur_col is not None:
                        c  = self._kf_cur_col
                        x0 = c * col_w_d
                        x1 = x0 + col_w_d if c < N_COLS - 1 else self.W
                        frame[:vid_h, x0:x1] = np.clip(
                            frame[:vid_h, x0:x1].astype(np.int32) + 50,
                            0, 255).astype(np.uint8)

                    # Bright flash on newly-improved column (lasts 0.5 s)
                    if self._kf_flash_col >= 0 and now - self._kf_flash_t < 0.5:
                        c  = self._kf_flash_col
                        x0 = c * col_w_d
                        x1 = x0 + col_w_d if c < N_COLS - 1 else self.W
                        frame[:vid_h, x0:x1] = np.clip(
                            frame[:vid_h, x0:x1].astype(np.int32) + 110,
                            0, 255).astype(np.uint8)

                    # Column dividers
                    for i in range(1, N_COLS):
                        cv2.line(frame, (i * col_w_d, 0),
                                 (i * col_w_d, vid_h), (80, 80, 80), 1)

                else:
                    # ── PHASE 2: composite with all 7 frames at once ───────
                    frame = self._draw_strobe_composite()

                self._draw_hud(frame, fps, ["SPACE = new shot"])

            # ── ROI selection overlay ─────────────────────────────────────
            if self._sel_mode:
                self._draw_sel_rect(frame)

            # ── Debug overlay ─────────────────────────────────────────────
            if self.debug:
                self._draw_debug(frame)

            # Pad camera-height frames to full window height
            if frame.shape[0] == self.H:
                padded = np.zeros((self.H + PANEL_H, self.W, 3), dtype=np.uint8)
                padded[:self.H] = frame
                frame = padded
            cv2.imshow("Putter Live", frame)

            # ── Key handling ──────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):         # Esc / Q → quit
                break

            elif key == ord(' '):                        # Space → start
                if self.state in (AppState.READY, AppState.REPLAY, AppState.KEYFRAMES):
                    self.state     = AppState.COUNTDOWN
                    self._cd_start = now

            elif key in (ord('c'), ord('C')):            # C → calibrate
                if self.state == AppState.READY and raw_angle is not None:
                    self.ref_angle = raw_angle
                    self.tgt_angle = raw_angle            # target line = face dir at address
                    self._ang_buf.clear()
                    print(f"[cal] ref_angle={self.ref_angle:.1f}°  "
                          f"tgt_angle={self.tgt_angle:.1f}°")

            elif key in (ord('d'), ord('D')):            # D → debug
                self.debug = not self.debug

            elif key in (ord('r'), ord('R')):            # R → reset
                self.state = AppState.READY
                self.records      = []
                self._rep_frames  = []
                self.result       = None
                self._pos_buf.clear()
                self._ang_buf.clear()
                self._kf_init     = False
                self._strobe_indices = []
                self._strobe_det  = None
                print("[reset]")

            elif key in (ord('f'), ord('F')):            # F → fallback ROI
                if self.state == AppState.READY:
                    self._sel_mode = True
                    print("[fallback] Click-drag to select putter head ROI.")

            elif key in (ord('m'), ord('M')):            # M → generate markers
                _save_markers(self._aruco_dict)

        self.cap.release()
        cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        PutterLive().run()
    except RuntimeError as e:
        print(f"\n[error] {e}\n")
        sys.exit(1)
