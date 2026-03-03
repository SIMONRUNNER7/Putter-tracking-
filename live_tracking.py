#!/usr/bin/env python3
"""
Runner Arc Analysis — 3-putt session recorder.

Workflow par putt :
  WAITING    → putter recule (backswing) → RECORDING
  RECORDING  → 0.5 s après que la balle quitte sa zone → REPLAY
  REPLAY     → lecture à 0.4× → STROBE
  STROBE     → vue stroboscopique 7 frames (ESPACE pour continuer)
  (× 3 puis RESULTS — moyenne des 3 arcs)

Touches : Q/Echap = quitter  |  ESPACE = avancer  |  R = reset total
          --lefty  pour les gauchers (déclenche sur mouvement vers la gauche)
"""
import sys
import os
import argparse
import math
import cv2
import numpy as np
from typing import Optional, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracking.putter_detector import PutterDetector

# ─── Modèle YOLO ──────────────────────────────────────────────────────────
YOLO_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "runs", "detect", "runs", "putter", "putter_detector", "weights", "best.pt"
)

# ─── Constantes ───────────────────────────────────────────────────────────
WARMUP_FRAMES  = 30
TOTAL_PUTTS    = 3
REPLAY_SPEED   = 0.4     # vitesse de lecture du replay
BALL_EXTRA_S   = 0.5     # secondes supplémentaires après départ balle
N_BEFORE       = 3       # frames avant l'impact dans la vue strobe
N_AFTER        = 3       # frames après l'impact
N_KEYFRAMES    = N_BEFORE + 1 + N_AFTER   # = 7

HEADER_H       = 62
FOOTER_H       = 58
LINE_Y_FRAC    = 0.52

# Seuils de déclenchement
BACK_PX        = 22      # px de déplacement pour valider le backswing
BACK_CONSEC    = 5       # frames consécutives nécessaires
BALL_LEAVE_PX  = 18      # déplacement de la balle pour "départ"

# Couleurs (BGR)
C_BLUE_BOX  = (200,  80,   0)   # rectangles putter
C_GREEN_LN  = (  0, 200,  60)   # cercles balle + ligne trajectoire
C_ARC_COLOR = (200, 130,  40)   # arc trajectoire club
C_SHAFT     = (  0,  40, 220)   # lignes shaft
C_WHITE     = (255, 255, 255)
C_DARK      = ( 15,  15,  15)
C_GRAY      = (170, 170, 170)
C_RED_UI    = (  0,  30, 215)   # rouge interface (arcs waiting, ARC ANALYSIS)
C_RECORDING = (  0,   0, 220)   # indicateur REC

# États
S_WAITING   = "waiting"
S_RECORDING = "recording"
S_COUNTDOWN = "countdown"
S_REPLAY    = "replay"
S_STROBE    = "strobe"
S_RESULTS   = "results"


# ═══════════════════════════════════════════════════════════════════════════
#  Détection balle (HoughCircles)
# ═══════════════════════════════════════════════════════════════════════════
def detect_ball(frame: np.ndarray,
                roi: Optional[Tuple[int, int, int, int]] = None
                ) -> Optional[Tuple[int, int, int]]:
    """Retourne (cx, cy, radius) de la première balle blanche trouvée."""
    if roi:
        x1, y1, x2, y2 = roi
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        search = frame[y1:y2, x1:x2]
        off_x, off_y = x1, y1
    else:
        search = frame
        off_x, off_y = 0, 0

    if search.size == 0:
        return None

    gray    = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=20,
        param1=50, param2=20,
        minRadius=4, maxRadius=35,
    )
    if circles is None:
        return None

    circles = np.uint16(np.around(circles[0]))
    for cx, cy, r in circles:
        mask = np.zeros(gray.shape, np.uint8)
        cv2.circle(mask, (int(cx), int(cy)), int(r), 255, -1)
        if cv2.mean(gray, mask=mask)[0] > 155:
            return (int(cx) + off_x, int(cy) + off_y, int(r))
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Détecteur de backswing
# ═══════════════════════════════════════════════════════════════════════════
class BackswingTrigger:
    """Détecte quand le putter commence à reculer."""

    def __init__(self, lefty: bool = False,
                 thresh_px: int = BACK_PX, consec: int = BACK_CONSEC):
        # RH : recul = déplacement vers la droite (+x)
        # LH : recul = déplacement vers la gauche (-x)
        self.sign    = -1 if lefty else +1
        self.thresh  = thresh_px
        self.consec  = consec
        self.start_x: Optional[float] = None
        self._count  = 0

    def reset(self, x: Optional[float] = None) -> None:
        self.start_x = x
        self._count  = 0

    def update(self, x: Optional[float]) -> bool:
        """Retourne True si le backswing est confirmé."""
        if x is None:
            return False
        if self.start_x is None:
            self.start_x = x
            return False
        delta = (x - self.start_x) * self.sign
        if delta > self.thresh / self.consec:
            self._count += 1
        else:
            self._count = max(0, self._count - 1)
        return self._count >= self.consec


# ═══════════════════════════════════════════════════════════════════════════
#  Extraction des 7 frames clés
# ═══════════════════════════════════════════════════════════════════════════
def find_impact_idx(buffer: List[np.ndarray],
                    ball_start: Optional[Tuple[float, float]]) -> int:
    """Trouve l'index approximatif de l'impact dans le buffer."""
    if ball_start is None or len(buffer) < 3:
        return len(buffer) * 6 // 10
    bx0, by0 = ball_start
    for i, frame in enumerate(buffer):
        b = detect_ball(frame)
        if b is not None:
            dist = math.hypot(b[0] - bx0, b[1] - by0)
            if dist > BALL_LEAVE_PX:
                return max(0, i - 1)
    return len(buffer) * 6 // 10


def extract_keyframes(buffer: List[np.ndarray],
                      impact_idx: int
                      ) -> List[np.ndarray]:
    """Retourne une liste de 7 frames (3 avant impact, impact, 3 après)."""
    n = len(buffer)
    total = N_BEFORE + 1 + N_AFTER

    if n < total:
        idxs = [int(i * (n - 1) / (total - 1)) for i in range(total)]
        return [buffer[min(i, n - 1)] for i in idxs]

    before_step = max(1, impact_idx // (N_BEFORE + 1))
    after_step  = max(1, (n - 1 - impact_idx) // (N_AFTER + 1))

    idxs = []
    for k in range(N_BEFORE, 0, -1):
        idxs.append(max(0, impact_idx - k * before_step))
    idxs.append(impact_idx)
    for k in range(1, N_AFTER + 1):
        idxs.append(min(n - 1, impact_idx + k * after_step))

    return [buffer[i] for i in idxs]


# ═══════════════════════════════════════════════════════════════════════════
#  Courbe lisse (polynôme degré 2)
# ═══════════════════════════════════════════════════════════════════════════
def draw_smooth_curve(frame: np.ndarray,
                      points: List[Tuple[int, int]],
                      color: Tuple[int, int, int],
                      thickness: int = 2) -> None:
    if len(points) < 2:
        return
    pts = np.array(points, dtype=float)
    xs, ys = pts[:, 0], pts[:, 1]
    x0, x1 = int(xs.min()), int(xs.max())
    if x1 <= x0:
        return
    degree = min(2, len(points) - 1)
    try:
        coeffs = np.polyfit(xs, ys, degree)
    except Exception:
        return
    px = np.linspace(x0, x1, 300, dtype=int)
    py = np.polyval(coeffs, px).astype(int)
    prev = None
    for xi, yi in zip(px, py):
        p = (int(xi), int(yi))
        if prev:
            cv2.line(frame, prev, p, color, thickness, cv2.LINE_AA)
        prev = p


# ═══════════════════════════════════════════════════════════════════════════
#  Vue stroboscopique
# ═══════════════════════════════════════════════════════════════════════════
def create_strobe(keyframes: List[np.ndarray],
                  detector: PutterDetector) -> np.ndarray:
    """
    Crée la vue stroboscopique 7 frames :
      - Fond = première frame assombrie
      - Crops des têtes putter collés à leur position réelle
      - Rectangles bleus, cercles verts, lignes shaft rouges
      - Arc bleu (trajectoire club), ligne verte (trajectoire balle)
    """
    bg = keyframes[0].copy()
    h, w = bg.shape[:2]
    bg = cv2.addWeighted(bg, 0.50, np.zeros_like(bg), 0.50, 0)

    putter_centers: List[Tuple[int, int]] = []
    ball_centers:   List[Tuple[int, int]] = []

    # Point d'ancrage du shaft = position estimée des mains (bas-centre du cadre)
    grip_x = w // 2
    grip_y = h - FOOTER_H - 5

    for i, frame in enumerate(keyframes):
        # ── Détection tête putter ─────────────────────────────────────────
        det = detector.detect(frame, i)

        if det.bbox is not None and det.confidence > 0.05:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            x1c = max(0, x1); y1c = max(0, y1)
            x2c = min(w, x2); y2c = min(h, y2)

            if x2c > x1c and y2c > y1c:
                # Coller le crop à sa position réelle
                crop = frame[y1c:y2c, x1c:x2c]
                bg[y1c:y2c, x1c:x2c] = crop

                # Rectangle bleu
                cv2.rectangle(bg, (x1c, y1c), (x2c, y2c),
                              C_BLUE_BOX, 2, cv2.LINE_AA)

                cx = (x1c + x2c) // 2
                cy = (y1c + y2c) // 2
                putter_centers.append((cx, cy))

                # Ligne shaft (rouge) : bas du rectangle → point grip
                shaft_start = (cx, y2c)
                cv2.line(bg, shaft_start, (grip_x, grip_y),
                         C_SHAFT, 2, cv2.LINE_AA)

        elif det.confidence > 0.05:
            cx, cy = int(det.head_x), int(det.head_y)
            putter_centers.append((cx, cy))

        # ── Détection balle ───────────────────────────────────────────────
        ball = detect_ball(frame)
        if ball:
            bx, by, br = ball
            cv2.circle(bg, (bx, by), br + 4, C_GREEN_LN, 2, cv2.LINE_AA)
            ball_centers.append((bx, by))

    # ── Arc club (bleu-cyan) ──────────────────────────────────────────────
    if len(putter_centers) >= 2:
        draw_smooth_curve(bg, putter_centers, C_ARC_COLOR, 2)

    # ── Ligne balle (vert) ────────────────────────────────────────────────
    if len(ball_centers) >= 2:
        draw_smooth_curve(bg, ball_centers, C_GREEN_LN, 2)

    return bg


# ═══════════════════════════════════════════════════════════════════════════
#  Résultats (moyenne des strobes)
# ═══════════════════════════════════════════════════════════════════════════
def create_results(strobes: List[np.ndarray],
                   frame_size: Tuple[int, int]) -> np.ndarray:
    """Blend les 3 vues stroboscopiques pour un résultat moyen."""
    h, w = frame_size
    acc = np.zeros((h, w, 3), dtype=np.float32)
    for s in strobes:
        acc += cv2.resize(s, (w, h)).astype(np.float32)
    result = (acc / len(strobes)).astype(np.uint8)

    # Label centré
    msg = "3-PUTT AVERAGE"
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
    cv2.putText(result, msg,
                ((w - tw) // 2, HEADER_H + (h - HEADER_H - FOOTER_H) // 2),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, C_WHITE, 2, cv2.LINE_AA)
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  UI : Header, Footer, Aim line, Arc waiting
# ═══════════════════════════════════════════════════════════════════════════
def draw_header(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, HEADER_H), C_DARK, -1)
    cv2.addWeighted(ov, 0.88, frame, 0.12, 0, frame)

    cv2.putText(frame, "< Back", (18, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_WHITE, 1, cv2.LINE_AA)

    (rw, _), _ = cv2.getTextSize("RUNNER",         cv2.FONT_HERSHEY_DUPLEX,  0.95, 2)
    (aw, _), _ = cv2.getTextSize("  ARC ANALYSIS", cv2.FONT_HERSHEY_SIMPLEX, 0.70, 1)
    tx = (w - rw - aw) // 2
    cv2.putText(frame, "RUNNER",         (tx,      42), cv2.FONT_HERSHEY_DUPLEX,  0.95, C_WHITE,  2, cv2.LINE_AA)
    cv2.putText(frame, "  ARC ANALYSIS", (tx + rw, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.70, C_RED_UI, 1, cv2.LINE_AA)

    gx, gy = w - 36, HEADER_H // 2
    cv2.circle(frame, (gx, gy), 13, C_WHITE, 1, cv2.LINE_AA)
    cv2.circle(frame, (gx, gy),  5, C_WHITE, 1, cv2.LINE_AA)
    for ang in range(0, 360, 45):
        r = math.radians(ang)
        cv2.line(frame,
                 (int(gx + 9  * math.cos(r)), int(gy + 9  * math.sin(r))),
                 (int(gx + 14 * math.cos(r)), int(gy + 14 * math.sin(r))),
                 C_WHITE, 2, cv2.LINE_AA)


_FOOTER_MSGS = {
    S_WAITING:   "PUTT 3 TIMES NATURALLY",
    S_RECORDING: "● REC",
    S_COUNTDOWN: "● REC",
    S_REPLAY:    "SLOW REPLAY  ×0.4",
    S_STROBE:    "ARC ANALYSIS  —  SPACE to continue",
    S_RESULTS:   "3-PUTT AVERAGE  —  R to restart",
}


def draw_footer(frame: np.ndarray, state: str, putt_count: int) -> None:
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, h - FOOTER_H), (w, h), C_DARK, -1)
    cv2.addWeighted(ov, 0.88, frame, 0.12, 0, frame)

    msg  = _FOOTER_MSGS.get(state, "")
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(msg, font, 0.72, 2)
    base_y = h - FOOTER_H + (FOOTER_H + th) // 2

    msg_color = C_RECORDING if state in (S_RECORDING, S_COUNTDOWN) else C_WHITE
    cv2.putText(frame, msg, ((w - tw) // 2, base_y), font, 0.72, msg_color, 2, cv2.LINE_AA)

    # Compteur X/3
    s_n = str(putt_count)
    s_s = "/"
    s_t = str(TOTAL_PUTTS)
    (nw, _), _ = cv2.getTextSize(s_n, font, 1.2, 2)
    (sw, _), _ = cv2.getTextSize(s_s, font, 0.85, 1)
    (tw2,_), _ = cv2.getTextSize(s_t, font, 1.2, 2)
    rx = w - 24 - nw - sw - tw2
    cv2.putText(frame, s_n, (rx,           base_y), font, 1.2,  C_WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, s_s, (rx + nw,      base_y - 3), font, 0.85, C_GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, s_t, (rx + nw + sw, base_y), font, 1.2,  C_WHITE, 2, cv2.LINE_AA)


def draw_aim_line(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    y = int(h * LINE_Y_FRAC)
    dash, gap, x = 20, 12, 0
    while x < w:
        cv2.line(frame, (x, y), (min(x + dash, w), y), C_WHITE, 1, cv2.LINE_AA)
        x += dash + gap
    cv2.arrowedLine(frame, (60, y), (10, y), C_WHITE, 2, cv2.LINE_AA, tipLength=0.45)


def draw_swing_arc(frame: np.ndarray) -> None:
    """Arcs rouges de référence (état WAITING)."""
    h, w = frame.shape[:2]
    cy  = int(h * LINE_Y_FRAC)
    amp = int((h - HEADER_H - FOOTER_H) * 0.30)
    x0, x1 = int(w * 0.04), int(w * 0.96)
    n = 300
    pts_up, pts_dn = [], []
    for i in range(n):
        t = i / (n - 1)
        x   = int(x0 + t * (x1 - x0))
        off = int(amp * 4 * t * (1 - t))
        pts_up.append((x, cy - off))
        pts_dn.append((x, cy + off))
    for i in range(1, n):
        cv2.line(frame, pts_up[i-1], pts_up[i], C_RED_UI, 2, cv2.LINE_AA)
        cv2.line(frame, pts_dn[i-1], pts_dn[i], C_RED_UI, 2, cv2.LINE_AA)
    tk = 26
    cv2.line(frame, (x0, cy - tk), (x0, cy + tk), C_RED_UI, 2, cv2.LINE_AA)
    cv2.line(frame, (x1, cy - tk), (x1, cy + tk), C_RED_UI, 2, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════
#  Caméra
# ═══════════════════════════════════════════════════════════════════════════
def list_cameras(max_index: int = 5) -> list:
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append({
                "index": i,
                "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps":    cap.get(cv2.CAP_PROP_FPS),
            })
            cap.release()
    return found


def pick_camera(forced_index: Optional[int]) -> int:
    if forced_index is not None:
        return forced_index
    cams = list_cameras()
    if not cams:
        print("[ERREUR] Aucune caméra détectée.")
        sys.exit(1)
    for c in cams:
        print(f"  index {c['index']} — {c['width']}x{c['height']} @ {c['fps']:.0f} fps")
    chosen = cams[-1]["index"] if len(cams) > 1 else cams[0]["index"]
    print(f"[INFO] Caméra choisie : {chosen}")
    return chosen


# ═══════════════════════════════════════════════════════════════════════════
#  Boucle principale
# ═══════════════════════════════════════════════════════════════════════════
def run_live(source, lefty: bool = False) -> None:
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERREUR] Impossible d'ouvrir : {source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    use_yolo = os.path.exists(YOLO_MODEL_PATH)
    print(f"[INFO] YOLO : {'chargé' if use_yolo else 'introuvable — fallback blob'}")

    def make_detector():
        return PutterDetector(
            target_line_angle=0.0,
            use_yolo=use_yolo,
            yolo_model_path=YOLO_MODEL_PATH if use_yolo else None,
        )

    detector = make_detector()

    # ── Warmup ────────────────────────────────────────────────────────────
    print(f"[INFO] Chauffe ({WARMUP_FRAMES} frames)…")
    buf = []
    for _ in range(WARMUP_FRAMES):
        ret, f = cap.read()
        if ret:
            buf.append(f)
    if buf:
        detector.initialize_background(buf)

    # ── Variables de session ───────────────────────────────────────────────
    state       = S_WAITING
    putt_count  = 0
    frame_idx   = WARMUP_FRAMES
    all_strobes: List[np.ndarray] = []

    # Par putt
    record_buf:  List[np.ndarray]      = []
    ball_start:  Optional[Tuple[float,float]] = None
    countdown    = 0
    replay_idx   = 0
    strobe_frame: Optional[np.ndarray] = None
    result_frame: Optional[np.ndarray] = None

    trigger = BackswingTrigger(lefty=lefty)

    print("[INFO] Prêt — place le putter et putte naturellement.")
    print("[INFO] Q/Echap=quitter | ESPACE=avancer | R=reset")

    while True:
        # ── REPLAY : lecture à vitesse réduite (gestion spéciale) ─────────
        if state == S_REPLAY:
            if replay_idx < len(record_buf):
                display = record_buf[replay_idx].copy()
                draw_header(display)
                draw_footer(display, state, putt_count)
                cv2.imshow("Runner Arc Analysis", display)
                wait_ms = max(1, int(1000 / fps / REPLAY_SPEED))
                key = cv2.waitKey(wait_ms) & 0xFF
                replay_idx += 1
                if key in (ord('q'), ord('Q'), 27):
                    break
                if key == ord(' '):
                    replay_idx = len(record_buf)   # sauter le replay
                continue
            else:
                # Replay terminé → calculer la strobe
                print("[INFO] Calcul vue stroboscopique…")
                impact_idx   = find_impact_idx(record_buf, ball_start)
                kframes      = extract_keyframes(record_buf, impact_idx)
                strobe_frame = create_strobe(kframes, make_detector())
                all_strobes.append(strobe_frame)
                state = S_STROBE
                continue

        # ── STROBE : affichage figé ────────────────────────────────────────
        if state == S_STROBE:
            display = strobe_frame.copy() if strobe_frame is not None else np.zeros_like(buf[0])
            draw_header(display)
            draw_footer(display, state, putt_count)
            cv2.imshow("Runner Arc Analysis", display)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            if key == ord(' ') or key == 13:
                putt_count += 1
                if putt_count >= TOTAL_PUTTS:
                    h, w = display.shape[:2]
                    result_frame = create_results(all_strobes, (h, w))
                    state = S_RESULTS
                else:
                    # Prochain putt
                    state = S_WAITING
                    record_buf.clear()
                    ball_start  = None
                    strobe_frame = None
                    trigger.reset()
            continue

        # ── RESULTS : affichage figé ───────────────────────────────────────
        if state == S_RESULTS:
            display = result_frame.copy() if result_frame is not None else np.zeros_like(buf[0])
            draw_header(display)
            draw_footer(display, state, putt_count)
            cv2.imshow("Runner Arc Analysis", display)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            if key in (ord('r'), ord('R')):
                state       = S_WAITING
                putt_count  = 0
                all_strobes.clear()
                record_buf.clear()
                ball_start   = None
                strobe_frame = None
                result_frame = None
                trigger.reset()
                detector     = make_detector()
            continue

        # ── Lecture caméra (WAITING / RECORDING / COUNTDOWN) ──────────────
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Fin de la source.")
            break

        h, w = frame.shape[:2]
        det  = detector.detect(frame, frame_idx)
        head_x = det.head_x if det.confidence > 0.10 else None

        # ── WAITING ────────────────────────────────────────────────────────
        if state == S_WAITING:
            if trigger.update(head_x):
                print("[INFO] Backswing détecté — enregistrement…")
                state = S_RECORDING
                record_buf.clear()
                b = detect_ball(frame)
                ball_start = (float(b[0]), float(b[1])) if b else None
                trigger.reset(head_x)

            draw_aim_line(frame)
            draw_swing_arc(frame)

        # ── RECORDING ─────────────────────────────────────────────────────
        elif state == S_RECORDING:
            record_buf.append(frame.copy())

            # Détection départ balle
            if ball_start is not None:
                b = detect_ball(frame)
                if b is not None:
                    dist = math.hypot(b[0] - ball_start[0],
                                      b[1] - ball_start[1])
                    if dist > BALL_LEAVE_PX:
                        state     = S_COUNTDOWN
                        countdown = int(fps * BALL_EXTRA_S)
                        print(f"[INFO] Balle partie — countdown {countdown} frames")

            # Indicateur REC (point rouge clignotant)
            if (frame_idx // 15) % 2 == 0:
                cv2.circle(frame, (w - 25, HEADER_H + 22), 9, C_RECORDING, -1)

        # ── COUNTDOWN ─────────────────────────────────────────────────────
        elif state == S_COUNTDOWN:
            record_buf.append(frame.copy())
            countdown -= 1

            # Indicateur REC
            if (frame_idx // 15) % 2 == 0:
                cv2.circle(frame, (w - 25, HEADER_H + 22), 9, C_RECORDING, -1)

            if countdown <= 0:
                print(f"[INFO] Enregistrement terminé : {len(record_buf)} frames")
                state      = S_REPLAY
                replay_idx = 0

        # ── Dessin commun (WAITING / RECORDING / COUNTDOWN) ───────────────
        draw_header(frame)
        draw_footer(frame, state, putt_count)

        frame_idx += 1
        cv2.imshow("Runner Arc Analysis", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('r'), ord('R')):
            state       = S_WAITING
            putt_count  = 0
            all_strobes.clear()
            record_buf.clear()
            ball_start   = None
            strobe_frame = None
            result_frame = None
            trigger.reset()
            detector     = make_detector()
            print("[INFO] Reset.")

    cap.release()
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner Arc Analysis")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--camera", type=int, default=None)
    group.add_argument("--video",  type=str)
    parser.add_argument("--list",  action="store_true")
    parser.add_argument("--lefty", action="store_true",
                        help="Gaucher (déclenche sur mouvement vers la gauche)")
    args = parser.parse_args()

    if args.list:
        for c in list_cameras():
            print(f"  --camera {c['index']}  →  {c['width']}x{c['height']} @ {c['fps']:.0f} fps")
        sys.exit(0)

    source = args.video if args.video else pick_camera(args.camera)
    run_live(source, lefty=args.lefty)
