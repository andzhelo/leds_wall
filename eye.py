"""Giant eye on the wall that watches you.

A single wide almond eye is painted across the full width of the 30x120
portrait panel (centered vertically). A MediaPipe person mask gives the
person's position in the camera frame, and the iris/pupil glide to look
straight at them. It blinks now and then, the pupil dilates when you come
close, and when the room is empty the eye slowly looks around on its own.

Same plumbing as mirror_rgb.py: threaded FreshestFrame camera, DDP to WLED,
and a hot-reloaded calibration.json. It SHARES calibration.json with the
mirror scripts and reuses their rotate / flip_code / color_order / brightness
keys, plus its own eye_* knobs below (all hot-reloaded while running):

  geometry (pixels on the 30x120 panel; "half" = half-size from center)
    eye_half_w     half-width of the eye socket   (15 = full 30-wide)
    eye_half_h     half-height of the eye socket   (~20 => ~40 tall almond)
    eye_cx eye_cy  socket center nudge in px (0 = dead center)
    eye_iris_r     iris radius
    eye_pupil_min  pupil radius when the person is far
    eye_pupil_max  pupil radius when the person is close (dilation range)
    eye_feather    edge softness in px (anti-aliasing for the LEDs)
  colours (0-255 RGB, dimmed by the shared "brightness")
    eye_iris_col   eye_sclera_col   eye_catchlight 1/0
  tracking
    eye_track      0..1 how far the iris travels toward the person
    eye_flip_x/y   1 or -1, flip gaze direction if it mirrors the wrong way
    eye_gaze_smooth 0..1 glide (higher = slower, smoother follow)
  life
    eye_blink 1/0   eye_blink_every (avg sec)   eye_blink_dur (sec)
    eye_idle 1/0    eye_idle_delay (frames of absence before it wanders)
    eye_dilate 1/0  (pupil grows as the person fills more of the frame)
"""
import cv2
import socket
import numpy as np
import json
import os
import threading
import time
import signal
import random
import mediapipe as mp

WLED_IP = "192.168.1.60"
UDP_PORT = 4048

MATRIX_WIDTH = 30
MATRIX_HEIGHT = 120
NUM_LEDS = MATRIX_WIDTH * MATRIX_HEIGHT

CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
MAX_PIXELS_PER_PACKET = 480

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH = os.path.join(BASE_DIR, "calibration.json")
# Per-lane physical wiring correction, SHARED with the games (wall.py / calib.py).
# Each vertical lane x is rolled by offsets[x] LEDs so straight rows render
# straight. Tune it live with `python3 calib.py`; the eye inherits the result.
WALL_CALIB_PATH = os.path.join(BASE_DIR, "wall_calib.json")

DEFAULT_CALIB = {
    "rotate": 0,
    "flip_code": 0,
    "brightness": 1.0,
    "color_order": "RGB",
    "mask_threshold": 0.4,
    "denoise": True,

    # --- eye geometry (pixels on the 30x120 wall) ---
    "eye_half_w": 15.0,     # full width of the panel
    "eye_half_h": 16.0,     # vertical radius (flatter than wide -> almond)
    "eye_cx": 0.0,
    "eye_cy": 0.0,
    # almond shape: >1 corners = pointier (real-eye canthi); upper lid taller
    # than lower for a natural asymmetric eye.
    "eye_corner": 1.15,
    "eye_upper": 1.0,
    "eye_lower": 0.72,
    "eye_edge_soft": 1.7,   # eyelid outline softness (x feather); hides lane offset
    "eye_iris_r": 8.0,
    "eye_pupil_min": 2.6,
    "eye_pupil_max": 6.0,
    "eye_feather": 1.3,

    # --- colours ---
    "eye_iris_col": [60, 140, 255],
    "eye_sclera_col": [255, 250, 238],
    "eye_catchlight": 1,

    # --- tracking ---
    "eye_track": 1.0,
    "eye_flip_x": 1,
    "eye_flip_y": 1,
    "eye_gaze_smooth": 0.82,

    # --- life ---
    "eye_blink": 1,
    "eye_blink_every": 4.5,
    "eye_blink_dur": 0.18,
    "eye_idle": 1,
    "eye_idle_delay": 50,
    "eye_dilate": 1,

    # render smoothing (small temporal blend to calm LED shimmer)
    "eye_render_smooth": 0.3,

    # lane correction ON (applied smoothly in the render, no pupil comb); 0 = off
    "eye_lane_correct": 1,
}

_CH = {"R": 0, "G": 1, "B": 2}
_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def load_calibration(state):
    calib, last_mtime = state
    try:
        mtime = os.path.getmtime(CALIB_PATH)
    except OSError:
        with open(CALIB_PATH, "w") as f:
            json.dump(DEFAULT_CALIB, f, indent=2)
        return dict(DEFAULT_CALIB), os.path.getmtime(CALIB_PATH)
    if mtime != last_mtime:
        try:
            with open(CALIB_PATH) as f:
                loaded = json.load(f)
            merged = dict(DEFAULT_CALIB)
            merged.update(loaded)
            print("Calibration reloaded:", json.dumps({k: merged[k] for k in merged if k.startswith("eye_") or k in ("rotate", "flip_code", "color_order", "brightness")}))
            return merged, mtime
        except (json.JSONDecodeError, OSError) as e:
            print("Calibration not reloaded (will retry):", e)
            return calib, last_mtime
    return calib, last_mtime


def load_lane_offsets(state):
    """Hot-reload wall_calib.json (same file calib.py writes). Zeros if absent."""
    offs, last_mtime = state
    try:
        mtime = os.path.getmtime(WALL_CALIB_PATH)
    except OSError:
        return [0] * MATRIX_WIDTH, None
    if mtime != last_mtime:
        try:
            with open(WALL_CALIB_PATH) as f:
                vals = [int(v) for v in json.load(f).get("offsets", [])]
            if len(vals) == MATRIX_WIDTH:
                print("Lane calibration loaded:", vals)
                return vals, mtime
            print(f"Lane calib ignored: expected {MATRIX_WIDTH} offsets, got {len(vals)}")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as ex:
            print("Lane calib not loaded (will retry):", ex)
    return offs, last_mtime


class FreshestFrame(threading.Thread):
    """Camera in its own thread, always the newest frame -> low latency."""
    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.start()

    def run(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame = f

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        self.join(timeout=1.0)


def smoothstep_alpha(signed_dist, feather):
    """1 inside, 0 outside, soft transition over `feather` px (for LED AA)."""
    return np.clip(signed_dist / max(1e-3, feather) + 0.5, 0.0, 1.0)


signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
try:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except Exception:
    pass

segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)

# coordinate grids on the panel (row-major y*W + x, y=0 top)
YY, XX = np.mgrid[0:MATRIX_HEIGHT, 0:MATRIX_WIDTH].astype(np.float32)

calib_state = load_calibration((dict(DEFAULT_CALIB), 0.0))
lane_state = load_lane_offsets(([0] * MATRIX_WIDTH, None))

print("Warming up camera + MediaPipe model (a few seconds)...")
cam = FreshestFrame(cap)
_wait0 = time.time()
while cam.read() is None:
    if time.time() - _wait0 > 8.0:
        raise SystemExit("ERROR: no camera frames within 8s -- is /dev/video0 in use?")
    time.sleep(0.05)
segmenter.process(cv2.cvtColor(cam.read(), cv2.COLOR_BGR2RGB))
print(f"Ready. The eye is watching, streaming to {WLED_IP} ... edit {CALIB_PATH} live. (Ctrl+C)")

# animated state
gaze_x, gaze_y = 0.0, 0.0        # smoothed gaze target in [-1, 1]
pupil_r_s = 4.0                  # smoothed pupil radius
smoothed = None
absent_frames = 999
next_blink = time.time() + random.uniform(2.0, 5.0)
blink_start = -10.0
wander_seed = random.uniform(0, 1000)

fps_t0, fps_n = time.time(), 0

try:
    while True:
        calib_state = load_calibration(calib_state)
        c = calib_state[0]
        now = time.time()

        frame = cam.read()
        if frame is None:
            time.sleep(0.005)
            continue

        rot = int(c.get("rotate", 0))
        if rot in _ROT:
            frame = cv2.rotate(frame, _ROT[rot])
        flip_code = int(c["flip_code"])
        if flip_code in (-1, 0, 1):
            frame = cv2.flip(frame, flip_code)
        h, w = frame.shape[:2]

        # --- person mask + centroid (where to look) ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = segmenter.process(rgb)
        target_x, target_y = gaze_x, gaze_y     # hold if no reading
        area = 0.0
        if results.segmentation_mask is not None:
            person = (results.segmentation_mask > float(c["mask_threshold"])).astype(np.uint8)
            if bool(c.get("denoise", True)):
                person = cv2.morphologyEx(person, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            area = float(person.mean())
            M = cv2.moments(person, binaryImage=True)
            if M["m00"] > (w * h * 0.004):       # enough of a person to trust
                px = M["m10"] / M["m00"]
                py = M["m01"] / M["m00"]
                target_x = int(c.get("eye_flip_x", 1)) * (2.0 * px / w - 1.0)
                target_y = int(c.get("eye_flip_y", 1)) * (2.0 * py / h - 1.0)
                absent_frames = 0
            else:
                absent_frames += 1

        # --- idle wander when nobody is around ---
        idle = bool(int(c.get("eye_idle", 1))) and absent_frames > int(c.get("eye_idle_delay", 50))
        if idle:
            t = now * 0.5 + wander_seed
            target_x = 0.7 * np.sin(t * 0.37) * np.sin(t * 0.11 + 1.0)
            target_y = 0.5 * np.sin(t * 0.29 + 1.3)

        # --- smooth the gaze so the iris glides ---
        gs = float(c.get("eye_gaze_smooth", 0.82))
        gaze_x = gs * gaze_x + (1.0 - gs) * float(np.clip(target_x, -1.0, 1.0))
        gaze_y = gs * gaze_y + (1.0 - gs) * float(np.clip(target_y, -1.0, 1.0))

        # --- pupil dilation from how close the person is (area in frame) ---
        pmin = float(c.get("eye_pupil_min", 2.6))
        pmax = float(c.get("eye_pupil_max", 6.5))
        if bool(int(c.get("eye_dilate", 1))) and not idle:
            frac = float(np.clip((area - 0.02) / (0.25 - 0.02), 0.0, 1.0))
            pupil_target = pmin + frac * (pmax - pmin)
        else:
            pupil_target = 0.5 * (pmin + pmax)
        pupil_r_s = 0.85 * pupil_r_s + 0.15 * pupil_target

        # --- blink schedule -> eyelid openness in [~0, 1] ---
        openness = 1.0
        if bool(int(c.get("eye_blink", 1))):
            dur = max(0.05, float(c.get("eye_blink_dur", 0.18)))
            if now >= next_blink and now - blink_start > dur:
                blink_start = now
                every = max(0.8, float(c.get("eye_blink_every", 4.5)))
                next_blink = now + random.uniform(0.6 * every, 1.4 * every)
            if now - blink_start < dur:
                phase = (now - blink_start) / dur        # 0..1
                openness = 1.0 - np.sin(np.pi * phase)   # 1 -> 0 -> 1 (closed at mid)

        # ------------------- render the eye at 30x120 -------------------
        feather = float(c.get("eye_feather", 1.3))
        half_w = max(2.0, float(c.get("eye_half_w", 15.0)))
        half_h = max(2.0, float(c.get("eye_half_h", 20.0)))
        cx0 = (MATRIX_WIDTH - 1) / 2.0 + float(c.get("eye_cx", 0.0))
        cy0 = (MATRIX_HEIGHT - 1) / 2.0 + float(c.get("eye_cy", 0.0))

        # Per-lane wiring correction, applied as a per-COLUMN vertical shift of
        # the render coordinate (not a roll of the finished frame). The eye is
        # drawn with soft edges, so this shift straightens the iris/outline on
        # the wall while the pupil stays smooth instead of combing into lines.
        lane_state = load_lane_offsets(lane_state)
        offs = lane_state[0]
        if bool(int(c.get("eye_lane_correct", 1))) and any(offs):
            YYc = YY - np.asarray(offs, np.float32)[None, :]   # shift each column
        else:
            YYc = YY

        nx = (XX - cx0) / half_w                 # -1..1 across the eye
        # almond lids: pointier corners (eye_corner>1) and an upper lid taller
        # than the lower one, like a real eye. Upper/lower handled separately.
        corner = max(0.3, float(c.get("eye_corner", 1.15)))
        curve = np.clip(1.0 - nx * nx, 0.0, 1.0) ** corner
        op = max(0.0, openness)
        upper = half_h * float(c.get("eye_upper", 1.0)) * curve * op
        lower = half_h * float(c.get("eye_lower", 0.72)) * curve * op
        vy = YYc - cy0
        # inside the lids when this is positive (upper arc above, lower arc below)
        signed = np.where(vy <= 0.0, upper + vy, lower - vy)
        corner_gate = np.clip((1.0 - nx * nx) / 0.05, 0.0, 1.0)
        edge_feather = feather * float(c.get("eye_edge_soft", 1.7))
        eye_alpha = smoothstep_alpha(signed, edge_feather) * corner_gate

        # iris + pupil follow the gaze, clipped by the eyelids
        iris_r = float(c.get("eye_iris_r", 8.0))
        max_tx = max(0.0, half_w - iris_r)
        max_ty = max(0.0, half_h * min(float(c.get("eye_upper", 1.0)),
                                       float(c.get("eye_lower", 0.72))) - iris_r)
        track = float(c.get("eye_track", 1.0))
        ix = cx0 + gaze_x * max_tx * track
        iy = cy0 + gaze_y * max_ty * track
        ir = np.sqrt((XX - ix) ** 2 + (YYc - iy) ** 2)
        iris_alpha = smoothstep_alpha(iris_r - ir, feather) * eye_alpha
        pupil_alpha = smoothstep_alpha(pupil_r_s - ir, feather) * eye_alpha

        sclera = np.array(c.get("eye_sclera_col", [255, 250, 238]), np.float32) / 255.0
        iris_col = np.array(c.get("eye_iris_col", [60, 140, 255]), np.float32) / 255.0

        canvas = eye_alpha[:, :, None] * sclera[None, None, :]
        # iris with a subtle limbal ring (darker rim) + brighter center
        radial = np.clip(1.0 - ir / max(1e-3, iris_r), 0.0, 1.0)
        iris_shaded = iris_col[None, None, :] * (0.55 + 0.45 * radial)[:, :, None]
        canvas = canvas * (1.0 - iris_alpha[:, :, None]) + iris_shaded * iris_alpha[:, :, None]
        # pupil (black)
        canvas = canvas * (1.0 - pupil_alpha[:, :, None])
        # catchlight: small specular dot up-left of the iris
        if bool(int(c.get("eye_catchlight", 1))):
            clx, cly = ix - iris_r * 0.35, iy - iris_r * 0.4
            cl = np.sqrt((XX - clx) ** 2 + (YYc - cly) ** 2)
            cl_alpha = smoothstep_alpha(max(1.4, iris_r * 0.22) - cl, feather) * eye_alpha
            canvas = canvas * (1.0 - cl_alpha[:, :, None]) + cl_alpha[:, :, None]

        canvas = np.clip(canvas, 0.0, 1.0)

        # small temporal smoothing to calm LED shimmer (gaze/blink already move)
        rs = float(c.get("eye_render_smooth", 0.3))
        smoothed = canvas if smoothed is None else rs * smoothed + (1.0 - rs) * canvas

        bright = max(0.0, min(1.0, float(c["brightness"])))
        out_rgb = np.clip(smoothed * bright * 255.0, 0, 255).astype(np.uint8)
        order = str(c["color_order"]).upper()
        perm = [_CH.get(ch, i) for i, ch in enumerate(order[:3])]
        out_rgb = out_rgb[:, :, perm]
        pixel_bytes = out_rgb.tobytes()

        for i in range(0, NUM_LEDS, MAX_PIXELS_PER_PACKET):
            chunk_pixels = min(MAX_PIXELS_PER_PACKET, NUM_LEDS - i)
            chunk_bytes = chunk_pixels * 3
            byte_offset = i * 3
            is_last = (i + MAX_PIXELS_PER_PACKET) >= NUM_LEDS
            # PUSH (0x01) only on the LAST packet so WLED renders the whole
            # frame atomically; pushing every packet tears the eye at the
            # 16-row packet boundaries (rows 48 & 64 cross the pupil).
            flags = 0x41 if is_last else 0x40
            header = bytes([
                flags, 0x00, 0x01, 0x00,
                (byte_offset >> 24) & 0xFF, (byte_offset >> 16) & 0xFF,
                (byte_offset >> 8) & 0xFF, byte_offset & 0xFF,
                (chunk_bytes >> 8) & 0xFF, chunk_bytes & 0xFF
            ])
            sock.sendto(header + pixel_bytes[byte_offset:byte_offset + chunk_bytes], (WLED_IP, UDP_PORT))

        fps_n += 1
        if time.time() - fps_t0 >= 2.0:
            print(f"  {fps_n / (time.time() - fps_t0):.1f} fps", flush=True)
            fps_t0, fps_n = time.time(), 0

except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    cam.stop()
    cap.release()
    segmenter.close()
    sock.close()
