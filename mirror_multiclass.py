import cv2
import socket
import numpy as np
import json
import os
import threading
import time
import signal
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --- 1. SYSTEM CONFIGURATION ---
WLED_IP = "192.168.1.61"
UDP_PORT = 4048

MATRIX_WIDTH = 30
MATRIX_HEIGHT = 120
NUM_LEDS = MATRIX_WIDTH * MATRIX_HEIGHT

CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
MAX_PIXELS_PER_PACKET = 480

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "selfie_multiclass.tflite")
CALIB_PATH = os.path.join(BASE_DIR, "calibration.json")

# Multiclass model categories:
#   0 background | 1 hair | 2 body-skin | 3 face-skin | 4 clothes | 5 others
DEFAULT_CALIB = {
    "rotate": 0,
    "flip_code": 0,

    # framing (same as before): zoom 1.0 = full height maps to wall; <1 shrinks
    # you on the panel with a margin; >1 zooms in. pan_x/pan_y move the window.
    "zoom": 1.0,
    "pan_x": 0.0,
    "pan_y": 0.0,
    "offset_x": 0,
    "offset_y": 0,

    "denoise": True,          # keep only the largest person blob
    "smoothing": 0.4,         # anti-flicker 0..1
    "brightness": 1.0,        # 0..1
    "color_order": "RGB",     # WLED byte order

    # --- SEMANTIC COLOURS [R,G,B] per body part (this is what gives detail) ---
    "col_skin":    [255, 200, 140],   # face + hands + arms  -> bright & warm
    "col_clothes": [0, 170, 50],      # your torso/legs      -> green
    "col_hair":    [150, 80, 30],     # hair                 -> brown
    "col_other":   [0, 120, 120],     # accessories/other
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
            print("Calibration reloaded:", json.dumps(merged))
            return merged, mtime
        except (json.JSONDecodeError, OSError) as e:
            print("Calibration not reloaded (will retry):", e)
            return calib, last_mtime
    return calib, last_mtime


def build_lut(c):
    """6x3 lookup table mapping category id -> RGB (0..1)."""
    lut = np.zeros((6, 3), np.float32)
    skin = np.array(c["col_skin"], np.float32) / 255.0
    lut[1] = np.array(c["col_hair"], np.float32) / 255.0
    lut[2] = skin                                   # body-skin (arms/hands)
    lut[3] = skin                                   # face-skin
    lut[4] = np.array(c["col_clothes"], np.float32) / 255.0
    lut[5] = np.array(c["col_other"], np.float32) / 255.0
    return lut


class FreshestFrame(threading.Thread):
    """Camera in its own thread, always keeps only the newest frame -> low latency."""
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


# Clean shutdown on SIGTERM (e.g. `timeout`, systemd stop) so the camera is
# released and doesn't stay busy for the next launch.
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

seg_opts = vision.ImageSegmenterOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    output_category_mask=True,
)
segmenter = vision.ImageSegmenter.create_from_options(seg_opts)

calib_state = load_calibration((dict(DEFAULT_CALIB), 0.0))
lut = build_lut(calib_state[0])
lut_mtime = calib_state[1]
smoothed = None
ASPECT = MATRIX_WIDTH / MATRIX_HEIGHT

print("Warming up camera + multiclass model (a few seconds)...")
cam = FreshestFrame(cap)
_wait0 = time.time()
while cam.read() is None:
    if time.time() - _wait0 > 8.0:
        raise SystemExit("ERROR: no camera frames within 8s -- is /dev/video0 in use?")
    time.sleep(0.05)
_warm = cam.read()
segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(_warm, cv2.COLOR_BGR2RGB)))
print(f"Ready. Streaming multiclass silhouette to {WLED_IP} ... edit {CALIB_PATH} live. (Ctrl+C)")

fps_t0, fps_n = time.time(), 0

try:
    while True:
        calib_state = load_calibration(calib_state)
        c = calib_state[0]
        if calib_state[1] != lut_mtime:               # rebuild colours on edit
            lut = build_lut(c)
            lut_mtime = calib_state[1]

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

        # --- Multiclass segmentation -> per-pixel category id ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        cat = result.category_mask.numpy_view()       # HxW uint8, 0..5
        h, w = cat.shape

        # colourise by body part (this is the "detail")
        look = lut[cat]                               # (h, w, 3) float 0..1
        person = cat > 0

        # keep only the largest person blob (kills stray specks)
        if bool(c["denoise"]):
            binm = cv2.morphologyEx(person.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(binm, 8)
            if n_lbl > 1:
                largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                if stats[largest, cv2.CC_STAT_AREA] > (w * h * 0.003):
                    look = look * (labels == largest)[:, :, None]

        # --- FIXED mirror window (1:4 aspect, zoom + pan), mapped to the wall ---
        zoom_raw = max(0.05, float(c["zoom"]))
        zoom = max(1.0, zoom_raw)
        crop_h = int(round(h / zoom))
        crop_w = int(round(crop_h * ASPECT))
        if crop_w > w:
            crop_w = w
            crop_h = int(round(w / ASPECT))
        crop_h = min(crop_h, h)
        mx, my = w - crop_w, h - crop_h
        x0 = int(round(mx / 2 + float(c["pan_x"]) * mx / 2))
        y0 = int(round(my / 2 + float(c["pan_y"]) * my / 2))
        x0 = max(0, min(x0, mx))
        y0 = max(0, min(y0, my))
        window = look[y0:y0 + crop_h, x0:x0 + crop_w]

        if zoom_raw < 1.0:
            tw = max(1, int(round(MATRIX_WIDTH * zoom_raw)))
            th = max(1, int(round(MATRIX_HEIGHT * zoom_raw)))
            small = cv2.resize(window, (tw, th), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((MATRIX_HEIGHT, MATRIX_WIDTH, 3), dtype=np.float32)
            cx = (MATRIX_WIDTH - tw) // 2 + int(c.get("offset_x", 0))
            cy = (MATRIX_HEIGHT - th) // 2 + int(c.get("offset_y", 0))
            dx0, dy0 = max(0, cx), max(0, cy)
            dx1, dy1 = min(MATRIX_WIDTH, cx + tw), min(MATRIX_HEIGHT, cy + th)
            if dx1 > dx0 and dy1 > dy0:
                canvas[dy0:dy1, dx0:dx1] = small[dy0 - cy:dy1 - cy, dx0 - cx:dx1 - cx]
        else:
            canvas = cv2.resize(window, (MATRIX_WIDTH, MATRIX_HEIGHT), interpolation=cv2.INTER_AREA)

        # --- Temporal smoothing ---
        sm = float(c["smoothing"])
        smoothed = canvas if smoothed is None else sm * smoothed + (1.0 - sm) * canvas

        # --- Brightness + channel order ---
        bright = max(0.0, min(1.0, float(c["brightness"])))
        out_rgb = np.clip(smoothed * bright * 255.0, 0, 255).astype(np.uint8)
        order = str(c["color_order"]).upper()
        perm = [_CH.get(ch, i) for i, ch in enumerate(order[:3])]
        pixel_bytes = out_rgb[:, :, perm].tobytes()

        # --- DDP output ---
        for i in range(0, NUM_LEDS, MAX_PIXELS_PER_PACKET):
            chunk_pixels = min(MAX_PIXELS_PER_PACKET, NUM_LEDS - i)
            chunk_bytes = chunk_pixels * 3
            byte_offset = i * 3
            is_last = (i + MAX_PIXELS_PER_PACKET) >= NUM_LEDS
            flags = 0x41 if is_last else 0x01
            header = bytes([
                flags, 0x00, 0x01, 0x00,
                (byte_offset >> 24) & 0xFF, (byte_offset >> 16) & 0xFF,
                (byte_offset >> 8) & 0xFF, byte_offset & 0xFF,
                (chunk_bytes >> 8) & 0xFF, chunk_bytes & 0xFF
            ])
            sock.sendto(header + pixel_bytes[byte_offset:byte_offset + chunk_bytes], (WLED_IP, UDP_PORT))

        fps_n += 1
        if time.time() - fps_t0 >= 2.0:
            print(f"  {fps_n / (time.time() - fps_t0):.1f} fps")
            fps_t0, fps_n = time.time(), 0

except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    cam.stop()
    cap.release()
    segmenter.close()
    sock.close()
