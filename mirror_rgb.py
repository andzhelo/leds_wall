"""Colour mirror: your REAL video colours inside a MediaPipe person mask.

Why this instead of background subtraction (mirror_led.py): MOG2 absorbs
anything that stands still (legs vanish after ~2 min) and crumbles on dark
clothes vs dark background. MediaPipe knows what a person is -- no warmup,
no absorption, clean single shape -- and the camera image itself supplies
the colours, so a hand on the chest reads naturally (skin vs shirt).

rgb_* knobs in calibration.json (hot-reloaded, shared with mirror_udp.py):
  rgb_mask   1 = person only on black (the mirror look), 0 = raw full video
  rgb_sat    saturation boost so clothes pop on LEDs (1 = natural)
  rgb_gamma  <1 brightens indoor mids so dark clothes still light up
  rgb_gain   overall brightness multiplier applied inside the mask
"""
import cv2
import socket
import numpy as np
import json
import os
import threading
import time
import signal
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

DEFAULT_CALIB = {
    "rotate": 0,
    "flip_code": 0,
    "zoom": 1.0,
    "pan_x": 0.0,
    "pan_y": 0.0,
    "offset_x": 0,
    "offset_y": 0,
    "brightness": 1.0,
    "color_order": "RGB",
    "smoothing": 0.2,
    "mask_threshold": 0.4,
    "denoise": True,
    "erode": 0,

    "rgb_mask": 1,
    "rgb_sat": 1.6,
    "rgb_gamma": 0.65,
    "rgb_gain": 1.3,
    # auto-brightness inside your silhouette: 1 = adapt to lighting (rgb_gain
    # is ignored), 0 = fixed rgb_gain
    "rgb_auto": 1,
    "rgb_auto_target": 0.85,
    # when nobody is in frame: 0 = wall off, 1 = faint slow breathing glow
    "rgb_idle": 1,
    "rgb_idle_bright": 0.06,
    "rgb_idle_col": [255, 190, 128],   # the glow colour (reads yellow-ish on wood)
    "rgb_idle_delay": 60,              # frames (~6s) of absence before the glow
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

calib_state = load_calibration((dict(DEFAULT_CALIB), 0.0))
smoothed = None
ASPECT = MATRIX_WIDTH / MATRIX_HEIGHT

print("Warming up camera + MediaPipe model (a few seconds)...")
cam = FreshestFrame(cap)
_wait0 = time.time()
while cam.read() is None:
    if time.time() - _wait0 > 8.0:
        raise SystemExit("ERROR: no camera frames within 8s -- is /dev/video0 in use?")
    time.sleep(0.05)
segmenter.process(cv2.cvtColor(cam.read(), cv2.COLOR_BGR2RGB))
print(f"Ready. Streaming COLOUR mirror to {WLED_IP} ... edit {CALIB_PATH} live. (Ctrl+C)")

fps_t0, fps_n = time.time(), 0
auto_gain = 1.0
absent_frames = 999

try:
    while True:
        calib_state = load_calibration(calib_state)
        c = calib_state[0]

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

        # --- Person mask (MediaPipe: robust, no warmup, never absorbs you) ---
        use_mask = bool(int(c.get("rgb_mask", 1)))
        if use_mask:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = segmenter.process(rgb)
            if results.segmentation_mask is None:
                continue
            person = (results.segmentation_mask > float(c["mask_threshold"])).astype(np.uint8)

            if bool(c["denoise"]):
                person = cv2.morphologyEx(person, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(person, 8)
                if n_lbl > 1:
                    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                    if stats[largest, cv2.CC_STAT_AREA] > (w * h * 0.003):
                        person = (labels == largest).astype(np.uint8)

            er = int(c.get("erode", 0))
            if er > 0:
                person = cv2.erode(person, np.ones((3, 3), np.uint8), iterations=er)
            person *= 255

            # presence: person occupies enough of the frame (sticky hysteresis:
            # easy to keep, hard to lose, so a distant/edge person stays "present")
            area = float(person.mean()) / 255.0
            if area > 0.005:
                absent_frames = 0
            elif area < 0.002:
                absent_frames += 1

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
        ys, xs = slice(y0, y0 + crop_h), slice(x0, x0 + crop_w)

        if zoom_raw < 1.0:
            tw = max(1, int(round(MATRIX_WIDTH * zoom_raw)))
            th = max(1, int(round(MATRIX_HEIGHT * zoom_raw)))
        else:
            tw, th = MATRIX_WIDTH, MATRIX_HEIGHT

        # --- Downscale FIRST, then colour math at 30x120 (cheap) ---
        vid = cv2.resize(frame[ys, xs], (tw, th), interpolation=cv2.INTER_AREA)

        sat = float(c.get("rgb_sat", 1.0))
        if sat != 1.0:                      # make clothes colours pop on LEDs
            hsv = cv2.cvtColor(vid, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat, 0, 255)
            vid = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        look = vid[:, :, ::-1].astype(np.float32) / 255.0   # BGR -> RGB
        gmm = float(c.get("rgb_gamma", 1.0))
        if gmm != 1.0:                      # brighten indoor mids
            look = np.power(look, gmm)

        alpha = None
        if use_mask:
            alpha = cv2.resize(person[ys, xs], (tw, th),
                               interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

        # auto-brightness: keep the person vivid whatever the room lighting
        if bool(int(c.get("rgb_auto", 1))):
            sel = look.max(axis=2)[alpha > 0.5] if alpha is not None else look.max(axis=2).ravel()
            if sel.size > 20:
                p95 = float(np.percentile(sel, 95))
                if p95 > 1e-3:
                    g = min(4.0, max(0.4, float(c.get("rgb_auto_target", 0.85)) / p95))
                    auto_gain = 0.9 * auto_gain + 0.1 * g   # no flicker
            look *= auto_gain
        else:
            look *= float(c.get("rgb_gain", 1.0))

        # compress highlights towards their own hue instead of clipping to white
        mx = look.max(axis=2)
        over = mx > 1.0
        if over.any():
            look[over] /= mx[over][:, None]

        if alpha is not None:
            look *= alpha[:, :, None]
        look = np.clip(look, 0.0, 1.0)

        if zoom_raw < 1.0:
            canvas = np.zeros((MATRIX_HEIGHT, MATRIX_WIDTH, 3), dtype=np.float32)
            cx = (MATRIX_WIDTH - tw) // 2 + int(c.get("offset_x", 0))
            cy = (MATRIX_HEIGHT - th) // 2 + int(c.get("offset_y", 0))
            dx0, dy0 = max(0, cx), max(0, cy)
            dx1, dy1 = min(MATRIX_WIDTH, cx + tw), min(MATRIX_HEIGHT, cy + th)
            if dx1 > dx0 and dy1 > dy0:
                canvas[dy0:dy1, dx0:dx1] = look[dy0 - cy:dy1 - cy, dx0 - cx:dx1 - cx]
        else:
            canvas = look

        # nobody there for a while -> faint slow breathing glow (attract mode)
        sm = float(c["smoothing"])
        if use_mask and bool(int(c.get("rgb_idle", 1))) and \
                absent_frames > int(c.get("rgb_idle_delay", 60)):
            idle_col = np.array(c.get("rgb_idle_col", [255, 190, 128]), np.float32) / 255.0
            wave = 0.5 + 0.5 * np.sin(2.0 * np.pi *
                                      (np.arange(MATRIX_HEIGHT, dtype=np.float32) / MATRIX_HEIGHT * 1.2
                                       - time.time() / 10.0))
            canvas = (idle_col[None, None, :] *
                      (float(c.get("rgb_idle_bright", 0.06)) * wave)[:, None, None]
                      ) * np.ones((1, MATRIX_WIDTH, 1), np.float32)
            sm = max(sm, 0.9)   # fade INTO the glow; a person still appears instantly
        smoothed = canvas if smoothed is None else sm * smoothed + (1.0 - sm) * canvas

        bright = max(0.0, min(1.0, float(c["brightness"])))
        out_rgb = np.clip(smoothed * bright * 255.0, 0, 255).astype(np.uint8)
        order = str(c["color_order"]).upper()
        perm = [_CH.get(ch, i) for i, ch in enumerate(order[:3])]
        pixel_bytes = out_rgb[:, :, perm].tobytes()

        for i in range(0, NUM_LEDS, MAX_PIXELS_PER_PACKET):
            chunk_pixels = min(MAX_PIXELS_PER_PACKET, NUM_LEDS - i)
            chunk_bytes = chunk_pixels * 3
            byte_offset = i * 3
            is_last = (i + MAX_PIXELS_PER_PACKET) >= NUM_LEDS
            flags = 0x41 if is_last else 0x40   # push only on the last packet (no tearing)
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
