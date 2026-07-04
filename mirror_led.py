"""The floor-POC look (led.py), ported to the wall.

MOG2 background subtraction -> solid filled silhouette in one colour.
Simple on purpose: this is the base we iterate on via calibration.json
(hot-reloaded every frame, shared with mirror_udp.py -- geometry keys
rotate/flip/zoom/pan are the same ones already tuned).
"""
import cv2
import socket
import numpy as np
import json
import os
import threading
import time
import signal

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
    # shared geometry (already tuned for the camera position)
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
    "erode": 0,

    # led.py-look knobs
    "col_solid": [255, 255, 255],  # the silhouette colour (POC was white)
    "led_bg_threshold": 50,        # MOG2 sensitivity (led.py used 50)
    "led_bg_learning": 0.0005,     # 0 = frozen background, never absorbs you
    "led_bg_warmup_sec": 3.0,      # empty-scene learning time -- step out!
    "led_bg_reset": 0,             # bump this number to relearn live
    "fill_holes": True,            # solid contour fill = the POC's clean look
    "min_blob": 0.003,             # drop blobs smaller than this frame fraction
    "close": 2,                    # bridge mask fragments before the fill (anti-holes)

    # depth: skin gets its own colour (set col_skin != col_solid to see it)
    "col_skin": [255, 255, 255],
    "skin_cr": [133, 175],
    "skin_cb": [77, 128],

    # depth: brightness follows your real luminance (same knobs as mirror_udp)
    "shading": 0.8,                # 0 = flat POC look, higher = 3D form
    "contrast": 1.8,
    "gamma": 0.7,
    "sharpen": 1.5,
    "luma_denoise": 3,
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

calib_state = load_calibration((dict(DEFAULT_CALIB), 0.0))
smoothed = None
ASPECT = MATRIX_WIDTH / MATRIX_HEIGHT

print("Warming up camera...")
cam = FreshestFrame(cap)
_wait0 = time.time()
while cam.read() is None:
    if time.time() - _wait0 > 8.0:
        raise SystemExit("ERROR: no camera frames within 8s -- is /dev/video0 in use?")
    time.sleep(0.05)


def learn_background(c):
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=float(c["led_bg_threshold"]), detectShadows=False)
    secs = float(c.get("led_bg_warmup_sec", 3.0))
    print(f"Learning background for {secs:.0f}s -- STEP OUT OF FRAME...")
    t0 = time.time()
    while time.time() - t0 < secs:
        f = cam.read()
        if f is not None:
            fgbg.apply(f, learningRate=-1)
        time.sleep(0.01)
    print("Background learned.")
    return fgbg


c0 = calib_state[0]
fgbg = learn_background(c0)
last_bg_reset = c0.get("led_bg_reset", 0)
print(f"Ready. Streaming POC-style silhouette to {WLED_IP} ... edit {CALIB_PATH} live. (Ctrl+C)")

fps_t0, fps_n = time.time(), 0

try:
    while True:
        calib_state = load_calibration(calib_state)
        c = calib_state[0]

        if c.get("led_bg_reset", 0) != last_bg_reset:
            last_bg_reset = c.get("led_bg_reset", 0)
            fgbg = learn_background(c)
        fgbg.setVarThreshold(float(c["led_bg_threshold"]))

        frame = cam.read()
        if frame is None:
            time.sleep(0.005)
            continue

        # --- Background removal (model sees the raw frame; mask follows geometry) ---
        fg_mask = fgbg.apply(frame, learningRate=float(c.get("led_bg_learning", 0.0)))

        rot = int(c.get("rotate", 0))
        if rot in _ROT:
            fg_mask = cv2.rotate(fg_mask, _ROT[rot])
            frame = cv2.rotate(frame, _ROT[rot])
        flip_code = int(c["flip_code"])
        if flip_code in (-1, 0, 1):
            fg_mask = cv2.flip(fg_mask, flip_code)
            frame = cv2.flip(frame, flip_code)
        h, w = fg_mask.shape

        fg_mask = cv2.medianBlur(fg_mask, 5)
        _, clean = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # bridge broken fragments into one body BEFORE the fill (anti-holes)
        cl = int(c.get("close", 2))
        if cl > 0:
            clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                                     iterations=cl)

        # --- THE SOLID FILL (the POC's clean look): keep big blobs, fill them solid ---
        if bool(c.get("fill_holes", True)):
            contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = float(c.get("min_blob", 0.003)) * w * h
            big = [ct for ct in contours if cv2.contourArea(ct) >= min_area]
            clean = np.zeros_like(clean)
            cv2.drawContours(clean, big, -1, 255, thickness=cv2.FILLED)

        er = int(c.get("erode", 0))
        if er > 0:
            clean = cv2.erode(clean, np.ones((3, 3), np.uint8), iterations=er)

        # --- Depth 1: skin detection (hands/face get col_skin) -- only when it
        # would be visible, i.e. col_skin differs from col_solid ---
        col = np.array(c["col_solid"], np.float32) / 255.0
        skin_col = np.array(c.get("col_skin", c["col_solid"]), np.float32) / 255.0
        want_skin = not np.array_equal(col, skin_col)
        shading = float(c.get("shading", 0.0))
        ycrcb = None
        skin = None
        if want_skin or shading > 0:
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        if want_skin:
            cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
            cr_lo, cr_hi = c["skin_cr"]
            cb_lo, cb_hi = c["skin_cb"]
            skin = (((cr >= cr_lo) & (cr <= cr_hi) & (cb >= cb_lo) & (cb <= cb_hi))
                    .astype(np.uint8) * 255)
            skin[clean == 0] = 0

        # --- Depth 2: hi-res luma prep (uint8, cheap) for brightness shading ---
        luma = None
        if shading > 0:
            luma = ycrcb[:, :, 0]
            ld = int(c.get("luma_denoise", 3))
            if ld >= 3 and ld % 2 == 1:
                luma = cv2.medianBlur(luma, ld)
            clip = float(c.get("contrast", 1.8))
            if clip > 0:
                luma = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(luma)
            sharpen = float(c.get("sharpen", 1.0))
            if sharpen > 0:
                blur = cv2.GaussianBlur(luma, (0, 0), 1.2)
                luma = cv2.addWeighted(luma, 1.0 + sharpen, blur, -sharpen, 0)

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

        # all float colour math happens at matrix size (30x120) -- cheap
        pa = cv2.resize(clean[ys, xs], (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        if skin is not None:
            sa = cv2.resize(skin[ys, xs], (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            sa = np.minimum(sa, pa)
            look = col * (pa - sa)[:, :, None] + skin_col * sa[:, :, None]
        else:
            look = col * pa[:, :, None]

        if shading > 0:
            gray = cv2.resize(luma[ys, xs], (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            body = pa > 0.5
            if int(body.sum()) > 10:
                lo, hi = np.percentile(gray[body], (3, 97))
                if hi > lo + 1e-3:
                    gray = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
            gmm = float(c.get("gamma", 0.8))
            if gmm != 1.0:
                gray = np.power(gray, gmm)
            look *= ((1.0 - shading) + shading * gray)[:, :, None]
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

        sm = float(c["smoothing"])
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
    sock.close()
