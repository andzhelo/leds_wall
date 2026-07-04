import cv2
import socket
import numpy as np
import json
import os
import threading
import time
import signal

# --- 1. SYSTEM CONFIGURATION ---
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

    "mask_threshold": 0.4,    # kept for calibration.json compat (unused here)
    "denoise": True,
    "erode": 2,
    "luma_denoise": 3,
    "smoothing": 0.2,
    "brightness": 1.0,
    "color_order": "RGB",

    "col_skin":    [255, 255, 255],
    "col_clothes": [255, 255, 255],
    "shading": 0.8,
    "contrast": 1.8,
    "gamma": 0.7,
    "sharpen": 1.5,
    "skin_cr": [133, 175],
    "skin_cb": [77, 128],

    # --- Background-subtraction knobs (no MediaPipe) ---
    # Sensitivity: lower = catches more of you (and more noise). Try 12-40.
    "bg_threshold": 24,
    # How fast the model keeps adapting AFTER warmup. 0 = frozen background
    # (never absorbs you if you stand still, but breaks when lighting changes).
    # Small values like 0.0005 slowly track lighting but will swallow you
    # after standing statue-still for a few minutes.
    "bg_learning": 0.0005,
    # Seconds of empty-scene learning at startup. STEP OUT OF FRAME during this!
    "bg_warmup_sec": 3.0,
    # Bump this number (any change) in calibration.json to relearn the
    # background live, e.g. after moving the camera or big light change.
    "bg_reset": 0,
    # Close-gap iterations: fills holes where clothes match the background.
    "close": 2,
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


def make_subtractor(c):
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=float(c["bg_threshold"]), detectShadows=True)
    return fgbg


def learn_background(fgbg, c):
    """Feed empty-scene frames at full learning rate. Step out of frame!"""
    secs = float(c.get("bg_warmup_sec", 3.0))
    print(f"Learning background for {secs:.0f}s -- STEP OUT OF FRAME...")
    t0 = time.time()
    n = 0
    while time.time() - t0 < secs:
        f = cam.read()
        if f is not None:
            fgbg.apply(f, learningRate=-1)
            n += 1
        time.sleep(0.01)
    print(f"Background learned ({n} frames).")


c0 = calib_state[0]
fgbg = make_subtractor(c0)
learn_background(fgbg, c0)
last_bg_reset = c0.get("bg_reset", 0)
print(f"Ready. Streaming bg-sub silhouette to {WLED_IP} ... edit {CALIB_PATH} live. (Ctrl+C)")

fps_t0, fps_n = time.time(), 0

try:
    while True:
        calib_state = load_calibration(calib_state)
        c = calib_state[0]

        # live re-learn trigger (change "bg_reset" in calibration.json)
        if c.get("bg_reset", 0) != last_bg_reset:
            last_bg_reset = c.get("bg_reset", 0)
            fgbg = make_subtractor(c)
            learn_background(fgbg, c)
        fgbg.setVarThreshold(float(c["bg_threshold"]))

        frame = cam.read()
        if frame is None:
            time.sleep(0.005)
            continue

        # --- Person mask (MOG2 background subtraction, no MediaPipe) ---
        fgmask = fgbg.apply(frame, learningRate=float(c.get("bg_learning", 0.0)))

        rot = int(c.get("rotate", 0))
        if rot in _ROT:
            frame = cv2.rotate(frame, _ROT[rot])
            fgmask = cv2.rotate(fgmask, _ROT[rot])
        flip_code = int(c["flip_code"])
        if flip_code in (-1, 0, 1):
            frame = cv2.flip(frame, flip_code)
            fgmask = cv2.flip(fgmask, flip_code)

        h, w = fgmask.shape
        person = fgmask > 200          # 255 = foreground; 127 = shadow -> dropped

        binm = person.astype(np.uint8)
        binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cl = int(c.get("close", 2))
        if cl > 0:                     # bridge holes where clothes match the wall
            binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8),
                                    iterations=cl)
        person = binm.astype(bool)

        if bool(c["denoise"]):
            n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(binm, 8)
            if n_lbl > 1:
                largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                if stats[largest, cv2.CC_STAT_AREA] > (w * h * 0.003):
                    person = person & (labels == largest)

        er = int(c.get("erode", 0))
        if er > 0:
            person = cv2.erode(person.astype(np.uint8), np.ones((3, 3), np.uint8),
                               iterations=er).astype(bool)

        # --- Skin detection (cheap YCrCb threshold) -> face/hands/arms ---
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
        cr_lo, cr_hi = c["skin_cr"]
        cb_lo, cb_hi = c["skin_cb"]
        skin = (cr >= cr_lo) & (cr <= cr_hi) & (cb >= cb_lo) & (cb <= cb_hi) & person

        # --- Hi-res luma prep (uint8, cheap) -- detail survives the downscale ---
        shading = float(c.get("shading", 0.65))
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

        # Downscale masks+luma to matrix size FIRST, then do the float colour
        # math on 30x120 instead of 640x480 (~10x faster, same look: the
        # INTER_AREA mask resize gives the same soft anti-aliased edges).
        if zoom_raw < 1.0:
            tw = max(1, int(round(MATRIX_WIDTH * zoom_raw)))
            th = max(1, int(round(MATRIX_HEIGHT * zoom_raw)))
        else:
            tw, th = MATRIX_WIDTH, MATRIX_HEIGHT
        pa = cv2.resize(person[ys, xs].astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA)
        sa = cv2.resize(skin[ys, xs].astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA)
        sa = np.minimum(sa, pa)

        skin_col = np.array(c["col_skin"], np.float32) / 255.0
        clothes_col = np.array(c["col_clothes"], np.float32) / 255.0
        look = clothes_col * (pa - sa)[:, :, None] + skin_col * sa[:, :, None]

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
