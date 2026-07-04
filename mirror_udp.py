import cv2
import socket
import numpy as np
import json
import os
import threading
import time
import signal
import mediapipe as mp

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

    "mask_threshold": 0.4,    # person-confidence cutoff
    "denoise": True,
    # erode shrinks the silhouette outline: opens the gap between touching parts
    # (e.g. two hands together) and sharpens the shape. 0 = off, try 1-3.
    "erode": 2,
    "luma_denoise": 3,        # median blur (odd, 0=off) to kill grain before boosting
    "smoothing": 0.2,         # temporal anti-flicker (lower = crisper motion)
    "brightness": 1.0,
    "color_order": "RGB",

    # WHITE "GHOST" look (the floor-POC look): the body is white, shaded by your
    # real brightness so fingers/face/folds show as depth. Set the two colours
    # differently (e.g. skin warm, clothes green) for a two-tone look instead.
    "col_skin":    [255, 255, 255],   # face + hands
    "col_clothes": [255, 255, 255],   # torso/legs
    "shading": 0.8,           # 0 = flat, ~0.8 = strong 3D depth
    "contrast": 1.8,          # local-contrast (CLAHE) boost, 0 = off; lower = less noise
    "gamma": 0.7,             # <1 brightens mids, >1 deepens shadows
    "sharpen": 1.5,           # unsharp-mask crispness, 0 = off (soft), try 0.5-2
    # Skin detector (YCrCb). Widen/narrow if your skin isn't caught/over-caught.
    "skin_cr": [133, 175],
    "skin_cb": [77, 128],
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
print(f"Ready. Streaming two-tone silhouette to {WLED_IP} ... edit {CALIB_PATH} live. (Ctrl+C)")

fps_t0, fps_n = time.time(), 0

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

        # --- Person mask (fast selfie segmentation) ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = segmenter.process(rgb)
        if results.segmentation_mask is None:
            continue
        mask = results.segmentation_mask
        h, w = mask.shape
        thr = float(c["mask_threshold"])
        person = mask > thr

        if bool(c["denoise"]):
            binm = cv2.morphologyEx(person.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(binm, 8)
            if n_lbl > 1:
                largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                if stats[largest, cv2.CC_STAT_AREA] > (w * h * 0.003):
                    person = person & (labels == largest)

        # erode to open gaps between touching parts (anti-merge) + crisp outline
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

        # --- Build the two-tone RGB look, shaded by luminance for body form ---
        skin_col = np.array(c["col_skin"], np.float32) / 255.0
        clothes_col = np.array(c["col_clothes"], np.float32) / 255.0
        look = np.zeros((h, w, 3), np.float32)
        look[person] = clothes_col
        look[skin] = skin_col
        shading = float(c.get("shading", 0.65))
        if shading > 0:
            luma = ycrcb[:, :, 0]                                   # 0..255 luma
            ld = int(c.get("luma_denoise", 3))
            if ld >= 3 and ld % 2 == 1:                             # kill sensor grain first
                luma = cv2.medianBlur(luma, ld)
            clip = float(c.get("contrast", 1.8))
            if clip > 0:                                            # local contrast reveals fingers/face
                luma = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(luma)
            sharpen = float(c.get("sharpen", 1.0))
            if sharpen > 0:                                         # unsharp mask -> crisp, not fuzzy
                blur = cv2.GaussianBlur(luma, (0, 0), 1.2)
                luma = cv2.addWeighted(luma, 1.0 + sharpen, blur, -sharpen, 0)
            gray = luma.astype(np.float32) / 255.0
            m = person
            if int(m.sum()) > 50:                                  # stretch over the body only
                lo, hi = np.percentile(gray[m], (3, 97))
                if hi > lo + 1e-3:
                    gray = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
            gmm = float(c.get("gamma", 0.8))
            if gmm != 1.0:
                gray = np.power(gray, gmm)
            look *= ((1.0 - shading) + shading * gray)[:, :, None]

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
    segmenter.close()
    sock.close()
