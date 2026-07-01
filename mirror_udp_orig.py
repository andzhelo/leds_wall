import cv2
import socket
import numpy as np
import json
import os
import mediapipe as mp

# --- 1. SYSTEM CONFIGURATION ---
WLED_IP = "192.168.1.60"
UDP_PORT = 4048

# LED Wall Dimensions (WLED 2D Matrix Mode)
MATRIX_WIDTH = 30
MATRIX_HEIGHT = 120
NUM_LEDS = MATRIX_WIDTH * MATRIX_HEIGHT

CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
MAX_PIXELS_PER_PACKET = 480

# --- 2. LIVE CALIBRATION ---
# Edit calibration.json while running; changes apply within ~1s, no restart.
CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")

DEFAULT_CALIB = {
    # Rotate the camera image to upright: 0, 90, 180, or 270 degrees.
    # Use 90 or 270 when the camera is physically mounted in portrait orientation.
    "rotate": 0,
    # Applied AFTER rotate. 0 = vertical flip, 1 = horizontal mirror, -1 = both,
    # 2 = no flip.
    "flip_code": 0,

    # --- MIRROR FRAMING (fixed, stable -- no per-frame resizing) ---
    # A fixed window of the camera maps straight onto the wall, so you appear at
    # natural size and move like a real mirror.
    # zoom: 1.0 = the full camera height maps to the wall. >1.0 zooms IN (you look
    #   bigger, tighter framing); e.g. 1.5 uses the middle 2/3 of the view.
    "zoom": 1.0,
    # pan the window to where you actually stand, range about -1..1 (0 = centred).
    # Your spot is to the left, so a negative pan_x will likely help.
    "pan_x": 0.0,
    "pan_y": 0.0,

    # --- LOOK ---
    "mask_threshold": 0.5,   # person-confidence cutoff 0..1
    "denoise": True,         # keep only the largest blob (kills stray specks)
    "soft_edges": True,      # true = glow at edges, false = crisp solid
    "smoothing": 0.4,        # anti-flicker 0..1 (higher = smoother, slight lag)
    "brightness": 1.0,       # 0..1
    "color": [84, 30, 32],   # silhouette colour as [R, G, B]
    # Byte order sent to WLED. Set by the colour test. Options: RGB GRB BRG RBG GBR BGR
    "color_order": "RGB",
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


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)

calib_state = load_calibration((dict(DEFAULT_CALIB), 0.0))
smoothed = None
ASPECT = MATRIX_WIDTH / MATRIX_HEIGHT  # 0.25

print(f"Streaming mirror silhouette to {WLED_IP} ... edit {CALIB_PATH} to calibrate live. (Ctrl+C to stop)")

try:
    while True:
        calib_state = load_calibration(calib_state)
        c = calib_state[0]

        ret, frame = cap.read()
        if not ret:
            break

        rot = int(c.get("rotate", 0))
        if rot in _ROT:
            frame = cv2.rotate(frame, _ROT[rot])
        flip_code = int(c["flip_code"])
        if flip_code in (-1, 0, 1):
            frame = cv2.flip(frame, flip_code)

        # --- Segment the person ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = segmenter.process(rgb)
        if results.segmentation_mask is None:
            continue
        mask = results.segmentation_mask
        h, w = mask.shape

        thr = float(c["mask_threshold"])
        if bool(c["soft_edges"]):
            denom = max(1e-3, 1.0 - thr)
            alpha = np.clip((mask - thr) / denom, 0.0, 1.0)
        else:
            b = (mask > thr).astype(np.uint8) * 255
            k = np.ones((5, 5), np.uint8)
            b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k)
            b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
            alpha = b.astype(np.float32) / 255.0

        # Remove stray noise (e.g. photos on the wall) by keeping only the
        # largest blob. This does NOT move or resize you -- framing stays fixed.
        if bool(c["denoise"]):
            binm = cv2.morphologyEx((mask > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(binm, 8)
            if n_lbl > 1:
                largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                if stats[largest, cv2.CC_STAT_AREA] > (w * h * 0.003):
                    alpha = alpha * (labels == largest)

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
        window = alpha[y0:y0 + crop_h, x0:x0 + crop_w]

        # Uniform scale to the panel (window is already 1:4, so no distortion).
        # zoom < 1.0 shrinks the figure on the panel, centred with a margin
        # (nudge with offset_x / offset_y in LED pixels, +offset_y = down).
        if zoom_raw < 1.0:
            tw = max(1, int(round(MATRIX_WIDTH * zoom_raw)))
            th = max(1, int(round(MATRIX_HEIGHT * zoom_raw)))
            small = cv2.resize(window, (tw, th), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((MATRIX_HEIGHT, MATRIX_WIDTH), dtype=np.float32)
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

        # --- Colourise + channel order ---
        bright = max(0.0, min(1.0, float(c["brightness"])))
        rgb_color = np.array(c["color"], dtype=np.float32)
        out_rgb = np.clip(smoothed[:, :, None] * rgb_color * bright, 0, 255).astype(np.uint8)
        order = str(c["color_order"]).upper()
        perm = [_CH.get(ch, i) for i, ch in enumerate(order[:3])]
        out = out_rgb[:, :, perm]
        pixel_bytes = out.tobytes()

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

except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    cap.release()
    segmenter.close()
    sock.close()
