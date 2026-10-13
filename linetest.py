"""Lane calibration helper: straight horizontal lines + a lane ruler.

Paints evenly spaced straight white rows across all 30 lanes, with the current
wall_calib.json correction applied, so any lane that STILL looks offset is one
that needs fixing. A colour ruler on the top rows marks every 5th lane so the
exact lane index is readable from a phone photo:

    lane 0 = red, 5 = green, 10 = blue, 15 = yellow, 20 = magenta, 25 = cyan,
    29 = white  (count: the markers are 5 lanes apart)

Run on the Pi:  ./venv/bin/python linetest.py          # corrected (default)
                ./venv/bin/python linetest.py --raw     # no correction (raw comb)

Then photograph the wall. Ctrl+C to stop.
"""
import argparse
import json
import os
import socket
import time

import numpy as np

WLED_IP = "192.168.1.60"
UDP_PORT = 4048
MATRIX_WIDTH = 30
MATRIX_HEIGHT = 120
NUM_LEDS = MATRIX_WIDTH * MATRIX_HEIGHT
MAX_PIXELS_PER_PACKET = 480

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WALL_CALIB_PATH = os.path.join(BASE_DIR, "wall_calib.json")

ROW_SPACING = 10          # a white line every 10 rows
RULER_COLORS = {
    0:  (255, 0, 0),      # red
    5:  (0, 255, 0),      # green
    10: (0, 80, 255),     # blue
    15: (255, 220, 0),    # yellow
    20: (255, 0, 220),    # magenta
    25: (0, 230, 230),    # cyan
    29: (255, 255, 255),  # white (far end)
}


def load_offsets():
    try:
        with open(WALL_CALIB_PATH) as f:
            off = [int(v) for v in json.load(f).get("offsets", [])]
        if len(off) == MATRIX_WIDTH:
            return off
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return [0] * MATRIX_WIDTH


def apply_correction(frame, offsets):
    if not any(offsets):
        return frame
    out = frame.copy()
    for x, k in enumerate(offsets):
        if k:
            out[:, x] = np.roll(frame[:, x], k, axis=0)
    return out


def build_lines():
    f = np.zeros((MATRIX_HEIGHT, MATRIX_WIDTH, 3), np.uint8)
    for y in range(ROW_SPACING, MATRIX_HEIGHT, ROW_SPACING):
        f[y, :, :] = (120, 120, 120)          # dim grey reference rows
    return f


def stamp_ruler(frame):
    # lane ruler: a 3-px tall colour block at the very top of marked lanes.
    # Stamped AFTER correction so markers always stay pinned to the top row.
    for x, col in RULER_COLORS.items():
        frame[0:3, x, :] = col
    return frame


def send(sock, frame):
    data = np.clip(frame, 0, 255).astype(np.uint8).reshape(-1).tobytes()
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
            (chunk_bytes >> 8) & 0xFF, chunk_bytes & 0xFF,
        ])
        sock.sendto(header + data[byte_offset:byte_offset + chunk_bytes], (WLED_IP, UDP_PORT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true", help="skip correction (show the raw wiring comb)")
    args = ap.parse_args()

    lines = build_lines()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("offsets:", load_offsets())
    print("mode:", "RAW (uncorrected)" if args.raw else "CORRECTED (residual comb = lanes to fix)")
    print(f"Streaming line test to {WLED_IP}. Photograph the wall. Ctrl+C to stop.")
    try:
        while True:
            offs = load_offsets()                 # hot-reload while you tune
            frame = lines if args.raw else apply_correction(lines, offs)
            frame = stamp_ruler(frame.copy())
            send(sock, frame)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
