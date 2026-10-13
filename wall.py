#!/usr/bin/env python3
"""
wall.py - shared transport + geometry + physical correction for the Niji wall.

The panel is a 30 (wide) x 120 (tall) 2D matrix, y=0 at the top, built from two
serpentine "snakes" of 1800 LEDs each:
    half 1 : GPIO16, columns 0..14  (left)
    half 2 : GPIO2,  columns 15..29 (right)

Half 1 has a wiring quirk: alternating vertical lanes are offset by 2 LEDs along
the lane relative to their neighbours, so a straight row renders as a comb on the
left half. apply_correction() rolls the offending lanes back by 2 to straighten
it. All tools (block_game, niji, mirror) should send through DDP here so the fix
lives in exactly one place.

    from wall import DDP, W, H
    ddp = DDP()
    ddp.send(frame)          # frame: (H, W, 3) uint8, corrected automatically
"""
import json
import os
import socket
import numpy as np

HOST = "192.168.1.60"     # WLED device IP
DDP_PORT = 4048
W = 30                    # columns
H = 120                   # rows (y=0 top)
LEDS_PER_PACKET = 480     # keep each UDP payload under the MTU

# ----------------------------------------------------------------------------
# Per-lane physical correction.
# Each of the 30 vertical lanes has its own small offset (from serpentine turns
# in the first snake). LANE_OFFSET[x] = how many LEDs to roll lane x along its
# length so a straight row renders straight. Tune it live with calib.py, which
# writes wall_calib.json; this loads that file so every tool is corrected.
# ----------------------------------------------------------------------------
CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wall_calib.json")


def load_offsets():
    try:
        with open(CALIB_FILE) as f:
            off = [int(v) for v in json.load(f).get("offsets", [])]
        if len(off) == W:
            return off
    except (OSError, ValueError, TypeError):
        pass
    return [0] * W


def save_offsets(offsets):
    with open(CALIB_FILE, "w") as f:
        json.dump({"offsets": [int(v) for v in offsets]}, f)


LANE_OFFSET = load_offsets()


def apply_correction(frame, offsets=None):
    offsets = LANE_OFFSET if offsets is None else offsets
    if not any(offsets):
        return frame
    out = frame.copy()
    for x, k in enumerate(offsets):
        if k:
            out[:, x] = np.roll(frame[:, x], k, axis=0)
    return out


class DDP:
    """Minimal DDP sender for WLED. Frames are (H, W, 3) uint8, y=0 at top.

    Applies the half-1 lane correction before sending unless correct=False.
    """

    def __init__(self, host=HOST, port=DDP_PORT, correct=True):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0
        self.correct = correct

    def send(self, frame):
        if self.correct:
            frame = apply_correction(frame)
        data = np.clip(frame, 0, 255).astype(np.uint8).reshape(-1).tobytes()
        total = len(data)
        offset = 0
        while offset < total:
            chunk = data[offset:offset + LEDS_PER_PACKET * 3]
            is_last = (offset + len(chunk)) >= total
            self.seq = (self.seq % 15) + 1
            flags = 0x40 | (0x01 if is_last else 0x00)  # version1 | push-on-last
            header = bytes([
                flags, self.seq,
                0x00,                # data type (default)
                0x01,                # destination id (1 = display)
                (offset >> 24) & 0xFF, (offset >> 16) & 0xFF,
                (offset >> 8) & 0xFF, offset & 0xFF,
                (len(chunk) >> 8) & 0xFF, len(chunk) & 0xFF,
            ])
            self.sock.sendto(header + chunk, self.addr)
            offset += len(chunk)
