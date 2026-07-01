#!/usr/bin/env python3
"""
niji.py - Custom realtime animations for a 30x120 WLED matrix (the "Niji" panel).

Streams per-pixel frames to WLED over DDP (port 4048). WLED is configured as a
2D matrix (w=30, h=120), so we render to a logical 30-wide x 120-tall canvas in
row-major order and let WLED handle the physical serpentine wiring.

Effects are designed specifically for a tall, narrow 30x120 portrait panel.

Usage:
    python3 niji.py list
    python3 niji.py fire
    python3 niji.py rain --fps 30 --bri 1.0 --duration 60
    python3 niji.py calibrate          # find out orientation / wiring
    python3 niji.py all --each 20      # cycle through every effect

Requires: numpy  (already present)
"""

import argparse
import math
import socket
import sys
import time

import numpy as np

# ----------------------------------------------------------------------------
# Panel geometry (logical canvas as WLED sees it)
# ----------------------------------------------------------------------------
HOST = "192.168.1.61"   # WLED device IP (see note below about pinning it)
DDP_PORT = 4048
W = 30           # columns (width)
H = 120          # rows (height); y=0 is TOP, y=H-1 is BOTTOM
N = W * H        # 3600

# DDP chunking: keep each UDP payload under typical MTU. 480 LEDs * 3 = 1440 B.
LEDS_PER_PACKET = 480

# Orientation (verified against the 'diag' photo IMG_5909, gravity-correct):
#   logical y=0 (red/green/blue rows) is at the physical TOP, y=119 at bottom;
#   logical x=0 is on the LEFT (confirmed by the diagonal x==y running top-left
#   to bottom-right). So NO flips are needed -- fire already rises, rain falls.
VFLIP = False    # flip rows before sending (not needed)
HFLIP = False    # flip columns left<->right (not needed)

# Runtime config for the text effect (set from CLI in main()).
CFG = {"text": "HELLO WORLD", "scale": 1}


# ----------------------------------------------------------------------------
# DDP transport
# ----------------------------------------------------------------------------
class DDP:
    """Minimal DDP sender for WLED. Frames are (H, W, 3) uint8, y=0 at top."""

    def __init__(self, host=HOST, port=DDP_PORT, vflip=VFLIP, hflip=HFLIP):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0
        self.vflip = vflip
        self.hflip = hflip

    def send(self, frame):
        if self.vflip:
            frame = frame[::-1]
        if self.hflip:
            frame = frame[:, ::-1]
        # frame: (H, W, 3) uint8 -> row-major flat bytes (logical LED order)
        data = np.clip(frame, 0, 255).astype(np.uint8).reshape(-1).tobytes()
        total = len(data)
        offset = 0
        while offset < total:
            chunk = data[offset:offset + LEDS_PER_PACKET * 3]
            is_last = (offset + len(chunk)) >= total
            self.seq = (self.seq % 15) + 1
            flags = 0x40 | (0x01 if is_last else 0x00)  # version1 | push-on-last
            header = bytes([
                flags,
                self.seq,
                0x00,                # data type (default)
                0x01,                # destination id (1 = display)
                (offset >> 24) & 0xFF,
                (offset >> 16) & 0xFF,
                (offset >> 8) & 0xFF,
                offset & 0xFF,
                (len(chunk) >> 8) & 0xFF,
                len(chunk) & 0xFF,
            ])
            self.sock.sendto(header + chunk, self.addr)
            offset += len(chunk)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def hsv_to_rgb(h, s, v):
    """Vectorized HSV->RGB. h,s,v in [0,1] numpy arrays. Returns (...,3) float."""
    h = (h % 1.0) * 6.0
    i = np.floor(h).astype(int)
    f = h - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    i = i % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def palette(t, stops):
    """Map t in [0,1] through a list of (pos, (r,g,b 0..255)) color stops.
    Returns normalized RGB in [0,1] (callers multiply by 255)."""
    t = np.clip(t, 0, 1)
    pos = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops], dtype=float) / 255.0
    out = np.empty(t.shape + (3,), dtype=float)
    for c in range(3):
        out[..., c] = np.interp(t, pos, cols[:, c])
    return out


# ----------------------------------------------------------------------------
# Effects.  Each is a generator yielding (H, W, 3) float frames in [0,255].
# ----------------------------------------------------------------------------
def fx_fire(fps):
    """Rising flames - hottest at the bottom, perfect for a tall panel."""
    heat = np.zeros((H, W))
    fire_pal = [
        (0.00, (0, 0, 0)),
        (0.25, (90, 0, 0)),
        (0.5, (220, 40, 0)),
        (0.75, (255, 150, 0)),
        (1.0, (255, 245, 180)),
    ]
    while True:
        # cool every cell a bit
        heat -= np.random.uniform(0.0, 0.06, (H, W))
        heat = np.clip(heat, 0, 1)
        # propagate upward (toward y=0): each cell pulls from the row below it
        below = np.roll(heat, 1, axis=0)
        below2 = np.roll(heat, 2, axis=0)
        side = np.roll(heat, 1, axis=1)
        heat = (below * 0.55 + below2 * 0.2 + side * 0.1 + heat * 0.15)
        # ignite the bottom rows
        heat[H - 1] = np.random.uniform(0.6, 1.0, W)
        heat[H - 2] = np.maximum(heat[H - 2], np.random.uniform(0.4, 0.9, W))
        yield palette(heat, fire_pal) * 255.0


def fx_rain(fps):
    """Matrix-style digital rain falling down the 30 columns."""
    canvas = np.zeros((H, W, 3))
    drops = np.random.randint(-H, 0, W).astype(float)   # head y per column
    speeds = np.random.uniform(0.4, 1.2, W)
    while True:
        canvas *= 0.80                                   # fade trails
        drops += speeds
        for x in range(W):
            y = int(drops[x])
            if 0 <= y < H:
                canvas[y, x] = (180, 255, 200)           # bright white-green head
                if y - 1 >= 0:
                    canvas[y - 1, x] = (0, 220, 70)
            if drops[x] > H + np.random.randint(0, 40):
                drops[x] = np.random.randint(-30, 0)
                speeds[x] = np.random.uniform(0.4, 1.2)
        yield canvas.copy()


def fx_plasma(fps):
    """Smooth flowing plasma across the whole panel."""
    ys, xs = np.mgrid[0:H, 0:W].astype(float)
    t = 0.0
    while True:
        t += 0.05
        v = (np.sin(xs / 5.0 + t)
             + np.sin(ys / 9.0 - t * 0.7)
             + np.sin((xs + ys) / 7.0 + t * 0.5)
             + np.sin(np.sqrt((xs - W / 2) ** 2 + (ys - H / 2) ** 2) / 6.0 - t))
        hue = (v / 4.0 + 1.0) / 2.0
        yield hsv_to_rgb(hue, np.ones_like(hue) * 0.9, np.ones_like(hue)) * 255.0


def fx_aurora(fps):
    """Vertical aurora curtains - greens, teals, violet drifting upward."""
    ys, xs = np.mgrid[0:H, 0:W].astype(float)
    t = 0.0
    aur = [
        (0.0, (0, 0, 0)),
        (0.4, (0, 60, 40)),
        (0.6, (0, 200, 120)),
        (0.8, (60, 255, 180)),
        (1.0, (160, 80, 255)),
    ]
    while True:
        t += 0.04
        band = (np.sin(xs / 6.0 + t)
                + 0.6 * np.sin(xs / 3.0 - t * 1.7)
                + np.sin(ys / 40.0 + t))
        curtain = (band + 2.6) / 5.2
        # brighter near top, fading down like real aurora
        vfade = np.clip(1.0 - ys / H * 0.6, 0, 1)
        intensity = np.clip(curtain, 0, 1) * vfade
        yield palette(intensity, aur) * 255.0


def fx_meteors(fps):
    """Shooting meteors streaking down the tall panel with glowing tails."""
    canvas = np.zeros((H, W, 3))
    meteors = []
    rng = np.random.default_rng()
    while True:
        canvas *= 0.78
        if rng.random() < 0.25 and len(meteors) < 8:
            hue = rng.random()
            meteors.append({"x": rng.uniform(0, W - 1), "y": -5.0,
                            "vx": rng.uniform(-0.3, 0.3),
                            "vy": rng.uniform(1.5, 3.5),
                            "col": hsv_to_rgb(np.array(hue), np.array(0.5),
                                              np.array(1.0))[0] * 255})
        for m in meteors:
            m["x"] += m["vx"]; m["y"] += m["vy"]
            xi, yi = int(round(m["x"])), int(round(m["y"]))
            if 0 <= xi < W and 0 <= yi < H:
                canvas[yi, xi] = np.maximum(canvas[yi, xi], m["col"])
        meteors = [m for m in meteors if m["y"] < H + 6]
        yield canvas.copy()


def fx_wipe(fps):
    """Rainbow color sweeping vertically with a soft leading edge."""
    ys = np.mgrid[0:H, 0:W][0].astype(float)
    t = 0.0
    while True:
        t += 0.01
        hue = (ys / H + t) % 1.0
        bright = 0.5 + 0.5 * np.sin(ys / 4.0 - t * 30)
        yield hsv_to_rgb(hue, np.ones_like(hue), bright) * 255.0


def _rgb1(h, s, v):
    """Single HSV->RGB color as a (3,) float array in 0..255 (v in 0..1)."""
    return hsv_to_rgb(np.array([h]), np.array([s]), np.array([v]))[0] * 255.0


def fx_fireworks(fps):
    """Rockets launch from the bottom and burst into showers of gravity sparks.
    Made for a tall panel - the long vertical flight path is the whole point."""
    canvas = np.zeros((H, W, 3))
    rockets, sparks = [], []
    rng = np.random.default_rng()
    g = 0.05
    while True:
        canvas *= 0.80
        if rng.random() < 0.13 and len(rockets) < 3:
            rockets.append({"x": rng.uniform(3, W - 3), "y": H - 1.0,
                            "vy": -rng.uniform(2.1, 2.9),
                            "apex": rng.uniform(12, 55), "hue": rng.random()})
        keep = []
        for r in rockets:
            r["y"] += r["vy"]; r["vy"] += g * 0.5
            xi, yi = int(r["x"]), int(r["y"])
            if 0 <= yi < H:
                canvas[yi, xi] = (255, 230, 180)
            if r["vy"] >= -0.4 or r["y"] <= r["apex"]:
                col = _rgb1(r["hue"], 0.7, 1.0)
                for _ in range(int(rng.integers(45, 75))):
                    a, sp = rng.uniform(0, 2 * np.pi), rng.uniform(0.3, 2.2)
                    sparks.append([r["x"], r["y"], np.cos(a) * sp, np.sin(a) * sp,
                                   0.0, rng.uniform(16, 34), col])
            else:
                keep.append(r)
        rockets = keep
        alive = []
        for s in sparks:
            s[0] += s[2]; s[1] += s[3]; s[3] += g; s[2] *= 0.96; s[3] *= 0.96; s[4] += 1
            xi, yi = int(round(s[0])), int(round(s[1]))
            if 0 <= xi < W and 0 <= yi < H:
                canvas[yi, xi] = np.maximum(canvas[yi, xi], s[6] * max(0.0, 1 - s[4] / s[5]))
            if s[4] < s[5] and s[1] < H:
                alive.append(s)
        sparks = alive[-600:]
        yield canvas.copy()


def fx_helix(fps):
    """A rotating DNA double helix climbing the panel, with connecting rungs
    and front/back depth shading."""
    t = 0.0
    cx = (W - 1) / 2.0
    amp = (W - 3) / 2.0 * 0.9
    ys = np.arange(H)
    while True:
        t += 0.13
        frame = np.zeros((H, W, 3))
        ph = ys * 0.17 + t
        x1, x2 = cx + amp * np.sin(ph), cx + amp * np.sin(ph + np.pi)
        d1, d2 = (np.cos(ph) + 1) / 2, (np.cos(ph + np.pi) + 1) / 2
        for y in range(H):
            if y % 7 == 0:                       # rung between the two strands
                a, b = sorted((x1[y], x2[y]))
                for xx in range(int(np.ceil(a)), int(np.floor(b)) + 1):
                    if 0 <= xx < W:
                        frame[y, xx] = (55, 55, 70)
            for xv, dep, col in ((x1[y], d1[y], (0, 200, 255)),
                                 (x2[y], d2[y], (255, 40, 160))):
                xi = int(round(xv))
                if 0 <= xi < W:
                    frame[y, xi] = np.maximum(frame[y, xi],
                                              np.array(col) * (0.35 + 0.65 * dep))
        yield frame


def fx_lightning(fps):
    """Forked lightning bolts crackling down the panel with full-frame flashes."""
    canvas = np.zeros((H, W, 3))
    rng = np.random.default_rng()
    glow = np.array((150, 180, 255))
    while True:
        canvas *= 0.55
        if rng.random() < 0.20:
            x, y, path = rng.uniform(0, W - 1), 0.0, []
            while y < H - 1:
                path.append((int(round(x)), int(y)))
                x = min(max(x + rng.uniform(-1.6, 1.6), 0), W - 1)
                y += rng.uniform(1.4, 3.2)
                if rng.random() < 0.13:           # spawn a branch
                    bx, by = x, y
                    for _ in range(int(rng.integers(4, 11))):
                        path.append((int(round(bx)), int(by)))
                        bx = min(max(bx + rng.uniform(-2, 2), 0), W - 1)
                        by += rng.uniform(1, 3)
                        if by >= H:
                            break
            for xi, yi in path:
                if 0 <= xi < W and 0 <= yi < H:
                    canvas[yi, xi] = (255, 255, 255)
                    for dx in (-1, 1):
                        if 0 <= xi + dx < W:
                            canvas[yi, xi + dx] = np.maximum(canvas[yi, xi + dx], glow)
            canvas += np.array((18, 22, 38))      # sky flash
        yield canvas.copy()


def fx_warp(fps):
    """Warp-speed starfield - stars streak outward from the center, Star-Wars
    hyperspace style."""
    rng = np.random.default_rng()
    n = 90
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    star = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n),
                            rng.uniform(0.1, 1, n)])
    canvas = np.zeros((H, W, 3))
    while True:
        canvas *= 0.55
        star[:, 2] -= 0.028
        r = star[:, 2] <= 0.05
        if r.any():
            k = int(r.sum())
            star[r, 0] = rng.uniform(-1, 1, k); star[r, 1] = rng.uniform(-1, 1, k)
            star[r, 2] = 1.0
        sx = cx + star[:, 0] / star[:, 2] * cx
        sy = cy + star[:, 1] / star[:, 2] * cy
        bright = np.clip(1 - star[:, 2], 0, 1)
        for i in range(n):
            xi, yi = int(round(sx[i])), int(round(sy[i]))
            if 0 <= xi < W and 0 <= yi < H:
                v = 110 + 145 * bright[i]
                canvas[yi, xi] = np.maximum(canvas[yi, xi], (v, v, v))
        yield canvas.copy()


def fx_metaballs(fps):
    """Gooey lava-lamp metaballs oozing and merging."""
    ys, xs = np.mgrid[0:H, 0:W].astype(float)
    rng = np.random.default_rng()
    nb = 5
    ph = rng.uniform(0, 2 * np.pi, (nb, 2))
    pal = [(0.0, (0, 0, 0)), (0.40, (40, 0, 60)), (0.62, (200, 0, 80)),
           (0.82, (255, 120, 0)), (1.0, (255, 240, 130))]
    t = 0.0
    while True:
        t += 0.05
        field = np.zeros((H, W))
        for i in range(nb):
            bx = W * 0.5 + W * 0.42 * np.sin(t * (0.3 + 0.09 * i) + ph[i, 0])
            by = H * 0.5 + H * 0.42 * np.sin(t * (0.21 + 0.07 * i) + ph[i, 1])
            field += 42.0 / ((xs - bx) ** 2 + (ys - by) ** 2 + 8.0)
        yield palette(np.clip(field, 0, 1), pal) * 255.0


def fx_vortex(fps):
    """Hypnotic spiral tunnel rotating into the center."""
    ys, xs = np.mgrid[0:H, 0:W].astype(float)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    ang = np.arctan2(ys - cy, xs - cx)
    rad = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    t = 0.0
    while True:
        t += 0.06
        spiral = np.sin(rad * 0.5 - t * 3 + ang * 3)
        hue = (ang / (2 * np.pi) + rad * 0.02 - t * 0.1) % 1.0
        yield hsv_to_rgb(hue, np.ones_like(hue) * 0.9, 0.5 + 0.5 * spiral) * 255.0


def fx_ripple(fps):
    """Rain droplets sending expanding ripples across a dark pool."""
    ys, xs = np.mgrid[0:H, 0:W].astype(float)
    rng = np.random.default_rng()
    drops = []
    pal = [(0.0, (0, 4, 20)), (0.5, (0, 60, 140)), (0.8, (0, 160, 210)),
           (1.0, (180, 240, 255))]
    t = 0.0
    while True:
        t += 1.0
        if rng.random() < 0.25:
            drops.append([rng.uniform(0, W - 1), rng.uniform(0, H - 1), t])
        field = np.zeros((H, W))
        keep = []
        for d in drops:
            age = t - d[2]
            front = np.sqrt((xs - d[0]) ** 2 + (ys - d[1]) ** 2) - age * 1.6
            field += np.sin(front) * np.exp(-0.15 * np.abs(front)) * np.exp(-age * 0.04)
            if age < 90:
                keep.append(d)
        drops = keep
        yield palette(np.clip((field + 1) / 2, 0, 1), pal) * 255.0


def fx_life(fps):
    """Conway's Game of Life with glowing color trails; auto-reseeds when it
    stalls so it never dies out."""
    rng = np.random.default_rng()
    grid = rng.random((H, W)) < 0.25
    canvas = np.zeros((H, W, 3))
    t, stale, prev_pop = 0, 0, 0
    while True:
        t += 1
        nb = sum(np.roll(np.roll(grid, dy, 0), dx, 1)
                 for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                 if not (dy == 0 and dx == 0))
        grid = (nb == 3) | (grid & (nb == 2))
        canvas *= 0.85
        hue = (t * 0.01) % 1.0
        canvas[grid] = _rgb1(hue, 0.8, 1.0)
        pop = int(grid.sum())
        stale = stale + 1 if abs(pop - prev_pop) < 3 else 0
        prev_pop = pop
        if pop < 8 or stale > 40:
            grid = rng.random((H, W)) < 0.28
            stale = 0
        yield canvas.copy()


# ----------------------------------------------------------------------------
# 5x7 bitmap font + text rendering (for fx_text / fx_clock)
# ----------------------------------------------------------------------------
_FONT = {
    " ": (".....", ".....", ".....", ".....", ".....", ".....", "....."),
    "0": (".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."),
    "1": ("..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "2": (".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"),
    "3": ("#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."),
    "4": ("...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."),
    "5": ("#####", "#....", "####.", "....#", "....#", "#...#", ".###."),
    "6": ("..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."),
    "7": ("#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."),
    "8": (".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."),
    "9": (".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."),
    "A": (".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "B": ("####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."),
    "C": (".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."),
    "D": ("####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."),
    "E": ("#####", "#....", "#....", "####.", "#....", "#....", "#####"),
    "F": ("#####", "#....", "#....", "####.", "#....", "#....", "#...."),
    "G": (".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".####"),
    "H": ("#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "I": (".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "J": ("..###", "...#.", "...#.", "...#.", "#..#.", "#..#.", ".##.."),
    "K": ("#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"),
    "L": ("#....", "#....", "#....", "#....", "#....", "#....", "#####"),
    "M": ("#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"),
    "N": ("#...#", "#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#"),
    "O": (".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "P": ("####.", "#...#", "#...#", "####.", "#....", "#....", "#...."),
    "Q": (".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"),
    "R": ("####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"),
    "S": (".####", "#....", "#....", ".###.", "....#", "....#", "####."),
    "T": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."),
    "U": ("#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "V": ("#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."),
    "W": ("#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"),
    "X": ("#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"),
    "Y": ("#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."),
    "Z": ("#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"),
    ":": (".....", "..#..", "..#..", ".....", "..#..", "..#..", "....."),
    ".": (".....", ".....", ".....", ".....", ".....", "..#..", "..#.."),
    ",": (".....", ".....", ".....", ".....", "..#..", "..#..", ".#..."),
    "-": (".....", ".....", ".....", ".###.", ".....", ".....", "....."),
    "!": ("..#..", "..#..", "..#..", "..#..", "..#..", ".....", "..#.."),
    "?": (".###.", "#...#", "...#.", "..#..", "..#..", ".....", "..#.."),
    "/": ("....#", "....#", "...#.", "..#..", ".#...", "#....", "#...."),
    "'": ("..#..", "..#..", ".....", ".....", ".....", ".....", "....."),
}


def _glyph(ch):
    return _FONT.get(ch.upper(), _FONT["?"])


def text_mask(s):
    """Render string s to a (7, width) float mask (1 = lit)."""
    parts = []
    for ch in s:
        parts.append(np.array([[1.0 if c == "#" else 0.0 for c in row]
                               for row in _glyph(ch)]))
        parts.append(np.zeros((7, 1)))        # 1px gap after each char
    return np.concatenate(parts, axis=1) if parts else np.zeros((7, 0))


def _scale(m, s):
    return np.kron(m, np.ones((s, s))) if s > 1 else m


def fx_text(fps):
    """Scrolling text, film-credits style: wraps to fit the narrow panel and
    scrolls upward in a vertical rainbow. Set words with  --text "MESSAGE"."""
    msg = CFG["text"].upper()
    sc = max(1, int(CFG["scale"]))
    gap = 3 * sc
    cw = (5 + 1) * sc                          # char cell width (px)
    per_line = max(1, (W + sc) // cw)          # chars that fit across
    lines, cur = [], ""
    for word in msg.split():
        while len(word) > per_line:            # hard-break overlong words
            if cur:
                lines.append(cur); cur = ""
            lines.append(word[:per_line]); word = word[per_line:]
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= per_line:
            cur += " " + word
        else:
            lines.append(cur); cur = word
    if cur:
        lines.append(cur)
    if not lines:
        lines = [" "]
    blocks = []
    for ln in lines:
        m = _scale(text_mask(ln), sc)
        h, wd = m.shape
        canvas = np.zeros((h, W))
        x0 = max(0, (W - wd) // 2)
        ww = min(wd, W)
        canvas[:, x0:x0 + ww] = m[:, :ww]
        blocks.append(canvas)
        blocks.append(np.zeros((gap, W)))
    content = np.concatenate(blocks, axis=0)
    ch = content.shape[0]
    col = np.zeros((ch, W, 3))                  # pre-color with vertical rainbow
    hue = (np.arange(ch) / 50.0) % 1.0
    rgb = hsv_to_rgb(hue, np.ones(ch), np.ones(ch)) * 255.0
    for y in range(ch):
        col[y][content[y] > 0] = rgb[y]
    speed = 0.6 * sc
    s = -float(H)
    while True:
        frame = np.zeros((H, W, 3))
        si = int(s)
        lo, hi = max(0, si), min(ch, si + H)
        if hi > lo:
            frame[lo - si:hi - si] = col[lo:hi]
        s += speed
        if s >= ch:
            s = -float(H)
        yield frame


def fx_clock(fps):
    """Big digital clock: HH over MM, blinking colon, and a seconds bar that
    fills up the left edge once per minute. 24-hour local time."""
    import time as _time
    sc = 2
    col = (0, 220, 255)

    def stamp(frame, s, ytop):
        m = _scale(text_mask(s), sc)
        h, wd = m.shape
        x0 = max(0, (W - wd) // 2)
        ww = min(wd, W)
        frame[ytop:ytop + h, x0:x0 + ww][m[:, :ww] > 0] = col

    while True:
        lt = _time.localtime()
        frame = np.zeros((H, W, 3))
        stamp(frame, "%02d" % lt.tm_hour, 20)
        stamp(frame, "%02d" % lt.tm_min, 70)
        if lt.tm_sec % 2 == 0:                  # blinking colon
            for yy in (52, 60):
                frame[yy, 14] = col
                frame[yy, 15] = col
        barh = int(round((lt.tm_sec / 59.0) * (H - 1)))
        if barh:
            frame[H - barh:H, 0] = (0, 160, 70)
        yield frame


def fx_calibrate(fps):
    """Diagnostic pattern: RED top edge, GREEN bottom edge, BLUE left edge,
    a single WHITE pixel walking the perimeter, and a dim moving column.
    Watch the wall and tell me what you actually see."""
    f = 0
    while True:
        frame = np.zeros((H, W, 3))
        frame[0, :] = (60, 0, 0)        # top row -> dim red
        frame[H - 1, :] = (0, 60, 0)    # bottom row -> dim green
        frame[:, 0] = (0, 0, 60)        # left col -> dim blue
        frame[:, W - 1] = (40, 40, 0)   # right col -> dim yellow
        # bright white marker walking down column 0 then across
        step = f % (H + W)
        if step < H:
            frame[step, 0] = (255, 255, 255)
        else:
            frame[H - 1, step - H] = (255, 255, 255)
        f += 1
        yield frame


def fx_diag(fps):
    """Static mapping diagnostic. Reveals exact logical->physical wiring so a
    corrected ledmap can be authored. Best photographed with --no-vflip.

    Layout (logical coords, x=0..29 width, y=0..119 height, y=0 = top):
      - 6x6 solid WHITE block at the (x0,y0) corner  -> marks logical origin
      - row y=0 RED, y=1 GREEN, y=2 BLUE             -> top three rows distinct
      - row y=119 dim CYAN, y=118 dim MAGENTA        -> bottom two rows distinct
      - column x=0 dim YELLOW (full height)          -> left edge
      - a CYAN diagonal (x=y for the first 30 rows)  -> detects any mirroring
      - a single bright dot at logical (x=15, y=0)   -> marks the left/right split
    """
    while True:
        frame = np.zeros((H, W, 3))
        frame[:, 0] = (30, 30, 0)            # left column, dim yellow
        frame[0, :] = (180, 0, 0)            # row 0  red
        frame[1, :] = (0, 180, 0)            # row 1  green
        frame[2, :] = (0, 0, 180)            # row 2  blue
        frame[H - 1, :] = (0, 60, 60)        # last row   dim cyan
        frame[H - 2, :] = (60, 0, 60)        # 2nd-last   dim magenta
        for i in range(min(W, 30)):          # diagonal in the top square
            frame[i, i] = (0, 200, 200)
        frame[0:6, 0:6] = (255, 255, 255)    # origin block
        frame[0, 15] = (255, 255, 255)       # split marker at column 15
        yield frame


EFFECTS = {
    "fire": fx_fire,
    "rain": fx_rain,
    "plasma": fx_plasma,
    "aurora": fx_aurora,
    "meteors": fx_meteors,
    "wipe": fx_wipe,
    "fireworks": fx_fireworks,
    "helix": fx_helix,
    "lightning": fx_lightning,
    "warp": fx_warp,
    "metaballs": fx_metaballs,
    "vortex": fx_vortex,
    "ripple": fx_ripple,
    "life": fx_life,
    "text": fx_text,
    "clock": fx_clock,
    "calibrate": fx_calibrate,
    "diag": fx_diag,
}


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------
def run(effect_name, fps, bri, duration, ddp):
    gen = EFFECTS[effect_name](fps)
    frame_dt = 1.0 / fps
    end = time.time() + duration if duration > 0 else float("inf")
    next_t = time.time()
    print(f"-> {effect_name}  ({fps} fps, bri {bri:.2f}"
          f"{', %ds' % duration if duration > 0 else ', ctrl-C to stop'})")
    while time.time() < end:
        frame = next(gen) * bri
        ddp.send(frame)
        next_t += frame_dt
        sleep = next_t - time.time()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_t = time.time()


def main():
    p = argparse.ArgumentParser(description="Custom WLED animations for the 30x120 Niji panel.")
    p.add_argument("effect", nargs="?", default="list",
                   help="effect name, 'list', or 'all'")
    p.add_argument("--host", default=HOST)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--bri", type=float, default=1.0, help="brightness 0..1")
    p.add_argument("--duration", type=int, default=0, help="seconds (0 = forever)")
    p.add_argument("--each", type=int, default=20, help="seconds per effect in 'all' mode")
    p.add_argument("--no-vflip", action="store_true", help="disable vertical flip (raw mapping)")
    p.add_argument("--hflip", action="store_true", help="flip left<->right")
    p.add_argument("--text", default=None, help="message for the 'text' effect")
    p.add_argument("--scale", type=int, default=1, help="font scale for 'text' (1-3)")
    args = p.parse_args()

    if args.text is not None:
        CFG["text"] = args.text
    CFG["scale"] = args.scale

    if args.effect == "list":
        print("Effects for the 30x120 Niji panel:\n")
        for k in EFFECTS:
            print(f"  python3 niji.py {k}")
        print("\n  python3 niji.py all          # cycle through them")
        print("  python3 niji.py calibrate    # verify orientation/wiring")
        return

    ddp = DDP(args.host, vflip=(VFLIP and not args.no_vflip), hflip=(HFLIP or args.hflip))

    try:
        if args.effect == "all":
            order = [k for k in EFFECTS if k not in ("calibrate", "diag")]
            while True:
                for name in order:
                    run(name, args.fps, args.bri, args.each, ddp)
        elif args.effect in EFFECTS:
            run(args.effect, args.fps, args.bri, args.duration, ddp)
        else:
            print(f"Unknown effect '{args.effect}'. Try: python3 niji.py list")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nstopped. (WLED reverts to its normal effect after ~2.5s)")


if __name__ == "__main__":
    main()
