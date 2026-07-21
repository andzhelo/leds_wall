#!/usr/bin/env python3
"""
camera_breakout.py - Breakout on the 30x120 Niji wall, YOU are the paddle.

Reuses the mirror's camera + MediaPipe SelfieSegmentation pipeline: the person
mask's horizontal centroid drives the paddle. Bricks sit at the top, the ball
falls down the tall panel, and you move left/right in front of the camera to
keep it in play. No buttons -- it auto-serves and auto-restarts.

Sends through wall.py, so the per-lane calibration applies here too.

    python3 camera_breakout.py
    python3 camera_breakout.py --no-flip      # if left/right feels reversed
    python3 camera_breakout.py --margin 0.2   # how far you must move (0..0.4)

Requires: numpy, opencv-python, mediapipe  (already used by the mirror)
"""
import argparse
import math
import time
import threading

import numpy as np
import cv2
import mediapipe as mp

from wall import DDP, W, H

# ---- playfield geometry (pixels) -------------------------------------------
BRICK_COLS = 5
BRICK_ROWS = 8
BRICK_W = W // BRICK_COLS          # 6
BRICK_H = 4
BRICK_Y0 = 10                      # top margin before bricks
PADDLE_Y = H - 4                   # paddle top row
PADDLE_TH = 3                      # paddle thickness (px)
PADDLE_HW = 4                      # paddle half-width (px) -> 8 wide
BALL_R = 1.0
BALL_SPEED0 = 34.0                 # px/sec at serve
BALL_SPEED_MAX = 70.0
LIVES = 3

BRICK_COLORS = [
    (220, 40, 40), (220, 110, 0), (210, 190, 0), (60, 200, 40),
    (0, 190, 200), (40, 90, 230), (150, 40, 220), (220, 60, 160),
]


# ----------------------------------------------------------------------------
# Camera -> paddle position (mask centroid), in its own thread for low latency.
# ----------------------------------------------------------------------------
class Paddler(threading.Thread):
    def __init__(self, seg_w=256, flip=True, margin=0.15, threshold=0.4):
        super().__init__(daemon=True)
        self.flip = flip
        self.margin = margin
        self.threshold = threshold
        self.seg_w = seg_w
        self.lock = threading.Lock()
        self.norm = 0.5            # smoothed paddle position 0..1
        self.present = False
        self.running = True
        self.fps = 0.0

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)

    def run(self):
        target = 0.5
        t0, n = time.time(), 0
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (self.seg_w, int(self.seg_w * h / w)))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mask = self.seg.process(rgb).segmentation_mask
            if mask is not None:
                person = mask > self.threshold
                cols = person.any(axis=0)
                area = float(person.mean())
                if area > 0.01 and cols.any():
                    xs = np.nonzero(cols)[0]
                    cx = float(xs.mean()) / person.shape[1]      # 0..1 across FOV
                    # expand the active band [margin, 1-margin] to full paddle range
                    m = self.margin
                    t = (cx - m) / max(1e-3, (1 - 2 * m))
                    t = min(1.0, max(0.0, t))
                    if self.flip:
                        t = 1.0 - t
                    target = t
                    present = True
                else:
                    present = False
            else:
                present = False

            with self.lock:
                # smooth toward target; hold last position when absent
                self.norm = 0.55 * self.norm + 0.45 * target
                self.present = present

            n += 1
            if time.time() - t0 >= 2.0:
                self.fps = n / (time.time() - t0)
                t0, n = time.time(), 0

    def paddle_x(self):
        with self.lock:
            return PADDLE_HW + self.norm * (W - 2 * PADDLE_HW), self.present

    def stop(self):
        self.running = False
        self.cap.release()
        self.seg.close()


# ----------------------------------------------------------------------------
# Game
# ----------------------------------------------------------------------------
class Breakout:
    def __init__(self):
        self.reset()

    def reset(self):
        self.bricks = np.ones((BRICK_ROWS, BRICK_COLS), dtype=bool)
        self.lives = LIVES
        self.score = 0
        self.speed = BALL_SPEED0
        self.state = "serve"       # serve | play | dead | won
        self.state_t = time.time()
        self.px = W / 2
        self._serve()

    def _serve(self):
        self.bx = self.px
        self.by = PADDLE_Y - 2
        ang = math.radians(-60 - 60 * np.random.rand())   # up-ish, random lean
        self.vx = math.cos(ang) * self.speed
        self.vy = math.sin(ang) * self.speed
        self.state = "serve"
        self.state_t = time.time()

    def _brick_at(self, x, y):
        if not (0 <= x < W) or not (BRICK_Y0 <= y < BRICK_Y0 + BRICK_ROWS * BRICK_H):
            return None
        c = int(x // BRICK_W)
        r = int((y - BRICK_Y0) // BRICK_H)
        if 0 <= r < BRICK_ROWS and 0 <= c < BRICK_COLS and self.bricks[r, c]:
            return r, c
        return None

    def update(self, dt, paddle_x):
        self.px = paddle_x
        now = time.time()

        if self.state == "serve":
            # ball rides the paddle briefly, then launches
            self.bx = self.px
            self.by = PADDLE_Y - 2
            if now - self.state_t > 1.2:
                self.state = "play"
            return
        if self.state in ("dead", "won"):
            if now - self.state_t > 2.0:
                if self.state == "won" or self.lives <= 0:
                    self.reset()
                else:
                    self._serve()
            return

        # --- play: integrate with sub-steps to avoid tunneling ---
        dt = min(dt, 0.1)
        dist = self.speed * dt
        steps = max(1, int(math.ceil(dist / 0.5)))
        for _ in range(steps):
            self._substep(dist / steps)
            if self.state != "play":
                break

    def _substep(self, d):
        v = math.hypot(self.vx, self.vy) or 1.0
        self.bx += self.vx / v * d
        self.by += self.vy / v * d

        # walls
        if self.bx < BALL_R:
            self.bx = BALL_R; self.vx = abs(self.vx)
        elif self.bx > W - BALL_R:
            self.bx = W - BALL_R; self.vx = -abs(self.vx)
        if self.by < BALL_R:
            self.by = BALL_R; self.vy = abs(self.vy)

        # bricks
        hit = self._brick_at(self.bx, self.by)
        if hit:
            self.bricks[hit] = False
            self.score += 1
            self.vy = -self.vy
            self.speed = min(BALL_SPEED_MAX, self.speed + 0.6)
            if not self.bricks.any():
                self.state = "won"; self.state_t = time.time()
            return

        # paddle
        if (self.vy > 0 and PADDLE_Y - BALL_R <= self.by <= PADDLE_Y + PADDLE_TH
                and self.px - PADDLE_HW <= self.bx <= self.px + PADDLE_HW):
            self.by = PADDLE_Y - BALL_R
            offset = (self.bx - self.px) / PADDLE_HW        # -1..1
            ang = math.radians(-90 + offset * 60)           # steer with hit point
            self.vx = math.cos(ang) * self.speed
            self.vy = -abs(math.sin(ang) * self.speed)
            return

        # fell past the bottom
        if self.by > H + BALL_R:
            self.lives -= 1
            self.state = "dead"; self.state_t = time.time()


# ----------------------------------------------------------------------------
# Render
# ----------------------------------------------------------------------------
def render(g, present):
    f = np.zeros((H, W, 3), dtype=np.uint8)

    # bricks
    for r in range(BRICK_ROWS):
        color = BRICK_COLORS[r % len(BRICK_COLORS)]
        for c in range(BRICK_COLS):
            if g.bricks[r, c]:
                y0 = BRICK_Y0 + r * BRICK_H
                x0 = c * BRICK_W
                f[y0:y0 + BRICK_H, x0:x0 + BRICK_W] = color
                f[y0, x0:x0 + BRICK_W] = tuple(min(255, v + 40) for v in color)

    # paddle
    px = int(round(g.px))
    x0, x1 = max(0, px - PADDLE_HW), min(W, px + PADDLE_HW)
    pcol = (120, 230, 255) if present else (120, 120, 120)
    f[PADDLE_Y:PADDLE_Y + PADDLE_TH, x0:x1] = pcol

    # ball
    bx, by = int(round(g.bx)), int(round(g.by))
    for dy in (0, 1):
        for dx in (0, 1):
            y, x = by + dy, bx + dx
            if 0 <= y < H and 0 <= x < W:
                f[y, x] = (255, 255, 255)

    # lives: dots in the top-left corner
    for i in range(g.lives):
        f[1, 1 + i * 2] = (0, 220, 0)

    if g.state == "dead":
        f = np.maximum(f // 2, np.array((40, 0, 0), np.uint8))
    elif g.state == "won":
        f = np.maximum(f // 2, np.array((0, 40, 0), np.uint8))
    return f


def main():
    global BALL_SPEED0
    ap = argparse.ArgumentParser(description="Camera-controlled Breakout on the Niji wall.")
    ap.add_argument("--no-flip", dest="flip", action="store_false",
                    help="disable mirror flip (use if left/right feels reversed)")
    ap.add_argument("--margin", type=float, default=0.15,
                    help="fraction of camera FOV as dead margin on each side (0..0.4)")
    ap.add_argument("--fps", type=int, default=60, help="render cap")
    ap.add_argument("--bri", type=float, default=1.0)
    ap.add_argument("--speed", type=float, default=BALL_SPEED0, help="serve ball speed px/s")
    args = ap.parse_args()

    BALL_SPEED0 = args.speed

    print("Warming up camera + MediaPipe...")
    pad = Paddler(flip=args.flip, margin=args.margin)
    pad.start()
    t0 = time.time()
    while not pad.present and pad.norm == 0.5 and time.time() - t0 < 8.0:
        time.sleep(0.1)          # give the segmenter a moment
    print("Ready. Move left/right to steer the paddle. Ctrl+C to quit.")

    ddp = DDP()
    game = Breakout()
    frame_dt = 1.0 / args.fps
    last = time.time()
    report = time.time()

    try:
        while True:
            now = time.time()
            dt = now - last
            last = now

            px, present = pad.paddle_x()
            game.update(dt, px)
            frame = render(game, present)
            if args.bri < 1.0:
                frame = (frame.astype(np.float32) * args.bri).astype(np.uint8)
            ddp.send(frame)

            if now - report >= 3.0:
                print(f"  cam {pad.fps:.0f} fps  score {game.score}  lives {game.lives}  "
                      f"state {game.state}", flush=True)
                report = now

            sleep = frame_dt - (time.time() - now)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        pad.stop()


if __name__ == "__main__":
    main()
