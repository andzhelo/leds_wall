#!/usr/bin/env python3
"""
snake.py - Snake on the 30x120 Niji LED wall, played with a gamepad.

The panel is 30 wide x 120 tall. At 3px per cell that's a 10 x 40 grid -- a
long, tall playfield. Uses the shared wall.py transport (so the per-lane
calibration applies here too) and the same evdev gamepad pattern as tetris.py.

    python3 snake.py
    python3 snake.py --dev /dev/input/event4 --fps 30 --wrap

Controls:
    left stick / D-pad   steer (can't reverse into yourself)
    A / Start            pause; restart after game over

Requires: numpy, evdev
"""
import argparse
import random
import threading
import time
from collections import deque

import numpy as np
import evdev
from evdev import ecodes as e

from wall import DDP, W, H

BLOCK = 3
COLS = W // BLOCK        # 10
ROWS = H // BLOCK        # 40

UP, DOWN, LEFT, RIGHT = (-1, 0), (1, 0), (0, -1), (0, 1)


# ----------------------------------------------------------------------------
# Gamepad: background thread keeps the latest desired direction + edge buttons.
# ----------------------------------------------------------------------------
def find_gamepad():
    for path in evdev.list_devices():
        try:
            dev = evdev.InputDevice(path)
        except OSError:
            continue
        keys = dev.capabilities().get(e.EV_KEY, [])
        if e.BTN_GAMEPAD in keys or e.BTN_SOUTH in keys:
            return path
    return None


class Gamepad(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True)
        self.dev = evdev.InputDevice(path)
        self.name = self.dev.name
        self.lock = threading.Lock()
        self.want = RIGHT          # latest desired direction
        self.actions = []          # edge-triggered (pause/restart)
        self._sx = self._axis_max(e.ABS_X)
        self._sy = self._axis_max(e.ABS_Y)
        self._x = 0
        self._y = 0
        self.running = True

    def _axis_max(self, code):
        try:
            info = self.dev.absinfo(code)
            return max(abs(info.min), abs(info.max)) or 1
        except OSError:
            return 32767

    def _set_axis(self, ax, val):
        # ax 'x' or 'y'; val -1/0/1. The axis that just went non-zero wins,
        # so a fresh push takes over even if the other axis is still held.
        if ax == "x":
            self._x = val
            if val:
                with self.lock:
                    self.want = LEFT if val < 0 else RIGHT
        else:
            self._y = val
            if val:
                with self.lock:
                    self.want = UP if val < 0 else DOWN

    def run(self):
        tx, ty = self._sx * 0.5, self._sy * 0.5
        for ev in self.dev.read_loop():
            if not self.running:
                break
            if ev.type == e.EV_KEY and ev.value == 1:
                if ev.code in (e.BTN_SOUTH, e.BTN_START, e.BTN_EAST):
                    with self.lock:
                        self.actions.append("action")
            elif ev.type == e.EV_ABS:
                if ev.code == e.ABS_X:
                    self._set_axis("x", -1 if ev.value < -tx else (1 if ev.value > tx else 0))
                elif ev.code == e.ABS_HAT0X:
                    self._set_axis("x", int(ev.value))
                elif ev.code == e.ABS_Y:
                    self._set_axis("y", -1 if ev.value < -ty else (1 if ev.value > ty else 0))
                elif ev.code == e.ABS_HAT0Y:
                    self._set_axis("y", int(ev.value))

    def get_want(self):
        with self.lock:
            return self.want

    def drain(self):
        with self.lock:
            a, self.actions = self.actions, []
            return a


# ----------------------------------------------------------------------------
# Game
# ----------------------------------------------------------------------------
class Snake:
    def __init__(self, wrap=False):
        self.wrap = wrap
        self.reset()

    def reset(self):
        mid_r, mid_c = ROWS // 2, COLS // 2
        # tail -> head, left to right; head is body[-1], so start moving RIGHT
        # (away from the body, not into it).
        self.body = deque([(mid_r, mid_c - 1), (mid_r, mid_c), (mid_r, mid_c + 1)])
        self.dir = RIGHT
        self.grow = 0
        self.food = self._place_food()
        self.dead = False
        self.paused = False
        self.score = 0

    def _place_food(self):
        empty = [(r, c) for r in range(ROWS) for c in range(COLS) if (r, c) not in self.body]
        return random.choice(empty) if empty else None

    def set_dir(self, d):
        # ignore a 180-degree reversal
        if (d[0] == -self.dir[0] and d[1] == -self.dir[1]):
            return
        self.dir = d

    def step(self):
        if self.dead or self.paused:
            return
        hr, hc = self.body[-1]
        nr, nc = hr + self.dir[0], hc + self.dir[1]
        if self.wrap:
            nr %= ROWS
            nc %= COLS
        elif not (0 <= nr < ROWS and 0 <= nc < COLS):
            self.dead = True
            return
        head = (nr, nc)
        # moving into own body kills, unless it's the tail cell that's about to move
        if head in self.body and not (head == self.body[0] and self.grow == 0):
            self.dead = True
            return
        self.body.append(head)
        if head == self.food:
            self.score += 1
            self.grow += 2
            self.food = self._place_food()
        if self.grow > 0:
            self.grow -= 1
        else:
            self.body.popleft()

    def step_period(self, fps):
        # speed up as the snake grows: ~5 steps/s at start, capped fast
        base = 0.16 - min(0.10, self.score * 0.004)
        return max(1, int(base * fps))


# ----------------------------------------------------------------------------
# Render
# ----------------------------------------------------------------------------
def draw_cell(frame, r, c, color):
    y, x = r * BLOCK, c * BLOCK
    frame[y:y + BLOCK, x:x + BLOCK] = color


def render(game, bri=1.0):
    frame = np.zeros((H, W, 3), dtype=np.uint8)

    # food: pulsing red
    if game.food:
        draw_cell(frame, game.food[0], game.food[1], (220, 20, 20))

    # snake: bright head fading to darker tail
    n = len(game.body)
    for i, (r, c) in enumerate(game.body):
        t = i / max(1, n - 1)                  # 0 tail -> 1 head
        g = int(70 + 185 * t)
        color = (0, g, int(30 * (1 - t)))
        draw_cell(frame, r, c, color)
    hr, hc = game.body[-1]
    draw_cell(frame, hr, hc, (180, 255, 180))  # bright head

    if game.paused:
        frame = frame // 3
    if game.dead:
        wash = np.zeros_like(frame)
        wash[:, :] = (40, 0, 0)
        frame = np.maximum(frame // 2, wash)

    if bri < 1.0:
        frame = (frame.astype(np.float32) * bri).astype(np.uint8)
    return frame


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Snake on the 30x120 Niji wall.")
    ap.add_argument("--dev", help="evdev path (auto if omitted)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bri", type=float, default=1.0)
    ap.add_argument("--wrap", action="store_true", help="wrap around edges instead of dying")
    args = ap.parse_args()

    path = args.dev or find_gamepad()
    if not path:
        raise SystemExit("No gamepad found. Turn on the controller / plug in the dongle.")
    pad = Gamepad(path)
    pad.start()
    print(f"Controller: {pad.name} ({path})")

    ddp = DDP()
    game = Snake(wrap=args.wrap)

    frame_dt = 1.0 / args.fps
    step_counter = 0

    try:
        while True:
            t0 = time.time()

            for _ in pad.drain():
                if game.dead:
                    game.reset()
                else:
                    game.paused = not game.paused

            if not game.dead and not game.paused:
                game.set_dir(pad.get_want())
                step_counter += 1
                if step_counter >= game.step_period(args.fps):
                    game.step()
                    step_counter = 0

            ddp.send(render(game, args.bri))

            dt = time.time() - t0
            if dt < frame_dt:
                time.sleep(frame_dt - dt)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        pad.running = False


if __name__ == "__main__":
    main()
