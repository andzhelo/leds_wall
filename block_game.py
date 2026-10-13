#!/usr/bin/env python3
"""
block_game.py - a falling-blocks stacker on the 30x120 Niji LED wall, played with a gamepad.

The panel is 30 wide x 120 tall. At 3px per block that's exactly a 10-wide
well (standard stacker width) with room to spare vertically, so we get a deep
36-row well below a small HUD that shows the next piece.

Reuses the DDP transport and geometry from niji.py -- nothing about the wall
changes here, we just render a different (H, W, 3) frame each tick.

Input: a gamepad exposed via evdev (tested with a Flydigi Vader 4 Pro in
XInput mode, which enumerates as "Microsoft X-Box 360 pad"). The device is
auto-detected; override with --dev /dev/input/eventN.

    python3 block_game.py
    python3 block_game.py --dev /dev/input/event4 --fps 30

Controls (left stick or D-pad for movement):
    left / right   move
    down           soft drop
    A              rotate clockwise
    B              rotate counter-clockwise
    X / Y (top)    hard drop
    Start / L/R    pause; restart after game over

Requires: numpy, evdev  (pip3 install evdev)
"""

import argparse
import random
import threading
import time

import numpy as np

from wall import DDP, W, H

# ----------------------------------------------------------------------------
# Board geometry (in blocks). One block = BLOCK x BLOCK pixels.
# ----------------------------------------------------------------------------
BLOCK = 3
COLS = W // BLOCK                 # 10
HUD_ROWS = 4                     # block rows reserved at the top for the HUD
ROWS = H // BLOCK - HUD_ROWS     # 36 playable rows
WELL_Y0 = HUD_ROWS * BLOCK       # first pixel row of the well

# ----------------------------------------------------------------------------
# Tetrominoes. Each piece is a list of (row, col) cells in its spawn state,
# defined inside a 4x4 box so rotation is a clean 90-degree turn about the box
# centre. Colours follow a standard colour-per-shape scheme.
# ----------------------------------------------------------------------------
PIECES = {
    "I": {"cells": [(1, 0), (1, 1), (1, 2), (1, 3)], "color": (0, 220, 230), "box": 4},
    "O": {"cells": [(0, 1), (0, 2), (1, 1), (1, 2)], "color": (230, 200, 0), "box": 4},
    "T": {"cells": [(0, 1), (1, 0), (1, 1), (1, 2)], "color": (170, 0, 220), "box": 3},
    "S": {"cells": [(0, 1), (0, 2), (1, 0), (1, 1)], "color": (0, 200, 40),  "box": 3},
    "Z": {"cells": [(0, 0), (0, 1), (1, 1), (1, 2)], "color": (230, 30, 30), "box": 3},
    "J": {"cells": [(0, 0), (1, 0), (1, 1), (1, 2)], "color": (30, 60, 230), "box": 3},
    "L": {"cells": [(0, 2), (1, 0), (1, 1), (1, 2)], "color": (230, 110, 0), "box": 3},
}
KICKS = [0, -1, 1, -2, 2]        # simple horizontal wall-kick offsets to try


def rotate_cells(cells, box, cw=True):
    """Rotate cells 90 degrees within a box x box bounding square."""
    out = []
    for r, c in cells:
        if cw:
            out.append((c, box - 1 - r))
        else:
            out.append((box - 1 - c, r))
    return out


class Piece:
    def __init__(self, kind):
        self.kind = kind
        spec = PIECES[kind]
        self.cells = list(spec["cells"])
        self.box = spec["box"]
        self.color = spec["color"]
        self.row = 0
        # spawn centred horizontally
        self.col = (COLS - self.box) // 2

    def world_cells(self, cells=None, row=None, col=None):
        cells = self.cells if cells is None else cells
        row = self.row if row is None else row
        col = self.col if col is None else col
        return [(row + r, col + c) for r, c in cells]


# ----------------------------------------------------------------------------
# Gamepad input. A background thread keeps the latest control state; the game
# loop reads held directions and drains an edge-triggered action queue. Same
# "freshest state, never block the render" idea as the mirror's camera thread.
# ----------------------------------------------------------------------------
import evdev
from evdev import ecodes as e

# action names pushed on button-press edges
ACT_ROT_CW = "rot_cw"
ACT_ROT_CCW = "rot_ccw"
ACT_HARD = "hard"
ACT_PAUSE = "pause"

BTN_ACTIONS = {
    e.BTN_SOUTH: ACT_ROT_CW,     # A
    e.BTN_EAST:  ACT_ROT_CCW,    # B
    e.BTN_NORTH: ACT_HARD,       # X (top on this pad)
    e.BTN_WEST:  ACT_HARD,       # Y
    e.BTN_START: ACT_PAUSE,
    e.BTN_TR:    ACT_PAUSE,
    e.BTN_TL:    ACT_PAUSE,
}


def find_gamepad():
    """Return the path of the first device that looks like a gamepad."""
    for path in evdev.list_devices():
        try:
            dev = evdev.InputDevice(path)
        except OSError:
            continue
        caps = dev.capabilities()
        keys = caps.get(e.EV_KEY, [])
        if e.BTN_GAMEPAD in keys or e.BTN_SOUTH in keys:
            return path
    return None


class Gamepad(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True)
        self.dev = evdev.InputDevice(path)
        self.name = self.dev.name
        self.lock = threading.Lock()
        self.actions = []            # edge-triggered, drained by the game loop
        self.dir_x = 0               # -1 / 0 / +1 from stick or hat
        self.soft = False            # down held
        self._axmax = self._axis_extent(e.ABS_X)
        self._aymax = self._axis_extent(e.ABS_Y)
        self._stick_x = 0            # last stick-derived direction
        self._hat_x = 0              # last hat-derived direction
        self.running = True

    def _axis_extent(self, code):
        try:
            info = self.dev.absinfo(code)
            return max(abs(info.min), abs(info.max)) or 1
        except OSError:
            return 32767

    def _push(self, act):
        with self.lock:
            self.actions.append(act)

    def run(self):
        thresh_x = self._axmax * 0.5
        thresh_y = self._aymax * 0.5
        for ev in self.dev.read_loop():
            if not self.running:
                break
            if ev.type == e.EV_KEY and ev.value == 1:
                act = BTN_ACTIONS.get(ev.code)
                if act:
                    self._push(act)
            elif ev.type == e.EV_ABS:
                if ev.code == e.ABS_X:
                    self._stick_x = -1 if ev.value < -thresh_x else (1 if ev.value > thresh_x else 0)
                    self._update_dir()
                elif ev.code == e.ABS_HAT0X:
                    self._hat_x = int(ev.value)
                    self._update_dir()
                elif ev.code == e.ABS_Y:
                    with self.lock:
                        self.soft = ev.value > thresh_y   # up is negative, down positive
                elif ev.code == e.ABS_HAT0Y:
                    with self.lock:
                        self.soft = ev.value > 0

    def _update_dir(self):
        with self.lock:
            self.dir_x = self._hat_x or self._stick_x

    def drain_actions(self):
        with self.lock:
            acts, self.actions = self.actions, []
            return acts

    def state(self):
        with self.lock:
            return self.dir_x, self.soft


# ----------------------------------------------------------------------------
# Game
# ----------------------------------------------------------------------------
class BlockGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.grid = [[None] * COLS for _ in range(ROWS)]
        self.bag = []
        self.piece = self._spawn()
        self.next_kind = self._draw_kind()
        self.lines = 0
        self.level = 0
        self.score = 0
        self.game_over = False
        self.paused = False

    def _draw_kind(self):
        if not self.bag:
            self.bag = list(PIECES.keys())
            random.shuffle(self.bag)
        return self.bag.pop()

    def _spawn(self, kind=None):
        return Piece(kind or self._draw_kind())

    def _collides(self, cells, row, col):
        for r, c in [(row + cr, col + cc) for cr, cc in cells]:
            if c < 0 or c >= COLS or r >= ROWS:
                return True
            if r >= 0 and self.grid[r][c] is not None:
                return True
        return False

    def move(self, dc):
        if not self._collides(self.piece.cells, self.piece.row, self.piece.col + dc):
            self.piece.col += dc
            return True
        return False

    def rotate(self, cw=True):
        rotated = rotate_cells(self.piece.cells, self.piece.box, cw)
        for k in KICKS:
            if not self._collides(rotated, self.piece.row, self.piece.col + k):
                self.piece.cells = rotated
                self.piece.col += k
                return True
        return False

    def _drop_one(self):
        """Move down one; lock and return False if it can't."""
        if not self._collides(self.piece.cells, self.piece.row + 1, self.piece.col):
            self.piece.row += 1
            return True
        self._lock()
        return False

    def soft_drop(self):
        self._drop_one()

    def hard_drop(self):
        while not self._collides(self.piece.cells, self.piece.row + 1, self.piece.col):
            self.piece.row += 1
        self._lock()

    def tick_gravity(self):
        self._drop_one()

    def _lock(self):
        for r, c in self.piece.world_cells():
            if r < 0:
                self.game_over = True
                return
            self.grid[r][c] = self.piece.color
        self._clear_lines()
        self.piece = self._spawn(self.next_kind)
        self.next_kind = self._draw_kind()
        if self._collides(self.piece.cells, self.piece.row, self.piece.col):
            self.game_over = True

    # classic line-clear points, scaled by the level you cleared them at
    _LINE_POINTS = {1: 40, 2: 100, 3: 300, 4: 1200}

    def _clear_lines(self):
        kept = [row for row in self.grid if any(cell is None for cell in row)]
        cleared = ROWS - len(kept)
        if cleared:
            self.score += self._LINE_POINTS.get(cleared, 0) * (self.level + 1)
            self.lines += cleared
            self.level = self.lines // 10
            self.grid = [[None] * COLS for _ in range(cleared)] + kept

    def gravity_period(self, fps):
        """Frames between gravity steps; speeds up with level."""
        secs = max(0.05, 0.8 - self.level * 0.07)
        return max(1, int(secs * fps))


# ----------------------------------------------------------------------------
# Rendering: game state -> (H, W, 3) uint8 frame
# ----------------------------------------------------------------------------
def draw_block(frame, br, bc, color, y0=WELL_Y0):
    """Fill one BLOCK x BLOCK cell with a slightly shaded block."""
    y = y0 + br * BLOCK
    x = bc * BLOCK
    r, g, b = color
    frame[y:y + BLOCK, x:x + BLOCK] = (r, g, b)
    frame[y, x:x + BLOCK] = (min(255, r + 40), min(255, g + 40), min(255, b + 40))


# 3x5 pixel font for the score readout (digits only)
DIGITS_3x5 = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("110", "010", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "001", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
}


def draw_number(frame, value, y=1, color=(200, 200, 210), right=W - 1, left_bound=13):
    """Right-align a number at the top-right; grow leftward, stop at left_bound."""
    x = right - 2                           # left column of the rightmost digit
    for ch in reversed(str(int(value))):
        if x < left_bound:
            break                           # ran into the next-piece preview
        glyph = DIGITS_3x5[ch]
        for dr in range(5):
            for dc in range(3):
                if glyph[dr][dc] == "1":
                    frame[y + dr, x + dc] = color
        x -= 4                              # 3px digit + 1px gap


def render(game, bri=1.0):
    frame = np.zeros((H, W, 3), dtype=np.uint8)

    # settled blocks
    for br in range(ROWS):
        for bc in range(COLS):
            color = game.grid[br][bc]
            if color:
                draw_block(frame, br, bc, color)

    # falling piece (skip cells above the top edge)
    if not game.game_over:
        for r, c in game.piece.world_cells():
            if 0 <= r < ROWS:
                draw_block(frame, r, c, game.piece.color)

    # HUD: next piece preview, top-left corner
    spec = PIECES[game.next_kind]
    for r, c in spec["cells"]:
        draw_block(frame, r, c, spec["color"], y0=0)

    # HUD: score in the top-right corner (up to 4 digits fit beside the preview)
    draw_number(frame, game.score)

    if game.paused:
        frame = frame // 3
    if game.game_over:
        # dim red wash so it's obvious from across the room
        wash = np.zeros_like(frame)
        wash[:, :] = (40, 0, 0)
        frame = np.maximum(frame // 2, wash)

    if bri < 1.0:
        frame = (frame.astype(np.float32) * bri).astype(np.uint8)
    return frame


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Falling-blocks stacker on the 30x120 Niji wall.")
    ap.add_argument("--dev", help="evdev path, e.g. /dev/input/event4 (auto if omitted)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bri", type=float, default=1.0)
    ap.add_argument("--das", type=int, default=6, help="frames before held-move auto-repeat")
    ap.add_argument("--arr", type=int, default=2, help="frames between auto-repeat moves")
    args = ap.parse_args()

    path = args.dev or find_gamepad()
    if not path:
        raise SystemExit("No gamepad found. Plug in the dongle / turn on the controller, "
                         "or pass --dev /dev/input/eventN.")
    pad = Gamepad(path)
    pad.start()
    print(f"Controller: {pad.name} ({path})")

    ddp = DDP()
    game = BlockGame()

    frame_dt = 1.0 / args.fps
    grav_counter = 0
    das_counter = 0
    last_dir = 0
    soft_counter = 0

    try:
        while True:
            t0 = time.time()

            for act in pad.drain_actions():
                if act == ACT_PAUSE:
                    if game.game_over:
                        game.reset()
                    else:
                        game.paused = not game.paused
                elif game.game_over or game.paused:
                    continue
                elif act == ACT_ROT_CW:
                    game.rotate(cw=True)
                elif act == ACT_ROT_CCW:
                    game.rotate(cw=False)
                elif act == ACT_HARD:
                    game.hard_drop()

            if not game.game_over and not game.paused:
                dir_x, soft = pad.state()

                # horizontal move with DAS/ARR auto-repeat
                if dir_x != 0:
                    if dir_x != last_dir:
                        game.move(dir_x)
                        das_counter = 0
                    else:
                        das_counter += 1
                        if das_counter >= args.das and (das_counter - args.das) % args.arr == 0:
                            game.move(dir_x)
                else:
                    das_counter = 0
                last_dir = dir_x

                # soft drop
                if soft:
                    soft_counter += 1
                    if soft_counter % max(1, args.arr) == 0:
                        game.soft_drop()
                else:
                    soft_counter = 0

                # gravity
                grav_counter += 1
                if grav_counter >= game.gravity_period(args.fps):
                    game.tick_gravity()
                    grav_counter = 0

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
