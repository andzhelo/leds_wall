#!/usr/bin/env python3
"""
space_invaders.py - Space Invaders on the 30x120 Niji LED wall, gamepad-played.

The panel is 30 wide x 120 tall. At 3px per cell that's a 10 x 40 grid; two
rows are reserved for the score HUD, leaving a 10 x 38 play field. The wall's
unusually tall, narrow shape actually suits this game well -- invaders get a
long march down towards the player instead of the usual few rows. Uses the
shared wall.py transport (per-lane calibration applies here too) and the same
evdev gamepad pattern as block_game.py/snake.py.

    python3 space_invaders.py
    python3 space_invaders.py --dev /dev/input/event4 --fps 30

Controls:
    left stick / D-pad   move left/right
    A / B / X / Y        fire (one shot on screen at a time, classic-style)
    Start / L / R        pause; restart after game over

Requires: numpy, evdev
"""
import argparse
import random
import threading
import time

import numpy as np
import evdev
from evdev import ecodes as e

from wall import DDP, W, H

# ----------------------------------------------------------------------------
# Board geometry (in blocks). One block = BLOCK x BLOCK pixels.
# ----------------------------------------------------------------------------
BLOCK = 3
COLS = W // BLOCK                  # 10
HUD_ROWS = 2                       # block rows reserved at the top for the HUD
ROWS = H // BLOCK - HUD_ROWS       # 38 playable rows
Y0 = HUD_ROWS * BLOCK              # first pixel row of the play field

PLAYER_ROW = ROWS - 2              # one row of buffer below the player

INV_ROWS, INV_COLS = 5, 6
INV_COL0 = 2                       # invaders occupy base cols 2..7
INV_ROW_COLOR = [
    (230, 30, 200),                 # front row
    (0, 220, 230), (0, 220, 230),
    (60, 230, 60), (60, 230, 60),
]

SHIELD_ROW = PLAYER_ROW - 6
SHIELD_HP = 3
SHIELDS = [(0, 1), (4, 5), (8, 9)]  # (col, col) pairs, 2 cols x 2 rows each

MAX_PLAYER_BULLETS = 1
MAX_ENEMY_BULLETS = 3
ENEMY_FIRE_CHANCE = 0.02            # per-frame chance to add an enemy bullet


# ----------------------------------------------------------------------------
# Gamepad: background thread keeps latest held direction + edge-triggered
# actions (fire/pause). Same pattern as block_game.py.
# ----------------------------------------------------------------------------
ACT_FIRE = "fire"
ACT_PAUSE = "pause"

BTN_ACTIONS = {
    e.BTN_SOUTH: ACT_FIRE,
    e.BTN_EAST: ACT_FIRE,
    e.BTN_NORTH: ACT_FIRE,
    e.BTN_WEST: ACT_FIRE,
    e.BTN_START: ACT_PAUSE,
    e.BTN_TR: ACT_PAUSE,
    e.BTN_TL: ACT_PAUSE,
}


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
        self.actions = []
        self.dir_x = 0
        self._axmax = self._axis_extent(e.ABS_X)
        self._stick_x = 0
        self._hat_x = 0
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

    def _update_dir(self):
        with self.lock:
            self.dir_x = self._hat_x or self._stick_x

    def run(self):
        thresh_x = self._axmax * 0.5
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

    def drain_actions(self):
        with self.lock:
            acts, self.actions = self.actions, []
            return acts

    def get_dir(self):
        with self.lock:
            return self.dir_x


# ----------------------------------------------------------------------------
# Game
# ----------------------------------------------------------------------------
class SpaceInvaders:
    def __init__(self):
        self.reset()

    def reset(self):
        self.score = 0
        self.lives = 3
        self.wave = 1
        self.paused = False
        self.game_over = False
        self.player_col = COLS // 2
        self.player_bullets = []   # list of [row, col]
        self.enemy_bullets = []    # list of [row, col]
        self.shields = {}          # (row, col) -> hp
        for c0, c1 in SHIELDS:
            for c in (c0, c1):
                for r in (SHIELD_ROW, SHIELD_ROW + 1):
                    self.shields[(r, c)] = SHIELD_HP
        self._spawn_wave()

    def _spawn_wave(self):
        self.alive = {(r, c) for r in range(INV_ROWS) for c in range(INV_COLS)}
        self.off_r = 0
        self.off_c = 0
        self.inv_dir = 1
        self._move_acc = 0.0
        self._pbullet_acc = 0.0
        self._ebullet_acc = 0.0

    def move_player(self, dx):
        self.player_col = max(0, min(COLS - 1, self.player_col + dx))

    def fire_player(self):
        if len(self.player_bullets) < MAX_PLAYER_BULLETS:
            self.player_bullets.append([PLAYER_ROW - 1, self.player_col])

    def _invader_period(self):
        # fewer aliens left -> faster march; higher wave -> faster still
        total = INV_ROWS * INV_COLS
        frac = len(self.alive) / total
        base = 4 + frac * 16
        return max(3, base - (self.wave - 1) * 2)

    def _frontline(self):
        """bottom-most alive invader in each occupied column, as shooters."""
        best = {}
        for r, c in self.alive:
            if c not in best or r > best[c]:
                best[c] = r
        return [(r, c) for c, r in best.items()]

    def _hit_shield(self, row, col):
        if (row, col) in self.shields:
            self.shields[(row, col)] -= 1
            if self.shields[(row, col)] <= 0:
                del self.shields[(row, col)]
            return True
        return False

    def step_invaders(self):
        if not self.alive:
            self.wave += 1
            self._spawn_wave()
            return
        min_c = min(c for _, c in self.alive) + INV_COL0 + self.off_c
        max_c = max(c for _, c in self.alive) + INV_COL0 + self.off_c
        if min_c + self.inv_dir < 0 or max_c + self.inv_dir > COLS - 1:
            self.inv_dir = -self.inv_dir
            self.off_r += 1
        else:
            self.off_c += self.inv_dir
        for r, c in self.alive:
            row, col = r + self.off_r, c + INV_COL0 + self.off_c
            self._hit_shield(row, col)
            if row >= PLAYER_ROW:
                self.game_over = True

    def maybe_enemy_fire(self):
        if len(self.enemy_bullets) >= MAX_ENEMY_BULLETS:
            return
        if random.random() >= ENEMY_FIRE_CHANCE:
            return
        shooters = self._frontline()
        if not shooters:
            return
        r, c = random.choice(shooters)
        row, col = r + self.off_r + 1, c + INV_COL0 + self.off_c
        self.enemy_bullets.append([row, col])

    def step_player_bullets(self):
        keep = []
        for b in self.player_bullets:
            b[0] -= 1
            if b[0] < 0:
                continue
            pos = tuple(b)
            if pos in self.shields:
                self._hit_shield(*pos)
                continue
            hit = None
            for r, c in self.alive:
                if r + self.off_r == b[0] and c + INV_COL0 + self.off_c == b[1]:
                    hit = (r, c)
                    break
            if hit:
                self.alive.discard(hit)
                self.score += 10
                continue
            keep.append(b)
        self.player_bullets = keep

    def step_enemy_bullets(self):
        keep = []
        for b in self.enemy_bullets:
            b[0] += 1
            if b[0] > PLAYER_ROW:
                continue
            pos = tuple(b)
            if pos in self.shields:
                self._hit_shield(*pos)
                continue
            if b[0] == PLAYER_ROW and b[1] == self.player_col:
                self.lives -= 1
                if self.lives <= 0:
                    self.game_over = True
                else:
                    self.player_col = COLS // 2
                continue
            keep.append(b)
        self.enemy_bullets = keep

    def tick(self, fps):
        if self.dead_or_paused():
            return
        self._move_acc += 1
        if self._move_acc >= self._invader_period():
            self._move_acc = 0
            self.step_invaders()
        self._pbullet_acc += 1
        if self._pbullet_acc >= 2:
            self._pbullet_acc = 0
            self.step_player_bullets()
        self._ebullet_acc += 1
        if self._ebullet_acc >= 3:
            self._ebullet_acc = 0
            self.step_enemy_bullets()
        self.maybe_enemy_fire()

    def dead_or_paused(self):
        return self.game_over or self.paused


# ----------------------------------------------------------------------------
# Render
# ----------------------------------------------------------------------------
def draw_cell(frame, r, c, color):
    y, x = Y0 + r * BLOCK, c * BLOCK
    frame[y:y + BLOCK, x:x + BLOCK] = color


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
    x = right - 2
    for ch in reversed(str(int(value))):
        if x < left_bound:
            break
        glyph = DIGITS_3x5[ch]
        for dr in range(5):
            for dc in range(3):
                if glyph[dr][dc] == "1":
                    frame[y + dr, x + dc] = color
        x -= 4


def draw_lives(frame, lives, color=(60, 220, 60)):
    for i in range(lives):
        x = 1 + i * 3
        frame[1:3, x:x + 2] = color


def render(game, bri=1.0):
    frame = np.zeros((H, W, 3), dtype=np.uint8)

    # shields
    for (r, c), hp in game.shields.items():
        t = hp / SHIELD_HP
        frame_color = (int(40 * t) + 20, int(160 * t) + 40, int(40 * t) + 20)
        draw_cell(frame, r, c, frame_color)

    # invaders
    for r, c in game.alive:
        color = INV_ROW_COLOR[r % len(INV_ROW_COLOR)]
        draw_cell(frame, r + game.off_r, c + INV_COL0 + game.off_c, color)

    # bullets
    for row, col in game.player_bullets:
        draw_cell(frame, row, col, (255, 255, 120))
    for row, col in game.enemy_bullets:
        draw_cell(frame, row, col, (230, 60, 30))

    # player ship
    if not game.game_over:
        draw_cell(frame, PLAYER_ROW, game.player_col, (120, 255, 160))

    # HUD
    draw_number(frame, game.score)
    draw_lives(frame, max(0, game.lives))

    if game.paused:
        frame = frame // 3
    if game.game_over:
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
    ap = argparse.ArgumentParser(description="Space Invaders on the 30x120 Niji wall.")
    ap.add_argument("--dev", help="evdev path (auto if omitted)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bri", type=float, default=1.0)
    ap.add_argument("--move-period", type=int, default=3, help="frames between held-move steps")
    args = ap.parse_args()

    path = args.dev or find_gamepad()
    if not path:
        raise SystemExit("No gamepad found. Turn on the controller / plug in the dongle.")
    pad = Gamepad(path)
    pad.start()
    print(f"Controller: {pad.name} ({path})")

    ddp = DDP()
    game = SpaceInvaders()

    frame_dt = 1.0 / args.fps
    move_counter = 0

    try:
        while True:
            t0 = time.time()

            for act in pad.drain_actions():
                if act == ACT_PAUSE:
                    if game.game_over:
                        game.reset()
                    else:
                        game.paused = not game.paused
                elif game.dead_or_paused():
                    continue
                elif act == ACT_FIRE:
                    game.fire_player()

            if not game.dead_or_paused():
                dx = pad.get_dir()
                if dx:
                    move_counter += 1
                    if move_counter >= args.move_period:
                        move_counter = 0
                        game.move_player(dx)
                else:
                    move_counter = 0
                game.tick(args.fps)

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
