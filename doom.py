#!/usr/bin/env python3
"""
doom.py - a Doom-like raycaster for the 30x120 Niji LED wall, gamepad-played.

30 columns is the entire horizontal angular resolution: one ray per column, so
at a 66 deg FOV each pixel is ~2.2 deg and a distant enemy is 2 px wide. The
panel's 1:4 portrait shape is also wrong for a forward view. Both problems are
solved by not making the whole wall the 3D view -- instead the tall shape is
used the way Doom's status bar always wanted to be:

    rows   0- 39   3D view (30 x 40), raycast walls + billboard sprites
    rows  40- 41   flash strip (white on fire, red on damage)
    rows  42- 97   automap (30 x 56), 2 px/cell, scrolls with the player
    rows  98-119   HUD: health bar, ammo bar, 8x8 face

Navigate by the automap, shoot by the view. Walls are flat-shaded by distance
and orientation rather than textured (texture at 30 columns is just noise), and
firing snaps to anything within AIM_TOL columns of centre because pixel-precise
aim is not possible here.

Uses the shared wall.py transport (per-lane calibration applies) and the same
evdev gamepad pattern as tetris.py/snake.py, extended with analog axes.

    python3 doom.py
    python3 doom.py --dev /dev/input/event4 --fps 30 --bri 0.8

Controls:
    left stick          move / strafe        (D-pad works too)
    right stick X       turn                 (shoulder L/R also turn)
    RT / A / B / X / Y  fire (hold to keep firing)
    Start               pause; restart after death

Requires: numpy, evdev
"""
import argparse
import math
import random
import threading
import time

import numpy as np
import evdev
from evdev import ecodes as e

from wall import DDP, W, H

# ----------------------------------------------------------------------------
# Screen layout (rows on the 120-tall panel)
# ----------------------------------------------------------------------------
VIEW_W, VIEW_H = W, 40
FLASH_Y, FLASH_H = 40, 2
MM_Y, MM_H = 42, 56
HUD_Y, HUD_H = 98, 22

MM_CELL = 2                            # automap pixels per map cell
MM_COLS = W // MM_CELL                 # 15 cells across
MM_ROWS = MM_H // MM_CELL              # 28 cells down

# ----------------------------------------------------------------------------
# Projection. PROJ makes the vertical and horizontal scales agree: a 1-unit-tall
# wall at distance 1 is PROJ px tall and 1 unit wide is PROJ px wide, so nothing
# is stretched and the leftover vertical space reads as floor and ceiling.
# ----------------------------------------------------------------------------
FOV = 66.0
PLANE_LEN = math.tan(math.radians(FOV / 2.0))     # ~0.649
PROJ = VIEW_W / (2.0 * PLANE_LEN)                 # ~23.1 px per unit at dist 1
HORIZON = VIEW_H // 2

MAX_DEPTH = 24.0

# ----------------------------------------------------------------------------
# Palette. At this resolution about four distinguishable hues is the budget:
# walls stay desaturated so the saturated enemy/pickup colours pop.
# ----------------------------------------------------------------------------
WALL_COLOR = {
    1: (140, 165, 215),     # blue-grey stone
    2: (205, 115, 60),      # rust
}
SIDE_DARKEN = 0.70          # E/W faces darker than N/S, the classic depth cue
CEIL_COLOR = (30, 33, 50)       # cool, and darker than the floor
FLOOR_COLOR = (66, 54, 42)      # warm brown; these fill most of the view, so
                                # near-black here makes the whole panel dead

IMP_COLOR = (240, 60, 30)
DEMON_COLOR = (230, 40, 160)
AMMO_COLOR = (255, 200, 40)
MEDKIT_COLOR = (245, 245, 250)
GUN_COLOR = (110, 110, 125)

# ----------------------------------------------------------------------------
# Level. 24x24. '1'/'2' are wall types, '.' is floor.
# ----------------------------------------------------------------------------
MAP_STR = [
    "111111111111111111111111",
    "1......11.......1......1",
    "1......1........1......1",
    "1...........1..........1",
    "1......1........1......1",
    "111.1111........1111.111",
    "1......2........2......1",
    "1......2........2......1",
    "1......2222..2222......1",
    "1......................1",
    "1..22..............22..1",
    "1..2................2..1",
    "1..2................2..1",
    "1..22..............22..1",
    "1......................1",
    "1......1111..1111......1",
    "1......1........1......1",
    "1......1........1......1",
    "111.1111........1111.111",
    "1......2........2......1",
    "1...........2..........1",
    "1......2........2......1",
    "1......2........2......1",
    "111111111111111111111111",
]
MAP = [[0 if c == "." else int(c) for c in row] for row in MAP_STR]
MAP_W, MAP_H = len(MAP[0]), len(MAP)

START_X, START_Y, START_A = 12.5, 9.5, math.pi / 2

# ----------------------------------------------------------------------------
# Tuning
# ----------------------------------------------------------------------------
MOVE_SPEED = 3.0            # cells/sec
TURN_SPEED = 2.6            # rad/sec
COLLIDE_R = 0.28

AIM_TOL = 2                 # columns either side of centre that count as a hit
FIRE_COOLDOWN = 6           # frames between shots
SHOT_DAMAGE = 34
START_AMMO = 60
MAX_AMMO = 99

SPRITE_W_UNITS = 0.70
SPRITE_H_UNITS = 0.90
MIN_SPRITE_PX = 2           # never let a monster fall below 2 columns

ENEMY_KINDS = {
    #        hp  speed  dmg  reach  color
    "imp":   (60, 1.20, 8,  1.20, IMP_COLOR),
    "demon": (110, 1.70, 14, 1.30, DEMON_COLOR),
}
ENEMY_ATTACK_COOLDOWN = 22  # frames


def is_wall(x, y):
    mx, my = int(x), int(y)
    if mx < 0 or my < 0 or mx >= MAP_W or my >= MAP_H:
        return True
    return MAP[my][mx] != 0


def open_cells():
    return [(x + 0.5, y + 0.5)
            for y in range(MAP_H) for x in range(MAP_W) if MAP[y][x] == 0]


def line_of_sight(x0, y0, x1, y1):
    """Coarse LOS: step along the segment looking for a wall."""
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return True
    steps = int(dist * 6) + 1
    for i in range(1, steps):
        t = i / steps
        if is_wall(x0 + dx * t, y0 + dy * t):
            return False
    return True


def shade(color, dist, extra=1.0):
    # Gentle falloff with a high floor: the wall runs at ~40% brightness, so a
    # steep 1/d curve turns anything past a few cells into black.
    f = extra / (1.0 + dist * 0.16)
    f = max(0.30, min(1.0, f))
    return (int(color[0] * f), int(color[1] * f), int(color[2] * f))


# ----------------------------------------------------------------------------
# Gamepad. Same background-thread pattern as tetris.py/space_invaders.py, but
# exposing normalised analog axes (-1..1) since turning needs to be smooth, plus
# a held-fire flag rather than only edge-triggered actions.
# ----------------------------------------------------------------------------
ACT_PAUSE = "pause"
DEADZONE = 0.22

BTN_FIRE = {e.BTN_SOUTH, e.BTN_EAST, e.BTN_NORTH, e.BTN_WEST}
BTN_TURN_L = {e.BTN_TL}
BTN_TURN_R = {e.BTN_TR}


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
        self.running = True

        self._range = {code: self._axis_extent(code)
                       for code in (e.ABS_X, e.ABS_Y, e.ABS_RX, e.ABS_RZ)}
        self._stick = {e.ABS_X: 0.0, e.ABS_Y: 0.0, e.ABS_RX: 0.0}
        self._hat_x = 0
        self._hat_y = 0
        self._btn_fire = False
        self._trig_fire = False
        self._btn_turn = 0

    def _axis_extent(self, code):
        try:
            info = self.dev.absinfo(code)
            return max(abs(info.min), abs(info.max)) or 1
        except OSError:
            return 32767

    @staticmethod
    def _dead(v):
        if abs(v) < DEADZONE:
            return 0.0
        # rescale so motion starts at zero just outside the deadzone
        return math.copysign((abs(v) - DEADZONE) / (1.0 - DEADZONE), v)

    def _push(self, act):
        with self.lock:
            self.actions.append(act)

    def run(self):
        for ev in self.dev.read_loop():
            if not self.running:
                break
            if ev.type == e.EV_KEY:
                held = ev.value != 0
                if ev.code in BTN_FIRE:
                    with self.lock:
                        self._btn_fire = held
                elif ev.code in BTN_TURN_L:
                    with self.lock:
                        self._btn_turn = -1 if held else 0
                elif ev.code in BTN_TURN_R:
                    with self.lock:
                        self._btn_turn = 1 if held else 0
                elif ev.code == e.BTN_START and ev.value == 1:
                    self._push(ACT_PAUSE)
            elif ev.type == e.EV_ABS:
                if ev.code in (e.ABS_X, e.ABS_Y, e.ABS_RX):
                    with self.lock:
                        self._stick[ev.code] = ev.value / self._range[ev.code]
                elif ev.code == e.ABS_RZ:      # right trigger, 0..max
                    with self.lock:
                        self._trig_fire = ev.value > self._range[e.ABS_RZ] * 0.4
                elif ev.code == e.ABS_HAT0X:
                    with self.lock:
                        self._hat_x = int(ev.value)
                elif ev.code == e.ABS_HAT0Y:
                    with self.lock:
                        self._hat_y = int(ev.value)

    def drain_actions(self):
        with self.lock:
            acts, self.actions = self.actions, []
            return acts

    def state(self):
        """(strafe, forward, turn, firing), each axis in -1..1."""
        with self.lock:
            sx = self._dead(self._stick[e.ABS_X]) or float(self._hat_x)
            # stick Y is negative up; forward should be positive
            fy = -self._dead(self._stick[e.ABS_Y]) or float(-self._hat_y)
            tx = self._dead(self._stick[e.ABS_RX]) or float(self._btn_turn)
            fire = self._btn_fire or self._trig_fire
        clamp = lambda v: max(-1.0, min(1.0, v))
        return clamp(sx), clamp(fy), clamp(tx), fire


# ----------------------------------------------------------------------------
# Entities
# ----------------------------------------------------------------------------
class Enemy:
    def __init__(self, x, y, kind):
        hp, speed, dmg, reach, color = ENEMY_KINDS[kind]
        self.x, self.y = x, y
        self.kind = kind
        self.hp = hp
        self.max_hp = hp
        self.speed = speed
        self.dmg = dmg
        self.reach = reach
        self.color = color
        self.alive = True
        self.hurt = 0           # frames of white flash after being shot
        self.cooldown = random.randint(0, ENEMY_ATTACK_COOLDOWN)
        self.wander = random.uniform(0, math.tau)

    def move(self, dx, dy):
        """Returns False if fully blocked, so the caller can pick a new heading."""
        moved = False
        if not is_wall(self.x + dx, self.y):
            self.x += dx
            moved = True
        if not is_wall(self.x, self.y + dy):
            self.y += dy
            moved = True
        return moved


class Pickup:
    def __init__(self, x, y, kind):
        self.x, self.y = x, y
        self.kind = kind        # "ammo" | "medkit"
        self.color = AMMO_COLOR if kind == "ammo" else MEDKIT_COLOR


# ----------------------------------------------------------------------------
# Game
# ----------------------------------------------------------------------------
class Doom:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x, self.y, self.a = START_X, START_Y, START_A
        self.health = 100
        self.ammo = START_AMMO
        self.kills = 0
        self.wave = 1
        self.paused = False
        self.game_over = False
        self.seen = set()
        self.fire_flash = 0
        self.hurt_flash = 0
        self.cooldown = 0
        self.bob = 0.0
        self.enemies = []
        self.pickups = []
        self._spawn_wave()

    # -- spawning ------------------------------------------------------------
    def _far_cells(self, min_dist, count):
        cells = [c for c in open_cells()
                 if math.hypot(c[0] - self.x, c[1] - self.y) >= min_dist]
        random.shuffle(cells)
        return cells[:count]

    def _spawn_wave(self):
        # The map is 400+ open cells and sight lines are short, so a thin
        # population means you wander without ever meeting anything.
        n_imp = min(14, 5 + 2 * self.wave)
        n_demon = min(6, max(0, self.wave - 1))
        spots = self._far_cells(7.0, n_imp + n_demon)
        for i, (cx, cy) in enumerate(spots):
            kind = "demon" if i < n_demon else "imp"
            self.enemies.append(Enemy(cx, cy, kind))
        for (cx, cy) in self._far_cells(4.0, 3):
            kind = "ammo" if random.random() < 0.6 else "medkit"
            self.pickups.append(Pickup(cx, cy, kind))

    # -- player --------------------------------------------------------------
    def move_player(self, strafe, forward, turn, dt):
        if turn:
            self.a = (self.a + turn * TURN_SPEED * dt) % math.tau
        dir_x, dir_y = math.cos(self.a), math.sin(self.a)
        # strafe axis is dir rotated +90 deg
        str_x, str_y = -dir_y, dir_x
        step = MOVE_SPEED * dt
        dx = (dir_x * forward + str_x * strafe) * step
        dy = (dir_y * forward + str_y * strafe) * step
        # axis-separated so sliding along a wall works instead of sticking
        if not is_wall(self.x + dx + math.copysign(COLLIDE_R, dx), self.y):
            self.x += dx
        if not is_wall(self.x, self.y + dy + math.copysign(COLLIDE_R, dy)):
            self.y += dy
        if forward or strafe:
            self.bob += step * 2.2

    def fire(self):
        if self.cooldown > 0 or self.ammo <= 0:
            return
        self.cooldown = FIRE_COOLDOWN
        self.ammo -= 1
        self.fire_flash = 3

        # Hitscan with aim assist: nearest visible enemy whose screen column is
        # within AIM_TOL of centre. Pixel-precise aim is impossible at W=30.
        best, best_d = None, 1e9
        for en in self.enemies:
            if not en.alive:
                continue
            col, depth = self.project(en.x, en.y)
            if col is None or depth >= best_d:
                continue
            if abs(col - VIEW_W / 2.0) > AIM_TOL + 0.5:
                continue
            if not line_of_sight(self.x, self.y, en.x, en.y):
                continue
            best, best_d = en, depth
        if best is not None:
            best.hp -= SHOT_DAMAGE
            best.hurt = 3
            if best.hp <= 0:
                best.alive = False
                self.kills += 1

    def hurt_player(self, dmg):
        self.health -= dmg
        self.hurt_flash = 5
        if self.health <= 0:
            self.health = 0
            self.game_over = True

    # -- camera --------------------------------------------------------------
    def camera(self):
        dir_x, dir_y = math.cos(self.a), math.sin(self.a)
        return dir_x, dir_y, -dir_y * PLANE_LEN, dir_x * PLANE_LEN

    def project(self, sx, sy):
        """World point -> (screen column, perpendicular depth), or (None, None)."""
        dir_x, dir_y, plane_x, plane_y = self.camera()
        rel_x, rel_y = sx - self.x, sy - self.y
        det = plane_x * dir_y - dir_x * plane_y
        if abs(det) < 1e-9:
            return None, None
        inv = 1.0 / det
        tx = inv * (dir_y * rel_x - dir_x * rel_y)
        ty = inv * (-plane_y * rel_x + plane_x * rel_y)
        if ty <= 0.05:
            return None, None
        return (VIEW_W / 2.0) * (1.0 + tx / ty), ty

    # -- world ---------------------------------------------------------------
    def tick(self, fps):
        if self.dead_or_paused():
            return
        dt = 1.0 / fps
        self.cooldown = max(0, self.cooldown - 1)
        self.fire_flash = max(0, self.fire_flash - 1)
        self.hurt_flash = max(0, self.hurt_flash - 1)

        for en in self.enemies:
            if not en.alive:
                continue
            en.hurt = max(0, en.hurt - 1)
            en.cooldown = max(0, en.cooldown - 1)
            dx, dy = self.x - en.x, self.y - en.y
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                continue
            if line_of_sight(en.x, en.y, self.x, self.y):
                if dist <= en.reach:
                    if en.cooldown == 0:
                        en.cooldown = ENEMY_ATTACK_COOLDOWN
                        self.hurt_player(en.dmg)
                else:
                    step = en.speed * dt
                    if not en.move(dx / dist * step, dy / dist * step):
                        en.wander = math.atan2(dy, dx) + random.choice((-1.4, 1.4))
            else:
                # No sight line: bias the drift towards the player so they close
                # in instead of milling about. Purely random wandering left the
                # map big enough that fights simply never started.
                want = math.atan2(dy, dx)
                diff = (want - en.wander + math.pi) % math.tau - math.pi
                en.wander += diff * 0.35 + random.uniform(-0.20, 0.20)
                step = en.speed * 0.70 * dt
                if not en.move(math.cos(en.wander) * step, math.sin(en.wander) * step):
                    # jammed against a wall between us and the player: sidestep,
                    # otherwise they press into it forever and fights never happen
                    en.wander += random.choice((-1.5, 1.5))

        # Safety net: running dry with no ammo box left means you can only stand
        # there and be eaten, so top the level up rather than softlock the wave.
        if self.ammo < 5 and not any(p.kind == "ammo" for p in self.pickups):
            for (cx, cy) in self._far_cells(3.0, 1):
                self.pickups.append(Pickup(cx, cy, "ammo"))

        for p in list(self.pickups):
            if math.hypot(p.x - self.x, p.y - self.y) < 0.55:
                if p.kind == "ammo":
                    self.ammo = min(MAX_AMMO, self.ammo + 20)
                else:
                    self.health = min(100, self.health + 25)
                self.pickups.remove(p)

        if not any(en.alive for en in self.enemies):
            self.wave += 1
            self.enemies = []
            self._spawn_wave()

    def dead_or_paused(self):
        return self.game_over or self.paused


# ----------------------------------------------------------------------------
# Render: 3D view
# ----------------------------------------------------------------------------
def cast(game):
    """One DDA ray per column. Returns per-column (dist, wall_type, side)."""
    dir_x, dir_y, plane_x, plane_y = game.camera()
    cols = []
    for x in range(VIEW_W):
        camera_x = 2.0 * x / VIEW_W - 1.0
        ray_x = dir_x + plane_x * camera_x
        ray_y = dir_y + plane_y * camera_x
        map_x, map_y = int(game.x), int(game.y)
        delta_x = abs(1.0 / ray_x) if ray_x != 0 else 1e30
        delta_y = abs(1.0 / ray_y) if ray_y != 0 else 1e30

        if ray_x < 0:
            step_x, side_x = -1, (game.x - map_x) * delta_x
        else:
            step_x, side_x = 1, (map_x + 1.0 - game.x) * delta_x
        if ray_y < 0:
            step_y, side_y = -1, (game.y - map_y) * delta_y
        else:
            step_y, side_y = 1, (map_y + 1.0 - game.y) * delta_y

        hit, side, wtype = False, 0, 1
        while not hit:
            if side_x < side_y:
                side_x += delta_x
                map_x += step_x
                side = 0
            else:
                side_y += delta_y
                map_y += step_y
                side = 1
            if map_x < 0 or map_y < 0 or map_x >= MAP_W or map_y >= MAP_H:
                break
            game.seen.add((map_x, map_y))       # automap reveals as you look
            if MAP[map_y][map_x] != 0:
                wtype = MAP[map_y][map_x]
                hit = True
            if min(side_x, side_y) > MAX_DEPTH:
                break

        if hit:
            dist = (side_x - delta_x) if side == 0 else (side_y - delta_y)
            dist = max(0.05, dist)
        else:
            dist = MAX_DEPTH
        cols.append((dist, wtype, side))
    return cols


def draw_view(frame, game):
    cols = cast(game)
    zbuf = [c[0] for c in cols]

    for x, (dist, wtype, side) in enumerate(cols):
        line_h = PROJ / dist
        top = int(HORIZON - line_h / 2.0)
        bot = int(HORIZON + line_h / 2.0)
        y0, y1 = max(0, top), min(VIEW_H, bot + 1)

        # ceiling and floor, slightly graded away from the horizon
        for y in range(0, min(y0, VIEW_H)):
            t = 0.45 + 0.55 * (y / max(1, HORIZON))
            frame[y, x] = tuple(int(c * t) for c in CEIL_COLOR)
        for y in range(max(0, y1), VIEW_H):
            t = 0.45 + 0.55 * ((y - HORIZON) / max(1, VIEW_H - HORIZON))
            frame[y, x] = tuple(int(c * t) for c in FLOOR_COLOR)

        if y1 > y0:
            base = WALL_COLOR.get(wtype, WALL_COLOR[1])
            frame[y0:y1, x] = shade(base, dist, SIDE_DARKEN if side else 1.0)

    draw_sprites(frame, game, zbuf)
    draw_gun(frame, game)


def draw_sprites(frame, game, zbuf):
    items = [(en, en.color, en.hurt > 0) for en in game.enemies if en.alive]
    items += [(p, p.color, False) for p in game.pickups]

    # far to near so nearer sprites overwrite
    order = []
    for obj, color, flash in items:
        col, depth = game.project(obj.x, obj.y)
        if col is None:
            continue
        order.append((depth, col, color, flash))
    order.sort(key=lambda t: -t[0])

    for depth, col, color, flash in order:
        h_px = max(MIN_SPRITE_PX, int(PROJ * SPRITE_H_UNITS / depth))
        w_px = max(MIN_SPRITE_PX, int(PROJ * SPRITE_W_UNITS / depth))
        # feet sit where the floor meets a wall at this distance
        bottom = int(HORIZON + (PROJ / depth) / 2.0)
        top = bottom - h_px
        x0 = int(col - w_px / 2.0)
        shown = (255, 255, 255) if flash else shade(color, depth)
        for x in range(x0, x0 + w_px):
            if x < 0 or x >= VIEW_W or depth >= zbuf[x]:
                continue
            for y in range(max(0, top), min(VIEW_H, bottom)):
                frame[y, x] = shown


def draw_gun(frame, game):
    """Small barrel at the bottom centre, bobbing with movement."""
    bob = int(round(math.sin(game.bob) * 1.2))
    cx = VIEW_W // 2
    top = VIEW_H - 5 + max(0, bob)
    frame[top:VIEW_H, cx - 1:cx + 1] = GUN_COLOR
    frame[VIEW_H - 2:VIEW_H, cx - 3:cx + 3] = tuple(int(c * 0.7) for c in GUN_COLOR)
    if game.fire_flash:
        t = game.fire_flash / 3.0
        muzzle = (int(255 * t), int(230 * t), int(140 * t))
        frame[max(0, top - 2):top, cx - 2:cx + 2] = muzzle


# ----------------------------------------------------------------------------
# Render: flash strip, automap, HUD
# ----------------------------------------------------------------------------
def draw_flash(frame, game):
    if game.hurt_flash:
        t = game.hurt_flash / 5.0
        frame[FLASH_Y:FLASH_Y + FLASH_H] = (int(220 * t), int(20 * t), int(15 * t))
    elif game.fire_flash:
        t = game.fire_flash / 3.0
        frame[FLASH_Y:FLASH_Y + FLASH_H] = (int(200 * t), int(180 * t), int(110 * t))
    else:
        frame[FLASH_Y:FLASH_Y + FLASH_H] = (10, 10, 14)


def draw_map(frame, game):
    cx0 = int(game.x) - MM_COLS // 2
    cy0 = int(game.y) - MM_ROWS // 2

    for ry in range(MM_ROWS):
        for rx in range(MM_COLS):
            mx, my = cx0 + rx, cy0 + ry
            py, px = MM_Y + ry * MM_CELL, rx * MM_CELL
            if mx < 0 or my < 0 or mx >= MAP_W or my >= MAP_H:
                color = (0, 0, 0)
            elif (mx, my) not in game.seen:
                color = (0, 0, 0)
            elif MAP[my][mx] != 0:
                base = WALL_COLOR.get(MAP[my][mx], WALL_COLOR[1])
                color = tuple(int(c * 0.80) for c in base)
            else:
                color = (16, 18, 26)
            frame[py:py + MM_CELL, px:px + MM_CELL] = color

    def plot(wx, wy, color, size=1):
        rx = (wx - cx0) * MM_CELL
        ry = (wy - cy0) * MM_CELL
        px, py = int(rx), int(MM_Y + ry)
        if 0 <= px < W - size + 1 and MM_Y <= py < MM_Y + MM_H - size + 1:
            frame[py:py + size, px:px + size] = color

    for p in game.pickups:
        if (int(p.x), int(p.y)) in game.seen:
            plot(p.x, p.y, p.color)
    for en in game.enemies:
        if en.alive:
            plot(en.x, en.y, en.color)

    # player + a pip showing facing
    plot(game.x, game.y, (255, 255, 255), size=2)
    plot(game.x + math.cos(game.a) * 1.1, game.y + math.sin(game.a) * 1.1,
         (90, 200, 255))


FACE_OK = (
    "..####..",
    ".######.",
    "#oo##oo#",
    "########",
    "###--###",
    "########",
    ".######.",
    "..####..",
)
FACE_HURT = (
    "..####..",
    ".######.",
    "#o####o#",
    "########",
    "##----##",
    "##----##",
    ".######.",
    "..####..",
)
FACE_DEAD = (
    "..####..",
    ".##..##.",
    "#o#..#o#",
    "###..###",
    "#------#",
    "#------#",
    ".######.",
    "..#..#..",
)


def draw_hud(frame, game):
    frame[HUD_Y] = (18, 18, 24)

    # health bar
    frac = max(0.0, game.health / 100.0)
    filled = int(round(28 * frac))
    if frac > 0.6:
        hcol = (60, 220, 80)
    elif frac > 0.3:
        hcol = (230, 200, 50)
    else:
        hcol = (230, 50, 40)
    frame[HUD_Y + 2:HUD_Y + 7, 1:29] = (22, 22, 28)
    if filled:
        frame[HUD_Y + 2:HUD_Y + 7, 1:1 + filled] = hcol

    # ammo bar
    afilled = int(round(28 * min(1.0, game.ammo / float(MAX_AMMO))))
    frame[HUD_Y + 8:HUD_Y + 12, 1:29] = (22, 22, 28)
    if afilled:
        frame[HUD_Y + 8:HUD_Y + 12, 1:1 + afilled] = AMMO_COLOR

    # face, tinted bloodier as health drops
    if game.game_over:
        face = FACE_DEAD
    elif frac > 0.55:
        face = FACE_OK
    else:
        face = FACE_HURT
    skin = (int(225 - 25 * (1 - frac)), int(180 - 120 * (1 - frac)),
            int(150 - 100 * (1 - frac)))
    if game.hurt_flash:
        skin = (255, 120, 100)
    y0, x0 = HUD_Y + 13, 11
    for r, row in enumerate(face):
        for c, ch in enumerate(row):
            if ch == "#":
                frame[y0 + r, x0 + c] = skin
            elif ch == "o":
                frame[y0 + r, x0 + c] = (25, 15, 15)
            elif ch == "-":
                frame[y0 + r, x0 + c] = (110, 25, 20)


def render(game, bri=1.0):
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    draw_view(frame, game)
    draw_flash(frame, game)
    draw_map(frame, game)
    draw_hud(frame, game)

    if game.hurt_flash:
        t = game.hurt_flash / 5.0
        wash = np.zeros_like(frame)
        wash[:VIEW_H] = (int(90 * t), 0, 0)
        frame = np.maximum(frame, wash)
    if game.paused:
        frame = frame // 3
    if game.game_over:
        wash = np.zeros_like(frame)
        wash[:, :] = (45, 0, 0)
        frame = np.maximum(frame // 2, wash)

    if bri < 1.0:
        frame = (frame.astype(np.float32) * bri).astype(np.uint8)
    return frame


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Doom-like raycaster on the 30x120 Niji wall.")
    ap.add_argument("--dev", help="evdev path (auto if omitted)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bri", type=float, default=1.0)
    args = ap.parse_args()

    path = args.dev or find_gamepad()
    if not path:
        raise SystemExit("No gamepad found. Turn on the controller / plug in the dongle.")
    pad = Gamepad(path)
    pad.start()
    print(f"Controller: {pad.name} ({path})")

    ddp = DDP()
    game = Doom()
    frame_dt = 1.0 / args.fps

    try:
        while True:
            t0 = time.time()

            for act in pad.drain_actions():
                if act == ACT_PAUSE:
                    if game.game_over:
                        game.reset()
                    else:
                        game.paused = not game.paused

            strafe, forward, turn, firing = pad.state()
            if not game.dead_or_paused():
                game.move_player(strafe, forward, turn, frame_dt)
                if firing:
                    game.fire()
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
