#!/usr/bin/env python3
"""
calib.py - interactive per-lane calibration for the Niji wall, using the gamepad.

Shows horizontal reference lines across all 30 lanes. Pick a lane and nudge its
offset until its dots line up with the neighbouring lanes; repeat across the
wall; save. The result is written to wall_calib.json and picked up by wall.py,
so tetris / niji / mirror all inherit the correction.

    python3 calib.py                 # auto-detect gamepad
    python3 calib.py --dev /dev/input/event4

Controls:
    left / right   select previous / next lane   (D-pad or left stick)
    up / down      nudge selected lane -1 / +1 LED
    A              save to wall_calib.json
    B              reset selected lane to 0
    Y (top)        reset ALL lanes to 0
    Start          quit
The selected lane is drawn in bright yellow; a small marker on the top row
shows which lane is selected. Console prints the full offset table on changes.
"""
import argparse
import threading
import time

import numpy as np
import evdev
from evdev import ecodes as e

import wall
from wall import DDP, W, H

REF_ROWS = list(range(6, H, 10))   # horizontal reference lines


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


class Pad(threading.Thread):
    """Edge-triggered gamepad reader: pushes discrete action strings."""

    def __init__(self, path):
        super().__init__(daemon=True)
        self.dev = evdev.InputDevice(path)
        self.name = self.dev.name
        self.lock = threading.Lock()
        self.q = []
        self._sx = self._axis_max(e.ABS_X)
        self._sy = self._axis_max(e.ABS_Y)
        self._lx = 0     # last discrete x dir
        self._ly = 0     # last discrete y dir
        self.running = True

    def _axis_max(self, code):
        try:
            info = self.dev.absinfo(code)
            return max(abs(info.min), abs(info.max)) or 1
        except OSError:
            return 32767

    def _push(self, a):
        with self.lock:
            self.q.append(a)

    def _edge_x(self, d):
        if d != self._lx:
            self._lx = d
            if d < 0:
                self._push("left")
            elif d > 0:
                self._push("right")

    def _edge_y(self, d):
        if d != self._ly:
            self._ly = d
            if d < 0:
                self._push("up")
            elif d > 0:
                self._push("down")

    def run(self):
        tx, ty = self._sx * 0.5, self._sy * 0.5
        for ev in self.dev.read_loop():
            if not self.running:
                break
            if ev.type == e.EV_KEY and ev.value == 1:
                if ev.code == e.BTN_SOUTH:
                    self._push("save")
                elif ev.code == e.BTN_EAST:
                    self._push("reset")
                elif ev.code in (e.BTN_NORTH, e.BTN_WEST):
                    self._push("reset_all")
                elif ev.code == e.BTN_START:
                    self._push("quit")
            elif ev.type == e.EV_ABS:
                if ev.code == e.ABS_X:
                    self._edge_x(-1 if ev.value < -tx else (1 if ev.value > tx else 0))
                elif ev.code == e.ABS_HAT0X:
                    self._edge_x(int(ev.value))
                elif ev.code == e.ABS_Y:
                    self._edge_y(-1 if ev.value < -ty else (1 if ev.value > ty else 0))
                elif ev.code == e.ABS_HAT0Y:
                    self._edge_y(int(ev.value))

    def drain(self):
        with self.lock:
            q, self.q = self.q, []
            return q


def render(offsets, sel):
    f = np.zeros((H, W, 3), dtype=np.uint8)
    for y in REF_ROWS:
        f[y, :] = (70, 70, 90)
    # selected lane's reference dots in bright yellow, plus a faint full column
    f[:, sel] = np.maximum(f[:, sel], (12, 12, 0))
    for y in REF_ROWS:
        f[y, sel] = (255, 220, 0)
    # top-row marker at the selected lane
    f[0, sel] = (0, 200, 255)
    # apply the live offsets so you see the corrected result
    return wall.apply_correction(f, offsets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", help="evdev path (auto if omitted)")
    args = ap.parse_args()

    path = args.dev or find_gamepad()
    if not path:
        raise SystemExit("No gamepad found. Turn on the controller / plug in the dongle.")
    pad = Pad(path)
    pad.start()
    print(f"Controller: {pad.name} ({path})")

    ddp = DDP(correct=False)          # we apply offsets ourselves in render()
    offsets = list(wall.LANE_OFFSET)
    sel = 0

    def show():
        print(f"lane {sel:2d}  offset {offsets[sel]:+d}   all={offsets}")

    show()
    try:
        while True:
            for a in pad.drain():
                if a == "left":
                    sel = (sel - 1) % W; show()
                elif a == "right":
                    sel = (sel + 1) % W; show()
                elif a == "up":
                    offsets[sel] -= 1; show()
                elif a == "down":
                    offsets[sel] += 1; show()
                elif a == "reset":
                    offsets[sel] = 0; show()
                elif a == "reset_all":
                    offsets = [0] * W; show()
                elif a == "save":
                    wall.save_offsets(offsets)
                    print(f"saved -> {wall.CALIB_FILE}")
                elif a == "quit":
                    raise KeyboardInterrupt
            ddp.send(render(offsets, sel))
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        pad.running = False


if __name__ == "__main__":
    main()
