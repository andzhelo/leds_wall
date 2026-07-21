#!/usr/bin/env python3
"""
ledtest.py - static diagnostic pattern for the 30x120 Niji wall.

Draws straight horizontal guide lines (full-width rows). Columns are wired as
vertical lanes, so any per-lane offset shows up as a comb. Use this to dial in
the half-1 correction in wall.py:

    python3 ledtest.py --raw     # correction OFF: you should see the comb
    python3 ledtest.py           # correction ON:  lines should be straight

If the comb is still there with correction ON, edit wall.py: flip SHIFT_PARITY
(0<->1); if it looks doubled/worse, flip the sign of SHIFT_AMOUNT. Re-run until
both halves are straight. Ctrl-C to stop.
"""
import argparse
import time
import numpy as np

import wall
from wall import DDP, W, H


def build():
    f = np.zeros((H, W, 3), dtype=np.uint8)
    # straight HORIZONTAL guide lines (constant y, full width). columns are
    # wired as vertical lanes, so a per-lane vertical offset combs these.
    for y in (10, 30, 50, 70, 90, 110):
        f[y, :] = (90, 90, 90)
    # top row coloured by COLUMN parity: even columns green, odd red.
    for x in range(W):
        f[0, x] = (0, 180, 0) if x % 2 == 0 else (180, 0, 0)
    # bright corner markers so orientation is unambiguous in the photo
    f[0, 0] = (255, 255, 255)          # top-left white
    f[H - 1, 0] = (0, 120, 255)        # bottom-left blue
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true", help="disable the half-1 correction")
    ap.add_argument("--parity", type=int, choices=(0, 1), help="override SHIFT_PARITY")
    ap.add_argument("--amount", type=int, help="override SHIFT_AMOUNT (e.g. -2, 2, -1)")
    args = ap.parse_args()

    # live overrides so you can sweep without editing wall.py
    if args.parity is not None:
        wall.SHIFT_PARITY = args.parity
    if args.amount is not None:
        wall.SHIFT_AMOUNT = args.amount

    ddp = DDP(correct=not args.raw)
    frame = build()
    print(f"correction {'OFF (raw)' if args.raw else 'ON'}  "
          f"parity={wall.SHIFT_PARITY} amount={wall.SHIFT_AMOUNT}. "
          "White=top-left, blue=bottom-left, green=even cols, red=odd cols. Ctrl-C to stop.")
    try:
        while True:
            ddp.send(frame)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nstopped.")
