# LED Wall

A 30 x 120 pixel LED wall (3600 addressable LEDs, driven by [WLED](https://kno.wled.ge/))
and the Python that makes it do interesting things: custom animations with a
phone-friendly control page, a live camera "mirror" that paints you onto the
wall, and a handful of arcade games you play on it with a gamepad.

Everything renders to the same target: a `(120, 30, 3)` uint8 NumPy frame, sent
to WLED over **DDP** (UDP port 4048). WLED is configured as a 2D matrix, so the
code only ever thinks in logical rows and columns — the physical serpentine
wiring is somebody else's problem.

```
  camera / effect / game  ->  (H, W, 3) frame  ->  DDP over UDP  ->  WLED  ->  wall
```

| | |
|---|---|
| Panel | 30 wide x 120 tall, portrait |
| Controller | WLED, 2D matrix mode, DDP enabled |
| Transport | DDP (UDP 4048), chunked at 480 LEDs/packet to stay under the MTU |
| Host | Raspberry Pi 4 (camera work) or any machine on the same LAN |
| Deps | `numpy`, plus `opencv-python` + `mediapipe` for the camera, `evdev` for the gamepad |

Set the panel's IP once at the top of `wall.py` / `niji.py` (`HOST`), or pass
`--host` on the command line.

---

## 1. WLED effects + web control page

Custom animations written for a tall, narrow panel — fire that actually rises
the full 120 rows, rain that has somewhere to fall, a helix that has room to
twist.

### `niji.py` — the effects engine

18 effects, each a generator yielding frames:

`fire` · `rain` (Matrix-style) · `plasma` · `aurora` · `meteors` · `wipe` ·
`fireworks` · `helix` · `lightning` · `warp` · `metaballs` · `vortex` ·
`ripple` · `life` (Conway) · `text` (scrolling message) · `clock` ·
plus `calibrate` and `diag` for checking orientation and wiring.

```bash
python3 niji.py list                            # show every effect
python3 niji.py fire
python3 niji.py rain --fps 30 --bri 0.6
python3 niji.py text --text "HAPPY BIRTHDAY" --scale 2
python3 niji.py all --each 20                   # cycle through everything
python3 niji.py clock --duration 3600           # run for an hour, then stop
```

Common flags: `--host` (panel IP), `--fps`, `--bri` (0..1), `--duration`
(0 = forever), `--hflip` / `--no-vflip` if your panel is mounted differently.
Stop with Ctrl-C; WLED falls back to its own effect a couple of seconds later.

### `niji_web.py` — control it from your phone

A tiny standard-library HTTP server (no Flask, no npm) that imports `niji.py`
and runs the DDP stream in a background thread. Tap an effect on the page and
it switches instantly — no process restart.

```bash
python3 niji_web.py        # serves on http://<this-machine>:8080
```

Open that URL on a phone on the same Wi-Fi. The page is a grid of big
emoji-labelled buttons (🔥 Fire, 🎆 Fireworks, 🌌 Aurora, 🕐 Clock, 🔤 Text …)
with brightness and speed controls; `GET /api/state` and `POST /api/set` are
there if you'd rather drive it from a script or a home-automation box.
