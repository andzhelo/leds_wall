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
with brightness and speed sliders and a field for the scrolling text;
`GET /api/state` and `POST /api/set` are there if you'd rather drive it from a
script or a home-automation box.

---

## 2. The silhouette mirror (camera)

Point a webcam at the wall's viewing spot and you appear on it, live. A person
segmentation model cuts you out of the video; everything outside the mask goes
black, so the wall shows a moving human shape and nothing else. Walk past it and
it walks past with you.

### `mirror_rgb.py` — the one to run

Your **real video colours** inside a MediaPipe SelfieSegmentation mask: a red
shirt is red on the wall, skin reads as skin, a hand raised over your chest is
visible as a separate shape.

```bash
python3 mirror_rgb.py
```

Requires `opencv-python` and `mediapipe`, and a camera on `/dev/video0`. It
grabs MJPG frames with `CAP_PROP_BUFFERSIZE=1` — on a Pi the driver otherwise
queues frames and you end up mirroring where you were a second ago.

### Tuning: `calibration.json`

The mirror re-reads `calibration.json` from its own directory **every frame**,
so you can sit at the wall, edit the file, and watch the change land — no
restart. It's created with defaults on first run.

| key | what it does |
|---|---|
| `rotate`, `flip_code` | orient the camera to match the wall |
| `zoom`, `pan_x`, `pan_y` | frame the person in the tall 30x120 aspect |
| `offset_x`, `offset_y` | nudge the output on the panel |
| `brightness`, `color_order` | global level, RGB/GRB channel order |
| `mask_threshold`, `smoothing`, `denoise`, `erode` | how tight and how stable the cutout is |
| `rgb_mask` | `1` = person on black (the mirror look), `0` = raw full video |
| `rgb_sat`, `rgb_gamma`, `rgb_gain` | make clothes pop on LEDs; gamma < 1 lifts dark indoor mids |
| `rgb_auto`, `rgb_auto_target` | auto-exposure so it survives changing room light |
| `rgb_idle`, `rgb_idle_bright`, `rgb_idle_delay` | after N seconds with nobody there, fade to a dim idle glow |

### Running it as a service

`mirror.service` is a systemd unit for a Pi: starts after the network is up,
restarts on failure, and stops with SIGTERM (the script traps it to release the
camera cleanly, otherwise the next start finds `/dev/video0` busy).

```bash
sudo cp mirror.service /etc/systemd/system/
sudo systemctl enable --now mirror
```

Adjust `User=`, `WorkingDirectory=` and the venv path inside it first.

### The other `mirror_*.py` files

Earlier takes, kept because they're each interesting on their own:

- `mirror_led.py`, `mirror_bgsub.py` — MOG2 background subtraction, one flat
  silhouette colour. No ML dependency and very fast, but MOG2 absorbs anything
  that holds still (stand there long enough and your legs disappear) and gives
  up on dark clothes against a dark background. That's what pushed the project
  to MediaPipe.
- `mirror_udp.py`, `mirror_udp_fast.py`, `mirror_udp_orig.py` — the MediaPipe
  mask as a solid colour fill, at various stages of latency surgery.
- `mirror_parts.py`, `mirror_multiclass.py` — multi-class segmentation, colouring
  hair / skin / clothes differently.
- `mirror_claude.py`, `test.py`, `t.py` — scratch copies from the way there.
