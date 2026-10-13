#!/usr/bin/env python3
"""
piano.py - an air piano: your silhouette on the Niji wall, keys you reach into.

You appear on the wall as a dim silhouette (the mirror pipeline), with a row of
bright keys floating just above your head. Raise a hand until it covers a key
and the key lights up and strikes a note. You can see your own hand land on the
key, so it feels like pressing something rather than waving at a sensor.

The trick that makes it feel connected: detection happens in *wall space*. The
mask is scaled to the 30x120 panel first, and a key fires when your silhouette
covers that key's cell -- the exact pixels you can see. What you see is what
triggers; there is no invisible sensor zone to aim at.

Notes are struck, not held: a sharp attack that rings out and decays, like a
marimba. That gives a clear moment of contact. Sweeping a hand sideways across
the keys plays a glissando.

Keys are a major scale (C D E F G A B), seven per lane, so the songs people
actually know are playable -- Twinkle, Ode to Joy, Happy Birthday,
Korobeiniki. --scale pentatonic (C D E G A) drops the F and the B, which makes every
combination consonant and free play foolproof, at the cost of most of the
songs. The root of each octave is marked brighter.

The keys live in two lanes running down the left and right edges: left is the
lower octave, right the upper one, and on both, pitch rises as you go up the
wall. They sit at head-to-hip height, where your hands are when your arms are
out -- which is the point, because that is what the camera can actually see.

There used to be a third lane along the top, floating above your head. It was
removed: raising your hands that high takes them outside the camera's crop
window, so the keys were unreachable no matter how they were drawn. --top-keys
10 brings it back if you reframe the camera to cover that space.

Pitch runs as one continuous climb through whichever lanes exist, so removing
a lane closes the range up instead of leaving a hole in the middle of it.
Reaching the edge columns needs the crop window wide enough for your arm span;
that is what --wide is for.

The camera view is cropped to the panel's own 1:4 shape before it is mapped,
the same way mirror_led.py does it, so you keep your proportions and read as a
person with arms. Frame yourself with --zoom / --pan-x / --pan-y.

    python3 piano.py
    python3 piano.py --pan-y -0.4        # slide the window up if you are cut off
    python3 piano.py --zoom 1.4          # crop in tighter
    python3 piano.py --sens 0.7          # easier to trigger
    python3 piano.py --key-row auto      # float the keys above your head instead
    python3 piano.py --song korobeiniki       # follow-the-light practice
    python3 piano.py --list-songs
    python3 piano.py --scale pentatonic  # no wrong notes, fewer songs
    python3 piano.py --calib             # terminal meters, no LEDs

Sends through wall.py, so the per-lane calibration applies here too.

Requires: numpy, opencv-python, mediapipe  (already used by the mirror)
          plus sounddevice (preferred) or pygame for sound.
              sudo apt install libportaudio2      # what sounddevice needs
"""
import argparse
import colorsys
import glob
import json
import math
import os
import re
import threading
import time
from collections import deque

import numpy as np
import cv2
import mediapipe as mp

from wall import DDP, W, H

# ---- keyboard geometry, in wall pixels --------------------------------------
# The top row is OFF by default. It floated above your head, which is exactly
# where the camera's crop window runs out -- hands raised that high leave the
# frame and stop being detected at all. The side lanes sit at head-to-hip
# height, inside the frame, so they are what the instrument is built on now.
# Bring the row back with --top-keys 10 if you reframe the camera for it.
TOP_ROW_KEYS = 10             # keys in the top row when it is switched on
TOP_KEYS = 0                  # how many are actually in use; 0 = no top row
KEY_W = W // TOP_ROW_KEYS     # 3px per top key, tiles the 30px width exactly
KEY_ROWS = 9                  # height of the top key row
HEAD_GAP = 5                  # rows of clearance between your head and the keys
KEY_ROW_DEFAULT = 12          # keys sit here unless --key-row auto is asked for
ASPECT = W / H                # 1:4, the panel's shape -- see KeyCam.to_wall

# Side lanes, running down the two long edges.
#
# The panel is four times taller than it is wide, so the long edges are the
# largest usable area on it: a top row can only ever hold 10 keys at 3px, while
# a side lane has 120 rows to spend. Reaching out sideways is also an easier,
# more sustainable gesture than holding your arms overhead -- and, decisively,
# it stays inside what the camera can actually see.
SIDE_KEYS = 7                 # per side; one octave of the scale each. Defaults
                              # to len(SCALE) so a lane is always exactly one
                              # octave -- 7 on the major scale, 5 on pentatonic.
SIDE_W = 4                    # lane width in columns
SIDE_KEY_H = 11               # rows per side key
SIDE_GAP = 3                  # rows below the top row's anchor before a lane
                              # starts -- kept even with no top row, so the
                              # lanes stay exactly where they already work

LANE_TOP, LANE_LEFT, LANE_RIGHT = 0, 1, 2
LANE_NAME = {LANE_TOP: "top", LANE_LEFT: "left", LANE_RIGHT: "right"}

# ---- look ------------------------------------------------------------------
SIL_COLOR = np.array((0.30, 0.45, 0.85), dtype=np.float32)   # silhouette tint
SIL_LEVEL = 0.15              # how bright the silhouette is (keys must win)
KEY_IDLE = 0.32               # brightness of an untouched key
CONTACT_COLOR = np.array((1.0, 1.0, 1.0), dtype=np.float32)  # you, inside a key
FLASH_TAU = 0.32              # seconds for the strike flash to fade
FLASH_REACH = 14              # how far the strike streak shoots up, in rows
# Side streaks fire inward across the 30px width, so two of them meet in the
# middle. At FLASH_REACH they overlapped, summed and blew out to white over the
# silhouette; 9 leaves a gap down the centre even when both lanes are struck.
SIDE_FLASH_REACH = 9
SIDE_FLASH_LEVEL = 0.6        # dimmer than a top streak, which has open panel
                              # above it rather than your own body behind it

# ---- song mode -------------------------------------------------------------
# Non-target keys are pushed down, but not out: at 0.06 they fell below the
# silhouette's own brightness, so your body outshone the keyboard and you lost
# the reference you aim against. Keep them at least as bright as SIL_LEVEL.
SONG_OTHER = 0.14
SONG_NEXT = 0.30              # the note after the target, so you can prepare
SONG_TARGET_MIN = 0.45        # the target pulses between this and full
SONG_PULSE_HZ = 2.2
SONG_BAR_ROWS = 3             # progress bar along the bottom edge
SONG_DONE_HOLD = 4.0          # seconds of celebration before it loops

PROFILE_BINS = 12             # height slices reported by --calib

# The mirror already has the camera geometry tuned for this room, in
# calibration.json. Reuse it so the piano is framed the same way instead of
# making you calibrate a second time. Anything passed on the command line wins.
CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "calibration.json")
_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def camera_candidates(skip):
    """Plausible camera indices actually present, lowest first.

    /dev/video10 and up on a Pi are the bcm2835 codec and ISP nodes -- they are
    v4l2 devices but not cameras, so probing them wastes time and can hand back
    a device that opens and never yields a frame.
    """
    found = []
    for path in glob.glob("/dev/video*"):
        m = re.fullmatch(r"/dev/video(\d+)", path)
        if m and int(m.group(1)) != skip and int(m.group(1)) < 10:
            found.append(int(m.group(1)))
    return sorted(found)


def open_camera(index):
    """Open a camera, falling back to whichever one is really there.

    USB cameras do not keep a stable index. Unplug one, reboot, or plug in
    another powered device and the same camera comes back as /dev/video1 while
    /dev/video0 simply does not exist -- which used to abort with "Is
    /dev/video0 there?" and leave you guessing. Probing costs a moment at
    startup and turns that into a one-line notice.

    A device is only accepted if it actually returns a frame; opening succeeds
    on plenty of nodes that then deliver nothing.
    """
    tried = []
    for idx in [index] + camera_candidates(index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # newest frame, no backlog
            except Exception:
                pass
            if cap.read()[0]:
                if idx != index:
                    print(f"camera {index} is not there; using camera {idx} "
                          f"instead (pass --camera {idx} to skip the search)")
                return cap, idx
        cap.release()
        tried.append(idx)
    raise SystemExit(
        f"No working camera found (tried {', '.join(map(str, tried))}). "
        f"Present: {' '.join(sorted(glob.glob('/dev/video*'))) or 'none'}. "
        f"Check it is plugged in with `lsusb`, and that you are in the "
        f"'video' group.")


def load_mirror_calib():
    """rotate / flip_code / zoom / pan from the mirror's calibration.json."""
    try:
        with open(CALIB_PATH) as f:
            c = json.load(f)
    except (OSError, ValueError):
        return {}
    return {k: c[k] for k in ("rotate", "flip_code", "zoom", "pan_x", "pan_y")
            if k in c}

# ---- scale -----------------------------------------------------------------
# Two scales, and the choice is a real trade-off rather than a preference.
#
# Pentatonic (C D E G A) has no wrong notes: every combination is consonant, so
# waving both hands around sounds musical. That is what it was built on. But it
# has no F and no B, and that rules out most of the tunes anyone actually wants
# to play -- Twinkle, Ode to Joy, Happy Birthday, Jingle Bells, Korobeiniki.
#
# Major (C D E F G A B) is the default now because the songs are the point. Free
# play is slightly riskier -- a B against a C is a real semitone clash -- but
# seven degrees is also one lane per octave, which is a tidier keyboard. Use
# --scale pentatonic to go back to jamming.
PENTATONIC = (0, 2, 4, 7, 9)          # semitone offsets of C D E G A
MAJOR = (0, 2, 4, 5, 7, 9, 11)        # ... of C D E F G A B
SCALES = {"pentatonic": PENTATONIC, "major": MAJOR}
SCALE_NAME = "major"
SCALE = MAJOR
C4 = 261.625565

SCALE_LETTERS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def set_scale(name):
    """Switch the scale. Call before configure_keys -- pitches derive from it."""
    global SCALE, SCALE_NAME
    if name not in SCALES:
        raise SystemExit(f"unknown --scale {name!r}; "
                         f"pick one of {', '.join(SCALES)}")
    SCALE_NAME, SCALE = name, SCALES[name]


def scale_letters():
    """'C D E F G A B' -- the degrees the current scale actually has."""
    return " ".join(SCALE_LETTERS[s] for s in SCALE)


def key_freq(i, octave=0):
    """Hz for key i, counting up the current scale from C4."""
    semis = SCALE[i % len(SCALE)] + 12 * (i // len(SCALE) + octave)
    return C4 * (2.0 ** (semis / 12.0))


def build_specs(top_keys, side_keys):
    """(lane, slot, scale_index, octave) for every key, in ascending pitch order.

    Pitch runs as one continuous climb up the scale through the instrument in
    playing order -- up the left lane, left-to-right along the top row if there
    is one, then up the right lane. Assigning it as a single run rather than
    fixing an octave per lane is what lets a lane be removed without leaving a
    hole: drop the top row and the two side lanes simply close up into the
    C4-A5 range the row used to cover, so the songs still fit.

    The base is anchored so the melody range starts at C4 either way: with a
    top row the left lane sits below it as bass, without one the left lane
    takes over that role itself.
    """
    base = -side_keys if top_keys else 0
    specs, n = [], base
    for slot in range(side_keys - 1, -1, -1):      # left lane, bottom key first
        specs.append((LANE_LEFT, slot, n, 0))
        n += 1
    for i in range(top_keys):
        specs.append((LANE_TOP, i, n, 0))
        n += 1
    for slot in range(side_keys - 1, -1, -1):      # right lane, bottom first
        specs.append((LANE_RIGHT, slot, n, 0))
        n += 1
    return specs


def build_freqs(specs, octave=0):
    return [key_freq(idx, oct_ + octave) for _, _, idx, oct_ in specs]


def build_colors(specs, freqs):
    """Each lane sweeps red -> violet in pitch order, low to high.

    One continuous gradient across all three lanes was the obvious first idea
    and it looked worse: the top row only spans the middle of a near-4-octave
    range, so it collapsed from a full rainbow into a narrow green-blue band.
    Per-lane keeps the top row exactly as it was and gives each side lane its
    own readable spread; which register you are in is told by which lane your
    hand is on, which is unambiguous when they sit on opposite edges.
    """
    out = [None] * len(specs)
    for lane in (LANE_TOP, LANE_LEFT, LANE_RIGHT):
        idx = sorted((i for i, s in enumerate(specs) if s[0] == lane),
                     key=lambda i: freqs[i])
        for rank, i in enumerate(idx):
            h = 0.86 * (rank / max(1, len(idx) - 1))
            r, g, b = colorsys.hsv_to_rgb(h, 0.95, 1.0)
            out[i] = np.array((r, g, b), dtype=np.float32)
    return out


def max_key_row():
    """Lowest row the top key row may sit at.

    The side lanes hang below the top row, so the whole instrument has to fit:
    pushing the top row further down would otherwise squeeze the lanes off the
    bottom of the panel, and clamping them back on would slide them up into the
    top row's own pixels. Bounding the top row instead keeps the lanes honest.
    """
    tail = SIDE_GAP + SIDE_KEYS * SIDE_KEY_H if SIDE_KEYS else 1
    return max(0, H - KEY_ROWS - tail)


def key_cells(key_y0):
    """Pixel rectangle (y slice, x slice) for every key.

    Side lanes hang off the top row rather than sitting at fixed rows, so the
    whole instrument moves as one when head-tracking repositions the keys --
    and the lanes land at roughly shoulder-to-hip height, which is where your
    hands are when your arms are out to the sides.
    """
    key_y0 = int(np.clip(key_y0, 0, max_key_row()))
    side_y0 = key_y0 + KEY_ROWS + SIDE_GAP
    cells = []
    for lane, slot, _, _ in KEY_SPECS:
        if lane == LANE_TOP:
            x0 = slot * KEY_W
            cells.append((slice(key_y0, key_y0 + KEY_ROWS), slice(x0, x0 + KEY_W)))
        else:
            y0 = side_y0 + slot * SIDE_KEY_H
            x0 = 0 if lane == LANE_LEFT else W - SIDE_W
            cells.append((slice(y0, y0 + SIDE_KEY_H), slice(x0, x0 + SIDE_W)))
    return cells


def configure_keys(side_keys, top_keys=TOP_KEYS):
    """Rebuild the key table. Called once from main before anything uses it."""
    global SIDE_KEYS, TOP_KEYS, KEY_W, KEYS, KEY_SPECS, KEY_FREQS, KEY_COLORS
    SIDE_KEYS = side_keys
    TOP_KEYS = top_keys
    KEY_W = W // top_keys if top_keys else W // TOP_ROW_KEYS
    KEY_SPECS = build_specs(top_keys, side_keys)
    KEYS = len(KEY_SPECS)
    KEY_FREQS = build_freqs(KEY_SPECS)
    KEY_COLORS = build_colors(KEY_SPECS, KEY_FREQS)


KEY_SPECS = build_specs(TOP_KEYS, SIDE_KEYS)
KEYS = len(KEY_SPECS)
KEY_FREQS = build_freqs(KEY_SPECS)
KEY_COLORS = build_colors(KEY_SPECS, KEY_FREQS)


# ----------------------------------------------------------------------------
# Songs: follow-the-light practice.
#
# Which of these you can play depends entirely on --scale. On pentatonic
# (C D E G A) there is no F and no B anywhere on the keyboard, so anything
# needing one cannot be played at any octave -- that used to rule out almost
# every famous tune, and is why the list was so short. On the major scale, which
# is now the default, all of them work.
#
# Everything below is written in C major / A minor and uses natural notes only,
# so no song needs a black key that the wall does not have. They are melody
# lines, not arrangements: pitch only, no rhythm, which is exactly what the
# follow-the-light trainer wants (it waits for you rather than running a clock).
# Add your own with --song-file.
# ----------------------------------------------------------------------------
NOTE_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

SONGS = {
    # --- pentatonic: these play on either scale ------------------------------
    "mary": ("Mary Had a Little Lamb",
             "E4 D4 C4 D4 E4 E4 E4  D4 D4 D4  E4 G4 G4 "
             "E4 D4 C4 D4 E4 E4 E4 E4  D4 D4 E4 D4 C4"),
    "macdonald": ("Old MacDonald Had a Farm",
                  "C4 C4 C4 G3 A3 A3 G3  E4 E4 D4 D4 C4 "
                  "C4 C4 C4 G3 A3 A3 G3  E4 E4 D4 D4 C4"),
    "amazing": ("Amazing Grace",
                "G3 C4 E4 C4 E4 D4 C4 A3 G3 "
                "G3 C4 E4 C4 E4 D4 G4 G4"),
    "mygirl": ("My Girl (The Temptations)",
               "C4 D4 E4 G4  C4 D4 E4 G4  E4 G4 A4 G4 E4 D4 C4"),

    # --- need the F and the B, so --scale major only -------------------------
    "twinkle": ("Twinkle Twinkle / ABC / Baa Baa Black Sheep",
                "C4 C4 G4 G4 A4 A4 G4  F4 F4 E4 E4 D4 D4 C4 "
                "G4 G4 F4 F4 E4 E4 D4  G4 G4 F4 F4 E4 E4 D4 "
                "C4 C4 G4 G4 A4 A4 G4  F4 F4 E4 E4 D4 D4 C4"),
    "ode": ("Ode to Joy (Beethoven)",
            "E4 E4 F4 G4 G4 F4 E4 D4 C4 C4 D4 E4 E4 D4 D4 "
            "E4 E4 F4 G4 G4 F4 E4 D4 C4 C4 D4 E4 D4 C4 C4"),
    "birthday": ("Happy Birthday",
                 "G3 G3 A3 G3 C4 B3  G3 G3 A3 G3 D4 C4 "
                 "G3 G3 G4 E4 C4 B3 A3  F4 F4 E4 C4 D4 C4"),
    "jingle": ("Jingle Bells",
               "E4 E4 E4  E4 E4 E4  E4 G4 C4 D4 E4 "
               "F4 F4 F4 F4 F4 E4 E4 E4 E4 D4 D4 E4 D4 G4"),
    "korobeiniki": ("Korobeiniki",
               "E5 B4 C5 D5 C5 B4 A4 A4 C5 E5 D5 C5 B4 C5 D5 E5 C5 A4 A4 "
               "D5 F5 A5 G5 F5 E5 C5 E5 D5 C5 B4 B4 C5 D5 E5 C5 A4 A4"),
    # Written in G rather than C: the pickup is the dominant *below* the tonic,
    # so in C it would need a G3 and the keyboard starts at C4. Shifting the
    # whole tune up a fifth puts the pickup on D4 and the whole span inside the
    # two octaves. It uses no F at all, so G costs no sharp.
    "starwars": ("Star Wars (Main Title)",
                 "D4 D4 D4  G4 D5 C5 B4 A4 G5 D5 "
                 "C5 B4 A4 G5 D5  C5 B4 C5 A4"),
    "thrones": ("Game of Thrones",
                "E4 A4 C5 D5 E4 A4 C5 D5 E4 A4 C5 D5 E4 A4 C5 B4 "
                "B4 E5 G5 A5 B4 E5 G5 A5 B4 E5 G5 A5 B4 E5 G5 E5"),
    "pirates": ("He's a Pirate (Pirates of the Caribbean)",
                "A4 A4 A4 A4 B4 C5 C5 C5 C5 D5 B4 B4 A4 A4 G4 A4 "
                "A4 A4 A4 A4 B4 C5 C5 C5 C5 D5 B4 B4 A4 "
                "C5 C5 C5 D5 E5 E5 E5 F5 D5 D5 C5 B4 A4"),

    "scale": ("Warm-up: up and down the keyboard", None),   # filled in below
}


def note_freq(name):
    """'C4' / 'A3' / 'F#5' -> Hz, with C4 = middle C."""
    m = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d+)", name.strip())
    if not m:
        raise SystemExit(f"cannot read note {name!r} (want e.g. C4, A3, F#5)")
    letter, acc, octave = m.group(1).upper(), m.group(2), int(m.group(3))
    semis = NOTE_SEMITONE[letter] + (1 if acc == "#" else -1 if acc == "b" else 0)
    return C4 * 2.0 ** ((semis + 12 * (octave - 4)) / 12.0)


def note_name(freq):
    """Nearest note name for a frequency, for listings and error messages."""
    semis = round(12 * math.log2(freq / C4))
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[semis % 12]}{4 + (semis // 12)}"


def resolve_song(notes, transpose=True):
    """Note names -> (key indices, octaves shifted) on the current keyboard.

    Two different failures hide behind "that note isn't here", and they deserve
    different answers. A note outside the scale can never be played at any
    octave -- and on the pentatonic scale the fix is a different scale, not a
    different octave, so the message says so. A note merely outside the
    keyboard's range is fixed by shifting the whole melody, so that is done
    automatically -- which
    octaves exist moves with the lane configuration, and a tune written around
    C4 should still play after the top row is removed rather than just error.

    Resolved against the unshifted keyboard, so --octave transposes what you
    hear without moving which key lights up.
    """
    if isinstance(notes, str):
        notes = notes.split()
    wanted = [note_freq(n) for n in notes]

    for name, f in zip(notes, wanted):
        if round(12 * math.log2(f / C4)) % 12 not in set(SCALE):
            hint = ("\nTry --scale major, which adds the F and the B."
                    if SCALE_NAME == "pentatonic"
                    else "\nThe keyboard has no black keys at any octave.")
            raise SystemExit(
                f"{name} is not in the scale, so that song cannot be played "
                f"here at any octave.\nThe keyboard is {SCALE_NAME} "
                f"({scale_letters()})." + hint)

    # smallest shift first, so a song that already fits is never moved
    order = [0] + [s for n in range(1, 5) for s in (n, -n)]
    for shift in (order if transpose else [0]):
        out = []
        for f in wanted:
            f2 = f * (2.0 ** shift)
            i = min(range(KEYS), key=lambda k: abs(math.log2(KEY_FREQS[k] / f2)))
            if 1200 * abs(math.log2(KEY_FREQS[i] / f2)) > 25:   # quarter-tone
                out = None
                break
            out.append(i)
        if out is not None:
            return out, shift

    raise SystemExit(
        f"that song spans {note_name(min(wanted))}-{note_name(max(wanted))}, "
        f"which does not fit this keyboard "
        f"({note_name(min(KEY_FREQS))}-{note_name(max(KEY_FREQS))}) even after "
        f"shifting it by whole octaves.\n--side-keys {SIDE_KEYS + 2} would "
        f"widen the range.")


def load_song(name, path=None):
    """(title, [key indices], octaves shifted) for a built-in or custom song."""
    if path:
        try:
            with open(path) as fh:
                custom = json.load(fh)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"could not read --song-file {path}: {exc}")
        if name not in custom:
            raise SystemExit(f"{path} has no song called {name!r}. "
                             f"It has: {', '.join(sorted(custom)) or '(nothing)'}")
        entry = custom[name]
        title = entry.get("title", name) if isinstance(entry, dict) else name
        notes = entry["notes"] if isinstance(entry, dict) else entry
        return (title,) + resolve_song(notes)

    if name not in SONGS:
        raise SystemExit(f"unknown song {name!r}. Try --list-songs.")
    title, notes = SONGS[name]
    if notes is None:                       # the generated warm-up
        return title, list(range(KEYS)) + list(range(KEYS - 2, 0, -1)), 0
    return (title,) + resolve_song(notes)


class SongTrainer:
    """Follow-the-light state machine.

    Free tempo on purpose: it waits for the right key rather than running a
    metronome. Your "fingers" are whole hands and the keys are 3px wide, so
    a clock would only ever be a source of failure.
    """

    def __init__(self, title, keys):
        self.title = title
        self.keys = keys
        self.pos = 0
        self.hits = 0
        self.wrong = 0
        self.done_at = None

    @property
    def target(self):
        return self.keys[self.pos] if self.pos < len(self.keys) else None

    @property
    def upcoming(self):
        return self.keys[self.pos + 1] if self.pos + 1 < len(self.keys) else None

    @property
    def progress(self):
        return self.pos / max(1, len(self.keys))

    def strike(self, i):
        """True if that was the note being waited for."""
        if self.target is None:
            return False
        if i == self.target:
            self.pos += 1
            self.hits += 1
            if self.target is None:
                self.done_at = time.time()
            return True
        self.wrong += 1
        return False

    def restart(self):
        self.pos = 0
        self.hits = 0
        self.wrong = 0
        self.done_at = None


# ----------------------------------------------------------------------------
# Audio. Percussive: every strike is a fresh attack that rings out and decays,
# so the sound lands at the moment your hand crosses into the key. Two backends
# -- a real-time synth on sounddevice, or one-shot samples on pygame.mixer.
# ----------------------------------------------------------------------------
def decay_time(freq):
    """Low notes ring longer than high ones, as struck bars really do.

    Kept short on purpose: you play this with whole hands, so notes land thick
    and fast. A long ring turns a glissando into mud, and a short one keeps
    each strike audible as a separate hit.
    """
    return float(np.clip(0.35 + 0.55 * (C4 / freq), 0.30, 0.95))


def _tone(phase, bright):
    """Fundamental plus harmonics that fade faster than it does.

    `bright` is the envelope itself, so as the note decays the upper partials
    die first and the tone mellows -- what makes a struck bar sound struck
    rather than like a held organ pipe.
    """
    return (np.sin(phase)
            + bright * (0.5 * np.sin(2 * phase) + 0.28 * np.sin(3 * phase))) / 1.6


class SoundDeviceSynth:
    """Polyphonic percussive synth, one voice per key."""

    ATTACK = 0.004               # seconds; short but not instant, or it clicks
    IDLE, ATTACKING, DECAYING = 0, 1, 2

    def __init__(self, freqs, volume=0.5, samplerate=44100, blocksize=256):
        import sounddevice as sd
        self.sd = sd
        self.sr = samplerate
        self.n = len(freqs)
        self.freqs = np.array(freqs, dtype=np.float64)
        self.decay = np.array([decay_time(f) for f in freqs], dtype=np.float64)
        self.phase = np.zeros(self.n, dtype=np.float64)
        self.env = np.zeros(self.n, dtype=np.float64)
        self.state = np.zeros(self.n, dtype=np.int8)
        # a full ten-note chord would clip; keep headroom that scales with it
        self.master = volume / math.sqrt(max(1, self.n) * 0.35)
        self.stream = sd.OutputStream(
            samplerate=samplerate, channels=1, blocksize=blocksize,
            dtype="float32", callback=self._callback)
        self.stream.start()
        self.name = f"sounddevice ({samplerate}Hz, {blocksize} frames)"

    def _callback(self, outdata, frames, time_info, status):
        mix = np.zeros(frames, dtype=np.float64)
        idx = np.arange(frames, dtype=np.float64)
        for i in range(self.n):
            if self.state[i] == self.IDLE and self.env[i] < 1e-4:
                continue
            e = self.env[i]
            if self.state[i] == self.ATTACKING:
                e_end = 1.0 + (e - 1.0) * math.exp(-frames / (self.sr * self.ATTACK))
                if e_end > 0.995:
                    self.state[i] = self.DECAYING
            else:
                e_end = e * math.exp(-frames / (self.sr * self.decay[i]))
                if e_end < 1e-3:
                    # -60dB and below is inaudible; ramping to a hard 0 here
                    # retires the voice instead of computing a silent tail
                    e_end, self.state[i] = 0.0, self.IDLE
            env = np.linspace(e, e_end, frames)          # click-free ramp
            inc = 2.0 * math.pi * self.freqs[i] / self.sr
            mix += _tone(self.phase[i] + inc * idx, env) * env
            self.phase[i] = (self.phase[i] + inc * frames) % (2.0 * math.pi)
            self.env[i] = e_end
            if self.state[i] == self.IDLE:
                self.phase[i] = 0.0
        outdata[:, 0] = np.tanh(mix * self.master).astype(np.float32)

    def strike(self, i):
        # re-attack from wherever the envelope is, so a fast retrigger on a
        # still-ringing note swells instead of clicking
        self.state[i] = self.ATTACKING

    def close(self):
        self.state[:] = self.IDLE
        self.env[:] = 0.0
        time.sleep(0.05)
        self.stream.stop()
        self.stream.close()


class PygameSynth:
    """Fallback: pre-rendered one-shot samples, one channel per key."""

    def __init__(self, freqs, volume=0.5, samplerate=44100):
        import pygame
        self.pygame = pygame
        pygame.mixer.pre_init(samplerate, -16, 1, 512)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(len(freqs) + 2)
        self.sounds, self.channels = [], []
        peak = 32767 * volume / math.sqrt(max(1, len(freqs)) * 0.35)
        for i, f in enumerate(freqs):
            tau = decay_time(f)
            n = int(samplerate * min(2.5, tau * 3))
            t = np.arange(n) / samplerate
            env = np.exp(-t / tau) * np.minimum(1.0, t / SoundDeviceSynth.ATTACK)
            buf = (_tone(2 * math.pi * f * t, env) * env * peak).astype(np.int16)
            self.sounds.append(pygame.sndarray.make_sound(np.ascontiguousarray(buf)))
            self.channels.append(pygame.mixer.Channel(i))
        self.name = f"pygame.mixer ({samplerate}Hz)"

    def strike(self, i):
        self.channels[i].play(self.sounds[i])

    def close(self):
        self.pygame.mixer.stop()
        self.pygame.mixer.quit()


class NullSynth:
    name = "none (silent)"

    def strike(self, i):
        pass

    def close(self):
        pass


def make_synth(freqs, volume, backend="auto", blocksize=256):
    order = {"auto": ("sounddevice", "pygame"), "sounddevice": ("sounddevice",),
             "pygame": ("pygame",), "none": ()}[backend]
    errors = []
    for name in order:
        try:
            if name == "sounddevice":
                return SoundDeviceSynth(freqs, volume, blocksize=blocksize)
            return PygameSynth(freqs, volume)
        except Exception as exc:                # missing lib, no ALSA device, ...
            errors.append(f"{name}: {exc}")
    if backend != "none":
        for line in errors:
            print(f"  audio backend unavailable -- {line}")
        print("  silent mode. For sound: sudo apt install libportaudio2")
    return NullSynth()


# ----------------------------------------------------------------------------
# Camera -> wall-space silhouette -> which keys are covered. Runs in its own
# thread so the render loop never blocks on a frame grab.
# ----------------------------------------------------------------------------
class KeyCam(threading.Thread):
    HEAD_MIN_COV = 0.04         # a wall row this covered counts as "you"
    HEAD_WINDOW = 64            # samples for the percentile (~6s)
    HEAD_PCTL = 92              # bias hard towards LOWER measurements (see below)
    HEAD_SMOOTH = 0.12          # EMA on the result, so the keys never jitter

    def __init__(self, seg_w=160, flip=True, zoom=1.0, key_row=KEY_ROW_DEFAULT,
                 fit="crop", pan_x=0.0, pan_y=0.0, rotate=0, flip_code=None,
                 wide=1.0,
                 on_thresh=0.16, off_thresh=0.10, adapt=True, adapt_tau=20.0,
                 max_hold=10.0, stuck_tau=3.0, camera=0, threshold=0.4):
        super().__init__(daemon=True)
        self.rotate = int(rotate)
        self.flip_code = 99 if flip_code is None else int(flip_code)
        self.flip = flip
        self.zoom = max(1.0, zoom)
        self.fit = fit
        self.wide = max(1.0, wide)
        self.pan_x = pan_x
        self.pan_y = pan_y
        self.fixed_row = key_row
        self.on_thresh = on_thresh
        self.off_thresh = off_thresh
        self.adapt = adapt
        self.adapt_tau = adapt_tau
        self.max_hold = max_hold
        self.stuck_tau = stuck_tau
        self.seg_w = seg_w
        self.threshold = threshold

        self.lock = threading.Lock()
        self.sil = np.zeros((H, W), dtype=bool)      # you, in wall pixels
        self.key_y0 = key_row if key_row is not None else 8
        self.state = np.zeros(KEYS, dtype=bool)
        self.energy = np.zeros(KEYS, dtype=np.float32)
        self.baseline = np.zeros(KEYS, dtype=np.float32)
        self.hold_t = np.zeros(KEYS, dtype=np.float32)
        self.raw = None
        self.head_row = None
        self.headroom = True
        self._tops = deque(maxlen=self.HEAD_WINDOW)
        self._baseline_at = self.key_y0
        self._resync = False
        self.person_frac = 0.0
        self.running = True
        self.fps = 0.0

        self.cap, self.camera = open_camera(camera)
        self.seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)

    # -- geometry -------------------------------------------------------------
    def to_wall(self, mask):
        """Segmentation mask -> (H, W) wall-space coverage, mirrored and framed.

        Default 'crop' takes a 1:4 window out of the camera view and maps that
        to the panel, exactly like mirror_led.py -- your proportions survive,
        so you read as a person with arms rather than a blob.

        'stretch' squeezes the whole 4:3 view onto the 1:4 panel instead. It
        buys you the full camera width (your whole arm span reaches the whole
        keyboard) at the cost of squashing you ~5x horizontally, which turns a
        person into an unrecognisable smear that covers every key at once.
        Kept only as an option; crop is what makes the instrument playable.
        """
        h, w = mask.shape
        if self.fit == "crop":
            crop_h = min(h, int(round(h / self.zoom)))
            # `wide` widens the window past the panel's own 1:4 shape so a full
            # wingspan fits. wide=1 is exactly proportional but clips your arms
            # at the elbow; wide=2 fits them at the cost of squashing you 2x
            # horizontally. Height is untouched either way.
            crop_w = min(w, int(round(crop_h * ASPECT * self.wide)))
            mx, my = w - crop_w, h - crop_h
            x0 = int(np.clip(round(mx / 2 + self.pan_x * mx / 2), 0, mx))
            y0 = int(np.clip(round(my / 2 + self.pan_y * my / 2), 0, my))
            mask = mask[y0:y0 + crop_h, x0:x0 + crop_w]
        elif self.zoom > 1.0:
            ch, cw = int(h / self.zoom), int(w / self.zoom)
            mask = mask[(h - ch) // 2:(h - ch) // 2 + ch,
                        (w - cw) // 2:(w - cw) // 2 + cw]
        wall = cv2.resize(mask, (W, H), interpolation=cv2.INTER_AREA)
        return wall[:, ::-1] if self.flip else wall

    def _track_head(self, sil):
        """Float the key row just above the top of your silhouette.

        A fixed row cannot work: stand closer and your head covers the keys,
        stand back and your hands never reach them. Following your silhouette
        works at any distance without tuning.

        The catch: a raised hand IS the topmost pixel, so naive tracking chases
        your hand and lifts the keys out of their own way. So instead of the
        latest measurement we take a high percentile over a few seconds. A hand
        raised in some frames only pulls the *low* end of that window, while a
        real move (you step closer, or leave) shifts every sample and is
        followed within a second or two. The percentile has to be high --
        while playing, hands are up most of the time, so they are the majority
        of samples, not outliers.
        """
        rows = sil.mean(axis=1)
        hit = np.nonzero(rows > self.HEAD_MIN_COV)[0]
        if len(hit) == 0:
            return                                   # nobody there; keep the keys
        self._tops.append(float(hit[0]))
        top = float(np.percentile(self._tops, self.HEAD_PCTL))
        self.head_row = top if self.head_row is None else \
            self.head_row + (top - self.head_row) * self.HEAD_SMOOTH

        y0 = int(round(self.head_row)) - HEAD_GAP - KEY_ROWS
        self.headroom = y0 >= 0
        y0 = int(np.clip(y0, 0, max_key_row()))
        if abs(y0 - self._baseline_at) > 2:
            # The keys moved to a different patch of the scene, so the baseline
            # describing the old patch is meaningless and would fire phantom
            # notes. Re-learn it. Slow drift while playing stays under this.
            self._baseline_at = y0
            self._resync = True
        self.key_y0 = y0

    def _key_cover(self, sil, y0):
        """Coverage per key. Generic over the key table, so the side lanes are
        detected by exactly the same rule as the top row -- what you see is
        still what triggers."""
        return np.array([sil[ys, xs].mean() for ys, xs in key_cells(y0)],
                        dtype=np.float32)

    # -- trigger --------------------------------------------------------------
    def update_state(self, raw, dt):
        """Per-key coverage -> which keys are pressed. Separate from the capture
        loop so the trigger behaviour can be tested offline."""
        with self.lock:
            self.raw = raw
            if self._resync:
                self._resync = False
                self.baseline = raw.copy()
                self.state = np.zeros(KEYS, dtype=bool)
                self.hold_t[:] = 0.0
            energy = np.maximum(0.0, raw - self.baseline)
            on = np.where(self.state, energy > self.off_thresh,
                          energy > self.on_thresh)      # hysteresis

            # Anything permanently covering a key (a lamp, a doorframe) settles
            # into the baseline and stops holding the key down. Adaptation
            # pauses while a key is pressed so it does not fight you, but only
            # up to max_hold -- a key "pressed" longer than any human would is
            # a static object, not a hand, and gets absorbed so it frees up.
            self.hold_t = np.where(on, self.hold_t + dt, 0.0)
            stuck = on & (self.hold_t > self.max_hold)
            if self.adapt:
                tau = np.where(stuck, self.stuck_tau, self.adapt_tau)
                k = np.where(~on | stuck, np.minimum(1.0, dt / tau), 0.0)
                self.baseline = (self.baseline + (raw - self.baseline) * k).astype(np.float32)

            self.state = on
            self.energy = energy
            return on

    def lock_row(self):
        """Freeze the key row where head-tracking has just put it.

        Floating keys adapt to any framing but are a moving target, and moving
        targets are hard to aim at. Fixed keys are learnable but only work if
        you happen to be framed with space above your head. Locking gets both:
        the row is measured from where you actually stand during warmup, then
        never moves again while you play.
        """
        self.fixed_row = self.key_y0
        return self.key_y0

    def calibrate(self, seconds=2.0):
        """Snapshot the empty scene, so whatever already covers a key at startup
        never counts as a press."""
        samples, t0 = [], time.time()
        while time.time() - t0 < seconds:
            with self.lock:
                if self.raw is not None:
                    samples.append(self.raw.copy())
            time.sleep(0.05)
        if samples:
            with self.lock:
                self.baseline = np.mean(samples, axis=0).astype(np.float32)
                self.state = np.zeros(KEYS, dtype=bool)
                self.hold_t[:] = 0.0
        return len(samples)

    def run(self):
        t0, n, last = time.time(), 0, time.time()
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            now = time.time()
            dt, last = now - last, now

            # Rotate BEFORE segmenting: the model is trained on upright people
            # and a sideways one segments badly, so this fixes the mask quality
            # as well as the orientation.
            if self.rotate in _ROT:
                frame = cv2.rotate(frame, _ROT[self.rotate])
            if self.flip_code in (-1, 0, 1):
                frame = cv2.flip(frame, self.flip_code)

            h, w = frame.shape[:2]
            small = cv2.resize(frame, (self.seg_w, max(1, int(self.seg_w * h / w))))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mask = self.seg.process(rgb).segmentation_mask
            if mask is None:
                continue

            sil = self.to_wall(mask) > self.threshold
            if self.fixed_row is None:
                self._track_head(sil)
            self.person_frac = float(sil.mean())
            with self.lock:
                self.sil = sil
            self.update_state(self._key_cover(sil, self.key_y0), dt)

            n += 1
            if now - t0 >= 2.0:
                self.fps = n / (now - t0)
                t0, n = now, 0

    def snapshot(self):
        """Silhouette, key row and pressed keys, consistent with each other."""
        with self.lock:
            return self.sil.copy(), self.key_y0, self.state.copy()

    def meters(self):
        with self.lock:
            return self.energy.copy(), self.state.copy()

    def stop(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()
        self.seg.close()


# ----------------------------------------------------------------------------
# Render
# ----------------------------------------------------------------------------
def render(sil, key_y0, held, flash, bri=1.0, song=None):
    f = np.zeros((H, W, 3), dtype=np.float32)

    # you, dim, behind everything
    f[sil] = SIL_COLOR * SIL_LEVEL

    cells = key_cells(key_y0)

    # Strike streaks shooting out of a struck key: up out of the top row, and
    # inward across the panel from a side lane. A side streak cannot go outward
    # -- the lane is already against the edge -- so it fires into the room
    # instead, which also keeps it clear of the other lane.
    for i in np.nonzero(flash > 0.01)[0]:
        lane = KEY_SPECS[i][0]
        ys, xs = cells[i]
        top = lane == LANE_TOP
        reach = int((FLASH_REACH if top else SIDE_FLASH_REACH) * flash[i])
        peak = flash[i] * (0.8 if top else SIDE_FLASH_LEVEL)
        if not reach:
            continue
        if top:
            y1 = max(0, ys.start - reach)
            if ys.start > y1:
                fade = np.linspace(peak, 0.0, ys.start - y1)[:, None, None]
                f[y1:ys.start, xs] += KEY_COLORS[i] * fade
        elif lane == LANE_LEFT:
            x1 = min(W, xs.stop + reach)
            if x1 > xs.stop:
                fade = np.linspace(peak, 0.0, x1 - xs.stop)[None, :, None]
                f[ys, xs.stop:x1] += KEY_COLORS[i] * fade
        else:
            x1 = max(0, xs.start - reach)
            if xs.start > x1:
                fade = np.linspace(0.0, peak, xs.start - x1)[None, :, None]
                f[ys, x1:xs.start] += KEY_COLORS[i] * fade

    # The keys, always on top so they stay readable over you.
    #
    # At 30 LEDs across, one LED is ~3cm of you: a hand is about 3 LEDs and a
    # finger is half of one, so fingers can never be drawn. Instead of a hand
    # that stops dead at the key, the part of you actually inside the key cell
    # is punched out in white below -- you see the contact patch itself, which
    # is the feedback that a finger-shaped blob could not give anyway.
    for i, (ys, xs) in enumerate(cells):
        lane, _, scale_idx, _ = KEY_SPECS[i]
        if song is None:
            idle = KEY_IDLE
        elif i == song["target"]:
            # the answer, pulsing so it reads as "here, now" rather than just
            # "bright" -- at 3px a static key competes with the strike flashes
            idle = SONG_TARGET_MIN + (1.0 - SONG_TARGET_MIN) * song["pulse"]
        elif i == song["next"]:
            idle = SONG_NEXT
        else:
            idle = SONG_OTHER
        level = idle + (1.0 - idle) * max(flash[i], 1.0 if held[i] else 0.0)
        f[ys, xs] = KEY_COLORS[i] * level
        root = scale_idx % len(SCALE) == 0
        if lane == LANE_TOP:
            # gap on every key's trailing edge, so all keys read the same size
            f[ys, xs.stop - 1] *= 0.25
            if held[i] or flash[i] > 0.5:
                f[ys.start, xs] = 1.0                 # bright lip on contact
            if root:
                # dimmer than CONTACT_COLOR on purpose: a white bar here would
                # be indistinguishable from your hand touching the key
                f[ys.stop - 1, xs] = np.maximum(f[ys.stop - 1, xs], 0.55)
        else:
            f[ys.stop - 1, xs] *= 0.25                # gap below each side key
            # The lip goes on the lane's inner edge, the side your hand arrives
            # from, so contact reads the same way it does on the top row.
            lip = xs.stop - 1 if lane == LANE_LEFT else xs.start
            if held[i] or flash[i] > 0.5:
                f[ys, lip] = 1.0
            if root:
                edge = xs.start if lane == LANE_LEFT else xs.stop - 1
                f[ys, edge] = np.maximum(f[ys, edge], 0.55)

    # your hand, punched through whichever keys it is touching
    for ys, xs in cells:
        touch = sil[ys, xs]
        if touch.any():
            f[ys, xs][touch] = CONTACT_COLOR

    if song is not None:
        draw_progress(f, song)

    return (np.clip(f * bri, 0.0, 1.0) * 255).astype(np.uint8)


def draw_progress(f, song):
    """How far through the song you are, along the bottom edge.

    The bottom of the panel is below the side lanes and below your silhouette's
    feet, so it is the one strip nothing else is using.
    """
    y0 = H - SONG_BAR_ROWS
    f[y0:H] = (0.03, 0.03, 0.05)
    done = int(round(W * song["progress"]))
    if song["target"] is None:                    # finished: whole bar cheers
        f[y0:H] = np.array((1.0, 1.0, 1.0)) * (0.35 + 0.65 * song["pulse"])
    elif done:
        f[y0:H, :done] = KEY_COLORS[song["target"]] * 0.8


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run_calib(cam):
    """Terminal meters. The wall itself is the real calibration display now;
    this is for when you cannot see it, or the LEDs are not the problem."""
    print("\nNOTE: --calib drives no LEDs; the wall stays dark. That is normal.")
    print("Stand where you will play and raise a hand into the key row.")
    print(f"A key fires when 'now' passes {cam.on_thresh * 100:.0f}%.")
    print("Ctrl+C to quit.\n")
    peak = np.zeros(KEYS, dtype=np.float32)
    prev_lines = 0
    try:
        while True:
            energy, state = cam.meters()
            peak = np.maximum(energy, peak * 0.97)
            sil, key_y0, _ = cam.snapshot()
            head = "-" if cam.head_row is None else f"{cam.head_row:.0f}"
            warn = "" if cam.headroom else "  NO HEADROOM: tilt camera up / step back"
            prof = [b.mean() for b in np.array_split(sil.mean(axis=1), PROFILE_BINS)]

            lines = [
                f" cam {cam.fps:4.0f}fps   person {cam.person_frac * 100:4.1f}%   "
                f"head row {head}   keys at row {key_y0}{warn}",
                f" rotate {cam.rotate}  flip_code {cam.flip_code}  fit {cam.fit}"
                f"  mirror {'on' if cam.flip else 'off'}  wide {cam.wide:.1f}"
                f"  zoom {cam.zoom:.2f}",
                "",
                " you, as the wall sees you  ('K' = the key row):",
            ]
            # rows covered by a side lane, for the 'S' marker below
            cells = key_cells(key_y0)
            side_rows = set()
            for i, (ys, _) in enumerate(cells):
                if KEY_SPECS[i][0] != LANE_TOP:
                    side_rows.update(range(ys.start, ys.stop))

            # every 3rd row, so a 120-row panel fits on screen
            for y in range(0, H, 3):
                band = sil[y:y + 3]
                if key_y0 <= y < key_y0 + KEY_ROWS:
                    mark = "K"
                elif y in side_rows:
                    mark = "S"
                else:
                    mark = " "
                lines.append(f"  {mark}{y:3d} |"
                             + "".join("#" if band[:, x].any() else "." for x in range(W))
                             + "|")

            # One meter row per lane -- all 20 keys on a single line runs off
            # the side of the terminal.
            lines.append("")
            for lane in (LANE_TOP, LANE_LEFT, LANE_RIGHT):
                idx = [i for i in range(KEYS) if KEY_SPECS[i][0] == lane]
                if not idx:
                    continue
                lines.append(f" {LANE_NAME[lane]:>5} now  "
                             + " ".join(f"{energy[i] * 100:3.0f}"
                                        f"{'#' if state[i] else ' '}" for i in idx))
                lines.append(f" {'':>5} peak " + " ".join(f"{peak[i] * 100:3.0f} "
                                                          for i in idx))
            if prev_lines:
                print(f"\033[{prev_lines}A", end="")
            print("\n".join(f"{ln:<78}" for ln in lines), flush=True)
            prev_lines = len(lines)
            time.sleep(0.15)
    except KeyboardInterrupt:
        print("\nstopped.")


def add_camera_args(ap):
    """Camera and framing options. Shared with notecatch.py so the two stay in
    step -- framing is fiddly to tune and having it drift between programs
    would mean tuning it twice."""
    ap.add_argument("--camera", type=int, default=0, help="camera index (/dev/videoN)")
    ap.add_argument("--mirror", choices=("auto", "on", "off"), default="auto",
                    help="horizontal mirror so your left hand lights the left of "
                         "the wall. 'auto' skips it when calibration.json already "
                         "flips horizontally, so the two cannot cancel out")
    ap.add_argument("--wide", type=float, default=2.0,
                    help="widen the crop window to fit your arms. 1.0 is exactly "
                         "proportional but clips them at the elbow; 2.0 (default) "
                         "fits raised hands; ~3.0 fits arms straight out")
    ap.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=None,
                    help="rotate the camera image so you stand upright; defaults "
                         "to the mirror's calibration.json if present")
    ap.add_argument("--flip-code", type=int, choices=(-1, 0, 1), default=None,
                    help="OpenCV flip applied after --rotate (same meaning as "
                         "the mirror's flip_code)")
    ap.add_argument("--fit", choices=("crop", "stretch"), default="crop",
                    help="crop: a 1:4 window, you keep your proportions (default). "
                         "stretch: whole camera view squashed onto the panel -- "
                         "more keyboard reach, but you become a blob")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="crop in tighter (>1 = closer)")
    ap.add_argument("--pan-x", type=float, default=0.0,
                    help="-1..1, slide the crop window left/right to centre you")
    ap.add_argument("--pan-y", type=float, default=0.0,
                    help="-1..1, slide the crop window up/down")
    ap.add_argument("--sens", type=float, default=1.0,
                    help="trigger sensitivity multiplier; lower = easier to trigger")
    ap.add_argument("--no-adapt", dest="adapt", action="store_false",
                    help="disable the slow baseline that ignores static objects")
    ap.add_argument("--warmup", type=float, default=2.0,
                    help="seconds spent learning the empty scene at startup (0 to skip)")
    ap.add_argument("--max-hold", type=float, default=10.0,
                    help="seconds a key may stay pressed before it is treated as "
                         "a static object and absorbed into the baseline")
    ap.add_argument("--seg-w", type=int, default=160,
                    help="width the frame is scaled to before segmentation; "
                         "lower is faster and lower latency")
    ap.add_argument("--fps", type=int, default=60, help="render cap")
    ap.add_argument("--bri", type=float, default=1.0)
    return ap


def resolve_framing(args):
    """Merge the command line with the mirror's calibration.json.

    Anything passed explicitly wins; otherwise the mirror's already-tuned
    geometry for this room is reused rather than calibrating a second time.
    """
    calib = load_mirror_calib()
    rotate = args.rotate if args.rotate is not None else calib.get("rotate", 0)
    flip_code = (args.flip_code if args.flip_code is not None
                 else calib.get("flip_code"))
    # flip_code 1 and -1 already mirror horizontally. Adding our own flip on top
    # would cancel it out and swap your hands, so auto only mirrors when the
    # calibration has not done it already.
    calib_mirrors = flip_code in (1, -1)
    return {
        "calib": calib,
        "rotate": rotate,
        "flip_code": flip_code,
        "zoom": args.zoom if args.zoom != 1.0 else float(calib.get("zoom", 1.0)),
        "pan_x": args.pan_x if args.pan_x else float(calib.get("pan_x", 0.0)),
        "pan_y": args.pan_y if args.pan_y else float(calib.get("pan_y", 0.0)),
        "calib_mirrors": calib_mirrors,
        "mirror": ((not calib_mirrors) if args.mirror == "auto"
                   else (args.mirror == "on")),
    }


def main():
    ap = argparse.ArgumentParser(description="Air piano on the 30x120 Niji wall.")
    add_camera_args(ap)
    ap.add_argument("--key-row", default="lock",
                    help="'lock' (default) places the keys above your head during "
                         "warmup and then holds them still; 'auto' keeps following "
                         "you; a number pins them to that wall row")
    ap.add_argument("--top-keys", type=int, default=TOP_KEYS,
                    help="keys in the row above your head. 0 (default) removes "
                         "it -- hands raised that high leave the camera frame. "
                         f"Use {TOP_ROW_KEYS} to bring it back")
    ap.add_argument("--scale", choices=tuple(SCALES), default=SCALE_NAME,
                    help="'major' (default, C D E F G A B) plays the famous "
                         "tunes; 'pentatonic' (C D E G A) has no wrong notes "
                         "but also no F or B, so most songs will not fit")
    ap.add_argument("--side-keys", type=int, default=None,
                    help="keys per side lane (0 = top row only, as before). "
                         "Defaults to one octave of the scale: 7 on major, 5 "
                         "on pentatonic. The left lane is the lower octave, "
                         "the right the upper; pitch rises upward on both")
    ap.add_argument("--song", help="follow-the-light practice: the next note to "
                                   "play pulses and everything else dims. "
                                   "See --list-songs")
    ap.add_argument("--list-songs", action="store_true",
                    help="list the built-in songs and exit")
    ap.add_argument("--song-file", help="JSON of your own songs, "
                                        '{"name": {"title": ..., "notes": "C4 D4 ..."}}')
    ap.add_argument("--octave", type=int, default=0, help="shift the scale by octaves")
    ap.add_argument("--volume", type=float, default=0.5, help="0..1 master volume")
    ap.add_argument("--audio", choices=("auto", "sounddevice", "pygame", "none"),
                    default="auto")
    ap.add_argument("--blocksize", type=int, default=256,
                    help="audio block size; raise to 512/1024 if sound crackles")
    ap.add_argument("--calib", action="store_true", help="terminal meters, no LEDs")
    args = ap.parse_args()

    # The scale sets how many keys a lane needs, so it has to land first.
    set_scale(args.scale)
    if args.side_keys is None:
        args.side_keys = len(SCALE)

    # Validate before anything reads the key table -- --list-songs builds it
    # too, so checking after its early return let bad values straight through.
    if args.side_keys < 0 or args.top_keys < 0:
        raise SystemExit("--side-keys / --top-keys cannot be negative")
    if args.top_keys and W % args.top_keys:
        raise SystemExit(f"--top-keys {args.top_keys} does not divide the "
                         f"{W}px width evenly; try {TOP_ROW_KEYS}, 6, 5 or 3")
    if not args.top_keys and not args.side_keys:
        raise SystemExit("--top-keys 0 and --side-keys 0 leaves no keys at all")
    if args.side_keys * SIDE_KEY_H > H - KEY_ROWS - SIDE_GAP:
        raise SystemExit(
            f"--side-keys {args.side_keys} needs "
            f"{args.side_keys * SIDE_KEY_H} rows; only "
            f"{H - KEY_ROWS - SIDE_GAP} are left below the top row")
    configure_keys(args.side_keys, args.top_keys)

    if args.list_songs:
        print(f"Keyboard: {note_name(min(KEY_FREQS))}-{note_name(max(KEY_FREQS))}, "
              f"{KEYS} keys, {SCALE_NAME} ({scale_letters()})"
              f"{', no top row' if not TOP_KEYS else ''}\n")
        for key in SONGS:
            try:
                title, keys, shift = load_song(key)
            except SystemExit:
                pitches = [note_freq(n) for n in SONGS[key][1].split()]
                off = sorted({SCALE_LETTERS[round(12 * math.log2(f / C4)) % 12]
                              for f in pitches
                              if round(12 * math.log2(f / C4)) % 12 not in set(SCALE)})
                why = (f"needs {'/'.join(off)}, not on a {SCALE_NAME} keyboard"
                       if off else
                       f"spans {note_name(min(pitches))}-"
                       f"{note_name(max(pitches))} -- too wide for this keyboard")
                print(f"  {key:<10} {SONGS[key][0]:<38} {why}")
                continue
            lanes = sorted({LANE_NAME[KEY_SPECS[i][0]] for i in keys})
            pitches = [KEY_FREQS[i] for i in keys]
            span = f"{note_name(min(pitches))}-{note_name(max(pitches))}"
            moved = f"  ({shift:+d} oct to fit)" if shift else ""
            print(f"  {key:<10} {title:<38} {len(keys):>3} notes  "
                  f"{span:<8} {'+'.join(lanes)}{moved}")
        print("\n  python3 piano.py --song mary")
        return

    row_mode = args.key_row.lower()
    if row_mode in ("auto", "lock"):
        key_row = None                      # track for now; 'lock' freezes later
    else:
        try:
            key_row = int(args.key_row)
        except ValueError:
            raise SystemExit("--key-row must be a number, 'auto' or 'lock'")
        if not 0 <= key_row <= max_key_row():
            raise SystemExit(
                f"--key-row must be 0..{max_key_row()}, 'auto' or 'lock'"
                + (f" (the side lanes need the {H - max_key_row() - KEY_ROWS} rows "
                   f"below the top row; --side-keys 0 frees them)"
                   if SIDE_KEYS else ""))

    fr = resolve_framing(args)
    calib = fr["calib"]
    if calib:
        print(f"Using the mirror's calibration.json: {calib}")
    rotate, flip_code = fr["rotate"], fr["flip_code"]
    zoom, pan_x, pan_y = fr["zoom"], fr["pan_x"], fr["pan_y"]
    calib_mirrors, mirror = fr["calib_mirrors"], fr["mirror"]

    print("Warming up camera + MediaPipe...")
    cam = KeyCam(camera=args.camera, flip=mirror, zoom=zoom,
                 fit=args.fit, pan_x=pan_x, pan_y=pan_y, wide=args.wide,
                 rotate=rotate, flip_code=flip_code,
                 key_row=key_row, seg_w=args.seg_w, adapt=args.adapt,
                 on_thresh=0.16 * args.sens, off_thresh=0.10 * args.sens,
                 max_hold=args.max_hold)
    cam.start()
    t0 = time.time()
    while cam.raw is None and time.time() - t0 < 8.0:
        time.sleep(0.1)
    if cam.raw is None:
        raise SystemExit("No frames from the camera after 8s -- check the camera.")
    if args.warmup > 0:
        if row_mode == "lock":
            print(f"Stand where you will play, hands DOWN, for {args.warmup:.1f}s "
                  "-- placing the keys above your head...")
            time.sleep(args.warmup)         # the tracker is running in the thread
            cam.lock_row()
        # baseline last, so it describes the scene at the final key position
        print("Reading the empty scene -- keep your hands off the keys...")
        cam.calibrate(max(1.0, args.warmup * 0.5))

    if args.calib:
        try:
            run_calib(cam)
        finally:
            cam.stop()
        return

    freqs = build_freqs(KEY_SPECS, args.octave)
    synth = make_synth(freqs, args.volume, args.audio, args.blocksize)
    print(f"Audio: {synth.name}")
    print("Keys: " + ("floating above your head" if cam.fixed_row is None
                      else f"fixed at wall row {cam.fixed_row}"))
    print(f"Framing: {args.fit}, rotate {rotate}, flip_code {flip_code}, "
          f"mirror {'on' if mirror else 'off'}"
          f"{' (calibration.json already mirrors)' if calib_mirrors else ''}, "
          f"wide {args.wide:.1f}, zoom {zoom:.2f}, pan {pan_x:+.2f}/{pan_y:+.2f}")
    print("  If your left hand lights the RIGHT of the wall, add --mirror "
          f"{'on' if not mirror else 'off'}")
    if not cam.headroom:
        print("  ! No space above your head -- the keys are jammed against the top")
        print("  ! of the wall and your head will hold them down. Step back, or")
        print("  ! try --pan-y -0.4 / --zoom 1.3 to reframe.")
    trainer = None
    if args.song:
        title, keys, shift = load_song(args.song, args.song_file)
        trainer = SongTrainer(title, keys)
        print(f"Song: {title} -- {len(keys)} notes"
              + (f", shifted {shift:+d} octave to fit the keyboard" if shift else ""))
        print("  The next note pulses; everything else dims. Free tempo, so it")
        print("  waits for you. A wrong key still sounds, it just does not advance.")
    print("Ready. Find yourself on the wall, then reach up into the keys. Ctrl+C to quit.")

    ddp = DDP()
    flash = np.zeros(KEYS, dtype=np.float32)
    prev = np.zeros(KEYS, dtype=bool)
    frame_dt = 1.0 / args.fps
    last = time.time()
    report = time.time()

    try:
        while True:
            now = time.time()
            dt = min(0.1, now - last)
            last = now

            sil, key_y0, held = cam.snapshot()
            for i in np.nonzero(held & ~prev)[0]:      # the moment of contact
                synth.strike(int(i))
                flash[i] = 1.0
                if trainer is not None:
                    trainer.strike(int(i))
            prev = held
            flash *= math.exp(-dt / FLASH_TAU)

            song_view = None
            if trainer is not None:
                if trainer.done_at and now - trainer.done_at > SONG_DONE_HOLD:
                    total = trainer.hits + trainer.wrong
                    print(f"  {trainer.title}: done -- {trainer.hits} notes, "
                          f"{trainer.wrong} wrong "
                          f"({100 * trainer.hits / max(1, total):.0f}% clean). "
                          f"Going round again.")
                    trainer.restart()
                song_view = {
                    "target": trainer.target,
                    "next": trainer.upcoming,
                    "progress": trainer.progress,
                    "pulse": 0.5 + 0.5 * math.sin(now * math.tau * SONG_PULSE_HZ),
                }

            ddp.send(render(sil, key_y0, held, flash, args.bri, song_view))

            if now - report >= 3.0:
                keys = "".join("#" if h else "." for h in held)
                print(f"  cam {cam.fps:.0f} fps  keys [{keys}]  row {key_y0}", flush=True)
                report = now

            sleep = frame_dt - (time.time() - now)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        synth.close()
        cam.stop()


if __name__ == "__main__":
    main()
