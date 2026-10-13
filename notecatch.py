#!/usr/bin/env python3
"""
notecatch.py - falling notes you catch with your hands, on the 30x120 wall.

Notes drop down five coloured tracks. Put your hand in a note's track as it
crosses the catch line and you catch it: the note sounds, the line flashes, and
your streak grows. Miss it and it falls on past you.

Why it is not just piano.py with movement: the piano's keys are two vertical
lanes down the edges, and a note cannot fall *along* one of those -- each lane
stacks five different pitches, so a falling note would change pitch on the way
down. This mode uses its own geometry, five tracks across the full width.

One track per scale degree, with octaves folded together: on the major scale
that is seven tracks (C D E F G A B) at 4px each, and on --scale pentatonic
five at 6px. Folding octaves is what keeps a track wide enough to catch -- one
track per *key* would mean fourteen tracks barely 2px across. You still hear
the song's real octave; only which track it falls down is folded.

Seven rather than five is what makes the interesting songs possible at all: the
F and the B are exactly the notes Twinkle, Ode to Joy, Happy Birthday, Jingle
Bells and Korobeiniki need, and a pentatonic wall has neither.

The colours match piano.py's lanes -- red C through violet at the top of the
octave. The same colour means the same note in both programs, so practising one
teaches the other.

Detection works the way the piano's does -- in wall space, against the note's
own drawn rectangle. What you see is exactly what you have to touch.

    python3 notecatch.py                       # Mary Had a Little Lamb
    python3 notecatch.py --song korobeiniki
    python3 notecatch.py --bpm 40 --fall 3.5   # slower, easier
    python3 notecatch.py --list-songs

Songs, framing and camera options are piano.py's -- see its --help. Note that
the built-in songs carry pitch but no rhythm, so notes arrive evenly at --bpm.

Requires: numpy, opencv-python, mediapipe, plus piano.py and wall.py.
"""
import argparse
import colorsys
import math
import time

import numpy as np

from wall import DDP, W, H
import piano
from piano import (KeyCam, SCALES, SONGS, add_camera_args, load_song,
                   make_synth, note_name, resolve_framing, scale_letters,
                   set_scale, C4, SIL_COLOR, SIL_LEVEL)

# ---- track geometry --------------------------------------------------------
# Filled in by configure_tracks() once --scale is known: 7 tracks of 4px on the
# major scale, 5 of 6px on pentatonic.
TRACKS = TRACK_W = MARGIN = 0
TRACK_COLORS = []
NOTE_H = 6                        # rows a falling note occupies

# The catch line sits below your head, at the height your hands reach when your
# arms are out -- the same reason the piano's top row had to go. Anchored to the
# head tracker so it follows you rather than needing to be tuned per person.
HIT_OFFSET = 44                   # rows below the key-row anchor
HIT_ROWS = 15                     # depth of the catch zone; this is the timing
                                  # window, and whole hands need a generous one
CATCH_COVER = 0.16                # silhouette fraction inside the note's cell

FALL_SECONDS = 3.0                # spawn -> catch line
LEAD_IN = 2.5                     # quiet seconds before the first note lands
MISS_SLACK = 4                    # rows past the zone before it counts as gone

# ---- look ------------------------------------------------------------------
TRACK_DIM = 0.05                  # the empty track, just visible
LINE_LEVEL = 0.34                 # the catch line at rest
NOTE_LEVEL = 0.85                 # a falling note
NOTE_ARMED = 1.0                  # a note inside the catch zone
HIT_FLASH_TAU = 0.28
MISS_FLASH_TAU = 0.45
BAR_ROWS = 3                      # progress along the bottom edge


def configure_tracks():
    """Lay the tracks out for whichever scale is in force.

    Seven tracks do not divide the 30px width, so they are 4px each and the two
    leftover columns become a dead margin at the edges rather than being handed
    to two arbitrary tracks -- unequal tracks would mean unequal difficulty, and
    the outermost ones are already the hardest to reach.
    """
    global TRACKS, TRACK_W, MARGIN, TRACK_COLORS
    TRACKS = len(piano.SCALE)
    TRACK_W = W // TRACKS
    MARGIN = (W - TRACKS * TRACK_W) // 2
    TRACK_COLORS = [track_color(t) for t in range(TRACKS)]


def track_color(t):
    """Same hue ramp piano.py gives one lane, so a note is the same colour in
    both programs -- red at the root of the octave, violet at the top of it."""
    r, g, b = colorsys.hsv_to_rgb(0.86 * t / max(1, TRACKS - 1), 0.95, 1.0)
    return np.array((r, g, b), dtype=np.float32)


def track_span(tr):
    """Columns of a track. Notes fill it edge to edge, and this is also what is
    tested for a catch -- the drawn rectangle and the hit box are the same
    thing. Tracks are told apart by hue and by a divider drawn into the empty
    background, which costs no catching width at 4px."""
    x0 = MARGIN + tr * TRACK_W
    return slice(x0, x0 + TRACK_W)


def track_of(freq):
    """Which track a pitch falls down, ignoring its octave."""
    return piano.SCALE.index(round(12 * math.log2(freq / C4)) % 12)


configure_tracks()


def note_rect(y):
    """Clamped (y0, y1) rows for a note centred at row y, or None if off-panel.

    Every use of a note's rectangle goes through this. Clamping by hand is what
    broke the first version: a note still above the panel gives y0 = -50, and
    slice(0, min(H, y0 + NOTE_H)) becomes slice(0, -44) -- a negative stop
    counts back from the end, so the "note" silently covered rows 0..76 and
    whited out the whole silhouette.
    """
    y0 = int(round(y - NOTE_H / 2.0))
    y1 = y0 + NOTE_H
    if y1 <= 0 or y0 >= H:
        return None
    return max(0, y0), min(H, y1)


class Note:
    __slots__ = ("key", "track", "t_hit", "state")

    def __init__(self, key, track, t_hit):
        self.key = key            # index into piano.KEY_FREQS, for the sound
        self.track = track
        self.t_hit = t_hit        # when its centre reaches the catch line
        self.state = "falling"    # falling | caught | missed


class Chart:
    """The song as a timed stream of falling notes."""

    def __init__(self, title, keys, bpm, fall_seconds, catch_cover=CATCH_COVER):
        self.title = title
        self.fall = fall_seconds
        self.catch_cover = catch_cover
        beat = 60.0 / bpm
        self.notes = [
            Note(k, track_of(piano.KEY_FREQS[k]), LEAD_IN + i * beat)
            for i, k in enumerate(keys)
        ]
        self.caught = 0
        self.missed = 0
        self.streak = 0
        self.best_streak = 0
        self.done_at = None

    @property
    def total(self):
        return len(self.notes)

    @property
    def settled(self):
        return self.caught + self.missed

    def y_of(self, note, t, hit_centre):
        """Row of a note's centre. Speed is set so it takes `fall` seconds to
        come down from the top, which is what makes the run-up readable."""
        speed = hit_centre / self.fall
        return hit_centre - (note.t_hit - t) * speed

    def update(self, t, sil, hit_top, hit_bottom, on_catch, on_miss):
        centre = (hit_top + hit_bottom) / 2.0
        for n in self.notes:
            if n.state != "falling":
                continue
            y = self.y_of(n, t, centre)
            if y - NOTE_H / 2 > hit_bottom + MISS_SLACK:
                n.state = "missed"
                self.missed += 1
                self.streak = 0
                on_miss(n)
                continue
            # Catchable only while it overlaps the zone, and only by touching
            # the note's own rectangle -- the thing you can actually see.
            rect = note_rect(y)
            if rect and y + NOTE_H / 2 >= hit_top and y - NOTE_H / 2 <= hit_bottom:
                cell = (slice(*rect), track_span(n.track))
                if sil[cell].size and sil[cell].mean() >= self.catch_cover:
                    n.state = "caught"
                    self.caught += 1
                    self.streak += 1
                    self.best_streak = max(self.best_streak, self.streak)
                    on_catch(n)
        if self.done_at is None and self.settled >= self.total:
            self.done_at = t

    def restart(self, t):
        beat_shift = t + LEAD_IN - self.notes[0].t_hit
        for n in self.notes:
            n.t_hit += beat_shift
            n.state = "falling"
        self.caught = self.missed = self.streak = 0
        self.done_at = None


# ----------------------------------------------------------------------------
# Render
# ----------------------------------------------------------------------------
def render(chart, t, sil, hit_top, hit_bottom, flash, miss_flash, bri=1.0):
    f = np.zeros((H, W, 3), dtype=np.float32)

    # empty tracks, so you can see where the notes will come down
    for tr in range(TRACKS):
        xs = track_span(tr)
        f[:, xs] = TRACK_COLORS[tr] * TRACK_DIM
        f[:, xs.stop - 1] *= 0.3               # divider on the trailing edge

    # you, dim, behind the notes
    f[sil] = np.maximum(f[sil], SIL_COLOR * SIL_LEVEL)

    # the catch line: a band per track, brightening as its note arrives
    for tr in range(TRACKS):
        xs = track_span(tr)
        level = LINE_LEVEL + (1.0 - LINE_LEVEL) * flash[tr]
        band = TRACK_COLORS[tr] * level
        if miss_flash[tr] > 0.01:
            band = np.maximum(band, np.array((1.0, 0.1, 0.05)) * miss_flash[tr])
        f[hit_top:hit_bottom + 1, xs] = band
        f[hit_top:hit_bottom + 1, xs.stop - 1] *= 0.35   # keep the tracks apart
    # bright rails top and bottom of the zone, so the line reads as a target
    for y in (hit_top, hit_bottom):
        f[y, MARGIN:W - MARGIN] = np.maximum(f[y, MARGIN:W - MARGIN], 0.55)

    # falling notes, drawn over the line so an arriving note is unmistakable
    centre = (hit_top + hit_bottom) / 2.0
    armed_cells = []
    for n in chart.notes:
        if n.state != "falling":
            continue
        rect = note_rect(chart.y_of(n, t, centre))
        if rect is None:
            continue
        y0, y1 = rect
        armed = y1 >= hit_top and y0 <= hit_bottom
        cell = (slice(y0, y1), track_span(n.track))
        f[cell] = TRACK_COLORS[n.track] * (NOTE_ARMED if armed else NOTE_LEVEL)
        if armed:
            f[y0, track_span(n.track)] = 1.0   # white lip: catch it now
            armed_cells.append(cell)

    # Your hand punched white through a note you are touching, as in the piano.
    # Only for armed notes: doing it to every note on screen made a hand look
    # like it was catching something it cannot yet reach.
    for cell in armed_cells:
        touch = sil[cell]
        if touch.any():
            f[cell][touch] = 1.0

    draw_bar(f, chart)
    return (np.clip(f * bri, 0.0, 1.0) * 255).astype(np.uint8)


def draw_bar(f, chart):
    """Bottom edge: how far through the song, and how clean the run has been."""
    y0 = H - BAR_ROWS
    f[y0:H] = (0.03, 0.03, 0.05)
    done = int(round(W * chart.settled / max(1, chart.total)))
    if chart.done_at is not None:
        acc = chart.caught / max(1, chart.total)
        f[y0:H] = np.array((1.0 - acc, acc, acc * 0.5)) * 0.9
    elif done:
        # green while the streak holds, amber once it has been broken
        col = (0.2, 1.0, 0.3) if chart.streak else (1.0, 0.6, 0.1)
        f[y0:H, :done] = col


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Catch falling notes on the 30x120 Niji wall.")
    add_camera_args(ap)
    ap.add_argument("--song", default="mary", help="which song to drop (see --list-songs)")
    ap.add_argument("--list-songs", action="store_true", help="list songs and exit")
    ap.add_argument("--song-file", help="JSON of your own songs (see piano.py --help)")
    ap.add_argument("--scale", choices=tuple(SCALES), default=piano.SCALE_NAME,
                    help="'major' (default) gives 7 tracks of 4px and plays "
                         "the famous tunes; 'pentatonic' gives 5 wider tracks "
                         "but has no F or B, so most songs will not fit")
    ap.add_argument("--bpm", type=float, default=52.0,
                    help="notes per minute; lower is easier (default 52)")
    ap.add_argument("--fall", type=float, default=FALL_SECONDS,
                    help="seconds a note takes to fall to the catch line")
    ap.add_argument("--hit-offset", type=int, default=HIT_OFFSET,
                    help="rows below the head anchor to put the catch line")
    ap.add_argument("--catch", type=float, default=CATCH_COVER,
                    help="fraction of a note you must cover to catch it")
    ap.add_argument("--octave", type=int, default=0, help="shift the scale by octaves")
    ap.add_argument("--volume", type=float, default=0.5, help="0..1 master volume")
    ap.add_argument("--audio", choices=("auto", "sounddevice", "pygame", "none"),
                    default="auto")
    ap.add_argument("--blocksize", type=int, default=256, help="audio block size")
    args = ap.parse_args()

    if args.bpm <= 0 or args.fall <= 0:
        raise SystemExit("--bpm and --fall must be positive")

    # The scale decides both the tracks and the pitches, so it lands first. The
    # piano's keyboard is what a song resolves against, one octave per lane.
    set_scale(args.scale)
    configure_tracks()
    piano.configure_keys(len(piano.SCALE), piano.TOP_KEYS)

    if args.list_songs:
        print(f"Tracks: {TRACKS} ({scale_letters()}) at {TRACK_W}px, "
              f"octaves folded together\n")
        for key in SONGS:
            try:
                title, keys, shift = load_song(key)
            except SystemExit as exc:
                print(f"  {key:<10} {SONGS[key][0]:<38} "
                      f"{str(exc).splitlines()[0]}")
                continue
            used = sorted({track_of(piano.KEY_FREQS[k]) for k in keys})
            print(f"  {key:<10} {title:<38} {len(keys):>3} notes  "
                  f"{len(used)}/{TRACKS} tracks")
        print("\n  python3 notecatch.py --song korobeiniki")
        return

    title, keys, shift = load_song(args.song, args.song_file)
    chart = Chart(title, keys, args.bpm, args.fall, args.catch)

    fr = resolve_framing(args)
    if fr["calib"]:
        print(f"Using the mirror's calibration.json: {fr['calib']}")
    print("Warming up camera + MediaPipe...")
    cam = KeyCam(camera=args.camera, flip=fr["mirror"], zoom=fr["zoom"],
                 fit=args.fit, pan_x=fr["pan_x"], pan_y=fr["pan_y"],
                 wide=args.wide, rotate=fr["rotate"], flip_code=fr["flip_code"],
                 key_row=None, seg_w=args.seg_w, adapt=args.adapt,
                 max_hold=args.max_hold)
    cam.start()
    t0 = time.time()
    while cam.raw is None and time.time() - t0 < 8.0:
        time.sleep(0.1)
    if cam.raw is None:
        raise SystemExit("No frames from the camera after 8s -- check the camera.")
    if args.warmup > 0:
        print(f"Stand where you will play for {args.warmup:.1f}s -- "
              "placing the catch line...")
        time.sleep(args.warmup)
        cam.lock_row()

    synth = make_synth(piano.build_freqs(piano.KEY_SPECS, args.octave),
                       args.volume, args.audio, args.blocksize)
    print(f"Audio: {synth.name}")
    print(f"Song: {title} -- {chart.total} notes at {args.bpm:g} bpm"
          + (f", shifted {shift:+d} octave to fit" if shift else ""))
    print(f"Notes fall down {TRACKS} tracks ({scale_letters()}). Put a hand in")
    print("a note's track as it crosses the bright line. Ctrl+C to quit.")

    ddp = DDP()
    flash = np.zeros(TRACKS, dtype=np.float32)
    miss_flash = np.zeros(TRACKS, dtype=np.float32)
    frame_dt = 1.0 / args.fps
    last = time.time()
    start = time.time()

    def on_catch(n):
        synth.strike(int(n.key))
        flash[n.track] = 1.0

    def on_miss(n):
        miss_flash[n.track] = 1.0

    try:
        while True:
            now = time.time()
            dt = min(0.1, now - last)
            last = now
            t = now - start

            sil, key_y0, _ = cam.snapshot()
            hit_top = int(np.clip(key_y0 + args.hit_offset, 0, H - HIT_ROWS - 1))
            hit_bottom = hit_top + HIT_ROWS

            chart.update(t, sil, hit_top, hit_bottom, on_catch, on_miss)

            if chart.done_at is not None and t - chart.done_at > 4.0:
                acc = 100.0 * chart.caught / max(1, chart.total)
                print(f"  {chart.title}: {chart.caught}/{chart.total} caught "
                      f"({acc:.0f}%), best streak {chart.best_streak}. Again!")
                chart.restart(t)

            flash *= math.exp(-dt / HIT_FLASH_TAU)
            miss_flash *= math.exp(-dt / MISS_FLASH_TAU)

            ddp.send(render(chart, t, sil, hit_top, hit_bottom,
                            flash, miss_flash, args.bri))

            slack = frame_dt - (time.time() - now)
            if slack > 0:
                time.sleep(slack)
    except KeyboardInterrupt:
        print(f"\nstopped. {chart.caught}/{chart.total} caught, "
              f"best streak {chart.best_streak}.")
    finally:
        cam.stop()
        synth.close()


if __name__ == "__main__":
    main()
