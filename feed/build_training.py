"""
Build the Training tab's practice examples: real cases whose answer is
known, shown without labels.

    python feed/build_training.py            # writes feed/training_out/*.json + *.png

Planets
  yes  confirmed/known planets the pipeline recovered, and fake planets
       injected into real quiet-star light curves (truth certain)
  no   signals the vetting rejected as data artifacts (glitches, sector-edge
       ramps, pulsating stars, lone events), and TIC 394015919, which
       vanished when checked against six more TESS sectors
Comets
  yes  a comet confirmed by a Sungrazer Project report, and fake comets
       injected into real SOHO frames
  no   tracks that move exactly with the background stars (they *are* stars)
       or don't move at all

Each example gets neutral stats (no vetting notes) plus a hidden answer and
an explanation shown after you answer. Upload as for queue cards, into the
page's `training` collection.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "feed" / "training_out"
sys.path[:0] = [str(ROOT / "planets"), str(ROOT / "comets")]


def write(item_id, item, plot):
    OUT.mkdir(exist_ok=True)
    shutil.copy(plot, OUT / f"{item_id}.png")
    (OUT / f"{item_id}.json").write_text(json.dumps(item, indent=1, ensure_ascii=False))


def planet_stats(r):
    return [
        {"label": "Period", "value": f"{r['period_days']:.3f} d"},
        {"label": "Depth", "value": f"{r['depth_ppm']:,.0f} ppm"},
        {"label": "Duration", "value": f"{r['duration_hours']:.1f} h"},
        {"label": "SDE", "value": f"{r['sde']:.1f}"},
        {"label": "SNR", "value": f"{r['detection_snr']:.1f}"},
        {"label": "Transits with data", "value": str(int(r['n_transits']))},
    ]


PROMPT_PLANET = "Is this a planet transit, or something else?"

# ---------------------------------------------------------------------------
# Planets
# ---------------------------------------------------------------------------

KNOWN = {  # validation runs (planets/results/known_planets_results.csv)
    "TIC 261136679": "Pi Mensae c, a super-Earth (confirmed 2018). Five evenly spaced shallow dips with a flat-bottomed fold.",
    "TIC 100100827": "WASP-18 b, a massive hot Jupiter. Deep, regular dips every 0.94 days; obvious even in the raw data.",
    "TIC 25155310": "WASP-126 b, a hot Jupiter. Deep, regular, flat-bottomed transits every 3.3 days.",
    "TIC 22529346": "WASP-121 b, an inflated hot Jupiter. Note the rounded bottom: limb darkening, a sign of a real planet.",
    "TIC 410153553": "LHS 3844 b, a rocky planet on an 11-hour orbit around a small M dwarf. Many short dips.",
}

ARTIFACTS = {  # (target, signal) from planets/results/hunt/signals.csv
    ("TIC 142086812", 1): "A single deep spike of a few points carries the whole signal. A glitch, not a transit.",
    ("TIC 3657911", 1): "Both 'transits' sit at the very start and end of the sector, where TESS data ramps. A classic edge artifact.",
    ("TIC 93716498", 1): "The folded curve waves up and down outside the 'dip': the star pulsates, and the box just fit one trough.",
    ("TIC 75160399", 1): "Only one dip has data; the 'period' is just a guess that puts the second one in a gap.",
    ("TIC 402104884", 1): "One event carries almost all the signal, next to a data gap. Not a repeating transit.",
    ("TIC 391099484", 1): "A single dip, with the second predicted transit in a gap. One event can't give a period.",
    ("TIC 70797900", 2): "One deep event only; the rest of the 'transits' have no data. Not a periodic signal.",
}


def planet_examples():
    known = pd.read_csv(ROOT / "planets/results/known_planets_results.csv")
    hunt = pd.read_csv(ROOT / "planets/results/hunt/signals.csv")
    hunt["has_detection_snr"] = True
    known["detection_snr"] = known.get("detection_snr", known["depth_snr"])
    n = 0
    for _, r in known.iterrows():
        if r["target"] in KNOWN:
            n += 1
            write(f"p-known-{n}", {"kind": "planet", "title": f"Practice star P{n}", "prompt": PROMPT_PLANET,
                                   "stats": planet_stats(r), "answer": "yes",
                                   "answer_detail": "A real, confirmed planet.", "explanation": KNOWN[r["target"]],
                                   "source": "TESS 2-min"}, ROOT / "planets" / r["plot"])
    toi = hunt[(hunt["target"] == "TIC 158490401") & (hunt["signal"] == 1)].iloc[0]
    n += 1
    write(f"p-known-{n}", {"kind": "planet", "title": f"Practice star P{n}", "prompt": PROMPT_PLANET,
                           "stats": planet_stats(toi), "answer": "yes", "answer_detail": "A real, known planet (TOI-7858.01).",
                           "explanation": "Two deep, identical, flat-bottomed dips on a quiet M dwarf: what a real giant planet looks like even with only two transits.",
                           "source": "TESS 2-min"}, ROOT / "planets" / toi["plot"])
    for (target, signal), why in ARTIFACTS.items():
        r = hunt[(hunt["target"] == target) & (hunt["signal"] == signal)].iloc[0]
        n += 1
        write(f"p-artifact-{n}", {"kind": "planet", "title": f"Practice star P{n}", "prompt": PROMPT_PLANET,
                                  "stats": planet_stats(r), "answer": "no", "answer_detail": "A data artifact.",
                                  "explanation": why, "source": "TESS 2-min"}, ROOT / "planets" / r["plot"])
    r = hunt[(hunt["target"] == "TIC 394015919") & (hunt["signal"] == 1)].iloc[0]
    n += 1
    write(f"p-artifact-{n}", {"kind": "planet", "title": f"Practice star P{n}", "prompt": PROMPT_PLANET,
                              "stats": planet_stats(r), "answer": "no", "answer_detail": "Not confirmed: noise.",
                              "explanation": "Looked plausible in one sector, but six more TESS sectors show nothing at this period. "
                                             "Clue in this plot: 30-minute dips are too short for a 9-day orbit around an M dwarf, and one 'transit' sits in a jittery stretch.",
                              "source": "TESS 2-min"}, ROOT / "planets" / r["plot"])
    return n


def injected_planets(start_n):
    """Fake planets in real quiet-star light curves, at moderate strength."""
    import injection_recovery as ir
    import transit_search as ts

    stars = pd.read_csv(ROOT / "planets/quiet_stars.csv")
    cases = [(0, 3.7, 900, "a 2.9 R⊕ planet"), (2, 1.9, 1500, "a 1.6 R⊕ planet"), (1, 6.3, 450, "a 2.4 R⊕ planet")]
    tmp = OUT / "tmp"
    n = start_n
    for star_i, period, depth_ppm, what in cases:
        star = stars.iloc[star_i]
        lc = ts.download_light_curve(star["target"], sectors=[int(star["sectors"])]).remove_nans().normalize()
        duration = ir.transit_duration_days(period, star["stellar_density"], 0.3)
        lc = ir.inject_transit(lc, period, lc.time.value.min() + 0.7 * period, depth_ppm / 1e6, duration, 0.3)
        n += 1
        r = ts.search_light_curve(lc, name=f"Practice star P{n}", plot_dir=tmp, min_period=0.3)
        write(f"p-injected-{n}", {"kind": "planet", "title": f"Practice star P{n}", "prompt": PROMPT_PLANET,
                                  "stats": planet_stats(r), "answer": "yes",
                                  "answer_detail": f"A planet signal: {what} we injected into this real star's data.",
                                  "explanation": "Evenly spaced dips of the same depth, a flat baseline around a flat-bottomed fold, "
                                                 "and a duration that fits the period. This is what the pipeline is looking for.",
                                  "source": "TESS 2-min + injected"}, r["plot"])
    shutil.rmtree(tmp, ignore_errors=True)
    return n


# ---------------------------------------------------------------------------
# Comets
# ---------------------------------------------------------------------------

def comet_stats(m):
    return [
        {"label": "Seen in", "value": f"{int(m['hits'])} frames"},
        {"label": "Signal", "value": f"{m['snr']:.1f} σ"},
        {"label": "Motion vs stars", "value": f"{m['speed_vs_stars']:.1f} px/h"},
        {"label": "Sunward", "value": f"{m['sunward']:+.2f}"},
        {"label": "Track scatter", "value": f"{m['rms_px']:.2f} px"},
    ]


PROMPT_COMET = "Is this a comet moving through the field, or something else?"


def comet_examples():
    import inject
    import scan
    import soho
    import track

    frames = soho.list_frames(dt.datetime(2026, 10, 23, 0, 0), dt.datetime(2026, 10, 23, 8, 0))
    stack = soho.load_frames(frames, progress=False).astype(np.float32)
    hours = soho.frame_hours(frames)
    sun = track.sun_centre(stack)

    def tracks_for(s):
        diff = track.difference_stack(s)
        tr = track.link(track.detect(diff, track.valid_mask(s)), hours)
        return track.classify_tracks(tr, sun, track.star_drift(tr)), diff

    tracks, diff = tracks_for(stack)
    n = 0
    items = []

    # The confirmed comet (Sungrazer report_cf_20261023080539).
    real = tracks[(tracks["kind"] == "mover") & (np.hypot(tracks["x_ref"] - 797, tracks["y_ref"] - 794) < 15)].iloc[0]
    items.append((real, diff, stack, "yes", "A real comet (Kreutz sungrazer).",
                  "Confirmed by a Sungrazer Project report the same morning. A faint fuzzy dot that stays in the circle in "
                  "every close-up, moving steadily toward the Sun (sunward +0.98), much faster than the stars."))
    # Stars: move exactly with the star field.
    for _, s in tracks[tracks["kind"] == "star"].sort_values("snr", ascending=False).iloc[[3, 12]].iterrows():
        items.append((s, diff, stack, "no", "A background star.",
                      f"It stays in the circle, but its motion relative to the stars is ~{s['speed_vs_stars']:.1f} px/h: "
                      "it drifts exactly like every other star as SOHO orbits the Sun. A comet moves differently from the star field."))
    # Fake comets injected into the same frames.
    rng = np.random.default_rng(5)
    noise = float(np.median(track.local_noise(diff[len(diff) // 2], track.valid_mask(stack))[track.valid_mask(stack)]))
    fakes = inject.draw_fakes(6, stack, hours, noise, rng, speed_range=(6, 20), snr_range=(6, 10))
    stack_f = inject.inject(stack, hours, fakes).astype(np.float32)
    tracks_f, diff_f = tracks_for(stack_f)
    found = inject.match(fakes, tracks_f)
    for (_, f), ok in zip(fakes.iterrows(), found):
        if not ok or sum(1 for it in items if it[3] == "yes") >= 3:
            continue
        movers = tracks_f[tracks_f["kind"] == "mover"]
        t = movers.iloc[np.argmin(np.hypot(movers["x_ref"] - (f["x0"] + f["vx"] * movers["t_ref"]),
                                           movers["y_ref"] - (f["y0"] + f["vy"] * movers["t_ref"])))]
        items.append((t, diff_f, stack_f, "yes", "A comet-like mover (a fake we injected).",
                      "A dot that keeps its place in the circle frame after frame while moving across the stars "
                      f"at {t['speed_vs_stars']:.0f} px/h: what a SOHO comet looks like."))

    for t, d, s, answer, detail, why in items:
        n += 1
        path = OUT / f"c-{n}.tmp.png"
        OUT.mkdir(exist_ok=True)
        scan.track_figure(s, d, hours, frames, t, sun, path)
        write(f"c-{n}", {"kind": "comet", "title": f"Practice track C{n}", "prompt": PROMPT_COMET,
                         "stats": comet_stats(t), "answer": answer, "answer_detail": detail, "explanation": why,
                         "source": "SOHO LASCO C3"}, path)
        path.unlink()
    return n


if __name__ == "__main__":
    shutil.rmtree(OUT, ignore_errors=True)
    n = planet_examples()
    n = injected_planets(n)
    c = comet_examples()
    # Shuffle the display order so answers don't come in blocks.
    rng = np.random.default_rng(11)
    files = sorted(OUT.glob("*.json"))
    for order, path in zip(rng.permutation(len(files)), files):
        item = json.loads(path.read_text())
        item["order"] = int(order)
        path.write_text(json.dumps(item, indent=1, ensure_ascii=False))
    print(f"{n} planet and {c} comet practice examples in {OUT}")
