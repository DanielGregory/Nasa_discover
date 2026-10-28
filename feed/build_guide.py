"""
Annotated example images for the Training tab's "How to read" guide.

    python feed/build_guide.py        # writes feed/guide_out/*.png

Each image is a real pipeline figure with callouts drawn in the plot's own
coordinates (so arrows land on the actual features): green = what a real
detection looks like, red = the giveaway of a fake.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "feed" / "guide_out"
sys.path[:0] = [str(ROOT / "planets"), str(ROOT / "comets")]

GOOD, BAD = "#1e7d4f", "#b3261e"


def callout(ax, xy, text, xytext, good=True, coords="data"):
    """Text box at `xytext` (axes fraction) with an arrow to `xy`."""
    colour = GOOD if good else BAD
    ax.annotate(("✓ " if good else "✗ ") + text, xy=xy, xycoords=coords,
                xytext=xytext, textcoords="axes fraction", fontsize=12.5, fontweight="bold",
                color="white", ha="left", va="center", zorder=20,
                bbox=dict(boxstyle="round,pad=0.35", fc=colour, ec="none", alpha=0.93),
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.2, shrinkA=0, shrinkB=2))


def note(ax, text, xytext, good=True):
    colour = GOOD if good else BAD
    ax.text(*xytext, ("✓ " if good else "✗ ") + text, transform=ax.transAxes, fontsize=12.5,
            fontweight="bold", color="white", ha="left", va="center", zorder=20,
            bbox=dict(boxstyle="round,pad=0.35", fc=colour, ec="none", alpha=0.93))


def banner(fig, text, good):
    fig.suptitle(text, fontsize=17, fontweight="bold", color=GOOD if good else BAD)


# ---------------------------------------------------------------------------
# Planets
# ---------------------------------------------------------------------------

def planet_figure(tic, sectors=None, min_period=0.5, mask=()):
    import transit_search as ts
    lc = ts.download_light_curve(tic, sectors=sectors)
    result, fig = ts.search_light_curve(lc, name=tic, min_period=min_period, mask_signals=mask,
                                        return_figure=True)
    for ax in fig.axes:
        if ax.get_legend():
            ax.get_legend().remove()
    return result, fig, fig.axes[:4]


def transit_times(r, ax):
    lo, hi = ax.get_xlim()
    k = np.arange(np.ceil((lo - r["t0_btjd"]) / r["period_days"]), np.floor((hi - r["t0_btjd"]) / r["period_days"]) + 1)
    return r["t0_btjd"] + k * r["period_days"]


def real_planet():
    r, fig, (raw, flat, pg, fold) = planet_figure("TIC 261136679")          # Pi Men c
    banner(fig, "REAL PLANET (Pi Mensae c): what good looks like", True)
    gap = 1339.1
    callout(raw, (gap, 1.0), "Even band of noise.\nThe gap = TESS sending data home", (0.55, 0.85))
    times = transit_times(r, flat)
    depth = r["depth_ppm"] / 1e6
    for tc in times[1:3]:
        callout(flat, (tc, 1 - 1.6 * depth), "A dip at EVERY red line,\nall about the same depth", (0.30, 0.13))
    p = r["period_days"]
    callout(pg, (p, pg.get_ylim()[1] * 0.93), "One tall, sharp spike", (0.35, 0.82))
    callout(pg, (p * 2, pg.get_ylim()[1] * 0.55), "Echoes at 2× (and ½×)\nare normal", (0.72, 0.62))
    callout(fold, (0, 1 - depth), "Flat-bottomed dip", (0.06, 0.18))
    callout(fold, (2.3, 1.0), "Flat outside the dip", (0.62, 0.86))
    return fig


def glitch():
    r, fig, (raw, flat, pg, fold) = planet_figure("TIC 142086812", min_period=0.3)
    banner(fig, "NOT A PLANET: one glitch doing all the work", False)
    callout(raw, (1348.5, 0.985), "Jittery, spiky stretch", (0.40, 0.15), good=False)
    times = transit_times(r, flat)
    x, y = (np.asarray(v, dtype=float) for v in flat.lines[0].get_data())
    t_low, f_low = x[np.nanargmin(y)], np.nanmin(y)          # the deepest point plotted
    deepest = times[np.argmin(np.abs(times - t_low))]
    callout(flat, (deepest, f_low), "One spike carries\nthe whole signal", (0.55, 0.13), good=False)
    others = [t for t in times if abs(t - deepest) > 1]
    for k, t in enumerate(others[:2]):
        callout(flat, (t, 1.0), "Nothing at this red line", (0.05 + 0.33 * k, 0.88), good=False)
    callout(pg, (r["period_days"], pg.get_ylim()[1] * 0.9), "Peak rises out of a messy,\nclimbing forest", (0.25, 0.80), good=False)
    callout(fold, (0, 1 - r["depth_ppm"] / 1e6 * 0.9), "'Dip' = a few stray points,\nscattered baseline", (0.58, 0.25), good=False)
    return fig


def edge_ramp():
    r, fig, (raw, flat, pg, fold) = planet_figure("TIC 3657911", min_period=0.3)
    banner(fig, "NOT A PLANET: dips at the edges of the data", False)
    lo, hi = raw.get_xlim()
    times = [t for t in transit_times(r, flat) if lo <= t <= hi]
    callout(flat, (times[0], 0.9985), "Right at the start\nof the data", (0.12, 0.13), good=False)
    callout(flat, (times[-1], 0.9985), "Right at the end", (0.72, 0.13), good=False)
    if len(times) > 2:
        callout(flat, (times[1], 1.0005), "This one falls in the gap:\nno data", (0.45, 0.88), good=False)
    callout(raw, (times[0], 1.003), "Data ramps where\neach orbit starts", (0.20, 0.88), good=False)
    callout(fold, (0, 1 - r["depth_ppm"] / 1e6), "Shallow, noisy box", (0.62, 0.20), good=False)
    return fig


def pulsator():
    r, fig, (raw, flat, pg, fold) = planet_figure("TIC 93716498", min_period=0.3)
    banner(fig, "NOT A PLANET: the star pulsates", False)
    note(flat, "Red lines everywhere: a very short 'period'", (0.25, 0.92), good=False)
    callout(fold, (-1.6, 1.0004), "Baseline waves up and down,\nnot flat", (0.03, 0.88), good=False)
    callout(fold, (0, 1 - r["depth_ppm"] / 1e6), "The 'dip' is just one\ntrough of the wave", (0.62, 0.15), good=False)
    callout(pg, (r["period_days"], pg.get_ylim()[1] * 0.9), "Comb of many similar peaks", (0.35, 0.82), good=False)
    return fig


# ---------------------------------------------------------------------------
# Comets
# ---------------------------------------------------------------------------

def comet_setup():
    import scan
    import soho
    import track
    frames = soho.list_frames(dt.datetime(2026, 10, 23, 0, 0), dt.datetime(2026, 10, 23, 8, 0))
    stack = soho.load_frames(frames, progress=False).astype(np.float32)
    hours = soho.frame_hours(frames)
    sun = track.sun_centre(stack)
    diff = track.difference_stack(stack)
    tr = track.link(track.detect(diff, track.valid_mask(stack)), hours)
    tracks = track.classify_tracks(tr, sun, track.star_drift(tr))

    def figure(t, title, good, notes):
        fig = scan.track_figure(stack, diff, hours, frames, t, sun, None)
        fig.set_size_inches(12, 7.4)
        fig.subplots_adjust(top=0.80, bottom=0.10)
        banner(fig, title, good)
        for text, ok, (x, y) in notes:
            fig.text(x, y, ("✓ " if ok else "✗ ") + text,
                     fontsize=12, fontweight="bold", color="white",
                     bbox=dict(boxstyle="round,pad=0.35", fc=GOOD if ok else BAD, ec="none"))
        return fig

    def pick(kind, x, y):
        cand = tracks[tracks["kind"] == kind]
        return cand.iloc[np.argmin(np.hypot(cand["x_ref"] - x, cand["y_ref"] - y))]

    real = pick("mover", 797, 794)
    star = tracks[tracks["kind"] == "star"].sort_values("snr", ascending=False).iloc[3]
    wander = pick("mover", 838, 440)
    return [
        ("comet_real", figure(real, "REAL COMET: what good looks like", True, [
            ("Heads toward the Sun (the + in the inset)", True, (0.01, 0.87)),
            ("Faint (5σ) is normal for SOHO comets", True, (0.37, 0.87)),
            (f"Moves {real['speed_vs_stars']:.0f} px/h vs the stars (stars ≈ 0)", True, (0.01, 0.035)),
            ("A dot in the circle in EVERY close-up", True, (0.37, 0.035))])),
        ("comet_star", figure(star, "NOT A COMET: it's a background star", False, [
            ("Close-ups look perfect...", True, (0.37, 0.035)),
            (f"...but it moves {star['speed_vs_stars']:.1f} px/h vs the stars: it moves WITH them", False, (0.01, 0.87))])),
        ("comet_wander", figure(wander, "NOT A COMET: the tracker joined unrelated spots", False, [
            ("Moves away from the Sun", False, (0.01, 0.87)),
            ("Sits next to a pair of stars", False, (0.37, 0.87)),
            ("Dot drifts off-centre: left at 04:43, right at 05:06", False, (0.37, 0.035))])),
    ]


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, make in [("planet_real", real_planet), ("planet_glitch", glitch),
                       ("planet_edge", edge_ramp), ("planet_pulsator", pulsator)]:
        fig = make()
        fig.savefig(OUT / f"{name}.png", dpi=85)
        plt.close(fig)
        print("wrote", name)
    for name, fig in comet_setup():
        fig.savefig(OUT / f"{name}.png", dpi=90)
        plt.close(fig)
        print("wrote", name)
