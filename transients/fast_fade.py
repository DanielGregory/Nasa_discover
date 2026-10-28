"""
Look for fast-fading new transients (kilonova-like) in Zwicky Transient
Facility (ZTF) alerts, via the ALeRCE broker.

A kilonova is the glow from radioactive debris after two neutron stars
merge (the 2017 one, AT2017gfo, came with gravitational waves). What sets it
apart from the thousands of supernovae ZTF sees:

  it fades FAST    ~0.5-1 mag per day; supernovae fade ~0.01-0.1 mag/day
  it reddens fast  blue at first, very red within days
  it's short-lived gone within about a week or two

So: take objects ZTF first saw in the last few days that ALeRCE's image
classifier calls supernova-like (a real new source, not a variable star,
AGN, asteroid or artifact), fit how fast each one fades in each band, and
rank the fastest faders.

    python fast_fade.py                 # objects first seen in the last 7 days
    python fast_fade.py --days 14 --max-objects 400

Being honest about the odds: kilonovae are rare (a handful per year might be
bright enough for ZTF, mostly found by following gravitational-wave alerts).
Most fast faders are flares from stars, afterglows, or noisy light curves.
Treat this as a learning tool and a way to practise vetting.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
API = "https://api.alerce.online/ztf/v1"
BANDS = {1: "g", 2: "r", 3: "i"}
COLOURS = {"g": "#2a9d5c", "r": "#c2185b", "i": "#8a3fb8"}

MIN_DECLINE = 0.3        # mag/day: kilonova-like fading
MIN_DECLINE_SIGNIFICANCE = 3

sys.path.insert(0, str(HERE.parent / "feed"))
import candidates as cards  # noqa: E402


def get(path, params=None, retries=3):
    for attempt in range(retries):
        try:
            response = requests.get(f"{API}/{path}", params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def mjd_now():
    return (dt.datetime.now(dt.timezone.utc) - dt.datetime(1858, 11, 17, tzinfo=dt.timezone.utc)).total_seconds() / 86400


def recent_objects(days=7, max_objects=200, min_detections=3):
    """New objects ALeRCE's stamp classifier calls supernova-like."""
    rows, page = [], 1
    since = mjd_now() - days
    while len(rows) < max_objects:
        data = get("objects", {"classifier": "stamp_classifier", "class": "SN",
                               "firstmjd": [since, mjd_now()],
                               "ndet": [min_detections, 10000],
                               "page": page, "page_size": 100,
                               "order_by": "firstmjd", "order_mode": "DESC"})
        items = data.get("items", [])
        rows += items
        # ALeRCE doesn't always say whether there's a next page: keep going
        # while pages come back full.
        if len(items) < 100:
            break
        page += 1
    return pd.DataFrame(rows[:max_objects])


def light_curve(oid):
    """Detections as a DataFrame: mjd, band, mag, err."""
    det = get(f"objects/{oid}/lightcurve").get("detections", [])
    lc = pd.DataFrame(det)
    if lc.empty:
        return lc
    lc = lc[lc["isdiffpos"].astype(str).isin(["1", "t", "True", "true"])] if "isdiffpos" in lc else lc
    return pd.DataFrame({"mjd": lc["mjd"], "band": lc["fid"].map(BANDS),
                         "mag": lc["magpsf"], "err": lc["sigmapsf"]}).sort_values("mjd")


def decline_rates(lc, window_days=10):
    """
    Fading speed after peak in each band: a straight-line fit of magnitude
    vs time to points from the brightest point up to `window_days` later.
    Returns {band: (rate mag/day, uncertainty, n points)}; positive = fading.
    """
    rates = {}
    for band, g in lc.groupby("band"):
        peak = g["mag"].idxmin()
        after = g[(g["mjd"] >= g.loc[peak, "mjd"]) & (g["mjd"] <= g.loc[peak, "mjd"] + window_days)]
        if len(after) < 2 or np.ptp(after["mjd"]) < 0.5:
            continue
        w = 1 / np.maximum(after["err"].to_numpy(), 0.02) ** 2
        t = after["mjd"].to_numpy() - after["mjd"].mean()
        slope = np.sum(w * t * (after["mag"] - np.average(after["mag"], weights=w))) / np.sum(w * t ** 2)
        slope_err = 1 / np.sqrt(np.sum(w * t ** 2))
        rates[band] = (float(slope), float(slope_err), len(after))
    return rates


def assess(lc):
    """Fastest significant fade, colour, age. Returns a dict."""
    rates = decline_rates(lc)
    fastest = max(rates.items(), key=lambda kv: kv[1][0], default=(None, (np.nan, np.nan, 0)))
    band, (rate, err, n) = fastest
    g, r = lc[lc["band"] == "g"], lc[lc["band"] == "r"]
    colour = np.nan
    if len(g) and len(r):
        # g - r from the latest pair of g and r points within 1 day.
        pairs = [(gm, rm) for gt, gm in zip(g["mjd"], g["mag"]) for rt, rm in zip(r["mjd"], r["mag"])
                 if abs(gt - rt) < 1]
        if pairs:
            colour = pairs[-1][0] - pairs[-1][1]
    return {"fade_band": band, "fade_rate": rate, "fade_err": err, "fade_points": n,
            "fade_sigma": rate / err if err and np.isfinite(err) else np.nan,
            "g_minus_r": colour, "age_days": float(np.ptp(lc["mjd"])) if len(lc) else np.nan,
            "peak_mag": float(lc["mag"].min()) if len(lc) else np.nan,
            "n_detections": len(lc)}


def is_fast_fader(a):
    return bool(np.isfinite(a["fade_rate"]) and a["fade_rate"] >= MIN_DECLINE
                and a["fade_sigma"] >= MIN_DECLINE_SIGNIFICANCE)


def score(a):
    fade = np.clip((a["fade_rate"] - MIN_DECLINE) / 0.7, 0, 1)
    sig = np.clip((a["fade_sigma"] - MIN_DECLINE_SIGNIFICANCE) / 7, 0, 1)
    red = np.clip((a["g_minus_r"] if np.isfinite(a["g_minus_r"]) else 0) / 1.0, 0, 1)
    young = 1.0 if a["age_days"] <= 10 else 0.5
    return round(float((0.5 * fade + 0.3 * sig + 0.2 * red) * young), 3)


def plot(oid, lc, a, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    t0 = lc["mjd"].min()
    for band, g in lc.groupby("band"):
        ax.errorbar(g["mjd"] - t0, g["mag"], g["err"], fmt="o", ms=6, color=COLOURS.get(band, "k"),
                    mec="white", mew=1, capsize=0, label=f"ZTF {band}")
    ax.invert_yaxis()
    ax.set_xlabel(f"Days since first detection (MJD {t0:.2f})")
    ax.set_ylabel("Magnitude (brighter up)")
    ax.set_title(f"{oid}: fading {a['fade_rate']:.2f} ± {a['fade_err']:.2f} mag/day in {a['fade_band']}",
                 fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def write_card(obj, lc, a):
    oid = obj["oid"]
    path = OUT / "plots" / f"{oid}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plot(oid, lc, a, path)
    colour = f"{a['g_minus_r']:+.2f}" if np.isfinite(a["g_minus_r"]) else "—"
    card = {
        "id": cards.card_id("transient", oid),
        "kind": "transient",
        "title": oid,
        "subtitle": f"fading {a['fade_rate']:.2f} mag/day in {a['fade_band']} · g−r {colour} · {a['age_days']:.1f} days old",
        "score": score(a),
        "summary": "A new ZTF source fading much faster than a typical supernova, as a kilonova would.",
        "stats": [
            {"label": "Fade rate", "value": f"{a['fade_rate']:.2f} ± {a['fade_err']:.2f} mag/d"},
            {"label": "Band", "value": a["fade_band"]},
            {"label": "g − r", "value": colour},
            {"label": "Peak", "value": f"{a['peak_mag']:.1f} mag"},
            {"label": "Detections", "value": str(a["n_detections"])},
            {"label": "Position", "value": f"{obj['meanra']:.4f}, {obj['meandec']:+.4f}"},
        ],
        "notes": ("Most fast faders are stellar flares, GRB afterglows or noisy points. Check: is it "
                  "on a galaxy (not a star)? Is there a recent gravitational-wave alert covering this sky?"),
        "vet_status": "possible new",
        "links": [{"label": "ALeRCE explorer", "url": f"https://alerce.online/object/{oid}"},
                  {"label": "TNS search", "url": f"https://www.wis-tns.org/search?ra={obj['meanra']}&decl={obj['meandec']}&radius=5&coords_unit=arcsec"}],
        "source": "ZTF alerts via ALeRCE",
        "found_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    cards.write_outbox(card, path)
    return card


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=float, default=7)
    parser.add_argument("--max-objects", type=int, default=200)
    parser.add_argument("--max-cards", type=int, default=5)
    args = parser.parse_args()

    objects = recent_objects(args.days, args.max_objects)
    print(f"{len(objects)} new supernova-like objects in the last {args.days:g} days")
    rows = []
    for _, obj in objects.iterrows():
        try:
            lc = light_curve(obj["oid"])
        except requests.RequestException:
            continue
        if len(lc) < 3:
            continue
        a = assess(lc)
        rows.append({"oid": obj["oid"], "ra": obj["meanra"], "dec": obj["meandec"], **a,
                     "fast": is_fast_fader(a)})
    table = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / "assessed.csv", index=False)
    fast = table[table["fast"]].copy() if len(table) else table
    print(f"{len(table)} with usable light curves, {len(fast)} fading faster than {MIN_DECLINE} mag/day")
    if len(fast):
        fast["score"] = [score(r) for _, r in fast.iterrows()]
        fast = fast.sort_values("score", ascending=False)
        print(fast[["oid", "fade_rate", "fade_sigma", "g_minus_r", "age_days", "peak_mag", "score"]].round(2).head(10).to_string(index=False))
        for _, r in fast.head(args.max_cards).iterrows():
            obj = objects[objects["oid"] == r["oid"]].iloc[0]
            write_card(obj, light_curve(r["oid"]), r)
        print(f"{min(len(fast), args.max_cards)} card(s) written to feed/outbox/")
