"""
Scan recent SOHO LASCO frames for moving objects and send candidates to the
Discovery Queue.

    python scan.py                       # last 24 hours of C3
    python scan.py --hours 48 --camera C3
    python scan.py --start 2026-10-23T00:00 --hours 10

Frames are processed in overlapping chunks (8 h, stepping 6 h). Each
'mover' becomes a card with a figure: the full frame with the track
marked, plus close-ups following the object frame by frame.

Movers that match a recent Sungrazer Project report (same place and
motion) are labelled "reported" rather than "possible new".
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

import soho  # noqa: E402
import track  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "feed"))
import candidates as cards  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
MAX_SPEED = {"C2": 150.0, "C3": 40.0}       # px/hour; C2 pixels are ~5x smaller
SUN_RADIUS_ARCSEC = 960.0
REPORTS_URL = "https://sungrazer.nrl.navy.mil/index.php/plain-text-reports"


# ---------------------------------------------------------------------------
# Known comets: recent Sungrazer reports
# ---------------------------------------------------------------------------

def sungrazer_reports(pages=3):
    """
    Recent reports from the Sungrazer Project's plain-text page: one row per
    reported position (date, camera, time, x, y). Positions are on the
    project's north-up 1024x1024 images.
    """
    rows = []
    for page in range(pages):
        text = requests.get(REPORTS_URL, params={"page": page}, timeout=60).text
        text = re.sub(r"<script.*?</script>|<style.*?</style>", "", text, flags=re.S)
        lines = [ln.strip() for ln in html.unescape(re.sub(r"<[^>]+>", "\n", text)).split("\n")
                 if ln.strip()]
        report = None
        for line in lines:
            if line.startswith("report_"):
                report = {"report": line}
            elif report is not None and re.match(r"^\d{4}-\d{2}-\d{2}$", line):
                report["date"] = line
            elif report is not None and line in ("C2", "C3", "CCOR1", "CCOR2"):
                report["camera"] = line
            elif report is not None and re.match(r"^\d{4} \d+ \d+$", line) and "date" in report:
                hhmm, x, y = line.split()
                rows.append({**report, "time": pd.Timestamp(f"{report['date']} {hhmm[:2]}:{hhmm[2:]}"),
                             "x": int(x), "y": int(y)})
    return pd.DataFrame(rows)


def to_north_up(x, y, sun_xy, rotation_deg, centre=(512, 512)):
    """
    Convert our (Helioviewer, camera-oriented) pixel position to the
    Sungrazer convention: north up, Sun at the image centre. Approximate
    (a few pixels); good enough to recognise a known comet.
    """
    a = np.radians(rotation_deg)
    dx, dy = x - sun_xy[0], y - sun_xy[1]
    return (centre[0] + dx * np.cos(a) - dy * np.sin(a),
            centre[1] + dx * np.sin(a) + dy * np.cos(a))


def match_reports(mover_time, xy_north_up, reports, camera, tol_px=40, tol_hours=1.0):
    """Names of Sungrazer reports with a position near this mover's."""
    if reports.empty:
        return []
    near = reports[(reports["camera"] == camera) &
                   ((reports["time"] - mover_time).abs() <= pd.Timedelta(hours=tol_hours))]
    hits = near[np.hypot(near["x"] - xy_north_up[0], near["y"] - xy_north_up[1]) < tol_px]
    return sorted(set(hits["report"]))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def stretch(image, low=1, high=99.7):
    lo, hi = np.percentile(image[image > 0], [low, high]) if np.any(image > 0) else (0, 1)
    return np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)


def track_figure(stack, hours, frames, mover, sun_xy, path, n_cutouts=8, half=32):
    """Full frame with the track marked, plus close-ups along the track."""
    first, last = int(mover["first_frame"]), int(mover["last_frame"])
    chosen = np.unique(np.linspace(first, last, min(n_cutouts, last - first + 1)).astype(int))
    mid = chosen[len(chosen) // 2]

    n_cols = (len(chosen) + 1) // 2
    fig = plt.figure(figsize=(12, 6.4))
    grid = fig.add_gridspec(2, 1 + n_cols, width_ratios=[2.2] + [1] * n_cols)
    ax = fig.add_subplot(grid[:, 0])
    ax.imshow(stretch(stack[mid]) ** 0.7, cmap="gray", origin="upper")
    t = hours[[first, last]] - mover["t_ref"]
    xs, ys = mover["x_ref"] + mover["vx"] * t, mover["y_ref"] + mover["vy"] * t
    ax.annotate("", xy=(xs[1], ys[1]), xytext=(xs[0], ys[0]),
                arrowprops=dict(arrowstyle="->", color="#f0a640", lw=2))
    ax.add_patch(plt.Circle((xs.mean(), ys.mean()), 40, fill=False, color="#f0a640", lw=1.5))
    ax.plot(*sun_xy, "+", color="#7aa2ff", ms=10)
    ax.set_title(f"{frames['date'].iloc[mid]:%Y-%m-%d %H:%M} UT (arrow: motion)", fontsize=10)
    ax.axis("off")

    for k, i in enumerate(chosen):
        sub = fig.add_subplot(grid[k // n_cols, 1 + k % n_cols])   # left to right, then down
        dt_h = hours[i] - mover["t_ref"]
        cx, cy = mover["x_ref"] + mover["vx"] * dt_h, mover["y_ref"] + mover["vy"] * dt_h
        x0, y0 = int(round(cx)) - half, int(round(cy)) - half
        cut = np.zeros((2 * half, 2 * half))
        ys_, xs_ = slice(max(y0, 0), max(min(y0 + 2 * half, 1024), 0)), slice(max(x0, 0), max(min(x0 + 2 * half, 1024), 0))
        piece = stack[i][ys_, xs_]
        cut[ys_.start - y0:ys_.start - y0 + piece.shape[0], xs_.start - x0:xs_.start - x0 + piece.shape[1]] = piece
        sub.imshow(stretch(cut, 2, 99.5), cmap="gray", origin="upper")
        sub.plot(half, half, marker="o", ms=14, mfc="none", mec="#f0a640", mew=1.2)
        sub.set_title(f"{frames['date'].iloc[i]:%H:%M}", fontsize=8)
        sub.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def scan_chunk(frames, camera, reports):
    stack = soho.load_frames(frames, progress=False).astype(np.float32)
    hours = soho.frame_hours(frames)
    sun_xy = track.sun_centre(stack)
    diff = track.difference_stack(stack)
    mask = track.valid_mask(stack)
    detections = track.detect(diff, mask)
    tracks = track.link(detections, hours, max_speed=MAX_SPEED[camera])
    tracks = track.classify_tracks(tracks, sun_xy, track.star_drift(tracks))
    movers = tracks[tracks["kind"] == "mover"].copy()

    rows = []
    rotation = float(frames["rotation"].iloc[0])
    r_sun_per_px = soho.ARCSEC_PER_PIXEL[camera] / SUN_RADIUS_ARCSEC
    for _, m in movers.iterrows():
        when = frames["date"].iloc[0] + pd.Timedelta(hours=float(m["t_ref"]))
        north_up = to_north_up(m["x_ref"], m["y_ref"], sun_xy, rotation)
        known = match_reports(when, north_up, reports, camera)
        rows.append({**m.drop(labels=["detections"]).to_dict(), "camera": camera, "time": when,
                     "x_north_up": north_up[0], "y_north_up": north_up[1],
                     "r_sun": m["r_sun_px"] * r_sun_per_px,
                     "known_reports": ", ".join(known), "sun_x": sun_xy[0], "sun_y": sun_xy[1]})
        tag = f"{camera}-{when:%Y%m%dT%H%M}-x{int(m['x_ref'])}-y{int(m['y_ref'])}"
        rows[-1]["plot"] = str(OUT / "plots" / f"{tag}.png")
        rows[-1]["id"] = cards.card_id("comet", tag)
        (OUT / "plots").mkdir(parents=True, exist_ok=True)
        track_figure(stack, hours, frames, m, sun_xy, rows[-1]["plot"])
    return rows


def comet_card(row):
    status = "reported" if row["known_reports"] else "possible new"
    speed_rsun = row["speed_vs_stars"] * soho.ARCSEC_PER_PIXEL[row["camera"]] / SUN_RADIUS_ARCSEC
    toward = "toward the Sun" if row["sunward"] > 0.7 else ("away from the Sun" if row["sunward"] < -0.7 else "across the field")
    return {
        "id": row["id"],
        "kind": "comet",
        "title": f"LASCO {row['camera']} mover, {row['time']:%b %d %H:%M} UT",
        "subtitle": f"{row['r_sun']:.1f} R☉ from the Sun · {speed_rsun:.2f} R☉/h {toward}",
        "score": float(row["score"]) * (0.3 if status == "reported" else 1.0),
        "summary": ("Matches a position in a recent Sungrazer report, so it's probably a known comet."
                    if status == "reported" else
                    "A faint object moving in a straight line against the stars, not matched to a recent Sungrazer report."),
        "stats": [
            {"label": "Seen in", "value": f"{int(row['hits'])} frames"},
            {"label": "Signal", "value": f"{row['snr']:.1f} σ"},
            {"label": "Motion", "value": f"{row['speed_vs_stars']:.1f} px/h vs stars"},
            {"label": "Sunward", "value": f"{row['sunward']:+.2f}"},
            {"label": "Track scatter", "value": f"{row['rms_px']:.2f} px"},
            {"label": "Position (north-up)", "value": f"x {row['x_north_up']:.0f}, y {row['y_north_up']:.0f}"},
        ],
        "notes": (f"Matches {row['known_reports']}." if row["known_reports"] else
                  "Check: does it look fuzzy rather than sharp (cosmic ray or star residual)? Does it keep moving in later frames?"),
        "vet_status": status,
        "links": [{"label": "Sungrazer reports", "url": REPORTS_URL},
                  {"label": "Helioviewer at this time",
                   "url": f"https://helioviewer.org/?date={row['time']:%Y-%m-%dT%H:%M:%S}.000Z&imageScale=56&imageLayers=%5BSOHO,LASCO,{row['camera']},white-light,1,100%5D"}],
        "source": f"SOHO LASCO {row['camera']} via Helioviewer",
        "found_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", default="C3", choices=["C2", "C3"])
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--start", help="UTC start, e.g. 2026-10-23T00:00 (default: now - hours)")
    args = parser.parse_args()

    end = dt.datetime.utcnow() if not args.start else \
        dt.datetime.fromisoformat(args.start) + dt.timedelta(hours=args.hours)
    start = end - dt.timedelta(hours=args.hours)
    frames = soho.list_frames(start, end, camera=args.camera)
    print(f"{len(frames)} {args.camera} frames from {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}")
    reports = sungrazer_reports()
    print(f"{len(reports)} positions in recent Sungrazer reports")

    rows = []
    chunk, step = pd.Timedelta(hours=8), pd.Timedelta(hours=6)
    t = frames["date"].min()
    while t < frames["date"].max():
        part = frames[(frames["date"] >= t) & (frames["date"] < t + chunk)].reset_index(drop=True)
        if len(part) >= 10:
            found = scan_chunk(part, args.camera, reports)
            print(f"  {t:%m-%d %H:%M}: {len(part)} frames, {len(found)} mover(s)", flush=True)
            rows += found
        t += step

    OUT.mkdir(exist_ok=True)
    table = pd.DataFrame(rows)
    if table.empty:
        print("No movers found.")
        sys.exit(0)
    # Overlapping chunks find the same object twice: keep the best per place and time.
    table = table.sort_values("score", ascending=False)
    table["bucket"] = [f"{r.time:%Y%m%d%H}-{int(r.x_north_up // 30)}-{int(r.y_north_up // 30)}" for r in table.itertuples()]
    table = table.drop_duplicates("bucket")
    table.to_csv(OUT / "movers.csv", index=False)
    for _, row in table.iterrows():
        cards.write_outbox(comet_card(row), row["plot"])
    print(f"{len(table)} mover card(s) written to feed/outbox/ "
          f"({(table['known_reports'] != '').sum()} match Sungrazer reports)")
