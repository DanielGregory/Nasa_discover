"""
Get SOHO LASCO coronagraph images.

LASCO is a set of coronagraphs on the SOHO spacecraft: a disk blocks the
Sun so the faint corona, and anything passing near the Sun, shows up.
  C2: 2-6 solar radii from the Sun's centre, a close-up view
  C3: 3.7-30 solar radii, wide view; most SOHO comets are found here

Frames are taken about every 12 minutes. We fetch them through the
Helioviewer API (api.helioviewer.org), which serves every LASCO frame since
1996 as a 1024x1024 JPEG 2000 image. Pixel values are the processed brightness
scaled to 0-255 (not calibrated physical units). That's fine for finding
moving dots, which is what Sungrazer hunters do by eye.

Downloaded frames are cached in comets/cache/frames/.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

API = "https://api.helioviewer.org/v2"
SOURCES = {"C2": 4, "C3": 5}
CACHE = Path(__file__).resolve().parent / "cache"

# C3 plate scale: 56 arcsec per pixel at 1024x1024; the Sun is ~17 px across.
ARCSEC_PER_PIXEL = {"C2": 11.9, "C3": 56.0}


def _get(path, params, retries=3):
    for attempt in range(retries):
        try:
            response = requests.get(f"{API}/{path}/", params=params, timeout=60)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def list_frames(start, end, camera="C3", step_minutes=6):
    """
    All frames between `start` and `end` (datetimes, UTC), as a DataFrame
    with columns id, date, and geometry (sun centre, rotation, scale).

    Helioviewer has no "list" call, so we ask for the frame closest to each
    time on a fine grid and keep the unique ones. Results are cached.
    """
    CACHE.mkdir(exist_ok=True)
    key = f"frames_{camera}_{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}.json"
    cached = CACHE / key
    if cached.exists():
        return pd.read_json(cached, convert_dates=["date"])

    rows = {}
    t = start
    while t <= end:
        info = _get("getClosestImage", {"date": f"{t:%Y-%m-%dT%H:%M:%S}Z",
                                        "sourceId": SOURCES[camera]}).json()
        if "id" in info:
            rows[info["id"]] = info
        t += dt.timedelta(minutes=step_minutes)

    frames = pd.DataFrame(rows.values())
    frames["date"] = pd.to_datetime(frames["date"])
    frames = frames[(frames["date"] >= start) & (frames["date"] <= end)]
    frames = frames.sort_values("date").reset_index(drop=True)
    frames.to_json(cached, date_format="iso")
    return frames


def load_frame(frame_id):
    """One frame as a 2-D float array (0-255), downloading if needed."""
    path = CACHE / "frames" / f"{frame_id}.jp2"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get("getJP2Image", {"id": frame_id}).content)
    # The JPEG 2000 file holds the 8-bit grayscale brightness directly. (The
    # PNG version is colourised with an unordered palette, so its pixel
    # values aren't brightness.)
    return np.asarray(Image.open(path), dtype=float)


def load_frames(frames, progress=True):
    """Stack of frames, shape (n_frames, 1024, 1024)."""
    stack = []
    for i, frame_id in enumerate(frames["id"]):
        stack.append(load_frame(frame_id))
        if progress and (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(frames)} frames", flush=True)
    return np.stack(stack)


def frame_hours(frames):
    """Time of each frame in hours since the first one."""
    t = frames["date"]
    return ((t - t.iloc[0]).dt.total_seconds() / 3600).to_numpy()
