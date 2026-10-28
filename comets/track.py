"""
Find faint moving objects (comets) in a sequence of LASCO frames.

What a Sungrazer hunter does by eye, step by step:

  1. Remove what doesn't move. The corona and the occulter's glare change
     slowly, so for each frame subtract the median of its neighbouring
     frames. What remains: noise, stars (drifting slowly), and anything
     moving faster, like a comet.
  2. Find bright spots. Smooth the difference image a little (a comet is
     a fuzzy few-pixel blob), estimate the local noise, and keep local
     maxima well above it.
  3. Link spots into tracks. A real object appears in frame after frame
     along a straight line at constant speed (over a few hours). Noise
     and cosmic rays don't line up, so we require most frames in a short
     window to agree with one straight-line motion.
  4. Reject the background sky. Stars all drift together (SOHO orbits the
     Sun with Earth), so the most common track velocity is the star drift.
     Anything moving with it, or not moving at all, is dropped.

What's left are candidate moving objects, scored by how significant and
how consistent they are, and whether they head toward the Sun (most SOHO
comets are Kreutz sungrazers diving toward the Sun).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# 1. Remove what doesn't move
# ---------------------------------------------------------------------------

def valid_mask(stack, saturated=250, edge=6):
    """
    Pixels usable in every frame: not behind the occulter or pylon, not off
    the edge of the field, and not next to saturated objects (planets and
    their bleed streaks).
    """
    good = np.all(stack > 0, axis=0)
    good = ndimage.binary_erosion(good, iterations=edge)
    bright = ndimage.binary_dilation(np.any(stack >= saturated, axis=0), iterations=8)
    return good & ~bright


def sun_centre(stack):
    """Centre of the occulting disk: the centroid of the dark central hole."""
    dark = np.all(stack == 0, axis=0)
    h, w = dark.shape
    labels, _ = ndimage.label(dark)
    centre_label = labels[h // 2, w // 2]
    if centre_label == 0:
        return w / 2, h / 2
    y, x = np.nonzero(labels == centre_label)
    return x.mean(), y.mean()


def difference_stack(stack, gap=4, half_window=10):
    """
    Each frame minus the median of frames between `gap` and `half_window`
    frames away on either side (about 48 min to 2 h at 12-min cadence).
    Static and slowly changing structure (corona, glare) cancels.

    Skipping the nearest frames matters for slow movers: a comet crawling
    2-3 px/hour is still in almost the same place 12-36 minutes later, so
    using those frames as "background" would subtract the comet itself.
    """
    n = len(stack)
    diff = np.empty_like(stack)
    for i in range(n):
        neighbours = [j for j in range(i - half_window, i + half_window + 1)
                      if 0 <= j < n and abs(j - i) >= gap]
        diff[i] = stack[i] - np.median(stack[neighbours], axis=0)
    return diff


# ---------------------------------------------------------------------------
# 2. Find bright spots
# ---------------------------------------------------------------------------

def local_noise(image, mask, block=32):
    """Robust noise (MAD -> sigma) in blocks, smoothed into a full map."""
    h, w = image.shape
    sigma = np.full((h // block, w // block), np.nan)
    for by in range(h // block):
        for bx in range(w // block):
            sl = (slice(by * block, (by + 1) * block), slice(bx * block, (bx + 1) * block))
            values = image[sl][mask[sl]]
            if values.size > block:
                sigma[by, bx] = 1.4826 * np.median(np.abs(values - np.median(values)))
    typical = np.nanmedian(sigma)
    sigma[np.isnan(sigma)] = typical
    # Floor: blocks that are mostly flat (e.g. clipped regions) would
    # otherwise give near-zero noise and absurd significances.
    sigma = np.maximum(sigma, 0.3 * typical)
    return np.kron(sigma, np.ones((block, block)))[:h, :w]


def detect(diff, mask, threshold=4.5, smooth=1.2):
    """
    Spots in each difference frame more than `threshold` sigma above the
    local noise. Returns a DataFrame: frame, x, y, snr, peak.
    """
    rows = []
    for i, frame in enumerate(diff):
        smoothed = ndimage.gaussian_filter(frame, smooth)
        snr = smoothed / local_noise(smoothed, mask)
        peaks = (snr > threshold) & mask & \
            (smoothed == ndimage.maximum_filter(smoothed, size=5))
        y, x = np.nonzero(peaks)
        cx, cy = _centroids(smoothed, x, y)
        rows.append(pd.DataFrame({"frame": i, "x": cx, "y": cy,
                                  "snr": snr[y, x], "peak": frame[y, x]}))
    return pd.concat(rows, ignore_index=True)


def _centroids(image, x, y, half=2):
    """Sub-pixel positions: brightness-weighted centre of a 5x5 box around
    each peak (positive values only). Whole-pixel positions would make
    velocities over a short track too rough to tell stars from movers."""
    h, w = image.shape
    cx, cy = x.astype(float), y.astype(float)
    dy, dx = np.mgrid[-half:half + 1, -half:half + 1]
    for k, (px, py) in enumerate(zip(x, y)):
        if half <= px < w - half and half <= py < h - half:
            box = np.clip(image[py - half:py + half + 1, px - half:px + half + 1], 0, None)
            total = box.sum()
            if total > 0:
                cx[k] = px + (box * dx).sum() / total
                cy[k] = py + (box * dy).sum() / total
    return cx, cy


# ---------------------------------------------------------------------------
# 3. Link spots into straight-line tracks
# ---------------------------------------------------------------------------

def link(detections, hours, window=8, min_hits=6, max_speed=40.0, tolerance=1.5):
    """
    Find straight-line, constant-speed tracks.

    For every window of `window` consecutive frames, each pair of detections
    in nearby frames proposes a velocity. We predict where the object would
    be in the other frames and count how many have a detection within
    `tolerance` pixels. Tracks seen in at least `min_hits` frames are kept.
    `max_speed` is in pixels per hour.

    Returns one row per track: reference time and position, velocity (px/h),
    hits, mean SNR, and the matched detection indices.
    """
    by_frame = {f: g for f, g in detections.groupby("frame")}
    trees = {f: cKDTree(g[["x", "y"]].to_numpy()) for f, g in by_frame.items()}
    n_frames = len(hours)
    tracks = []

    for start in range(0, n_frames - window + 1):
        frames = [f for f in range(start, start + window) if f in by_frame]
        if len(frames) < min_hits:
            continue
        for a_i, a in enumerate(frames[:2]):           # seed from the first frames
            for b in frames[a_i + 1:a_i + 3]:
                dt = hours[b] - hours[a]
                if dt <= 0:
                    continue
                A, B = by_frame[a], by_frame[b]
                pairs = trees[a].query_ball_tree(trees[b], r=max_speed * dt)
                for ia, matches in enumerate(pairs):
                    for ib in matches:
                        xa, ya = A.iloc[ia][["x", "y"]]
                        xb, yb = B.iloc[ib][["x", "y"]]
                        vx, vy = (xb - xa) / dt, (yb - ya) / dt
                        hits = []
                        for f in frames:
                            px = xa + vx * (hours[f] - hours[a])
                            py = ya + vy * (hours[f] - hours[a])
                            d, k = trees[f].query([px, py])
                            if d <= tolerance:
                                hits.append(by_frame[f].index[k])
                        if len(hits) >= min_hits:
                            tracks.append(_fit_track(detections.loc[hits], hours))
    if not tracks:
        return pd.DataFrame(columns=["t_ref", "x_ref", "y_ref", "vx", "vy", "hits", "snr"])
    return _merge_duplicates(pd.DataFrame(tracks))


def _fit_track(points, hours):
    """Least-squares straight line through a track's detections."""
    t = hours[points["frame"].to_numpy()]
    t_ref = t.mean()
    vx, x0 = np.polyfit(t - t_ref, points["x"], 1)
    vy, y0 = np.polyfit(t - t_ref, points["y"], 1)
    resid = np.hypot(points["x"] - (x0 + vx * (t - t_ref)), points["y"] - (y0 + vy * (t - t_ref)))
    rms = float(np.sqrt(np.mean(resid ** 2)))
    # Velocity uncertainty of a straight-line fit: position scatter over
    # the spread of times (with a floor of 0.2 px for centroid error).
    v_err = max(rms, 0.2) / (np.std(t) * np.sqrt(len(t)))
    return {"t_ref": t_ref, "x_ref": x0, "y_ref": y0, "vx": vx, "vy": vy,
            "v_err": v_err, "hits": len(points), "snr": points["snr"].mean(),
            "rms_px": rms,
            "first_frame": int(points["frame"].min()), "last_frame": int(points["frame"].max()),
            "detections": list(points.index)}


def _merge_duplicates(tracks, position_tol=4.0, velocity_tol=1.5):
    """The same object is found from many seed pairs and windows: keep one
    (the one with most hits) per object."""
    tracks = tracks.sort_values(["hits", "snr"], ascending=False).reset_index(drop=True)
    keep = []
    for i, row in tracks.iterrows():
        duplicate = False
        for j in keep:
            other = tracks.loc[j]
            dt = row["t_ref"] - other["t_ref"]
            # Compare positions at the same moment.
            dx = row["x_ref"] - (other["x_ref"] + other["vx"] * dt)
            dy = row["y_ref"] - (other["y_ref"] + other["vy"] * dt)
            if np.hypot(dx, dy) < position_tol + 0.5 * abs(dt) and \
                    np.hypot(row["vx"] - other["vx"], row["vy"] - other["vy"]) < velocity_tol:
                duplicate = True
                break
        if not duplicate:
            keep.append(i)
    return tracks.loc[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Reject the background sky and score what's left
# ---------------------------------------------------------------------------

def star_drift(tracks):
    """
    The velocity shared by most tracks: the stars. Found as the densest
    point in velocity space (a 2-D histogram peak), refined by the median
    of tracks near it.
    """
    if len(tracks) < 5:
        return np.array([np.nan, np.nan])
    v = tracks[["vx", "vy"]].to_numpy()
    tree = cKDTree(v)
    counts = np.array([len(tree.query_ball_point(p, r=0.5)) for p in v])
    centre = v[np.argmax(counts)]
    near = np.hypot(*(v - centre).T) < 0.5
    return np.median(v[near], axis=0)


def classify_tracks(tracks, sun_xy, drift, star_tol=0.5, min_speed=0.6):
    """
    Label each track: 'star' (moves with the star field, within 3x its
    velocity uncertainty + `star_tol` px/h), 'static' (barely moves), or
    'mover' (a candidate). Adds speed, direction relative to the Sun, and
    a 0-1 score.
    """
    tracks = tracks.copy()
    v = tracks[["vx", "vy"]].to_numpy()
    tracks["speed"] = np.hypot(*v.T)
    rel = v - drift if np.all(np.isfinite(drift)) else v
    tracks["speed_vs_stars"] = np.hypot(*rel.T)

    # Angle between motion (relative to the stars) and the direction to the Sun.
    to_sun = np.column_stack([sun_xy[0] - tracks["x_ref"], sun_xy[1] - tracks["y_ref"]])
    cos = np.sum(rel * to_sun, axis=1) / (np.hypot(*rel.T) * np.hypot(*to_sun.T) + 1e-9)
    tracks["sunward"] = cos            # +1 = straight at the Sun
    tracks["r_sun_px"] = np.hypot(*to_sun.T)

    like_stars = tracks["speed_vs_stars"] < star_tol + 3 * tracks["v_err"]
    tracks["kind"] = np.where(tracks["speed"] < min_speed + 3 * tracks["v_err"], "static",
                              np.where(like_stars, "star", "mover"))
    hit_frac = tracks["hits"] / (tracks["last_frame"] - tracks["first_frame"] + 1)
    tracks["score"] = np.round(
        np.clip((tracks["snr"] - 4) / 6, 0, 1) * 0.5
        + np.clip(hit_frac, 0, 1) * 0.2
        + np.clip(1 - tracks["rms_px"] / 2, 0, 1) * 0.15
        + np.clip(tracks["sunward"], 0, 1) * 0.15, 3)
    tracks.loc[tracks["kind"] != "mover", "score"] = 0.0
    return tracks


def centring(diff, hours, mover, box=6, smooth=1.2, tolerance=2.0):
    """
    Does one object actually follow the fitted track? In each frame, find
    the brightest spot within `box` px of the predicted position (in the
    smoothed difference image) and measure how far it is from that
    position. A real mover stays within ~1 px; a track stitched from
    unrelated spots (star residuals, noise) wanders by several px.
    Returns (median offset in px, fraction of frames within `tolerance` px).
    """
    offsets = []
    for i in range(int(mover["first_frame"]), int(mover["last_frame"]) + 1):
        dt_h = hours[i] - mover["t_ref"]
        x = int(round(mover["x_ref"] + mover["vx"] * dt_h))
        y = int(round(mover["y_ref"] + mover["vy"] * dt_h))
        if not (box <= x < diff.shape[2] - box and box <= y < diff.shape[1] - box):
            continue
        patch = ndimage.gaussian_filter(diff[i, y - box - 3:y + box + 4, x - box - 3:x + box + 4], smooth)[3:-3, 3:-3]
        yy, xx = np.unravel_index(np.argmax(patch), patch.shape)
        offsets.append(np.hypot(xx - box, yy - box))
    if not offsets:
        return np.nan, 0.0
    offsets = np.array(offsets)
    return float(np.median(offsets)), float(np.mean(offsets <= tolerance))
