"""
Injection-recovery for the comet tracker, the same idea as for planets:
add fake comets with known positions, speeds and brightness to real LASCO
frames, run the unchanged tracker, and count what comes back.

Many fakes go into one sequence at once (spread apart), so one tracker run
tests a dozen or more of them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import track


def add_blob(image, x, y, amplitude, sigma=1.2):
    """Add a fuzzy round spot (2-D Gaussian) centred at (x, y), in place."""
    h, w = image.shape
    r = int(4 * sigma) + 1
    x0, y0 = int(round(x)), int(round(y))
    if not (-r < x0 < w + r and -r < y0 < h + r):
        return                                   # off the image
    ys, xs = np.mgrid[max(y0 - r, 0):min(y0 + r + 1, h), max(x0 - r, 0):min(x0 + r + 1, w)]
    blob = amplitude * np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))
    # Nothing shows through the occulter, the pylon or outside the field.
    image[ys, xs] += np.where(image[ys, xs] > 0, blob, 0)


def draw_fakes(n, stack, hours, noise, rng, speed_range=(2.0, 25.0), snr_range=(2.0, 12.0)):
    """
    Random fake comets: start positions inside the usable field (and not
    too close to each other), speeds in px/hour mostly toward the Sun with
    some scatter, and peak brightness = snr x the frame-difference noise.
    """
    mask = track.valid_mask(stack)
    sun = track.sun_centre(stack)
    span = hours[-1] - hours[0]
    ys, xs = np.nonzero(mask)
    fakes = []
    while len(fakes) < n:
        k = rng.integers(len(xs))
        x, y = float(xs[k]), float(ys[k])
        speed = np.exp(rng.uniform(*np.log(speed_range)))
        to_sun = np.arctan2(sun[1] - y, sun[0] - x)
        angle = to_sun + rng.normal(0, 0.6)
        vx, vy = speed * np.cos(angle), speed * np.sin(angle)
        # Start position at t = 0 so the path is centred on (x, y).
        x0, y0 = x - vx * span / 2, y - vy * span / 2
        if any(np.hypot(x - f["x_mid"], y - f["y_mid"]) < 60 for f in fakes):
            continue
        fakes.append({"x0": x0, "y0": y0, "x_mid": x, "y_mid": y, "vx": vx, "vy": vy,
                      "speed": speed, "snr_peak": np.exp(rng.uniform(*np.log(snr_range)))})
    fakes = pd.DataFrame(fakes)
    fakes["amplitude"] = fakes["snr_peak"] * noise
    return fakes


def inject(stack, hours, fakes):
    """Copy of the stack with every fake comet drawn into every frame."""
    out = stack.copy()
    for _, f in fakes.iterrows():
        for i, t in enumerate(hours):
            add_blob(out[i], f["x0"] + f["vx"] * t, f["y0"] + f["vy"] * t, f["amplitude"])
    return np.clip(out, 0, 255)


def run_tracker(stack, hours):
    """The full tracker, exactly as the search uses it."""
    diff = track.difference_stack(stack)
    mask = track.valid_mask(stack)
    detections = track.detect(diff, mask)
    tracks = track.link(detections, hours)
    drift = track.star_drift(tracks)
    return track.classify_tracks(tracks, track.sun_centre(stack), drift)


def match(fakes, tracks, pos_tol=3.0, vel_tol=2.0):
    """Was each fake found as a 'mover' with the right position and velocity?"""
    movers = tracks[tracks["kind"] == "mover"]
    found = []
    for _, f in fakes.iterrows():
        ok = False
        for _, t in movers.iterrows():
            x = f["x0"] + f["vx"] * t["t_ref"]
            y = f["y0"] + f["vy"] * t["t_ref"]
            if np.hypot(t["x_ref"] - x, t["y_ref"] - y) < pos_tol and \
                    np.hypot(t["vx"] - f["vx"], t["vy"] - f["vy"]) < vel_tol:
                ok = True
                break
        found.append(ok)
    return np.array(found)


def injection_test(stack, hours, n_fakes=20, n_runs=5, seed=0):
    """Several runs of `n_fakes` fakes each. One row per fake, with `recovered`."""
    rng = np.random.default_rng(seed)
    diff = track.difference_stack(stack)
    mask = track.valid_mask(stack)
    noise = float(np.median(track.local_noise(diff[len(diff) // 2], mask)[mask]))
    rows = []
    for run in range(n_runs):
        fakes = draw_fakes(n_fakes, stack, hours, noise, rng)
        tracks = run_tracker(inject(stack, hours, fakes), hours)
        fakes["recovered"] = match(fakes, tracks)
        fakes["run"] = run
        rows.append(fakes)
        print(f"  run {run + 1}/{n_runs}: {fakes['recovered'].sum()}/{n_fakes} recovered", flush=True)
    return pd.concat(rows, ignore_index=True)
