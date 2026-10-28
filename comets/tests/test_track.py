"""Offline tests for the comet tracker on synthetic frames."""

import numpy as np

import inject
import track


def synthetic_sequence(n=16, size=256, seed=0):
    """Noise + a disk 'occulter' + drifting stars, like a tiny C3 field."""
    rng = np.random.default_rng(seed)
    hours = np.arange(n) * 0.2                      # 12-minute cadence
    yy, xx = np.mgrid[:size, :size]
    occulter = np.hypot(xx - size / 2, yy - size / 2) < 20
    stars = rng.uniform(30, size - 30, (40, 2))
    stack = np.empty((n, size, size))
    for i, t in enumerate(hours):
        frame = 100 + rng.normal(0, 3, (size, size))
        for sx, sy in stars:
            inject.add_blob(frame, sx + 3.0 * t, sy, 40)   # stars drift 3 px/h in x
        frame[occulter] = 0
        stack[i] = frame
    return stack, hours


def test_tracker_finds_moving_object_and_ignores_stars():
    stack, hours = synthetic_sequence()
    fake = {"x0": 200.0, "y0": 60.0, "vx": -12.0, "vy": 10.0}
    for i, t in enumerate(hours):
        inject.add_blob(stack[i], fake["x0"] + fake["vx"] * t, fake["y0"] + fake["vy"] * t, 25)
    tracks = inject.run_tracker(stack, hours)
    movers = tracks[tracks["kind"] == "mover"]
    assert len(movers) >= 1
    best = movers.sort_values("snr", ascending=False).iloc[0]
    assert abs(best["vx"] - fake["vx"]) < 1.5 and abs(best["vy"] - fake["vy"]) < 1.5
    # Everything else moving should be classified as stars.
    assert (tracks["kind"] == "star").sum() >= 20
    assert best["sunward"] > 0.5                    # heading toward the centre


def test_no_movers_without_injection():
    stack, hours = synthetic_sequence(seed=1)
    tracks = inject.run_tracker(stack, hours)
    assert (tracks["kind"] == "mover").sum() == 0


def test_star_drift_is_the_common_velocity():
    import pandas as pd
    v = np.vstack([np.random.default_rng(2).normal([3, 0], 0.1, (30, 2)), [[-10, 5], [8, 8]]])
    tracks = pd.DataFrame({"vx": v[:, 0], "vy": v[:, 1]})
    drift = track.star_drift(tracks)
    assert np.allclose(drift, [3, 0], atol=0.2)
