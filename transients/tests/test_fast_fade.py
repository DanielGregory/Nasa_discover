"""Offline tests: the fade-rate logic separates a kilonova-like light curve
from a normal supernova."""

import numpy as np
import pandas as pd

import fast_fade as ff


def light_curve(times, mags, band="r", err=0.05):
    return pd.DataFrame({"mjd": times, "band": band, "mag": mags, "err": err})


def test_kilonova_like_fade_is_flagged():
    # AT2017gfo-like: peaks at 17.5 and fades ~0.8 mag/day in r.
    t = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0])
    lc = pd.concat([light_curve(t, 17.5 + 0.8 * t, "r"),
                    light_curve(t, 17.3 + 1.1 * t, "g")])
    a = ff.assess(lc)
    assert ff.is_fast_fader(a)
    assert a["fade_band"] == "g" and abs(a["fade_rate"] - 1.1) < 0.05
    assert a["g_minus_r"] > 0.5            # reddening as it fades


def test_supernova_is_not_flagged():
    # SN 2023ixf (ZTF23aaklqou) faded ~0.03 mag/day after peak.
    t = np.arange(0, 20, 2.0)
    a = ff.assess(light_curve(t, 12.0 + 0.03 * t))
    assert not ff.is_fast_fader(a)


def test_noisy_short_baseline_not_flagged():
    # Two points 0.3 days apart can't define a fade rate.
    a = ff.assess(light_curve(np.array([0.0, 0.3]), np.array([19.0, 19.5]), err=0.3))
    assert not ff.is_fast_fader(a)
