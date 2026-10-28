"""
Offline sanity checks: inject a known transit into fake data and make sure
the pipeline gets it back. No internet needed. Run with:  pytest -q
"""

import lightkurve as lk
import numpy as np
import pandas as pd

import transit_search as ts


def fake_light_curve(period=6.27, depth=300e-6, duration_days=3 / 24,
                     t0=1.5, noise=150e-6, days=27.0, seed=42):
    """A TESS-like sector: 2-min cadence, white noise, spot modulation, a
    mid-sector data gap, and box-shaped transits."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, days, 2 / 60 / 24) + 1325.0
    t = t[np.abs(t - t.mean()) > 0.5]                 # downlink gap
    stellar = 1 + 2e-3 * np.sin(2 * np.pi * t / 3.7)   # starspots, P_rot=3.7 d
    phase = (t - (1325.0 + t0) + period / 2) % period - period / 2
    transit = np.where(np.abs(phase) < duration_days / 2, 1 - depth, 1.0)
    flux = stellar * transit + rng.normal(0, noise, t.size)
    flux[rng.choice(t.size, 20)] += 5e-3              # cosmic-ray spikes
    return lk.LightCurve(time=t, flux=flux, flux_err=np.full(t.size, noise))


def test_recovers_injected_period_depth_duration():
    lc = fake_light_curve()
    result = ts.search_light_curve(lc, name="injected")
    assert ts.period_match(result["period_days"], 6.27) == "match"
    assert abs(result["depth_ppm"] - 300) < 60
    assert abs(result["duration_hours"] - 3) <= 1
    assert result["is_candidate"]
    assert result["odd_even_sigma"] < 3


def test_pure_noise_is_not_a_candidate():
    lc = fake_light_curve(depth=0)
    result = ts.search_light_curve(lc, name="noise")
    assert not result["is_candidate"]


def test_flattening_preserves_transit_depth():
    """The robust trend shouldn't absorb transits, with or without a transit
    mask, even on a spotted star with cosmic-ray spikes. (lightkurve's
    Savitzky-Golay flatten() lost ~20-75% of the depth here.)"""
    lc = fake_light_curve(depth=1e-3, noise=1e-4)
    period, t0, duration = 6.27, 1326.5, 3 / 24
    phase = (lc.time.value - t0 + period / 2) % period - period / 2
    in_transit = np.abs(phase) < duration / 4    # central part of each dip

    _, blind = ts.clean_light_curve(lc)
    mask = lc.create_transit_mask(period=period, transit_time=t0,
                                  duration=2 * duration)
    _, masked = ts.clean_light_curve(lc, transit_mask=mask)

    def measured_depth(flat):
        keep = np.isin(lc.time.value, flat.time.value)
        return 1 - np.median(flat.flux.value[in_transit[keep]])

    assert abs(measured_depth(blind) - 1e-3) < 1e-4
    assert abs(measured_depth(masked) - 1e-3) < 1e-4


def test_trend_does_not_step_at_gaps():
    """Each side of a data gap gets its own trend; no step leaks across."""
    t = np.concatenate([np.arange(0, 5, 0.002), np.arange(6, 11, 0.002)])
    flux = np.where(t < 5.5, 1.0, 1.01)          # offset after the gap
    trend = ts.robust_trend(t, flux)
    assert np.allclose(trend[t < 5.5], 1.0) and np.allclose(trend[t > 5.5], 1.01)


def test_period_match_flags_aliases():
    assert ts.period_match(6.27, 6.27) == "match"
    assert ts.period_match(12.54, 6.27) == "alias 2x"
    assert ts.period_match(3.135, 6.27) == "alias 1/2x"
    assert ts.period_match(4.0, 6.27) == "miss"


def test_search_many_writes_csv_and_records_errors(tmp_path, monkeypatch):
    """Stub out the download so the batch runner can be tested offline."""
    def fake_download(target, **kwargs):
        if target == "missing star":
            raise ValueError("No TESS data")
        return fake_light_curve(depth=1e-3, noise=1e-4)

    monkeypatch.setattr(ts, "download_light_curve", fake_download)
    targets = pd.DataFrame({"target": ["star A", "missing star"],
                            "known_period_days": [6.27, 3.0]})
    csv_path = tmp_path / "results.csv"
    ts.search_many(targets, csv_path=csv_path, plot_dir=tmp_path / "plots")

    table = pd.read_csv(csv_path).fillna("")
    assert list(table["period_check"]) == ["match", ""]
    assert table.loc[1, "error"].startswith("ValueError")
    assert (tmp_path / "plots" / "star_A.png").exists()
