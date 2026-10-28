"""Tests for injection_recovery.py, plus regression tests for two pipeline
bugs that injection-recovery runs exposed."""

import numpy as np

import injection_recovery as ir
import transit_search as ts
from tests.test_synthetic import fake_light_curve


def test_duration_matches_sun_like_star():
    # Earth around the Sun: ~13 h. Pi Men c (P = 6.27 d): ~3.3 h.
    assert abs(ir.transit_duration_days(365.25) * 24 - 13) < 0.01
    assert abs(ir.transit_duration_days(6.27) * 24 - 3.35) < 0.05
    # Denser star or grazing transit -> shorter.
    assert ir.transit_duration_days(5, stellar_density=10) < ir.transit_duration_days(5)
    assert ir.transit_duration_days(5, impact=0.8) < ir.transit_duration_days(5)


def test_inject_transit_has_requested_depth():
    lc = fake_light_curve(depth=0, noise=1e-9)
    flat_before = lc.flux.value.copy()
    injected = ir.inject_transit(lc, period=3.0, t0=1326.0, depth=1e-3,
                                 duration=0.15)
    ratio = injected.flux.value / flat_before
    assert abs(1 - ratio.min() - 1e-3) < 1e-9
    assert np.isclose(np.median(ratio), 1.0)


def test_short_period_deep_planet_recovered():
    """Regression: BLS used to prefer an unphysical 8-hour box at ~1 d over
    this real 0.8-day, 1.7-hour transit (fixed by MAX_DUTY_CYCLE)."""
    lc = fake_light_curve(depth=0).normalize()
    lc = ir.inject_transit(lc, period=0.79975, t0=1325.433, depth=1.417e-3,
                           duration=1.66 / 24, impact=0.18)
    result = ts.search_light_curve(lc)
    assert ts.period_match(result["period_days"], 0.79975) == "match"
    assert result["is_candidate"]
    # Both passes should measure close to the injected depth (box-averaged
    # over ingress/egress, so a little shallower).
    assert 1200 < result["depth_ppm_blind"] < 1500
    assert 1200 < result["depth_ppm"] < 1500


def test_injections_classified():
    lc = fake_light_curve(depth=0)
    results = ir.run_injections(lc, n=4, seed=3, n_workers=2)
    assert len(results) == 4
    assert set(results["status"]) <= {"recovered", "alias", "other signal",
                                      "below threshold"}
    assert results["expected_snr"].notna().all()


def test_search_multi_finds_second_planet():
    """A deep planet shouldn't hide a shallower one at another period."""
    lc = fake_light_curve(period=3.1, depth=2e-3, duration_days=2 / 24, noise=1e-4)
    lc = ir.inject_transit(lc, period=7.7, t0=1327.2, depth=6e-4, duration=3 / 24)
    results = ts.search_multi(lc, name="two planets")
    found = [r["period_days"] for r in results if r["is_candidate"]]
    assert len(found) >= 2
    assert ts.period_match(found[0], 3.1) == "match"
    assert ts.period_match(found[1], 7.7) == "match"
    assert results[1]["signal"] == 2 and results[1]["target"] == "two planets"
