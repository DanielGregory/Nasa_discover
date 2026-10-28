"""Offline tests for the known-signal matching and size checks in vetting.py."""

import numpy as np
import pandas as pd

import vetting as v

KNOWN = pd.DataFrame({
    "tic": [100, 100, 200],
    "period": [6.27, 2.0, 3.0],
    "name": ["Star-100 b", "TOI-1.02", "TOI-2.01"],
    "source": ["confirmed", "TOI", "TOI"],
    "disposition": ["CP", "PC", "PC"],
    "ra": [10.0, 10.0, 10.0 + 30 / 3600],     # star 200 is 30" east of star 100
    "dec": [0.0, 0.0, 0.0],
})


def test_same_period_is_known_including_aliases():
    for period in (6.27, 12.54, 3.135):
        same, other, _ = v.match_known(100, period, 10.0, 0.0, KNOWN)
        assert list(same["name"]) == ["Star-100 b"]
        assert list(other["name"]) == ["TOI-1.02"]


def test_new_period_on_known_host_is_not_rejected():
    same, other, _ = v.match_known(100, 4.4, 10.0, 0.0, KNOWN)
    assert same.empty
    assert len(other) == 2          # reported as "already hosts", not a match


def test_signal_from_nearby_star_is_flagged():
    _, _, neighbour = v.match_known(100, 3.0, 10.0, 0.0, KNOWN)
    assert list(neighbour["name"]) == ["TOI-2.01"]
    _, _, far = v.match_known(100, 3.0, 10.0, 1.0, KNOWN)   # 1 degree away
    assert far.empty


def test_planet_radius_from_depth():
    # Earth across the Sun: ~84 ppm.
    radius, _ = v.planet_radius(84, 1.0, 0.05)
    assert np.isclose(radius, 1.0, atol=0.01)
    # A 25% eclipse can't be a planet.
    radius, _ = v.planet_radius(250_000, 1.0)
    assert radius > v.MAX_PLANET_RADIUS_EARTH


def test_duration_check():
    # Earth-Sun: 13 h central transit.
    assert v.duration_check(365.25, 13, 1.0) == ""
    assert "too long" in v.duration_check(365.25, 25, 1.0)
    assert "grazing" in v.duration_check(365.25, 2, 1.0)
    # M dwarf (density 10), 9-day orbit: ~1.8 h central.
    assert "too long" in v.duration_check(9, 4, 10.0)
