"""
Vetting: is a BLS signal plausibly a *new* planet?

A detection alone means little. Before anyone looks at a candidate we ask:

  1. How big would the planet be?  depth ≈ (R_planet / R_star)², with R_star
     from the TESS Input Catalog (TIC). Anything over ~2 Jupiter radii is a
     star, not a planet (an eclipsing binary).
  2. Could a neighbour star be the source?  TESS pixels are 21″ wide, so
     nearby stars share the aperture. A neighbour can only produce the dip if
     it contributes at least `depth` of the total light (a 100% eclipse of
     it). We list every neighbour bright enough to do that.
  3. Is it already known?  Cross-match against confirmed planets (NASA
     Exoplanet Archive), TOIs (Archive) and community TOIs (CTOIs, ExoFOP),
     by TIC ID *and period*. A known host with a *different* period is not
     rejected: an extra planet in a known system is a great find.

vet(result) runs all three and returns flags plus a one-line summary.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import transit_search as ts

EARTH_RADII_PER_SUN = 109.1
EARTH_RADII_PER_JUPITER = 11.2
MAX_PLANET_RADIUS_EARTH = 2 * EARTH_RADII_PER_JUPITER   # bigger = a star

TESS_PIXEL_ARCSEC = 21.0
NEIGHBOUR_RADIUS_ARCSEC = 3 * TESS_PIXEL_ARCSEC          # typical aperture reach

ARCHIVE_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
CTOI_CSV = "https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_MAX_AGE_DAYS = 1.0


# ---------------------------------------------------------------------------
# 1. Star and planet size (TIC)
# ---------------------------------------------------------------------------

def tic_number(target):
    """'TIC 261136679' or 261136679 -> 261136679."""
    return int(str(target).upper().replace("TIC", "").strip())


def star_info(target):
    """Stellar parameters from the TESS Input Catalog, via MAST."""
    from astroquery.mast import Catalogs

    row = Catalogs.query_criteria(catalog="TIC", ID=tic_number(target))[0]
    return {
        "tic": tic_number(target),
        "ra": float(row["ra"]), "dec": float(row["dec"]),
        "tmag": float(row["Tmag"]), "teff": float(row["Teff"]),
        "stellar_radius": float(row["rad"]), "stellar_radius_err": float(row["e_rad"]),
        "stellar_mass": float(row["mass"]),
        "stellar_density": float(row["rho"]),       # in solar units
        # TIC's estimate of the fraction of light in the aperture from other
        # stars, relative to the target (0 = isolated).
        "contamination_ratio": float(row["contratio"]),
    }


def planet_radius(depth_ppm, stellar_radius, stellar_radius_err=np.nan):
    """
    Planet radius in Earth radii from transit depth, with uncertainty from the
    stellar radius only. This is a lower limit if light from other stars
    dilutes the dip.
    """
    ratio = np.sqrt(max(depth_ppm, 0) / 1e6)
    radius = EARTH_RADII_PER_SUN * stellar_radius * ratio
    err = EARTH_RADII_PER_SUN * stellar_radius_err * ratio
    return radius, err


# ---------------------------------------------------------------------------
# 2. Neighbour stars that could fake the signal
# ---------------------------------------------------------------------------

def neighbours_that_could_cause(star, depth_ppm, radius_arcsec=NEIGHBOUR_RADIUS_ARCSEC):
    """
    TIC stars within `radius_arcsec` bright enough to produce the observed
    depth by being totally eclipsed. Their flux relative to the target is
    10^(-0.4 * ΔTmag); if that is below the depth, the neighbour can't
    produce the dip even with a 100% eclipse.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astroquery.mast import Catalogs

    centre = SkyCoord(star["ra"], star["dec"], unit="deg")
    near = Catalogs.query_region(centre, radius=radius_arcsec * u.arcsec,
                                 catalog="TIC").to_pandas()
    near = near[near["ID"].astype(int) != star["tic"]].copy()
    near["separation_arcsec"] = near["dstArcSec"]
    near["flux_ratio"] = 10 ** (-0.4 * (near["Tmag"] - star["tmag"]))
    # Fraction of the *total* light this neighbour supplies.
    near["light_fraction"] = near["flux_ratio"] / (1 + near["flux_ratio"])
    able = near[near["light_fraction"] >= depth_ppm / 1e6]
    return able[["ID", "Tmag", "separation_arcsec", "light_fraction"]] \
        .sort_values("light_fraction", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Is it already known?
# ---------------------------------------------------------------------------

def _cached_csv(name, fetch):
    """Download a catalogue at most once a day; keep a copy in cache/."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{name}.csv"
    if not path.exists() or (time.time() - path.stat().st_mtime) / 86400 > CACHE_MAX_AGE_DAYS:
        path.write_text(fetch())
    return pd.read_csv(path)


def _archive_query(adql):
    response = requests.get(ARCHIVE_TAP, params={"query": adql, "format": "csv"}, timeout=120)
    response.raise_for_status()
    return response.text


def known_signals():
    """
    One table of every known signal: confirmed planets, TOIs and CTOIs, with
    columns tic, period, name, source, disposition, ra, dec.
    """
    planets = _cached_csv("confirmed_planets", lambda: _archive_query(
        "select pl_name, tic_id, pl_orbper, ra, dec from pscomppars where tic_id is not null"))
    tois = _cached_csv("tois", lambda: _archive_query(
        "select toi, tid, pl_orbper, ra, dec, tfopwg_disp from toi"))
    ctois = _cached_csv("ctois", lambda: requests.get(CTOI_CSV, timeout=120).text)

    frames = [
        pd.DataFrame({"tic": planets["tic_id"].str.replace("TIC ", "").astype(int),
                      "period": planets["pl_orbper"], "name": planets["pl_name"],
                      "source": "confirmed", "disposition": "CP",
                      "ra": planets["ra"], "dec": planets["dec"]}),
        pd.DataFrame({"tic": tois["tid"].astype(int), "period": tois["pl_orbper"],
                      "name": "TOI-" + tois["toi"].astype(str), "source": "TOI",
                      "disposition": tois["tfopwg_disp"], "ra": tois["ra"], "dec": tois["dec"]}),
        pd.DataFrame({"tic": ctois["TIC ID"].astype(int), "period": ctois["Period (days)"],
                      "name": "CTOI-" + ctois["CTOI"].astype(str), "source": "CTOI",
                      "disposition": ctois["User Disposition"],
                      "ra": ctois["RA"], "dec": ctois["Dec"]}),
    ]
    return pd.concat(frames, ignore_index=True)


def match_known(tic, period, ra, dec, known=None, radius_arcsec=NEIGHBOUR_RADIUS_ARCSEC):
    """
    Returns (same_signal, other_signals_on_star, signals_on_neighbours), each
    a DataFrame of known signals:
      same_signal          same star, same period (or 2x / 1/2x alias)
      other_signals_on_star  same star, different period -> possible new planet
      signals_on_neighbours  a known signal on a star near enough to leak in
    """
    known = known_signals() if known is None else known
    on_star = known[known["tic"] == tic]
    is_same = on_star["period"].apply(lambda p: ts.period_match(period, p) not in ("", "miss"))
    # Neighbours: small-angle separation in arcsec.
    dra = (known["ra"] - ra) * np.cos(np.radians(dec))
    sep = 3600 * np.hypot(dra, known["dec"] - dec)
    nearby = known[(sep < radius_arcsec) & (known["tic"] != tic)]
    nearby_same = nearby[nearby["period"].apply(
        lambda p: ts.period_match(period, p) not in ("", "miss"))]
    return on_star[is_same], on_star[~is_same], nearby_same


def duration_check(period_days, duration_hours, stellar_density):
    """
    Is the transit duration possible for a planet around this star?
    A central transit lasts ~13 h x (P / 1 yr)^(1/3) x density^(-1/3)
    (Kepler's law; density in solar units). Grazing transits are shorter,
    so only "much too long" is strong evidence; "very short" just needs a
    grazing orbit and is worth noting.
    """
    if not np.isfinite(stellar_density) or stellar_density <= 0:
        return ""
    central = 13.0 * (period_days / 365.25) ** (1 / 3) * stellar_density ** (-1 / 3)
    if duration_hours > 1.5 * central:
        return (f"{duration_hours:.1f} h is too long for a planet here (max ~{central:.1f} h): "
                "maybe a blended binary or wrong stellar parameters")
    if duration_hours < 0.3 * central:
        return f"{duration_hours:.1f} h is short (central transit ~{central:.1f} h): needs a grazing orbit"
    return ""


# ---------------------------------------------------------------------------
# 4. Data artifacts
# ---------------------------------------------------------------------------

def artifact_reasons(result):
    """
    Reasons to think the signal is bad data rather than an astrophysical
    dip, from transit_search.transit_quality(). Thresholds come from
    looking at the impostors in our first real hunt.
    """
    reasons = []
    if result.get("n_transits", 0) < 2:
        reasons.append("only one transit has data")
    if result.get("max_event_share", 0) > 0.7:
        reasons.append(f"one transit carries {result['max_event_share']:.0%} of the signal")
    if result.get("bad_data_fraction", 0) >= 0.5:
        reasons.append("most transits sit at a data gap, sector edge or noisy stretch")
    if result.get("median_mean_ratio", 1) < 0.5:
        reasons.append("dip comes from a few extreme points, not a smooth transit")
    if result.get("baseline_wiggle", 0) > 0.4:
        reasons.append("the star varies at this period outside the 'transit' (pulsation or spots?)")
    return reasons


# ---------------------------------------------------------------------------
# All checks together
# ---------------------------------------------------------------------------

def vet(result, known=None):
    """
    Vet one search result (a row from ts.search_target / search_many).
    Returns a dict of vetting columns, including `vet_status`:
      known            matches a known planet/TOI/CTOI on this star
      known neighbour  matches a known signal on a nearby star (contamination)
      artifact         bad data (one event, edge/noisy transits, spiky dip) or
                       a wavy baseline (stellar variability, not a transit)
      not a planet     too big to be a planet, or odd/even depths differ
      possible new     a new signal (on a known host, says `vet_notes`)
      weak             doesn't pass the candidate thresholds
    """
    star = star_info(result["target"])
    depth = result["depth_ppm"]
    radius, radius_err = planet_radius(depth, star["stellar_radius"], star["stellar_radius_err"])
    same, other, neighbour_match = match_known(star["tic"], result["period_days"],
                                               star["ra"], star["dec"], known)
    blenders = neighbours_that_could_cause(star, depth)

    notes = []
    if radius > MAX_PLANET_RADIUS_EARTH:
        notes.append(f"radius {radius:.0f} R⊕ is too big for a planet")
    if result["odd_even_sigma"] > 3 and result["odd_even_diff_pct"] > 20:
        notes.append("odd/even depths differ (eclipsing binary at 2x period?)")
    if len(blenders):
        notes.append(f"{len(blenders)} neighbour(s) bright enough to cause the dip")
    if len(other):
        notes.append("star already hosts: " + ", ".join(other["name"].astype(str)))
    artifact = artifact_reasons(result)
    notes += artifact
    duration_note = duration_check(result["period_days"], result["duration_hours"],
                                   star["stellar_density"])
    if duration_note:
        notes.append(duration_note)

    if len(same):
        status = "known"
        notes.insert(0, "matches " + ", ".join(same["name"].astype(str)))
    elif len(neighbour_match):
        status = "known neighbour"
        notes.insert(0, "same period as " + ", ".join(neighbour_match["name"].astype(str))
                     + " on a nearby star")
    elif not result["is_candidate"]:
        status = "weak"
    elif artifact:
        status = "artifact"
    elif radius > MAX_PLANET_RADIUS_EARTH or "odd/even" in " ".join(notes):
        status = "not a planet"
    else:
        status = "possible new"

    return {
        **{k: star[k] for k in ("tmag", "teff", "stellar_radius", "contamination_ratio")},
        "planet_radius_earth": radius,
        "planet_radius_err": radius_err,
        "n_bright_neighbours": len(blenders),
        "vet_status": status,
        "vet_notes": "; ".join(notes),
    }
