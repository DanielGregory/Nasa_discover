"""
Injection-recovery tests: how sensitive is the transit search, really?

The idea: take a real light curve of a star with no known signal, add
("inject") a fake planet transit of known period and size, run the *exact
same* search we'd run for real, and see whether the fake planet comes back.
Repeat a few hundred times across many periods and sizes, and you get a
**sensitivity map**: the fraction of planets of each period/size the
pipeline would have found around this star.

Why it matters:
  - "No planet found" only means something if you know what you could have
    found. A map lets you say "we'd have recovered 90% of 2 R_earth planets
    with P < 5 d around this star, and didn't see one."
  - It exposes pipeline weaknesses (flattening eating long transits,
    aliases, edge effects) with ground truth to compare against.

Main pieces:
  transit_duration_days()  -> physically sensible duration for (P, star)
  inject_transit()         -> add a trapezoid-shaped transit to a light curve
  run_injections()         -> many injections in parallel, one row each
  plot_recovery_map()      -> recovery fraction vs period and planet radius
  plot_recovery_vs_snr()   -> recovery fraction vs expected signal-to-noise
  find_quiet_stars()       -> pick test stars from the TESS Input Catalog
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import transit_search as ts

EARTH_RADII_PER_SUN = 109.1

# Sequential blue ramp (light = low recovery, dark = high recovery).
RECOVERY_CMAP = LinearSegmentedColormap.from_list(
    "recovery", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


# ---------------------------------------------------------------------------
# Making a fake planet
# ---------------------------------------------------------------------------

def transit_duration_days(period, stellar_density=1.0, impact=0.0):
    """
    Approximate total transit duration for a circular orbit.

    For a central transit across the Sun, a planet at P = 365 d takes ~13 h,
    and duration scales as P^(1/3) * density^(-1/3) (Kepler's third law).
    A non-central transit (impact parameter b > 0) crosses a shorter chord,
    shrinking the duration by sqrt(1 - b^2).

    stellar_density is in units of the Sun's density (M dwarfs: ~5-20).
    """
    hours = 13.0 * (period / 365.25) ** (1 / 3) * stellar_density ** (-1 / 3)
    return hours * np.sqrt(1 - impact ** 2) / 24


def transit_shape(time, period, t0, depth, duration, ingress):
    """
    Trapezoid transit: flat bottom of `depth`, linear ingress/egress of
    length `ingress`. Real transits are rounder (limb darkening), but a
    trapezoid is honest enough and, unlike a box, doesn't flatter BLS
    by matching its model exactly.
    """
    phase = (time - t0 + period / 2) % period - period / 2   # days from mid-transit
    distance_from_edge = duration / 2 - np.abs(phase)        # >0 means in transit
    fraction = np.clip(distance_from_edge / ingress, 0, 1)
    return 1 - depth * fraction


def inject_transit(lc, period, t0, depth, duration, impact=0.0):
    """Return a copy of `lc` with a transit multiplied into its flux."""
    radius_ratio = np.sqrt(depth)
    # Ingress gets longer for grazing (high impact parameter) transits.
    ingress = min(duration * radius_ratio / max(1 - impact ** 2, 0.05), duration / 2)
    model = transit_shape(lc.time.value, period, t0, depth, duration, ingress)
    injected = lc.copy()
    injected.flux = injected.flux * model
    return injected


# ---------------------------------------------------------------------------
# Running many injections
# ---------------------------------------------------------------------------

def draw_injections(lc, n, stellar_radius, stellar_density, period_range=(0.5, 13.0),
                    depth_range_ppm=(30, 5000), seed=0):
    """
    Draw `n` random fake planets. Period and depth are drawn uniformly in
    log space, so short/long and small/large get equal attention.
    Planet radius follows from depth = (R_planet / R_star)^2.
    """
    rng = np.random.default_rng(seed)
    t_start = lc.time.value.min()

    period = np.exp(rng.uniform(*np.log(period_range), n))
    depth = np.exp(rng.uniform(*np.log(depth_range_ppm), n)) / 1e6
    impact = rng.uniform(0, 0.9, n)
    t0 = t_start + rng.uniform(0, 1, n) * period
    duration = transit_duration_days(period, stellar_density, impact)

    return pd.DataFrame({
        "injected_period": period,
        "injected_t0": t0,
        "injected_depth_ppm": depth * 1e6,
        "injected_duration_hours": duration * 24,
        "impact": impact,
        "planet_radius_earth": EARTH_RADII_PER_SUN * stellar_radius * np.sqrt(depth),
    })


def classify(injection, found):
    """
    Did the search get the fake planet back?

      recovered        right period (within 1%), right transit times, and it
                       passed the candidate thresholds (SDE, SNR)
      alias            found at 2x or 1/2x the period (the search noticed,
                       but you'd need a human to fix the period)
      other signal     a candidate was found, but at an unrelated period
                       (noise or systematics beat the injected planet)
      below threshold  no candidate passed SDE/SNR cuts
    """
    if not found["is_candidate"]:
        return "below threshold"

    match = ts.period_match(found["period_days"], injection["injected_period"])
    if match == "match":
        period = injection["injected_period"]
        offset = (found["t0_btjd"] - injection["injected_t0"] + period / 2) % period - period / 2
        if abs(offset) * 24 < injection["injected_duration_hours"]:
            return "recovered"
        return "other signal"
    if match.startswith("alias"):
        return "alias"
    return "other signal"


def _one_injection(args):
    """Worker: inject one planet and run the full production search."""
    time, flux, flux_err, injection, search_kwargs = args
    lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)
    lc = inject_transit(lc, injection["injected_period"], injection["injected_t0"],
                        injection["injected_depth_ppm"] / 1e6,
                        injection["injected_duration_hours"] / 24, injection["impact"])
    found = ts.search_light_curve(lc, **search_kwargs)
    return {
        **injection,
        "found_period": found["period_days"],
        "found_t0": found["t0_btjd"],
        "found_depth_ppm": found["depth_ppm"],
        "sde": found["sde"],
        "depth_snr": found["depth_snr"],
        "status": classify(injection, found),
    }


def expected_snr(lc, injections):
    """
    Rough signal-to-noise we *expect* for each injection, before searching:
        SNR ~ depth / point_scatter * sqrt(number of in-transit points)
    Recovery vs. this number should look like a step from 0 to 1 near the
    detection threshold, whatever the star. That makes it a useful way
    to compare stars.
    """
    _, flat = ts.clean_light_curve(lc)
    # Robust point-to-point scatter (median absolute deviation -> sigma).
    flux = flat.flux.value
    scatter = 1.4826 * np.median(np.abs(flux - np.median(flux)))

    time = flat.time.value
    snrs = []
    for _, inj in injections.iterrows():
        P, t0 = inj["injected_period"], inj["injected_t0"]
        phase = (time - t0 + P / 2) % P - P / 2
        n_in = np.sum(np.abs(phase) < inj["injected_duration_hours"] / 48)
        snrs.append(inj["injected_depth_ppm"] / 1e6 / scatter * np.sqrt(n_in))
    return np.array(snrs)


def run_injections(lc, n=200, stellar_radius=1.0, stellar_density=1.0,
                   period_range=(0.5, 13.0), depth_range_ppm=(30, 5000),
                   seed=0, n_workers=None, search_kwargs=None):
    """
    Inject `n` fake planets one at a time into `lc` and search each with
    ts.search_light_curve(). Runs in parallel across CPU cores. Returns one
    row per injection.

    `lc` should be a star with no detectable signal of its own; check with
    baseline_check() first. Otherwise the search finds the real signal
    instead of the injected one, and the injection counts as "other signal".
    """
    search_kwargs = search_kwargs or {}
    lc = lc.remove_nans().normalize()
    injections = draw_injections(lc, n, stellar_radius, stellar_density,
                                 period_range, depth_range_ppm, seed)

    arrays = (lc.time.value, lc.flux.value, lc.flux_err.value)
    tasks = [(*arrays, row, search_kwargs) for row in injections.to_dict("records")]
    n_workers = n_workers or os.cpu_count()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        rows = []
        for i, row in enumerate(pool.map(_one_injection, tasks), start=1):
            rows.append(row)
            if i % 20 == 0 or i == n:
                done = sum(r["status"] == "recovered" for r in rows)
                print(f"  {i}/{n} injections, {done} recovered", flush=True)

    results = pd.DataFrame(rows)
    results["expected_snr"] = expected_snr(lc, injections)
    results["recovered"] = results["status"] == "recovered"
    return results


# ---------------------------------------------------------------------------
# Choosing test stars
# ---------------------------------------------------------------------------

def baseline_check(lc, **search_kwargs):
    """
    Search the star *before* injecting anything. A good injection target
    has no candidate of its own. Also reports the 1-hour CDPP (combined
    differential photometric precision): the noise, in ppm, on a 1-hour
    average. It's the standard single-number "how quiet is this star".
    """
    result = ts.search_light_curve(lc, **search_kwargs)
    _, flat = ts.clean_light_curve(lc)
    cdpp = flat.estimate_cdpp(transit_duration=30)   # 30 x 2-min = 1 hour
    return {
        "cdpp_1h_ppm": float(cdpp.value),
        "baseline_period": result["period_days"],
        "baseline_sde": result["sde"],
        "baseline_snr": result["depth_snr"],
        "has_own_signal": result["is_candidate"],
    }


def find_quiet_stars(n_wanted=2, tmag=(8.5, 10.5), teff=(5300, 6000),
                     radius=(0.8, 1.3), ra=(60, 120), dec=(-75, -60), max_tries=15):
    """
    Pick dwarf stars from the TESS Input Catalog that have SPOC 2-minute
    data and no signal of their own. `radius` (in solar radii) keeps out
    slightly evolved stars, which TIC can still label "DWARF". A bigger
    star dilutes the same planet into a shallower dip. The default sky box is near the
    southern ecliptic pole, where TESS observed for many sectors, which
    helps later multi-sector searches.

    This only rejects stars whose light curve shows a candidate. It does not
    check the TOI list or the Exoplanet Archive (yet).
    """
    from astroquery.mast import Catalogs

    tic = Catalogs.query_criteria(catalog="TIC", Tmag=list(tmag), Teff=list(teff),
                                  rad=list(radius), lumclass="DWARF", objType="STAR",
                                  ra=list(ra), dec=list(dec))
    tic = tic.to_pandas().sort_values("Tmag")
    print(f"{len(tic)} TIC stars match; checking up to {max_tries}")

    picked = []
    for _, star in tic.head(max_tries).iterrows():
        target = f"TIC {star['ID']}"
        try:
            lc = ts.download_light_curve(target, max_sectors=1)
        except Exception as exc:
            print(f"  {target}: skipped ({type(exc).__name__})")
            continue
        check = baseline_check(lc)
        print(f"  {target}: CDPP {check['cdpp_1h_ppm']:.0f} ppm, "
              f"own signal: {check['has_own_signal']}")
        if check["has_own_signal"]:
            continue
        picked.append({"target": target, "tmag": star["Tmag"], "teff": star["Teff"],
                       "stellar_radius": star["rad"], "stellar_mass": star["mass"],
                       "stellar_density": star["rho"],
                       "sectors": ",".join(map(str, lc.meta["SECTORS"])), **check})
        if len(picked) == n_wanted:
            break
    return pd.DataFrame(picked)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_recovery_map(results, title="", period_bins=6, radius_bins=6, ax=None):
    """
    Heatmap of recovery fraction in period vs planet-radius bins (log axes).
    Each cell shows the fraction recovered and, below it, how many
    injections landed there. Small counts mean a noisy fraction.
    """
    p_edges = np.geomspace(results["injected_period"].min(),
                           results["injected_period"].max() * 1.0001, period_bins + 1)
    r_edges = np.geomspace(results["planet_radius_earth"].min(),
                           results["planet_radius_earth"].max() * 1.0001, radius_bins + 1)
    p_idx = np.digitize(results["injected_period"], p_edges) - 1
    r_idx = np.digitize(results["planet_radius_earth"], r_edges) - 1

    total = np.zeros((radius_bins, period_bins))
    found = np.zeros((radius_bins, period_bins))
    np.add.at(total, (r_idx, p_idx), 1)
    np.add.at(found, (r_idx, p_idx), results["recovered"].astype(float))
    fraction = np.divide(found, total, out=np.full_like(found, np.nan), where=total > 0)

    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 6))
    mesh = ax.pcolormesh(p_edges, r_edges, fraction, cmap=RECOVERY_CMAP,
                         vmin=0, vmax=1, edgecolors="white", linewidth=2)
    for i in range(radius_bins):
        for j in range(period_bins):
            if total[i, j] == 0:
                continue
            x = np.sqrt(p_edges[j] * p_edges[j + 1])
            y = np.sqrt(r_edges[i] * r_edges[i + 1])
            ink = "white" if fraction[i, j] > 0.55 else "#1a1a1a"
            ax.text(x, y, f"{fraction[i, j]:.0%}\nn={int(total[i, j])}",
                    ha="center", va="center", fontsize=8, color=ink)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Injected period [days]")
    ax.set_ylabel("Injected planet radius [Earth radii]")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(plt.NullFormatter())
    ax.set_xticks([t for t in (0.5, 1, 2, 3, 5, 8, 13) if p_edges[0] <= t <= p_edges[-1]])
    ax.set_title(title)
    plt.colorbar(mesh, ax=ax, label="Fraction recovered")
    return ax


def plot_recovery_vs_snr(results, title="", ax=None, n_bins=12):
    """
    Recovery fraction vs expected SNR. Should rise from ~0 to ~1 around
    the SNR threshold (7). Wherever it stays below 1 at high SNR,
    the pipeline is losing planets it should find. That's the place to dig.
    """
    snr = results["expected_snr"].clip(lower=0.5)
    edges = np.geomspace(snr.min(), snr.max() * 1.0001, n_bins + 1)
    idx = np.digitize(snr, edges) - 1
    centers = np.sqrt(edges[:-1] * edges[1:])
    frac = np.array([results["recovered"][idx == k].mean() if np.any(idx == k) else np.nan
                     for k in range(n_bins)])
    counts = np.bincount(idx, minlength=n_bins)[:n_bins]

    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4))
    ax.axvline(ts.MIN_SNR, color="0.6", lw=1, ls="--")
    ax.text(ts.MIN_SNR * 1.05, 0.04, f"SNR threshold ({ts.MIN_SNR:g})", color="0.4", fontsize=8)
    ax.plot(centers, frac, "-o", color="#2a78d6", lw=2, ms=8,
            markeredgecolor="white", markeredgewidth=2)
    for x, y, c in zip(centers, frac, counts):
        if np.isfinite(y):
            ax.annotate(f"n={c}", (x, y), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=7, color="0.4")
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.12)
    ax.set_xlabel("Expected SNR of injected signal")
    ax.set_ylabel("Fraction recovered")
    ax.grid(alpha=0.25)
    ax.set_title(title)
    return ax
