"""
A small, readable transit-search pipeline for TESS light curves.

The pipeline has four steps, each its own function so you can run and inspect
them one at a time in a notebook:

    1. download_light_curve()  -> get PDCSAP flux from NASA's MAST archive
    2. clean_light_curve()     -> drop NaNs, flatten variability, clip outliers
    3. run_bls()               -> Box Least Squares search for periodic dips
    4. plot_diagnostics()      -> raw, flattened, periodogram, folded curve

search_target() chains all four for one star; search_many() loops over a
list of stars and writes one CSV row per star.
"""

from __future__ import annotations

import re
from pathlib import Path

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares

# Trial transit durations, in days (30 minutes to 8 hours). Hot Jupiters and
# small planets around Sun-like stars mostly last 1-6 hours; ultra-short-
# period planets around small stars can be under an hour.
DEFAULT_DURATIONS = np.array([0.5, 1, 1.5, 2, 3, 4, 5, 6, 8]) / 24.0

# Rough "is this worth a closer look" thresholds. They're conventions, not
# laws: SDE ~ 7-9 and SNR ~ 7 are typical cut-offs in TESS/Kepler searches.
MIN_SDE = 9.0
MIN_SNR = 7.0


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------

def download_light_curve(target, sectors=None, max_sectors=1,
                         author="SPOC", exptime=120):
    """
    Download a TESS light curve for `target` (e.g. "Pi Mensae" or
    "TIC 261136679") and stitch the sectors into one LightCurve.

    We use the SPOC pipeline's 2-minute-cadence PDCSAP flux: NASA has
    already removed most spacecraft systematics from it.

    `max_sectors` limits how much data we fetch. More sectors means more
    transits (more signal) but also a longer time baseline, and the BLS
    period grid grows with the square of the baseline. One sector (~27 days)
    is plenty to find planets with periods up to ~13 days.
    """
    search = lk.search_lightcurve(target, mission="TESS", author=author,
                                  exptime=exptime, sector=sectors)
    if len(search) == 0:
        raise ValueError(f"No TESS {author} {exptime}s light curves for {target!r}")

    if sectors is None and max_sectors is not None:
        search = search[:max_sectors]

    collection = search.download_all(quality_bitmask="default")
    # stitch() normalises each sector to median 1 before joining them,
    # so sector-to-sector offsets don't look like signals.
    lc = collection.stitch()
    lc.meta["SECTORS"] = [single.meta.get("SECTOR") for single in collection]
    return lc


# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------

def window_length_in_cadences(lc, window_days):
    """Convert a smoothing window from days to an odd number of cadences."""
    cadence_days = np.nanmedian(np.diff(lc.time.value))
    n = int(round(window_days / cadence_days))
    return n + 1 if n % 2 == 0 else n


def clean_light_curve(lc, window_days=1.0, sigma_upper=4.0, sigma_lower=20.0,
                      transit_mask=None):
    """
    Returns (normalized, flattened) light curves.

    - Remove NaNs (gaps, flagged cadences) and normalise to median 1.
    - Flatten: fit a smooth trend (Savitzky-Golay filter) and divide it out,
      removing slow stellar variability (spots, rotation) while keeping
      short dips. The window must be much longer than a transit (rule of
      thumb: >= 3x the duration) or the filter will "eat" the transit.
      `transit_mask` (boolean array, True = in transit) tells the filter to
      ignore those points when fitting the trend; see search_light_curve().
    - Clip outliers *after* flattening (on a variable star the spread before
      flattening is dominated by the variability, so spikes hide in it).
      Clip *asymmetrically*: transits are dips, so we aggressively clip
      upward spikes (cosmic rays, flares) but only remove absurdly low
      points. A symmetric 4-sigma clip can delete the transits themselves!
    """
    normalized = lc.remove_nans().normalize()

    window = window_length_in_cadences(normalized, window_days)
    flattened = normalized.flatten(window_length=window, polyorder=2,
                                   mask=transit_mask)
    flattened = flattened.remove_outliers(sigma_upper=sigma_upper,
                                          sigma_lower=sigma_lower)
    return normalized, flattened


# ---------------------------------------------------------------------------
# 3. Box Least Squares search
# ---------------------------------------------------------------------------

def run_bls(lc, min_period=0.5, max_period=None, durations=DEFAULT_DURATIONS,
            frequency_factor=1.0, max_grid_size=2_000_000):
    """
    Search a flattened light curve for a periodic box-shaped dip.

    BLS slides a box of every trial (period, duration, phase) across the
    folded light curve and asks how much better a "flat line with a box dip"
    fits than a flat line. The best (period, duration, phase) wins.

    Returns (result_dict, periodogram, bls_model).
    """
    t = lc.time.value
    y = lc.flux.value
    dy = lc.flux_err.value
    baseline = t.max() - t.min()

    # Require at least two transits inside the data by default.
    if max_period is None:
        max_period = baseline / 2

    # BLS needs every trial duration to be shorter than the shortest period.
    durations = np.asarray(durations)
    durations = durations[durations < min_period]

    bls = BoxLeastSquares(t, y, dy)
    periods = bls.autoperiod(durations, minimum_period=min_period,
                             maximum_period=max_period,
                             frequency_factor=frequency_factor)
    if len(periods) > max_grid_size:
        raise ValueError(
            f"Period grid has {len(periods):,} points (baseline {baseline:.0f} d). "
            "Use fewer/contiguous sectors, a narrower period range, or a "
            "larger frequency_factor.")

    periodogram = bls.power(periods, durations, objective="likelihood")

    best = np.argmax(periodogram.power)
    period = periodogram.period[best]
    duration = periodogram.duration[best]
    t0 = periodogram.transit_time[best]
    stats = bls.compute_stats(period, duration, t0)

    # Signal Detection Efficiency: how far the tallest peak stands above the
    # typical periodogram level, in standard deviations.
    power = periodogram.power
    sde = (power[best] - np.mean(power)) / np.std(power)

    depth, depth_err = stats["depth"]
    odd, odd_err = stats["depth_odd"]
    even, even_err = stats["depth_even"]

    result = {
        "period_days": period,
        "t0_btjd": t0,
        "duration_hours": duration * 24,
        "depth_ppm": depth * 1e6,
        "depth_snr": depth / depth_err,
        "sde": sde,
        "bls_power": power[best],
        "n_transits": int(np.sum(stats["per_transit_count"] > 0)),
        # Odd/even test: an eclipsing binary at 2x the true period shows
        # alternating deep (primary) and shallow (secondary) eclipses.
        "depth_odd_ppm": odd * 1e6,
        "depth_even_ppm": even * 1e6,
        "odd_even_sigma": abs(odd - even) / np.hypot(odd_err, even_err),
        # At very high SNR a tiny (harmless) difference can be many sigma,
        # so also report it as a fraction of the depth. EBs differ by tens
        # of percent; a few percent is usually systematics.
        "odd_even_diff_pct": 100 * abs(odd - even) / depth,
        "n_points": len(t),
        "baseline_days": baseline,
    }
    result["is_candidate"] = bool(result["sde"] >= MIN_SDE
                                  and result["depth_snr"] >= MIN_SNR)
    return result, periodogram, bls


# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------

def plot_diagnostics(raw, flattened, periodogram, result, title=""):
    """Four-panel summary figure. Returns the matplotlib Figure."""
    period = result["period_days"]
    t0 = result["t0_btjd"]
    duration_hours = result["duration_hours"]

    fig, axes = plt.subplots(4, 1, figsize=(11, 13))
    ax_raw, ax_flat, ax_pg, ax_fold = axes

    raw.plot(ax=ax_raw, normalize=False, color="k", lw=0.5)
    ax_raw.set_title("1. Raw (normalised PDCSAP) light curve")

    flattened.plot(ax=ax_flat, normalize=False, color="k", lw=0.5)
    # Mark where BLS thinks the transits are.
    t_min, t_max = flattened.time.value.min(), flattened.time.value.max()
    n_first = np.ceil((t_min - t0) / period)
    for transit_time in t0 + period * np.arange(n_first, n_first + 1000):
        if transit_time > t_max:
            break
        ax_flat.axvline(transit_time, color="C3", alpha=0.3, lw=2)
    ax_flat.set_title("2. Flattened light curve (red lines: BLS transit times)")

    ax_pg.plot(periodogram.period, periodogram.power, "k", lw=0.6)
    ax_pg.axvline(period, color="C3", alpha=0.4, lw=4)
    # Aliases at P/2 and 2P are a classic sign you're looking at a real
    # periodic signal (or an eclipsing binary at the wrong period).
    for harmonic in (0.5, 2):
        ax_pg.axvline(period * harmonic, color="C3", ls="--", alpha=0.3)
    ax_pg.set_xscale("log")
    ax_pg.set_xlabel("Period [days]")
    ax_pg.set_ylabel("BLS power")
    ax_pg.set_title(f"3. BLS periodogram: best P = {period:.4f} d, "
                    f"SDE = {result['sde']:.1f}")

    folded = flattened.fold(period=period, epoch_time=t0)
    phase_hours = folded.time.value * 24
    window = max(3 * duration_hours, 3)
    in_view = np.abs(phase_hours) < window
    ax_fold.plot(phase_hours[in_view], folded.flux.value[in_view], ".",
                 color="0.6", ms=2, label="data")
    binned = folded[in_view].bin(time_bin_size=duration_hours / 24 / 6)
    ax_fold.plot(binned.time.value * 24, binned.flux.value, "o", color="k",
                 ms=4, label="binned")
    # The box model BLS fitted.
    depth = result["depth_ppm"] / 1e6
    half = duration_hours / 2
    ax_fold.plot([-window, -half, -half, half, half, window],
                 [1, 1, 1 - depth, 1 - depth, 1, 1], color="C3", lw=2,
                 label="BLS box model")
    ax_fold.set_xlim(-window, window)
    ax_fold.set_xlabel("Hours from mid-transit")
    ax_fold.set_ylabel("Normalised flux")
    ax_fold.legend(loc="lower right")
    ax_fold.set_title(f"4. Folded at best period: depth = {result['depth_ppm']:.0f} ppm, "
                      f"duration = {duration_hours:.2f} h, SNR = {result['depth_snr']:.1f}")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Whole pipeline
# ---------------------------------------------------------------------------

def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")


def search_light_curve(lc, name="target", window_days=1.0, min_period=0.5,
                       max_period=None, durations=DEFAULT_DURATIONS,
                       plot_dir=None, show=False):
    """
    Clean + search + plot an already-loaded LightCurve. Returns a dict.

    Two passes:
      1. Flatten blindly and run BLS to find the strongest signal.
      2. Mask that signal's transits, flatten again (so the trend fit can't
         partly absorb the dips), and re-run BLS. Pass 2 gives unbiased
         depths; pass 1 alone tends to under-estimate them.
    """
    normalized, flattened = clean_light_curve(lc, window_days=window_days)
    first, _, _ = run_bls(flattened, min_period=min_period,
                          max_period=max_period, durations=durations)

    transit_mask = normalized.create_transit_mask(
        period=first["period_days"], transit_time=first["t0_btjd"],
        duration=2 * first["duration_hours"] / 24)   # pad by 2x to be safe
    if transit_mask.mean() > 0.3:
        # A junk pass-1 "signal" (pure noise) can mask most of the data,
        # leaving nothing to fit the trend with. Fall back to no mask.
        transit_mask = None
    normalized, flattened = clean_light_curve(lc, window_days=window_days,
                                              transit_mask=transit_mask)
    result, periodogram, _ = run_bls(flattened, min_period=min_period,
                                     max_period=max_period, durations=durations)
    result = {"target": name, **result}

    if plot_dir is not None or show:
        fig = plot_diagnostics(normalized, flattened, periodogram, result, title=name)
        if plot_dir is not None:
            plot_dir = Path(plot_dir)
            plot_dir.mkdir(parents=True, exist_ok=True)
            path = plot_dir / f"{safe_filename(name)}.png"
            fig.savefig(path, dpi=110)
            result["plot"] = str(path)
        if show:
            plt.show()
        else:
            plt.close(fig)
    return result


def search_target(target, sectors=None, max_sectors=1, **kwargs):
    """Download a star's TESS data and run the full search on it."""
    lc = download_light_curve(target, sectors=sectors, max_sectors=max_sectors)
    result = search_light_curve(lc, name=target, **kwargs)
    result["sectors"] = ",".join(str(s) for s in lc.meta.get("SECTORS", []))
    return result


def search_many(targets, csv_path="results.csv", plot_dir="plots", **kwargs):
    """
    Run search_target() on every star in `targets` and save one row per star
    to `csv_path`. A failure on one star (no data, download error) is
    recorded in the `error` column instead of stopping the whole run.

    `targets` can be a list of names, or a DataFrame with a "target" column
    (any other columns, e.g. a published period, are carried into the CSV).
    """
    if isinstance(targets, pd.DataFrame):
        rows_in = targets.to_dict("records")
    else:
        rows_in = [{"target": t} for t in targets]

    rows_out = []
    for row in rows_in:
        target = row["target"]
        # Optional per-star settings from the table override the defaults,
        # e.g. a smaller min_period for an ultra-short-period planet.
        per_star = {key: row[key] for key in ("min_period", "max_period", "window_days")
                    if key in row and pd.notna(row[key])}
        print(f"Searching {target} ...", flush=True)
        try:
            result = search_target(target, plot_dir=plot_dir, **{**kwargs, **per_star})
            result["error"] = ""
        except Exception as exc:  # keep going; record what went wrong
            result = {"target": target, "error": f"{type(exc).__name__}: {exc}"}
            print(f"  failed: {result['error']}")
        rows_out.append({**row, **result})

    table = pd.DataFrame(rows_out)
    if "known_period_days" in table.columns:
        table = compare_to_known(table)
    table.to_csv(csv_path, index=False)
    print(f"Saved {len(table)} rows to {csv_path}")
    return table


# ---------------------------------------------------------------------------
# Validation against published planets
# ---------------------------------------------------------------------------

def period_match(found, known, tolerance=0.01):
    """
    Compare a BLS period to a published one. Returns "match", "alias 2x",
    "alias 1/2x" (BLS often locks onto a harmonic), "miss", or "" if there
    is nothing to compare (e.g. the search failed).
    """
    if not (np.isfinite(found) and np.isfinite(known)):
        return ""
    for ratio, label in ((1, "match"), (2, "alias 2x"), (0.5, "alias 1/2x")):
        if abs(found / (known * ratio) - 1) < tolerance:
            return label
    return "miss"


def compare_to_known(table):
    """Add a `period_check` column comparing found vs. known period."""
    table = table.copy()
    found = table.get("period_days", pd.Series(np.nan, index=table.index))
    table["period_check"] = [period_match(f, k) for f, k in
                             zip(found.astype(float), table["known_period_days"].astype(float))]
    return table
