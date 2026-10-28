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
from types import SimpleNamespace

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares

# Trial transit durations, in days (30 minutes to 8 hours). Hot Jupiters and
# small planets around Sun-like stars mostly last 1-6 hours; ultra-short-
# period planets around small stars can be under an hour.
DEFAULT_DURATIONS = np.array([0.5, 1, 1.5, 2, 3, 4, 5, 6, 8]) / 24.0

# "Is this worth a closer look" thresholds. SDE ~ 7-9 and SNR ~ 7 are
# typical cut-offs in TESS/Kepler searches. We chose SDE 8 from our own
# injection tests: on 105 pure-noise trials, SDE >= 7.5 still gave no false
# alarms (SDE >= 7 gave 2%), while going from 9 to 8 raised recovery of
# near-threshold planets (SNR 7-15) from 65% to 71%. Most misses at 9 were
# found at the right period but cut by SDE alone.
MIN_SDE = 8.0
MIN_SNR = 7.0

# Longest transit we'll consider, as a fraction of the orbital period.
MAX_DUTY_CYCLE = 0.15


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

def robust_local_level(dt, values, degree=2, c=5.0, iterations=5):
    """
    The trend value at dt = 0 from a robust polynomial fit to (dt, values).

    A quadratic (degree 2) follows curved stellar variability (starspots); a
    plain average would sit above or below a curving trend. "Robust" means
    iteratively reweighted with Tukey's biweight: points far from the fit
    (a transit dip, a flare) get less weight, and beyond c x MAD none at
    all. So a transit barely pulls the trend down and isn't partly erased.
    """
    X = np.vander(dt, degree + 1)        # columns: ..., dt, 1
    weight = np.ones(len(values))
    coeffs = np.zeros(degree + 1)
    for _ in range(iterations):
        sw = np.sqrt(weight)
        coeffs, *_ = np.linalg.lstsq(X * sw[:, None], values * sw, rcond=None)
        resid = values - X @ coeffs
        mad = np.median(np.abs(resid[weight > 0]))
        if mad == 0:
            break
        u = resid / (c * mad)
        weight = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
    return coeffs[-1]


def robust_trend(time, flux, window_days=1.0, mask=None, gap_days=0.5):
    """
    The star's slow trend: at each moment, a robust quadratic fit to the
    flux within +/- window_days/2, ignoring `mask`ed points (True = skip,
    e.g. known transits).

    Three choices matter:
      - Robust (biweight weights), so transit dips don't drag the trend
        down and get partly erased.
      - The window never crosses a data gap. Each stretch of data is
        handled on its own, so the two sides of a gap never mix.
      - No interpolation across masked points or gaps. lightkurve's
        Savitzky-Golay flatten() does both, and injection tests caught it
        creating 500-1000 ppm steps next to transits.

    For speed the trend is evaluated on a grid (every window/30) and
    interpolated. The trend is smooth on that scale, so this costs nothing.
    """
    mask = np.zeros(len(time), dtype=bool) if mask is None else np.asarray(mask)
    trend = np.full(len(time), np.nan)
    half = window_days / 2
    step = window_days / 30

    # Split into segments at gaps.
    breaks = np.where(np.diff(time) > gap_days)[0] + 1
    for seg in np.split(np.arange(len(time)), breaks):
        t, f, use = time[seg], flux[seg], ~mask[seg]
        grid = np.arange(t[0], t[-1] + step, step)
        values = np.full(len(grid), np.nan)
        lo = np.searchsorted(t, grid - half)
        hi = np.searchsorted(t, grid + half)
        for i, (a, b) in enumerate(zip(lo, hi)):
            keep = use[a:b]
            if keep.sum() < 20:
                continue
            dt = t[a:b][keep] - grid[i]
            # Near a segment edge the window is one-sided, and a curve fitted
            # to one side bends to follow any dip there. Use a straight line
            # unless there's data on both sides.
            both_sides = min(np.sum(dt < 0), np.sum(dt > 0)) > 0.3 * len(dt) / 2
            values[i] = robust_local_level(dt, f[a:b][keep], degree=2 if both_sides else 1)
        good = np.isfinite(values)
        if good.sum() >= 2:
            trend[seg] = np.interp(t, grid[good], values[good])
        else:
            # Segment too short or fully masked: use its plain median.
            trend[seg] = np.median(f[use]) if use.any() else np.median(f)
    return trend


def clean_light_curve(lc, window_days=1.0, sigma_upper=4.0, sigma_lower=20.0,
                      transit_mask=None):
    """
    Returns (normalized, flattened) light curves.

    - Remove NaNs (gaps, flagged cadences) and normalise to median 1.
    - Flatten: estimate the star's slow trend with a sliding robust
      quadratic fit (robust_trend) and divide it out. That removes stellar variability
      (spots, rotation) while keeping short dips. The window must be much
      longer than a transit (rule of thumb: >= 3x the duration) or the
      trend starts to follow the transit.
      `transit_mask` (boolean array, True = in transit) leaves those points
      out of the trend estimate; see search_light_curve().
    - Clip outliers *after* flattening (on a variable star the spread before
      flattening is dominated by the variability, so spikes hide in it).
      Clip *asymmetrically*: transits are dips, so we aggressively clip
      upward spikes (cosmic rays, flares) but only remove absurdly low
      points. A symmetric 4-sigma clip can delete the transits themselves!
    """
    normalized = lc.remove_nans().normalize()

    trend = robust_trend(normalized.time.value, normalized.flux.value,
                           window_days=window_days, mask=transit_mask)
    flattened = normalized.copy()
    flattened.flux = normalized.flux / trend
    flattened.flux_err = normalized.flux_err / trend
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

    durations = np.sort(np.asarray(durations))

    bls = BoxLeastSquares(t, y, dy)
    periods = bls.autoperiod(durations, minimum_period=min_period,
                             maximum_period=max_period,
                             frequency_factor=frequency_factor)
    if len(periods) > max_grid_size:
        raise ValueError(
            f"Period grid has {len(periods):,} points (baseline {baseline:.0f} d). "
            "Use fewer/contiguous sectors, a narrower period range, or a "
            "larger frequency_factor.")

    periodogram = _bls_power_physical(bls, periods, durations)

    best = np.argmax(periodogram.power)
    period = periodogram.period[best]
    duration = periodogram.duration[best]
    t0 = periodogram.transit_time[best]

    # Signal Detection Efficiency: how far the tallest peak stands above the
    # typical periodogram level, in standard deviations.
    power = periodogram.power
    sde = (power[best] - np.mean(power)) / np.std(power)

    result = {
        "period_days": period,
        "t0_btjd": t0,
        "duration_hours": duration * 24,
        **transit_stats(lc, period, duration, t0),
        "sde": sde,
        "bls_power": power[best],
        "n_points": len(t),
        "baseline_days": baseline,
    }
    # The SNR used to *decide* whether this is a candidate.
    result["detection_snr"] = result["depth_snr"]
    result["is_candidate"] = bool(result["sde"] >= MIN_SDE
                                  and result["detection_snr"] >= MIN_SNR)
    return result, periodogram, bls


def _bls_power_physical(bls, periods, durations, n_chunks=12):
    """
    Run BLS, but at each period only try durations a planet could have.

    A transit can only last a small fraction of the orbit (the "duty
    cycle"): about 4-12% at P = 0.5 d around a Sun-like star, and less at
    longer periods. Without this limit, BLS happily fits an 8-hour box to
    a 1-day period. A long box catches more data, so its score can beat
    a real short transit. Injection tests caught exactly that.

    astropy uses one duration list for every period, so we split the
    period grid into chunks and give each chunk its own allowed durations.
    """
    edges = np.geomspace(periods.min(), periods.max(), n_chunks + 1)
    chunk = np.clip(np.digitize(periods, edges) - 1, 0, n_chunks - 1)
    parts = []
    for k in range(n_chunks):
        chunk_periods = periods[chunk == k]
        if len(chunk_periods) == 0:
            continue
        allowed = durations[durations <= MAX_DUTY_CYCLE * chunk_periods.min()]
        if len(allowed) == 0:
            allowed = durations[:1]
        parts.append(bls.power(chunk_periods, allowed, objective="likelihood"))

    return SimpleNamespace(**{key: np.concatenate([part[key] for part in parts])
                              for key in ("period", "power", "duration", "transit_time")})


def transit_stats(lc, period, duration, t0):
    """
    Depth, SNR, transit count and odd/even depths of a box-shaped transit at
    a *fixed* ephemeris (period, duration and t0 all in days).
    """
    bls = BoxLeastSquares(lc.time.value, lc.flux.value, lc.flux_err.value)
    stats = bls.compute_stats(period, duration, t0)
    depth, depth_err = stats["depth"]
    odd, odd_err = stats["depth_odd"]
    even, even_err = stats["depth_even"]
    return {
        "depth_ppm": depth * 1e6,
        "depth_snr": depth / depth_err,
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
    }


def transit_quality(lc, period, t0, duration, gap_days=0.5, edge_days=0.3, noisy=1.5):
    """
    Checks on the individual transits and the folded light curve, which
    catch the commonest impostors in a real search (a strong period can come
    from one or two bad stretches of data, or from stellar pulsations):

      max_event_share    fraction of the total signal (SNR^2) in the single
                         strongest transit. Near 1: one event (a glitch, a
                         lone dip) is doing all the work.
      bad_data_fraction  fraction of transits that sit within `edge_days`
                         (+ half a duration) of a data gap or sector edge,
                         where TESS data ramps, or in a stretch whose
                         scatter is `noisy` x the typical (pointing jitter).
      median_mean_ratio  in-transit median depth / mean depth. A real dip
                         lowers every point (~1); a few extreme points give a
                         deep mean but a shallow median (<< 1).
      baseline_wiggle    scatter of the folded, binned light curve *outside*
                         the transit, relative to the depth. A transit has a
                         flat baseline (~0.1-0.2); pulsations or spots at the
                         same period make it wavy (~0.5+).
    """
    t, f = lc.time.value, lc.flux.value
    scatter = 1.4826 * np.median(np.abs(f - np.median(f)))
    starts = np.concatenate([[t[0]], t[1:][np.diff(t) > gap_days]])
    ends = np.concatenate([t[:-1][np.diff(t) > gap_days], [t[-1]]])
    boundaries = np.concatenate([starts, ends])

    n_first = np.floor((t[0] - t0) / period)
    centres = t0 + period * np.arange(n_first, n_first + (t[-1] - t[0]) / period + 2)
    snr2, bad, in_points = [], [], []
    for tc in centres:
        inside = np.abs(t - tc) < duration / 2
        if inside.sum() < 3:
            continue
        depth = 1 - np.mean(f[inside])
        snr2.append(max(depth, 0) ** 2 * inside.sum() / scatter ** 2)
        near_edge = np.min(np.abs(boundaries - tc)) < edge_days + duration / 2
        around = (np.abs(t - tc) < 0.5) & ~inside
        local = 1.4826 * np.median(np.abs(f[around] - np.median(f[around]))) if around.sum() > 20 else scatter
        bad.append(near_edge or local > noisy * scatter)
        in_points.append(f[inside])
    if not snr2:
        return {"max_event_share": np.nan, "bad_data_fraction": np.nan,
                "median_mean_ratio": np.nan, "baseline_wiggle": np.nan}
    points = np.concatenate(in_points)
    mean_depth = 1 - np.mean(points)

    # Folded baseline: bin out-of-transit phases at the transit duration.
    phase = ((t - t0) / period + 0.5) % 1 - 0.5           # -0.5..0.5, transit at 0
    out = np.abs(phase * period) > duration
    n_bins = max(int(period / duration), 4)
    edges = np.linspace(-0.5, 0.5, n_bins + 1)
    idx = np.digitize(phase[out], edges)
    binned = np.array([np.mean(f[out][idx == k]) for k in range(1, n_bins + 1) if np.sum(idx == k) > 5])
    noise_per_bin = scatter / np.sqrt(max(len(f[out]) / n_bins, 1))
    # Subtract the scatter expected from noise alone, then compare to depth.
    excess = np.sqrt(max(np.var(binned) - noise_per_bin ** 2, 0)) if len(binned) > 3 else 0.0

    return {
        "max_event_share": float(max(snr2) / max(sum(snr2), 1e-12)),
        "bad_data_fraction": float(np.mean(bad)),
        "median_mean_ratio": float((1 - np.median(points)) / mean_depth) if mean_depth > 0 else np.nan,
        "baseline_wiggle": float(excess / mean_depth) if mean_depth > 0 else np.nan,
    }


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


def signal_mask(lc, signals, pad=2.0):
    """True for cadences inside any of `signals` [(period, t0, duration_days)],
    with each transit window widened by `pad`."""
    mask = np.zeros(len(lc), dtype=bool)
    for period, t0, duration in signals:
        mask |= lc.create_transit_mask(period=period, transit_time=t0,
                                       duration=pad * duration)
    return mask


def search_light_curve(lc, name="target", window_days=1.0, min_period=0.5,
                       max_period=None, durations=DEFAULT_DURATIONS,
                       mask_signals=(), plot_dir=None, show=False, return_figure=False):
    """
    Clean + search + plot an already-loaded LightCurve. Returns a dict, or
    (dict, Figure) with `return_figure=True` (for annotated guide images).

    Two passes, with separate jobs:
      1. Detect: flatten blindly and run BLS. The period, t0, SDE and
         `detection_snr` (and so `is_candidate`) come from this pass only.
      2. Measure: mask the transits found in pass 1, flatten again so the
         trend fit can't partly absorb the dips, and re-measure depth, SNR
         and odd/even at the *same* ephemeris. Blind flattening makes
         transits look too shallow; this pass fixes that.

    Pass 2 must not search again. Under the mask the trend is only
    interpolated, which can bend the masked stretch up or down. A new search
    would then "re-discover" whatever pass 1 picked, even noise. Injection-
    recovery tests caught exactly that, so pass 2 only measures.

    `mask_signals` lists signals already found [(period, t0, duration_days)].
    Their transits are kept out of the trend fit and then dropped before BLS,
    so the search looks for something else (see search_multi).
    """
    normalized = lc.remove_nans().normalize()
    prior = signal_mask(normalized, mask_signals)

    def flatten_and_drop_prior(transit_mask):
        _, flat = clean_light_curve(lc, window_days=window_days,
                                    transit_mask=transit_mask if transit_mask.any() else None)
        return flat[~signal_mask(flat, mask_signals)]

    flattened = flatten_and_drop_prior(prior)
    result, periodogram, _ = run_bls(flattened, min_period=min_period,
                                     max_period=max_period, durations=durations)
    result["depth_ppm_blind"] = result["depth_ppm"]

    period = result["period_days"]
    duration = result["duration_hours"] / 24
    transit_mask = signal_mask(normalized, [(period, result["t0_btjd"], duration)])
    # A long-duration junk signal can mask most of the data, leaving too
    # little to fit the trend with; then keep the blind numbers.
    if (transit_mask | prior).mean() <= 0.3:
        flattened = flatten_and_drop_prior(transit_mask | prior)
        result.update(transit_stats(flattened, period, duration, result["t0_btjd"]))
    result.update(transit_quality(flattened, period, result["t0_btjd"], duration))

    result = {"target": name, **result}

    if return_figure:
        return result, plot_diagnostics(normalized, flattened, periodogram, result, title=name)
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


def search_multi(lc, name="target", max_signals=3, plot_dir=None, **kwargs):
    """
    Find up to `max_signals` periodic signals in one light curve.

    After each detection, that signal's transits are masked and the search
    runs again. This is how a second or third planet shows up: the strongest
    signal (often an already-known planet) no longer hides the weaker ones.
    Stops at the first search that isn't a candidate.

    Returns a list of result dicts, with `signal` = 1, 2, 3...
    """
    results, found = [], []
    for signal in range(1, max_signals + 1):
        label = name if signal == 1 else f"{name} signal {signal}"
        result = search_light_curve(lc, name=label, mask_signals=found,
                                    plot_dir=plot_dir, **kwargs)
        result.update(target=name, signal=signal)
        results.append(result)
        if not result["is_candidate"]:
            break
        found.append((result["period_days"], result["t0_btjd"],
                      result["duration_hours"] / 24))
        if signal_mask(lc.remove_nans(), found).mean() > 0.3:
            break   # most of the data is masked; anything left is unreliable
    return results


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
