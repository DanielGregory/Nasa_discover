# NASA Discover: a TESS transit search, built to learn the method

A small, readable pipeline that downloads public NASA TESS light curves,
cleans them, and searches for transiting planets with Box Least Squares
(BLS).

**Current goal: show that the pipeline recovers known planets.** Use it for discovery
only after it reliably finds planets we already know about.

```
transit_search.py              the whole pipeline (download → clean → BLS → plots → CSV)
validate_known_planets.py      runs the pipeline on known_planets.csv
known_planets.csv              known planet hosts with their published periods
notebooks/01_recover_known_planet.ipynb   step-by-step walkthrough on Pi Mensae c
tests/test_synthetic.py        offline tests: inject a fake transit, check we recover it
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                                   # offline tests (~1 min)
python validate_known_planets.py         # downloads TESS data, writes results/
jupyter lab notebooks/                   # the walkthrough
```

In Python:

```python
import transit_search as ts

result = ts.search_target("Pi Men", show=True)          # one star
table  = ts.search_many(["Pi Men", "WASP-18"], csv_path="results.csv")  # many stars
```

Each CSV row contains the best period, mid-transit time (`t0_btjd`), duration,
depth, depth SNR, SDE, number of transits, and odd/even depths. If you supply
`known_period_days`, it also adds a `period_check` column (`match`,
`alias 2x`, `alias 1/2x`, `miss`).

---

## What is a transit?

When a planet passes in front of its star as seen from Earth, it blocks a
small fraction of the starlight. The star dims a little for a few hours, then
recovers, and this repeats every orbit.

```
flux
1.000 ──────────┐          ┌──────────────────┐          ┌──────
                │          │                  │          │
0.9997          └──────────┘                  └──────────┘
                 ← duration →
                ←──────────── period ─────────→
```

Three numbers describe it:

- **Depth** ≈ (R_planet / R_star)². Jupiter in front of the Sun gives about 1%
  (10,000 ppm). Earth in front of the Sun gives about 0.008% (84 ppm).
- **Period**: the orbital period, i.e. the time between transits.
- **Duration**: how long the planet takes to cross the disk. This depends on
  the period, the star's size, and how centrally the planet crosses
  (the "impact parameter").

TESS photographs most of the sky every 2 minutes (or every 20 s to 30 min,
depending on the data product) for ~27-day "sectors". A **light curve** is
one star's brightness over that time.

## The pipeline, step by step

1. **Download** (`download_light_curve`). We use SPOC **PDCSAP** flux. NASA's
   pipeline has already removed most spacecraft systematics (pointing jitter,
   scattered light) and corrected for light from nearby stars. Cadences
   flagged as bad (for example during momentum dumps) are dropped.
2. **Clean** (`clean_light_curve`).
   - *Remove NaNs*: gaps and flagged cadences.
   - *Flatten*: stars vary on their own. Starspots rotate in and out of
     view, and stars pulsate and flare. We fit a smooth Savitzky–Golay trend
     and divide it out. The smoothing window must be **much longer than a
     transit** (we use 1 day, against transits of a few hours). Otherwise the
     filter follows the dip and erases it.
   - *Clip outliers asymmetrically*: transits are dips, so we clip upward
     spikes (cosmic rays, flares) hard but leave low points alone. A
     symmetric clip can delete the transit you are looking for.
3. **Search** (`run_bls`). See below.
4. **Two passes** (`search_light_curve`). Even a 1-day window partly
   absorbs transits, so the depth comes out too small. After the first BLS
   pass we know when the transits are. We mask them, flatten again so the
   trend is fitted only to out-of-transit data, and search again. On
   synthetic data this raises the measured depth from ~190 ppm to the true
   300 ppm.
5. **Plot** (`plot_diagnostics`). Four panels: raw, flattened (with
   predicted transit times), periodogram, folded light curve.

## How Box Least Squares works

BLS (Kovács, Zucker & Mazeh 2002) models a transit as a **box**: the flux is
at one level out of transit and a lower level in transit. For every trial
combination of

- **period** P (thousands of values),
- **duration** (we try 0.5–8 hours),
- **phase** (when in the orbit the box sits),

it folds the light curve at P, places the box, and computes how much better
"flat line + box" fits the data than a flat line alone. The improvement is
the **BLS power** (here the change in log-likelihood). Plotting the best
power for each period gives the **periodogram**. A real periodic transit
shows up as a sharp peak, usually with smaller peaks at P/2, 2P, and so on.

Why a box rather than a realistic transit shape? It is fast enough to test
millions of combinations. It is also close enough to the real shape to find
the signal. Detailed modelling (limb darkening, ingress/egress) comes later,
once you have a candidate.

**The period grid.** Periods are spaced evenly in frequency, with spacing
set by the shortest duration and the total time span. A longer time span
needs a finer grid, and the number of trial periods grows roughly with the
**square** of the span. That is why the defaults use one sector, and why
stitching sectors from years apart is slow.

### Measures of signal strength

| Quantity | Meaning | Rule of thumb |
|---|---|---|
| `sde` | Signal Detection Efficiency: (peak − mean power) / std of power. How far the peak stands above the rest of the periodogram. | ≳ 9 interesting |
| `depth_snr` | depth / uncertainty on depth | ≳ 7 interesting |
| `n_transits` | individual transits that have data | ≥ 2 to know a period; ≥ 3 preferred |
| `odd_even_sigma` | difference between odd- and even-numbered transit depths, in σ | ≳ 3 suspicious (see below) |

`is_candidate` is `True` when SDE ≥ 9 and SNR ≥ 7. These are thresholds
for "look closer", not proof.

## Real planets vs. false positives

**Most transit-like signals are not planets.** Before believing a signal,
work through these checks:

**Eclipsing binaries (EBs).** These are two stars orbiting each other. They are the most
common impostor.
- *Odd/even depth difference.* If the two stars differ, the primary and
  secondary eclipses have different depths. BLS often finds half the true
  period, which folds the two eclipses on top of each other. A large
  `odd_even_sigma` is a warning sign. Check the periodogram peak at 2P
  as well.
- *Secondary eclipse.* Fold at the period and look at phase 0.5. A dip
  there means the companion emits its own light, so it is probably a star.
  (Very hot Jupiters can show a tiny secondary eclipse too, but only tens
  of ppm.)
- *Shape.* Planets give flat-bottomed, "U"-shaped transits. Grazing EBs
  give "V" shapes with no flat bottom.
- *Too deep / too long.* A depth over a few percent, or a duration longer
  than a planet could have around that star, implies a companion
  bigger than a planet.

**Blended (background) eclipsing binaries.** TESS pixels are big (21″),
so light from several stars mixes in one aperture. A deep EB a few pixels
away, diluted by your bright target, can look like a shallow planet
transit. Checking for this requires the pixel data (target pixel files):
does the dip come from the target star? lightkurve can help here.

**Stellar variability and noise.**
- Flattening with a window that is too short creates fake dips, and so
  does leftover spot modulation.
- Spacecraft effects (momentum dumps, scattered light at the start of an
  orbit) cause dips at the same time in many stars. If several unrelated
  stars "share" a transit time, it is the spacecraft.
- A single dip is not a period. Check `n_transits` and look at each
  transit in panel 2: one bad cadence can dominate a fold.
- Peaks at exactly **13.7 days** (the TESS orbit) or at 1 day and its
  fractions are suspicious.

**Aliases.** The BLS peak can be at P/2 or 2P of the true period. Panel 3
marks both. If the folded curve at 2P shows two dips of equal depth, the
true period is P.

**What astronomers do next** (outside this project's scope): check that the
signal appears in independent sectors, inspect the pixels for the source of
the dip (centroid tests), compare against Gaia for nearby stars, fit a
physical transit model, and finally confirm with radial velocities. NASA's
TESS team runs a version of this "vetting" before it publishes
a TOI (TESS Object of Interest).

## Validation plan

1. ✅ Offline: inject synthetic transits, check period/depth/duration come
   back (`pytest`).
2. ☐ Run `validate_known_planets.py`: every planet in `known_planets.csv`
   should come back as `match`.
3. ☐ Inject fake transits into **real** light curves of quiet stars at
   different depths and periods to measure the smallest signal the pipeline
   can recover.
4. ☐ Only then: run it on stars without known planets.

## Data and credits

TESS data come from the Mikulski Archive for Space Telescopes (MAST) via
[lightkurve](https://docs.lightkurve.org/). BLS is
[`astropy.timeseries.BoxLeastSquares`](https://docs.astropy.org/en/stable/timeseries/bls.html).
Published periods in `known_planets.csv` are approximate and come from the discovery
papers. Check them against the
[NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/).
