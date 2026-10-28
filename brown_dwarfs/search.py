"""
Search NASA's WISE data for cold, nearby brown dwarfs.

Brown dwarfs are "failed stars": too small to fuse hydrogen, so they cool
and fade forever. The coldest ones (T and Y types, a few hundred degrees C)
are nearly invisible in optical light but glow in the infrared. Two
signatures make them stand out in WISE's two short-wavelength bands:

  colour  W1 - W2 is large: methane absorbs W1 (3.4 um) light but not W2
          (4.6 um). Ordinary stars have W1 - W2 ~ 0.
  motion  the nearby ones move across the sky fast enough to see in WISE
          images taken years apart (that's what Backyard Worlds volunteers
          look for in flipbooks).

We query CatWISE2020 (a catalogue of ~1.9 billion WISE sources with motions
measured over 2010-2018) at NASA's IRSA, in sky tiles, keep sources that
are red, moving significantly and free of image artifacts, then check them
against Gaia DR3 (anything Gaia sees well in optical light with a large
parallax is a known nearby star, not a cold brown dwarf).

    python search.py --tiles 5                      # 5 random 10x10 deg tiles
    python search.py --ra 120 130 --dec -10 0       # one specific tile

Candidates become Discovery Queue cards with an unWISE flipbook (flipbook.py).
"""

from __future__ import annotations

import argparse
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
TAP = "https://irsa.ipac.caltech.edu/TAP/sync"

# Selection. W1-W2 > 1.5 picks mid/late T and Y dwarfs; > 1.0 would add
# early T dwarfs but many more red background galaxies. Sources brighter
# than W2 = 8 are saturated in WISE, which fakes a red colour.
MIN_W1W2 = 1.5
MIN_W2_SNR = 10
MIN_PM = 0.3            # arcsec/year
MIN_PM_SIGNIFICANCE = 5


def tap(query, retries=3):
    for attempt in range(retries):
        try:
            response = requests.get(TAP, params={"QUERY": query, "FORMAT": "csv"}, timeout=600)
            response.raise_for_status()
            return pd.read_csv(StringIO(response.text))
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def query_tile(ra_range, dec_range):
    """Red, significantly moving, clean CatWISE2020 sources in one tile."""
    pm = "SQRT(pmra*pmra + pmdec*pmdec)"
    pm_err = "SQRT(sigpmra*sigpmra + sigpmdec*sigpmdec)"
    return tap(f"""
        SELECT source_name, ra, dec, w1mpro, w2mpro, w1sigmpro, w2sigmpro,
               w1mpro - w2mpro AS w1w2, w2snr, pmra, pmdec, sigpmra, sigpmdec,
               {pm} AS pm, ab_flags, cc_flags, par_pm, par_pmsig
        FROM catwise_2020
        WHERE ra BETWEEN {ra_range[0]} AND {ra_range[1]}
          AND dec BETWEEN {dec_range[0]} AND {dec_range[1]}
          AND w1mpro - w2mpro > {MIN_W1W2}
          AND w2snr > {MIN_W2_SNR}
          AND {pm} > {MIN_PM}
          AND {pm} > {MIN_PM_SIGNIFICANCE} * {pm_err}
          AND ab_flags = '00'
          AND w2mpro > 8
    """)


def gaia_neighbours(candidates, radius_arcsec=3.0):
    """
    For each candidate, the nearest Gaia DR3 source within `radius_arcsec`
    (G magnitude and parallax), or NaN. Cold brown dwarfs are usually too
    faint for Gaia; a bright Gaia match with a big parallax means a known
    nearby star or warmer dwarf.
    """
    rows = []
    for _, c in candidates.iterrows():
        match = tap(f"""
            SELECT TOP 1 source_id, phot_g_mean_mag, parallax, parallax_error,
                   DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {c['ra']}, {c['dec']})) * 3600 AS sep
            FROM gaia_dr3_source
            WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {c['ra']}, {c['dec']}, {radius_arcsec / 3600}))=1
            ORDER BY sep
        """)
        rows.append(match.iloc[0].to_dict() if len(match) else {})
    columns = ["source_id", "phot_g_mean_mag", "parallax", "parallax_error", "sep"]
    gaia = pd.DataFrame(rows, index=candidates.index).reindex(columns=columns).add_prefix("gaia_")
    return candidates.join(gaia)


def merge_split_detections(candidates, radius_arcsec=12.0, years=8.0):
    """
    Very fast movers can appear as several CatWISE entries along their path
    (WISE 0855-0714 shows up four times, up to 16" apart). Keep the
    highest-SNR entry within max(`radius_arcsec`, its motion over `years`)
    that moves the same way.
    """
    candidates = candidates.sort_values("w2snr", ascending=False).reset_index(drop=True)
    keep = []
    for i, c in candidates.iterrows():
        dup = False
        for j in keep:
            k = candidates.loc[j]
            sep = 3600 * np.hypot((c["ra"] - k["ra"]) * np.cos(np.radians(c["dec"])), c["dec"] - k["dec"])
            if sep < max(radius_arcsec, k["pm"] * years) and np.sign(c["pmra"]) == np.sign(k["pmra"]):
                dup = True
                break
        if not dup:
            keep.append(i)
    return candidates.loc[keep].reset_index(drop=True)


def score(c):
    """Ranking: redder, faster and more significant first (0-1)."""
    redness = np.clip((c["w1w2"] - MIN_W1W2) / 2.0, 0, 1)
    motion = np.clip(np.log10(c["pm"] / MIN_PM) / 1.0, 0, 1)
    snr = np.clip((c["w2snr"] - MIN_W2_SNR) / 30, 0, 1)
    gaia_star = np.isfinite(c.get("gaia_phot_g_mean_mag", np.nan))
    return round(float((0.45 * redness + 0.35 * motion + 0.2 * snr) * (0.3 if gaia_star else 1.0)), 3)


def random_tiles(n, seed=None, size=10):
    """Random 10x10 degree tiles, avoiding the crowded Galactic plane."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    rng = np.random.default_rng(seed)
    tiles = []
    while len(tiles) < n:
        ra = rng.uniform(0, 360 - size)
        dec = np.degrees(np.arcsin(rng.uniform(-1, np.sin(np.radians(90 - size)))))
        b = SkyCoord(ra + size / 2, dec + size / 2, unit="deg").galactic.b.to(u.deg).value
        if abs(b) > 15:
            tiles.append(((round(ra, 1), round(ra + size, 1)), (round(dec, 1), round(dec + size, 1))))
    return tiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiles", type=int, default=3)
    parser.add_argument("--ra", type=float, nargs=2)
    parser.add_argument("--dec", type=float, nargs=2)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-cards", type=int, default=10)
    args = parser.parse_args()

    tiles = [(tuple(args.ra), tuple(args.dec))] if args.ra else random_tiles(args.tiles, args.seed)
    found = []
    for ra_range, dec_range in tiles:
        t = query_tile(ra_range, dec_range)
        print(f"tile RA {ra_range} Dec {dec_range}: {len(t)} red movers", flush=True)
        found.append(t)
    candidates = merge_split_detections(pd.concat(found, ignore_index=True))
    if candidates.empty:
        print("No candidates.")
        sys.exit(0)
    candidates = gaia_neighbours(candidates)
    candidates["score"] = candidates.apply(score, axis=1)
    candidates = candidates.sort_values("score", ascending=False)
    OUT.mkdir(exist_ok=True)
    candidates.to_csv(OUT / "candidates.csv", index=False)
    print(candidates[["source_name", "w2mpro", "w1w2", "pm", "w2snr", "gaia_phot_g_mean_mag", "score"]]
          .head(15).round(2).to_string(index=False))

    import flipbook
    for _, c in candidates.head(args.max_cards).iterrows():
        flipbook.write_card(c)
    print(f"{min(len(candidates), args.max_cards)} card(s) written to feed/outbox/")
