"""
Step 1 of the project: prove the pipeline recovers planets we already know.

    python validate_known_planets.py

Downloads one TESS sector per star in known_planets.csv, runs the search,
and writes results/known_planets_results.csv plus a diagnostic plot per
star in results/plots/. The `period_check` column says whether the BLS
period matches the published one ("match"), lands on a harmonic
("alias 2x" / "alias 1/2x"), or misses.
"""

from pathlib import Path

import pandas as pd

import transit_search as ts

if __name__ == "__main__":
    out = Path("results")
    out.mkdir(exist_ok=True)

    targets = pd.read_csv("known_planets.csv")
    table = ts.search_many(targets, csv_path=out / "known_planets_results.csv",
                           plot_dir=out / "plots", max_sectors=1)

    columns = ["planet", "known_period_days", "period_days", "period_check",
               "depth_ppm", "duration_hours", "sde", "depth_snr",
               "odd_even_sigma", "odd_even_diff_pct", "error"]
    print(table[[c for c in columns if c in table.columns]].to_string(index=False))
