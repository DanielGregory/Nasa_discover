"""
Map the pipeline's sensitivity with injection-recovery tests.

    python run_injection_recovery.py              # 300 injections per star
    python run_injection_recovery.py --n 50       # quick look
    python run_injection_recovery.py --replot     # redraw plots from saved CSVs

For each star in quiet_stars.csv (made with injection_recovery.find_quiet_stars),
injects fake planets into its real TESS light curve and runs the normal
search on each. Writes, per star:
    results/injections/<star>.csv      one row per injection
    results/injections/<star>.png      recovery map + recovery vs SNR
and a cross-star comparison, results/injections/recovery_vs_radius.png.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import injection_recovery as ir
import transit_search as ts

# Categorical colours, in fixed order (one per star).
STAR_COLORS = ["#2a78d6", "#eb6834", "#1fa37a", "#8f5cd6"]


def run_star(star, n, out_dir, seed, replot=False):
    name = star["target"]
    csv_path = out_dir / f"{ts.safe_filename(name)}.csv"
    if replot:
        results = pd.read_csv(csv_path)
    else:
        lc = ts.download_light_curve(name, sectors=[int(s) for s in str(star["sectors"]).split(",")])
        print(f"{name}: R = {star['stellar_radius']:.2f} Rsun, "
              f"CDPP = {star['cdpp_1h_ppm']:.0f} ppm, {n} injections")
        results = ir.run_injections(lc, n=n, stellar_radius=star["stellar_radius"],
                                    stellar_density=star["stellar_density"], seed=seed)
        results.insert(0, "target", name)
        results.to_csv(csv_path, index=False)

    fig, (ax_map, ax_snr) = plt.subplots(1, 2, figsize=(15, 5.8),
                                         gridspec_kw={"width_ratios": [1.15, 1]})
    label = (f"{name}  (R = {star['stellar_radius']:.2f} R☉, Teff = {star['teff']:.0f} K, "
             f"1-h CDPP = {star['cdpp_1h_ppm']:.0f} ppm)")
    ir.plot_recovery_map(results, title="Recovery fraction", ax=ax_map)
    ir.plot_recovery_vs_snr(results, title="Recovery vs. expected SNR", ax=ax_snr)
    fig.suptitle(label, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / f"{ts.safe_filename(name)}.png", dpi=110)
    plt.close(fig)
    return results


def plot_radius_comparison(all_results, stars, path, n_bins=8):
    """Recovery fraction vs planet radius, one line per star."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    radii = all_results["planet_radius_earth"]
    edges = np.geomspace(radii.min(), radii.max() * 1.0001, n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    for color, (_, star) in zip(STAR_COLORS, stars.iterrows()):
        res = all_results[all_results["target"] == star["target"]]
        idx = np.digitize(res["planet_radius_earth"], edges) - 1
        frac = [res["recovered"][idx == k].mean() if (idx == k).sum() >= 5 else np.nan
                for k in range(n_bins)]
        ax.plot(centers, frac, "-o", color=color, lw=2, ms=8,
                markeredgecolor="white", markeredgewidth=2,
                label=f"{star['target']} ({star['stellar_radius']:.2f} R☉)")
    ax.axvline(1.0, color="0.6", ls="--", lw=1)
    ax.text(1.03, 0.03, "Earth", color="0.4", fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks([0.3, 0.5, 1, 2, 3, 5, 8])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("Injected planet radius [Earth radii]")
    ax.set_ylabel("Fraction recovered (P = 0.5–13 d)")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Same noise level, smaller star → smaller planets detectable")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=300, help="injections per star")
    parser.add_argument("--stars", default="quiet_stars.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replot", action="store_true",
                        help="redraw plots from saved CSVs instead of running injections")
    args = parser.parse_args()

    out_dir = Path("results/injections")
    out_dir.mkdir(parents=True, exist_ok=True)
    stars = pd.read_csv(args.stars)

    all_results = pd.concat([run_star(star, args.n, out_dir, args.seed + i, args.replot)
                             for i, (_, star) in enumerate(stars.iterrows())])
    all_results.to_csv(out_dir / "all_injections.csv", index=False)
    plot_radius_comparison(all_results, stars, out_dir / "recovery_vs_radius.png")

    summary = all_results.groupby("target")["status"].value_counts().unstack(fill_value=0)
    print(summary.to_string())
