"""
unWISE flipbooks: the same patch of sky in two WISE eras, side by side.

unWISE (unwise.me) serves sharp coadds of WISE images. We use:
  allwise  the original WISE mission, 2010-2011
  neo7     everything through 2020 (WISE + 7 years of NEOWISE)
A source that hasn't moved looks the same in both. A nearby brown dwarf
has shifted, and a fast one smears into a streak in neo7, because neo7
averages over ten years of motion. The difference image (neo7 - allwise)
shows the old position as a dark spot and the new one as a bright spot or
trail.
"""

from __future__ import annotations

import datetime as dt
import io
import sys
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402
from astropy.io import fits  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
CUTOUT = "https://unwise.me/cutout_fits"
PIXEL_ARCSEC = 2.75

sys.path.insert(0, str(HERE.parent / "feed"))
import candidates as cards  # noqa: E402
import reports  # noqa: E402


def cutout(ra, dec, version, band=2, size=64):
    """One unWISE cutout (2-D array), cached."""
    path = CACHE / f"{version}_w{band}_{ra:.5f}_{dec:+.5f}_{size}.fits"
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        response = requests.get(CUTOUT, params={"version": version, "ra": ra, "dec": dec,
                                                "size": size, "bands": band, "file_img_m": "on"},
                                timeout=120)
        response.raise_for_status()
        with tarfile.open(fileobj=io.BytesIO(response.content)) as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith("img-m.fits"))
            path.write_bytes(tar.extractfile(member).read())
    return fits.getdata(path).astype(float)


def figure(candidate, path, size=64):
    ra, dec = candidate["ra"], candidate["dec"]
    old, new = cutout(ra, dec, "allwise", size=size), cutout(ra, dec, "neo7", size=size)
    lo, hi = np.percentile(np.concatenate([old.ravel(), new.ravel()]), [5, 99.7])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, image, title in ((axes[0], old, "WISE 2010–11 (W2)"), (axes[1], new, "2010–2020 combined (W2)")):
        ax.imshow(image, origin="lower", cmap="gray_r", vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=10)
    diff = new - old
    lim = np.percentile(np.abs(diff), 99.5)
    axes[2].imshow(diff, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[2].set_title("later − earlier", fontsize=10)
    # Expected motion over ~6 years (mean epochs 2010.5 -> ~2016), in pixels.
    years = 6.0
    dx = -candidate["pmra"] * years / PIXEL_ARCSEC     # RA increases to the left
    dy = candidate["pmdec"] * years / PIXEL_ARCSEC
    c = size / 2 - 0.5
    for ax in axes:
        ax.add_patch(plt.Circle((c, c), 5, fill=False, color="#f0a640", lw=1.2))
        ax.annotate("", xy=(c - dx, c - dy), xytext=(c, c),
                    arrowprops=dict(arrowstyle="->", color="#c79af0", lw=1.5))
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(f"{candidate['source_name']}  (circle: CatWISE position; arrow: catalogue motion × 6 yr)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def write_card(candidate):
    name = candidate["source_name"]
    plot = HERE / "results" / "plots" / f"{name}.png"
    plot.parent.mkdir(parents=True, exist_ok=True)
    figure(candidate, plot)
    gaia_g = candidate.get("gaia_phot_g_mean_mag", np.nan)
    in_gaia = np.isfinite(gaia_g)
    card = {
        "id": cards.card_id("bd", name),
        "kind": "brown_dwarf",
        "title": f"CWISE {name}",
        "subtitle": (f"W1−W2 = {candidate['w1w2']:.2f} · moving {candidate['pm']:.2f}″/yr"
                     f" · W2 = {candidate['w2mpro']:.1f}"),
        "score": float(candidate["score"]),
        "summary": ("Very red in WISE and moving: the signature of a cold, nearby brown dwarf."
                    if not in_gaia else
                    "Red and moving, but Gaia also sees it in visible light, so it's probably a warmer dwarf or a known nearby star."),
        "stats": [
            {"label": "W1 − W2", "value": f"{candidate['w1w2']:.2f} mag"},
            {"label": "W2", "value": f"{candidate['w2mpro']:.2f} mag"},
            {"label": "Motion", "value": f"{candidate['pm']:.2f}″/yr"},
            {"label": "Motion S/N", "value": f"{candidate['pm'] / np.hypot(candidate['sigpmra'], candidate['sigpmdec']):.0f}"},
            {"label": "W2 S/N", "value": f"{candidate['w2snr']:.0f}"},
            {"label": "Gaia", "value": f"G = {gaia_g:.1f}" if in_gaia else "not detected"},
        ],
        "notes": ("Check the flipbook: does a dot shift (or smear) along the arrow? A static red dot "
                  "is more likely a galaxy; a shift opposite to the arrow is suspicious."),
        "vet_status": "possible new" if not in_gaia else "in Gaia",
        "links": [
            {"label": "Legacy Survey viewer",
             "url": f"https://www.legacysurvey.org/viewer?ra={candidate['ra']:.5f}&dec={candidate['dec']:.5f}&layer=unwise-neo7&zoom=14"},
            {"label": "SIMBAD (is it known?)",
             "url": f"https://simbad.cds.unistra.fr/simbad/sim-coo?Coord={candidate['ra']:.5f}%20{candidate['dec']:+.5f}&Radius=10&Radius.unit=arcsec"},
            {"label": "Backyard Worlds", "url": "https://www.zooniverse.org/projects/marckuchner/backyard-worlds-planet-9"},
        ],
        "source": "CatWISE2020 (IRSA) + unWISE",
        **({"report": reports.brown_dwarf_report(candidate)} if not in_gaia else {}),
        "found_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    cards.write_outbox(card, plot)
    return card
