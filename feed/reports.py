"""
Report drafts for the Discovery Queue's "Report it" panel.

Each function returns a dict stored on the card as `report`:
    destination   who to report to
    form_url      where to submit
    deadline      how urgent it is
    instructions  what to do before submitting
    checklist     [{"text", "done"}]: vetting already done / still to do
    fields        [{"label", "value"}]: values to copy into the form

The person submits from their own account; Claude never submits.
Uncertainties here are rough estimates from our own pipeline and say so.
"""

from __future__ import annotations

import numpy as np

BTJD_OFFSET = 2457000.0           # TESS times are BJD - 2457000


def planet_report(r, sector_check=""):
    """ExoFOP-TESS CTOI draft for a planet candidate (a search result row)."""
    snr = max(float(r["detection_snr"]), 1.0)
    duration_d = float(r["duration_hours"]) / 24
    # Rough timing precision: a transit's mid-time is known to about
    # duration / (2 x SNR); spread over the transits observed for the period.
    t0_err = duration_d / (2 * snr) * np.sqrt(max(int(r["n_transits"]), 1))
    n_cycles = max(int(r["n_transits"]) - 1, 1)
    period_err = np.sqrt(2) * t0_err / n_cycles
    depth_err = float(r["depth_ppm"]) / snr
    tic = str(r["target"]).replace("TIC", "").strip()
    confirmed_elsewhere = sector_check == "confirmed in other sectors"
    return {
        "destination": "ExoFOP-TESS (CTOI)",
        "form_url": "https://exofop.ipac.caltech.edu/tess/",
        "deadline": "No rush: nobody is racing you for a particular star, but check it isn't a TOI by then.",
        "instructions": ("Log in to ExoFOP, check the target page for new TOIs/CTOIs, then upload a CTOI with "
                         "these values. Uncertainties are rough estimates from our pipeline; say so in the notes."),
        "checklist": [
            {"text": "Not a known planet, TOI or CTOI (checked when found)", "done": True},
            {"text": "Found again in another TESS sector", "done": confirmed_elsewhere},
            {"text": "Passed artifact checks (single event, edges, wavy baseline)", "done": True},
            {"text": "Odd/even depths consistent", "done": float(r.get("odd_even_diff_pct", 0) or 0) <= 20},
            {"text": "Neighbouring stars considered (see card notes)", "done": True},
            {"text": "Re-check the ExoFOP target page right before submitting", "done": False},
        ],
        "fields": [
            {"label": "TIC ID", "value": tic},
            {"label": "Period (days)", "value": f"{r['period_days']:.5f} ± {period_err:.5f}"},
            {"label": "Epoch (BJD_TDB)", "value": f"{float(r['t0_btjd']) + BTJD_OFFSET:.4f} ± {t0_err:.4f}"},
            {"label": "Depth (ppm)", "value": f"{r['depth_ppm']:.0f} ± {depth_err:.0f}"},
            {"label": "Duration (hours)", "value": f"{r['duration_hours']:.2f} ± 0.5 (search grid step)"},
            {"label": "Planet radius (Earth radii)",
             "value": f"{float(r.get('planet_radius_earth', np.nan)):.2f}"},
            {"label": "Notes",
             "value": (f"Found with a BLS search of TESS SPOC 2-min PDCSAP data (sector {r.get('sectors', '?')}); "
                       f"SDE {r['sde']:.1f}, SNR {snr:.1f}, {int(r['n_transits'])} transits. "
                       + ("Recovered independently in other sectors. " if confirmed_elsewhere else "")
                       + "Uncertainties are approximate. Amateur search; follow-up welcome.")},
        ],
    }


def comet_report(row):
    """Sungrazer Project report draft for a SOHO comet candidate (a scan.py row)."""
    group = "Kreutz (likely: heading straight for the Sun)" if row["sunward"] > 0.8 else "Non-group / unsure"
    positions = row.get("positions") or []
    return {
        "destination": "Sungrazer Project",
        "form_url": "https://sungrazer.nrl.navy.mil/index.php/report",
        "deadline": "Report within hours. Other hunters search the same images, and credit goes to the first report.",
        "instructions": ("Open the same frames in the Sungrazer project's own images and re-measure the comet's "
                         "x, y in at least 4 frames with their measuring tool (our positions are converted "
                         "approximately, good to ~25 px). Then submit."),
        "checklist": [
            {"text": "Moves against the stars and toward the Sun", "done": True},
            {"text": "Stays on its track in the close-ups", "done": True},
            {"text": "Not in recent Sungrazer reports (checked at scan time)", "done": True},
            {"text": "Re-measure positions on the project's images", "done": False},
            {"text": "Check the newest reports page right before submitting", "done": False},
        ],
        "fields": [
            {"label": "Telescope", "value": f"LASCO {row['camera']}"},
            {"label": "Date (UT)", "value": f"{row['time']:%Y-%m-%d}"},
            {"label": "Group", "value": group},
            {"label": "Positions (HHMM x y, approximate)", "value": "\n".join(positions) or "see card"},
        ],
    }


def brown_dwarf_report(c):
    """Backyard Worlds route for a brown dwarf candidate (a search.py row)."""
    return {
        "destination": "Backyard Worlds: Planet 9",
        "form_url": "https://www.zooniverse.org/projects/marckuchner/backyard-worlds-planet-9",
        "deadline": "No rush.",
        "instructions": ("Check SIMBAD at the position for a known object first. If nothing is known, post it "
                         "on the project's Talk board with these details so the science team can check it "
                         "against their lists and follow it up."),
        "checklist": [
            {"text": "Very red in WISE (W1-W2) and moving", "done": True},
            {"text": "Motion visible in the flipbook", "done": False},
            {"text": "Not in SIMBAD within ~10 arcsec", "done": False},
        ],
        "fields": [
            {"label": "Name / position", "value": f"CWISE {c['source_name']} (RA {c['ra']:.5f}, Dec {c['dec']:+.5f})"},
            {"label": "Proper motion", "value": f"{c['pm']:.2f} arcsec/yr (pmRA {c['pmra']:.3f}, pmDec {c['pmdec']:.3f})"},
            {"label": "Photometry", "value": f"W1-W2 = {c['w1w2']:.2f}, W2 = {c['w2mpro']:.2f}"},
        ],
    }


def transient_report(obj, a):
    """Transient Name Server draft for a fast-fading ZTF transient."""
    return {
        "destination": "Transient Name Server (TNS)",
        "form_url": "https://www.wis-tns.org/",
        "deadline": "Within a day: fast transients fade, and follow-up needs to start quickly.",
        "instructions": ("Search TNS at the position first (it may already have a name). If it's new, report it "
                         "as an AT with the ZTF photometry; mention the fast fade in the remarks."),
        "checklist": [
            {"text": f"Fades {a['fade_rate']:.2f} mag/day at {a['fade_sigma']:.0f}σ", "done": True},
            {"text": "Sits on or near a galaxy, not a star (check the ALeRCE images)", "done": False},
            {"text": "Not already on TNS", "done": False},
        ],
        "fields": [
            {"label": "Position (RA, Dec)", "value": f"{obj['meanra']:.5f}, {obj['meandec']:+.5f}"},
            {"label": "Discovery data", "value": f"ZTF ({obj['oid']}), peak {a['peak_mag']:.1f} mag"},
            {"label": "Remarks", "value": f"Fast fade {a['fade_rate']:.2f} ± {a['fade_err']:.2f} mag/day in {a['fade_band']}"},
        ],
    }
