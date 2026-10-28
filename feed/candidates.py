"""
Turn search results into Discovery Queue cards.

Every search (planets, comets, brown dwarfs, transients) produces the same
card shape, so one page can show them all:

    {
      "kind":       "planet" | "comet" | "brown_dwarf" | "transient",
      "title":      short name, e.g. "TIC 261136679",
      "subtitle":   one line of key numbers,
      "score":      0-1 ranking (higher = review first),
      "summary":    one plain sentence,
      "stats":      [{"label": ..., "value": ...}, ...],
      "notes":      vetting flags,
      "vet_status": e.g. "possible new", "known",
      "links":      [{"label": ..., "url": "https://..."}],
      "source":     which data product,
      "found_at":   ISO date,
      "plot_asset": id of the uploaded plot (filled in when uploaded),
    }

write_outbox() saves each card as JSON next to its plot in feed/outbox/.
Claude then uploads the plots and writes the cards to the page's database.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

OUTBOX = Path(__file__).resolve().parent / "outbox"


def card_id(*parts):
    """A database-safe id: letters, digits and - . _ only."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", "-".join(str(p) for p in parts)).strip("-")


def _fmt(value, digits=1):
    return "—" if value is None or not np.isfinite(value) else f"{value:,.{digits}f}"


def planet_score(result, vetting):
    """
    Ranking only (what to look at first), not a probability of being real.
    Signal strength sets the base; vetting problems scale it down.
    """
    strength = 0.5 * np.clip((result["sde"] - 7) / 15, 0, 1) \
        + 0.5 * np.clip(np.log10(max(result["detection_snr"], 1) / 7) / 1.5, 0, 1)
    penalty = {"possible new": 1.0, "weak": 0.4, "not a planet": 0.2,
               "known neighbour": 0.15, "known": 0.1}.get(vetting["vet_status"], 0.5)
    if vetting["n_bright_neighbours"]:
        penalty *= 0.85
    return float(np.round(strength * penalty, 3))


def planet_card(result, vetting, source="TESS SPOC 2-min"):
    tic = str(result["target"]).replace("TIC", "").strip()
    radius = vetting["planet_radius_earth"]
    sectors = str(result.get("sectors", "") or "")
    status = vetting["vet_status"]
    summary = {
        "possible new": "A periodic dip not matched to any known planet, TOI or CTOI on this star.",
        "known": "Matches a known planet or TOI. Shown to check the pipeline, not a new find.",
        "known neighbour": "Same period as a known signal on a nearby star, so probably light leaking in from it.",
        "not a planet": "Too big or too uneven to be a planet; most likely an eclipsing binary.",
        "weak": "Below the detection thresholds; probably noise.",
    }.get(status, "")
    return {
        "id": card_id("planet", f"tic{tic}", f"p{result['period_days']:.4f}"),
        "kind": "planet",
        "title": f"TIC {tic}",
        "subtitle": (f"P = {result['period_days']:.4f} d · {result['depth_ppm']:,.0f} ppm"
                     f" · ~{_fmt(radius)} R⊕"),
        "score": planet_score(result, vetting),
        "summary": summary,
        "stats": [
            {"label": "Period", "value": f"{result['period_days']:.5f} d"},
            {"label": "Depth", "value": f"{result['depth_ppm']:,.0f} ppm"},
            {"label": "Duration", "value": f"{result['duration_hours']:.1f} h"},
            {"label": "Radius", "value": f"{_fmt(radius, 2)} ± {_fmt(vetting['planet_radius_err'], 2)} R⊕"},
            {"label": "SDE", "value": _fmt(result["sde"])},
            {"label": "SNR", "value": _fmt(result["detection_snr"])},
            {"label": "Transits", "value": str(int(result["n_transits"]))},
            {"label": "Odd/even", "value": f"{_fmt(result['odd_even_diff_pct'], 0)}% ({_fmt(result['odd_even_sigma'])}σ)"},
            {"label": "Star", "value": f"T = {_fmt(vetting['tmag'])}, {_fmt(vetting['teff'], 0)} K, {_fmt(vetting['stellar_radius'], 2)} R☉"},
            {"label": "Neighbours", "value": f"{vetting['n_bright_neighbours']} could blend"},
        ],
        "notes": vetting["vet_notes"],
        "vet_status": status,
        "links": [
            {"label": "ExoFOP", "url": f"https://exofop.ipac.caltech.edu/tess/target.php?id={tic}"},
        ],
        "source": source + (f", sector {sectors}" if sectors else ""),
        "found_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def write_outbox(card, plot_path=None, outbox=OUTBOX):
    """Save a card (and a copy of its plot) for upload. Returns the JSON path."""
    outbox.mkdir(parents=True, exist_ok=True)
    if plot_path:
        shutil.copy(plot_path, outbox / f"{card['id']}.png")
    path = outbox / f"{card['id']}.json"
    path.write_text(json.dumps(card, indent=1, ensure_ascii=False))
    return path
