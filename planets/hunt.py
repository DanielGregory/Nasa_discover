"""
Hunt for new planets and send candidates to the Discovery Queue.

    python hunt.py --hosts 20 --fresh 20      # 20 known hosts + 20 new stars

Two kinds of targets, both M dwarfs (small stars make small planets easy
to see; see the injection-recovery results):

  hosts   stars with exactly one known planet or TOI. search_multi() finds
          the known one first, masks it, and keeps looking. Extra planets
          in known systems are the most likely place for a new find.
  fresh   M dwarfs with 2-minute TESS data and no known planet, TOI or CTOI.

Each signal that passes the detection thresholds is vetted (vetting.vet).
Anything that isn't simply a known planet becomes a card in
../feed/outbox/ for upload to the Discovery Queue. Writes every signal to
results/hunt/signals.csv and plots to results/hunt/plots/.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import transit_search as ts
import vetting

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "feed"))
import candidates as cards  # noqa: E402

OUT = Path("results/hunt")
# Stars already hunted (committed to the repo, so scheduled runs move on).
SEARCHED = Path(__file__).resolve().parent / "searched_stars.csv"


def already_searched():
    if not SEARCHED.exists():
        return set()
    return set(pd.read_csv(SEARCHED)["tic"].astype(int))
# What reaches the Discovery Queue. Chosen from the first reviews: weak,
# noise-like single-sector signals weren't worth the person's time.
SEND_TO_QUEUE = {"possible new"}
MIN_SNR_SINGLE_SECTOR = 10     # with nothing else to check, demand a stronger signal


def host_targets(n, known, skip=frozenset()):
    """M dwarfs (Teff < 4000 K, T < 13) hosting exactly one known signal."""
    tois = pd.read_csv(vetting.CACHE_DIR / "tois.csv")
    extra = vetting._cached_csv("toi_stars", lambda: vetting._archive_query(
        "select tid, st_teff, st_rad, st_tmag from toi"))
    stars = extra.drop_duplicates("tid")
    stars = stars[(stars["st_teff"] < 4000) & (stars["st_tmag"] < 13)]
    counts = known.groupby("tic").size()
    single = counts[counts == 1].index
    # Exclude stars whose only TOI is a known false positive.
    false_pos = set(tois.loc[tois["tfopwg_disp"].isin(["FP", "FA"]), "tid"])
    stars = stars[stars["tid"].isin(single) & ~stars["tid"].isin(false_pos)
                  & ~stars["tid"].isin(skip)]
    return [f"TIC {t}" for t in stars.sort_values("st_tmag")["tid"].head(n)]


def fresh_targets(n, known, skip=frozenset(), dec=(-90, -30), max_checked=1000):
    """
    Bright M dwarfs from the TIC with no known planet, TOI or CTOI, that
    have SPOC 2-minute data. Only TESS's pre-selected targets have 2-minute
    data, so we check each star until we have `n`.
    """
    import lightkurve as lk
    from astroquery.mast import Catalogs

    tic = Catalogs.query_criteria(catalog="TIC", Teff=[2800, 3900], rad=[0.1, 0.6],
                                  Tmag=[7, 12], lumclass="DWARF", objType="STAR",
                                  dec=list(dec)).to_pandas()
    tic = tic[~tic["ID"].astype(int).isin(set(known["tic"]) | set(skip))].sort_values("Tmag")
    picked = []
    for tic_id in tic["ID"].head(max_checked):
        target = f"TIC {tic_id}"
        if len(lk.search_lightcurve(target, mission="TESS", author="SPOC", exptime=120)):
            picked.append(target)
            if len(picked) == n:
                break
    return picked


def hunt_one(target, kind):
    """Download, search for up to 3 signals, vet each. Returns signal rows."""
    try:
        lc = ts.download_light_curve(target, max_sectors=1)
    except Exception as exc:
        return [{"target": target, "kind": kind, "error": f"{type(exc).__name__}: {exc}"}]
    sectors = ",".join(str(s) for s in lc.meta.get("SECTORS", []))
    rows = []
    known = vetting.known_signals()
    # min_period 0.3 d: ultra-short-period planets are common around M
    # dwarfs, and starting at 0.5 d finds them at twice their period.
    for result in ts.search_multi(lc, name=target, plot_dir=OUT / "plots", min_period=0.3):
        row = {**result, "kind": kind, "sectors": sectors, "error": ""}
        if result["is_candidate"]:
            try:
                row.update(vetting.vet(result, known))
                if row["vet_status"] == "possible new":
                    check, note = vetting.other_sectors_check(
                        target, result, [int(s) for s in sectors.split(",") if s])
                    row["sector_check"] = check
                    row["vet_notes"] = "; ".join(filter(None, [row["vet_notes"], note]))
                    if check == "not confirmed":
                        row["vet_status"] = "not confirmed"
                    elif check != "confirmed in other sectors" and \
                            result["detection_snr"] < MIN_SNR_SINGLE_SECTOR:
                        row["vet_status"] = "weak"
                        row["vet_notes"] += f"; SNR {result['detection_snr']:.1f} is too weak with no second sector"
            except Exception as exc:
                row["error"] = f"vetting failed: {type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hosts", type=int, default=20)
    parser.add_argument("--fresh", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--targets", help="CSV with target,kind columns (e.g. a previous "
                                          "signals.csv) instead of choosing new stars")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    known = vetting.known_signals()
    if args.targets:
        listed = pd.read_csv(args.targets).drop_duplicates("target")
        targets = list(zip(listed["target"], listed["kind"]))
    else:
        skip = already_searched()
        targets = [(t, "host") for t in host_targets(args.hosts, known, skip)]
        targets += [(t, "fresh") for t in fresh_targets(args.fresh, known, skip)]
    print(f"{len(targets)} targets", flush=True)

    rows, fresh_done = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(hunt_one, t, k): (t, k) for t, k in targets}
        for future in as_completed(futures):
            target, kind = futures[future]
            new_rows = future.result()
            rows += new_rows
            found = [r for r in new_rows if r.get("is_candidate")]
            status = new_rows[0].get("error") or \
                ", ".join(f"{r['period_days']:.3f} d: {r.get('vet_status', '?')}" for r in found) or "nothing"
            print(f"  {target} ({kind}): {status}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "signals.csv", index=False)

    # Remember which stars were searched successfully.
    done = table[table["error"].fillna("") == ""].drop_duplicates("target")
    record = pd.DataFrame({"tic": done["target"].str.replace("TIC", "").str.strip().astype(int),
                           "kind": done["kind"], "sectors": done["sectors"],
                           "searched": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")})
    if SEARCHED.exists():
        record = pd.concat([pd.read_csv(SEARCHED), record]).drop_duplicates("tic", keep="first")
    record.to_csv(SEARCHED, index=False)

    ran = table[table["error"].fillna("") == ""].drop_duplicates("target")
    print(f"\nSearched {len(ran)} stars "
          f"({(ran['kind'] == 'host').sum()} hosts, {(ran['kind'] == 'fresh').sum()} fresh)")

    sent = 0
    for _, row in table.iterrows():
        if row.get("vet_status") in SEND_TO_QUEUE:
            card = cards.planet_card(row, row, source="TESS SPOC 2-min")
            if row["signal"] > 1:
                card["summary"] = f"Signal {int(row['signal'])} on this star, found after masking stronger ones. " + card["summary"]
            cards.write_outbox(card, row.get("plot"))
            sent += 1
    print(f"{sent} card(s) written to feed/outbox/")
    print(table.groupby("vet_status").size().to_string() if "vet_status" in table else "")
