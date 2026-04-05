"""
tbbt_parser.py
==============
Parses The Big Ban Theory (TBBT) dataset from the 18245670/ folder and builds
two clean datasets for downstream analysis.

Data availability note
----------------------
TBBT contains only SURVIVING (non-removed) comments from each community — the
`removal_reason` field listed in the DATASHEET is absent from all records.
Therefore we CANNOT compute a removal-rate-based C signal from TBBT.

What we CAN compute and use:
  1. Daily comment volume + mean score → T_proxy dynamics and engagement changes
  2. Quarantine events (6 cases, all have in-before AND in-after):
       Pre/post engagement ratio → operationalises Prediction 1 via volume/score
  3. Ban out-after data (17 cases): which external communities absorbed displaced
       users → extends the spillover analysis (Q_spillover) beyond the 2015 case

Outputs (written to <out_dir>/)
-------
  tbbt_daily.parquet          — daily volume + mean score per intervention-slice
  tbbt_intervention_dates.csv — estimated intervention date per intervention
  tbbt_spillover_communities.csv — top destination communities from out-after data
  tbbt_quarantine_outcomes.csv  — pre/post engagement metrics for 6 quarantine events

Usage
-----
    python tbbt_parser.py --tbbt_dir ../18245670 --out_dir ./output
"""

import argparse
import json
import logging
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Known intervention dates from the TBBT paper and public records.
# Used when an intervention only has in-before (no in-after to derive the date from).
KNOWN_DATES = {
    "2015-fatpeoplehate": "2015-06-10",
    "2017-incels":        "2017-11-07",
    "2017-physical_removal": "2017-08-14",
    "2018-braincels":     "2018-09-30",
    "2018-darknetmarkets":"2018-03-20",
    "2018-greatawakening":"2018-09-12",
    "2018-gunsforsale":   "2018-03-14",
    "2018-milliondollarextreme": "2018-09-12",
    "2018-sanctionedsuicide":    "2018-03-14",
    "2018-theredpill":    "2018-09-30",
    "2019-chapotraphouse":"2019-07-26",
    "2019-the_donald":    "2019-06-26",
    "2020-chapotraphouse":"2020-06-29",
    "2020-consumeproduct":"2020-06-29",
    "2020-darkhumorandmemes": "2020-06-29",
    "2020-debatealtright":"2020-06-29",
    "2020-gendercritical":"2020-06-29",
    "2020-shitneoconssay":"2020-06-29",
    "2020-the_donald":    "2020-02-26",
    "2022-askreddit":     "2022-04-18",
    "2022-chodi":         "2022-01-31",
    "2022-genzedong":     "2022-10-03",
    "2022-science":       "2022-02-07",
    "2023-goblin":        "2023-02-17",
    "2018-greatawakening": "2018-09-12",   # migration version, same date
}


def _read_jsonl(zf: zipfile.ZipFile, path: str):
    """Yield parsed JSON objects from a line-delimited JSON file inside a zip."""
    with zf.open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _build_daily_agg(zf: zipfile.ZipFile, path: str, slice_label: str):
    """
    Aggregate a single TBBT file (one slice) into daily stats.
    Returns a list of dicts: {intervention_id, slice, day, n_comments, score_sum, n_scored}
    """
    # intervention_id and subreddit from path: e.g. 'ban/2017-incels/2017-incels-in-before'
    parts = path.split("/")
    itype = parts[0]          # ban / quarantine / migration / post_removal
    iid   = parts[1]          # e.g. '2017-incels'

    day_agg = defaultdict(lambda: {"n": 0, "score_sum": 0.0, "n_scored": 0,
                                    "ts_min": None, "ts_max": None})
    subreddit_counter = defaultdict(int)

    for rec in _read_jsonl(zf, path):
        ts = rec.get("created_utc")
        if not ts:
            continue
        try:
            ts = int(ts)
        except (ValueError, TypeError):
            continue

        day = pd.Timestamp(ts, unit="s", tz="UTC").floor("D")
        key = str(day)
        day_agg[key]["n"] += 1
        score = rec.get("score")
        if score is not None:
            try:
                day_agg[key]["score_sum"] += float(score)
                day_agg[key]["n_scored"]  += 1
            except (ValueError, TypeError):
                pass

        # Track min/max timestamp to help derive intervention date
        if day_agg[key]["ts_min"] is None or ts < day_agg[key]["ts_min"]:
            day_agg[key]["ts_min"] = ts
        if day_agg[key]["ts_max"] is None or ts > day_agg[key]["ts_max"]:
            day_agg[key]["ts_max"] = ts

        sub = rec.get("subreddit", "")
        subreddit_counter[sub] += 1

    # Dominant subreddit in this file
    subreddit = max(subreddit_counter, key=subreddit_counter.get) if subreddit_counter else ""

    rows = []
    for day_str, v in day_agg.items():
        rows.append({
            "intervention_type": itype,
            "intervention_id":   iid,
            "subreddit":         subreddit,
            "slice":             slice_label,    # "in_before", "in_after", "out_before", "out_after"
            "day":               pd.Timestamp(day_str),
            "n_comments":        v["n"],
            "mean_score":        v["score_sum"] / max(v["n_scored"], 1) if v["n_scored"] > 0 else np.nan,
            "ts_min":            v["ts_min"],
            "ts_max":            v["ts_max"],
        })
    return rows


def parse_tbbt(tbbt_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_map = {
        "ban":          tbbt_dir / "ban.zip",
        "migration":    tbbt_dir / "migration.zip",
        "quarantine":   tbbt_dir / "quarantine.zip",
        "post_removal": tbbt_dir / "post_removal.zip",
    }

    all_rows = []
    intervention_meta = {}   # iid → {type, subreddit, slices_available, intervention_date}

    for itype, zip_path in zip_map.items():
        if not zip_path.exists():
            log.warning(f"  Not found: {zip_path}")
            continue

        log.info(f"Processing {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            entries = [
                n for n in zf.namelist()
                if not n.endswith("/")
                and ".DS_Store" not in n
                and "__MACOSX" not in n
            ]

            # Group entries by intervention id
            by_iid = defaultdict(list)
            for e in entries:
                parts = e.split("/")
                if len(parts) >= 3:
                    by_iid[parts[1]].append(e)

            for iid, file_paths in by_iid.items():
                slices_found = []
                for fpath in file_paths:
                    fname = fpath.split("/")[-1]  # e.g. '2017-incels-in-before'
                    # Determine slice label
                    if "in-before" in fname:
                        slice_label = "in_before"
                    elif "in-after" in fname:
                        slice_label = "in_after"
                    elif "out-before" in fname:
                        slice_label = "out_before"
                    elif "out-after" in fname:
                        slice_label = "out_after"
                    else:
                        continue

                    log.info(f"  Parsing {iid} / {slice_label} ...")
                    rows = _build_daily_agg(zf, fpath, slice_label)
                    all_rows.extend(rows)
                    slices_found.append(slice_label)

                if iid not in intervention_meta:
                    intervention_meta[iid] = {
                        "intervention_type": itype,
                        "intervention_id":   iid,
                        "slices": slices_found,
                    }
                else:
                    intervention_meta[iid]["slices"].extend(slices_found)

    if not all_rows:
        log.error("No data parsed — check tbbt_dir path and zip contents.")
        return

    daily = pd.DataFrame(all_rows)
    daily["day"] = pd.to_datetime(daily["day"], utc=True)
    daily = daily.sort_values(["intervention_id", "slice", "day"])

    # -----------------------------------------------------------------------
    # Derive intervention dates
    # -----------------------------------------------------------------------
    dates = {}
    for iid, meta in intervention_meta.items():
        # Prefer known dates
        if iid in KNOWN_DATES:
            dates[iid] = pd.Timestamp(KNOWN_DATES[iid], tz="UTC")
            continue
        # Fall back: boundary between in_before and in_after timestamps
        ib = daily[(daily["intervention_id"] == iid) & (daily["slice"] == "in_before")]
        ia = daily[(daily["intervention_id"] == iid) & (daily["slice"] == "in_after")]
        if not ib.empty and not ia.empty:
            est = (ib["day"].max() + ia["day"].min()) / 2
            dates[iid] = pd.Timestamp(est, tz="UTC").floor("D")
        else:
            dates[iid] = pd.NaT

    date_df = pd.DataFrame([
        {"intervention_id": iid, "intervention_type": intervention_meta[iid]["intervention_type"],
         "subreddit": daily[daily["intervention_id"] == iid]["subreddit"].mode().iloc[0]
                      if iid in daily["intervention_id"].values else "",
         "intervention_date": d,
         "has_in_after":  "in_after"  in intervention_meta[iid]["slices"],
         "has_out_after": "out_after" in intervention_meta[iid]["slices"],
         "n_slices":      len(intervention_meta[iid]["slices"])}
        for iid, d in dates.items()
    ])

    # -----------------------------------------------------------------------
    # Save tbbt_daily.parquet
    # -----------------------------------------------------------------------
    out_daily = out_dir / "tbbt_daily.parquet"
    daily.to_parquet(out_daily, index=False)
    log.info(f"Saved: {out_daily}  ({len(daily):,} rows, {daily['intervention_id'].nunique()} interventions)")

    # -----------------------------------------------------------------------
    # Save intervention date metadata
    # -----------------------------------------------------------------------
    out_dates = out_dir / "tbbt_intervention_dates.csv"
    date_df.to_csv(out_dates, index=False)
    log.info(f"Saved: {out_dates}")

    # -----------------------------------------------------------------------
    # Build quarantine outcomes: pre/post engagement for Prediction 1 test
    # -----------------------------------------------------------------------
    quarantine_records = []
    quarantine_iids = [iid for iid, meta in intervention_meta.items()
                       if meta["intervention_type"] == "quarantine"
                       and "in_before" in meta["slices"]
                       and "in_after" in meta["slices"]]

    for iid in quarantine_iids:
        int_date = dates.get(iid)
        sub_data = daily[daily["intervention_id"] == iid]

        ib = sub_data[sub_data["slice"] == "in_before"].sort_values("day")
        ia = sub_data[sub_data["slice"] == "in_after"].sort_values("day")

        if ib.empty or ia.empty:
            continue

        # Pre-intervention: last 30 days of in_before
        ib_last30 = ib.tail(30)
        # Post-intervention: first 30 and 60 days of in_after
        ia_first30 = ia.head(30)
        ia_first60 = ia.head(60)

        # OLS slope of daily volume in pre-intervention window (fragility proxy)
        if len(ib_last30) >= 7:
            y = ib_last30["n_comments"].values.astype(float)
            x = np.arange(len(y), dtype=float) - np.arange(len(y)).mean()
            pre_volume_slope = float(np.sum(x * (y - y.mean())) / (np.sum(x**2) + 1e-9))
        else:
            pre_volume_slope = np.nan

        # OLS slope of mean_score in pre-intervention (T proxy trend)
        if len(ib_last30) >= 7:
            y = ib_last30["mean_score"].fillna(method="ffill").values.astype(float)
            x = np.arange(len(y), dtype=float) - np.arange(len(y)).mean()
            pre_score_slope = float(np.sum(x * (y - y.mean())) / (np.sum(x**2) + 1e-9))
        else:
            pre_score_slope = np.nan

        # Post-intervention: volume ratio (post/pre) and score ratio
        pre_mean_vol   = ib_last30["n_comments"].mean() if not ib_last30.empty else np.nan
        post_mean_vol30 = ia_first30["n_comments"].mean() if not ia_first30.empty else np.nan
        post_mean_vol60 = ia_first60["n_comments"].mean() if not ia_first60.empty else np.nan

        pre_mean_score   = ib_last30["mean_score"].mean() if not ib_last30.empty else np.nan
        post_mean_score30 = ia_first30["mean_score"].mean() if not ia_first30.empty else np.nan

        vol_ratio_30 = (post_mean_vol30 / pre_mean_vol) if pre_mean_vol and pre_mean_vol > 0 else np.nan
        score_ratio_30 = (post_mean_score30 / pre_mean_score) if pre_mean_score and pre_mean_score > 0 else np.nan

        quarantine_records.append({
            "intervention_id":    iid,
            "intervention_date":  int_date,
            "n_pre_days":         len(ib),
            "n_post_days":        len(ia),
            "pre_volume_slope":   pre_volume_slope,    # fragility proxy: declining = fragile
            "pre_score_slope":    pre_score_slope,     # T-proxy trend
            "pre_mean_volume":    pre_mean_vol,
            "post_vol_ratio_30d": vol_ratio_30,        # outcome: vol drop = worse post-shock
            "post_score_ratio_30d": score_ratio_30,
        })

    quarantine_df = pd.DataFrame(quarantine_records)
    out_quar = out_dir / "tbbt_quarantine_outcomes.csv"
    quarantine_df.to_csv(out_quar, index=False)
    log.info(f"Saved: {out_quar}  ({len(quarantine_df)} quarantine events)")

    # -----------------------------------------------------------------------
    # Build spillover community table from out-after slices
    # -----------------------------------------------------------------------
    out_after_data = daily[daily["slice"] == "out_after"]
    if not out_after_data.empty:
        # Top destination communities per ban event (by total post-ban comment volume)
        spillover = (out_after_data
                     .groupby(["intervention_id", "subreddit"])["n_comments"]
                     .sum()
                     .reset_index()
                     .sort_values(["intervention_id", "n_comments"], ascending=[True, False]))
        spillover = spillover.merge(
            date_df[["intervention_id", "intervention_type"]], on="intervention_id", how="left")
        out_spill = out_dir / "tbbt_spillover_communities.csv"
        spillover.to_csv(out_spill, index=False)
        log.info(f"Saved: {out_spill}  ({len(spillover):,} rows)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    log.info("\n=== TBBT Parse Summary ===")
    log.info(f"Total interventions parsed: {date_df['intervention_id'].nunique()}")
    tc = date_df["intervention_type"].value_counts()
    for t, n in tc.items():
        log.info(f"  {t:<20}: {n}")
    log.info(f"Quarantine events with pre+post: {len(quarantine_df)}")
    log.info(f"Ban events with out-after: {date_df['has_out_after'].sum()}")
    if not quarantine_df.empty:
        log.info("\nQuarantine events:")
        for _, row in quarantine_df.iterrows():
            log.info(f"  {row['intervention_id']:30s}  "
                     f"pre_vol_slope={row['pre_volume_slope']:+.2f}  "
                     f"vol_ratio_30d={row['post_vol_ratio_30d']:.3f}"
                     if pd.notna(row.get('post_vol_ratio_30d')) else
                     f"  {row['intervention_id']:30s}  vol_ratio_30d=NaN")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tbbt_dir", required=True,
                        help="Path to the 18245670/ folder containing ban.zip etc.")
    parser.add_argument("--out_dir",  default="./output")
    args = parser.parse_args()

    parse_tbbt(Path(args.tbbt_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
