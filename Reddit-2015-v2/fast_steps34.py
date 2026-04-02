"""
fast_steps34.py
===============
Vectorized replacement for Steps 3 and 4 of reddit_pipeline_safe.py.

The original pipeline uses iterrows() over 30M comments — too slow and
kills the process on Windows (OOM / timeout).

This script replaces that with pandas vectorized groupby operations:
  - Processes 500K-row chunks
  - Merges C_smooth/T_norm from daily_metrics by (subreddit, day_str)
  - Accumulates per-chunk aggregates as DataFrames (not dicts)
  - Final groupby over accumulated aggregates — no iterrows anywhere

Produces identical outputs:
  output/user_thresholds.parquet
  output/panel.parquet

Usage:
    python fast_steps34.py --out_dir ./output
"""

import argparse
import gc
import logging
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

BANNED_SUBREDDITS = {
    "fatpeoplehate", "CoonTown", "transfags", "neofag", "hamplanet",
}
BAN_DATE = pd.Timestamp("2015-06-10", tz="UTC")
EXCLUDE  = {"[deleted]", "AutoModerator"}
CHUNK    = 500_000


# ---------------------------------------------------------------------------
# Step 3: User thresholds — vectorized
# ---------------------------------------------------------------------------

def build_user_thresholds(out_dir: Path, daily_metrics: pd.DataFrame) -> pd.DataFrame:
    out_path = out_dir / "user_thresholds.parquet"
    if out_path.exists():
        log.info("user_thresholds.parquet already exists — loading.")
        return pd.read_parquet(out_path)

    log.info("Building user thresholds (vectorized)...")
    c_path = out_dir / "comments_filtered.csv"

    # Build C_smooth lookup as a DataFrame for fast merging
    dm = daily_metrics.copy()
    dm["day_str"] = dm["day"].dt.strftime("%Y-%m-%d")
    ct_df = dm[["subreddit", "day_str", "C_smooth"]].drop_duplicates(
        subset=["subreddit", "day_str"]
    )

    # Accumulate per-chunk aggregates — two separate aggregations:
    #   1. All posts: n, n_removed, score_sum, C_sum
    #   2. Removed posts only: C_removed_sum, C_removed_n
    agg_all_list     = []
    agg_removed_list = []

    n_chunks = 0
    for chunk in pd.read_csv(
        c_path, chunksize=CHUNK,
        usecols=["author", "subreddit", "created_utc", "is_removed", "score"]
    ):
        chunk = chunk[~chunk["author"].isin(EXCLUDE)].copy()
        chunk["day_str"] = pd.to_datetime(
            chunk["created_utc"], unit="s", utc=True
        ).dt.strftime("%Y-%m-%d")
        chunk["score"] = pd.to_numeric(chunk["score"], errors="coerce").fillna(0)

        # Merge C_smooth by (subreddit, day_str)
        chunk = chunk.merge(ct_df, on=["subreddit", "day_str"], how="left")
        chunk["C_smooth"] = chunk["C_smooth"].fillna(0)

        # Aggregate over ALL posts in this chunk
        agg_all = (
            chunk.groupby(["author", "subreddit"], sort=False)
            .agg(
                n          = ("is_removed", "count"),
                n_removed  = ("is_removed", "sum"),
                score_sum  = ("score",      "sum"),
                C_sum      = ("C_smooth",   "sum"),
            )
            .reset_index()
        )
        agg_all_list.append(agg_all)

        # Aggregate over REMOVED posts only (for C_removed_sum)
        removed = chunk[chunk["is_removed"] == True]
        if len(removed) > 0:
            agg_rem = (
                removed.groupby(["author", "subreddit"], sort=False)
                .agg(
                    C_removed_sum = ("C_smooth", "sum"),
                    C_removed_n   = ("C_smooth", "count"),
                )
                .reset_index()
            )
            agg_removed_list.append(agg_rem)

        n_chunks += 1
        del chunk, removed
        gc.collect()

        if n_chunks % 10 == 0:
            log.info(f"  ... processed {n_chunks * CHUNK / 1e6:.0f}M rows")

    log.info(f"  Chunk pass complete ({n_chunks} chunks). Consolidating...")

    # Consolidate all-posts aggregates
    all_df = pd.concat(agg_all_list, ignore_index=True)
    del agg_all_list
    gc.collect()

    all_final = (
        all_df.groupby(["author", "subreddit"], sort=False)
        .agg(
            n         = ("n",         "sum"),
            n_removed = ("n_removed", "sum"),
            score_sum = ("score_sum", "sum"),
            C_sum     = ("C_sum",     "sum"),
        )
        .reset_index()
    )
    del all_df
    gc.collect()
    log.info(f"  All-posts aggregated: {len(all_final):,} user×subreddit pairs")

    # Consolidate removed-posts aggregates
    if agg_removed_list:
        rem_df = pd.concat(agg_removed_list, ignore_index=True)
        del agg_removed_list
        gc.collect()

        rem_final = (
            rem_df.groupby(["author", "subreddit"], sort=False)
            .agg(
                C_removed_sum = ("C_removed_sum", "sum"),
                C_removed_n   = ("C_removed_n",   "sum"),
            )
            .reset_index()
        )
        del rem_df
        gc.collect()
    else:
        rem_final = pd.DataFrame(
            columns=["author", "subreddit", "C_removed_sum", "C_removed_n"]
        )

    # Merge
    merged = all_final.merge(rem_final, on=["author", "subreddit"], how="left")
    merged["C_removed_sum"] = merged["C_removed_sum"].fillna(0)
    merged["C_removed_n"]   = merged["C_removed_n"].fillna(0).astype(int)
    del all_final, rem_final
    gc.collect()

    # Filter minimum activity
    merged = merged[merged["n"] >= 5].copy()
    log.info(f"  After min-5-posts filter: {len(merged):,} pairs")

    # Compute derived columns — all vectorized
    merged["removal_rate"]    = merged["n_removed"] / merged["n"]
    merged["avg_score"]       = merged["score_sum"] / merged["n"]
    merged["avg_C"]           = merged["C_sum"]     / merged["n"]
    merged["avg_C_on_removed"] = np.where(
        merged["C_removed_n"] > 0,
        merged["C_removed_sum"] / merged["C_removed_n"],
        0.0
    )
    merged["theta_proxy"] = (1 - merged["removal_rate"]) * merged["avg_C_on_removed"]

    # User type classification
    merged["user_type"] = "COMPLIER"
    merged.loc[merged["removal_rate"] >= 0.05, "user_type"] = "JOINER"
    merged.loc[
        (merged["removal_rate"] >= 0.30) & (merged["avg_C"] >= 0.1),
        "user_type"
    ] = "INITIATOR"

    # Tag users from banned subreddits
    banned_authors = set(
        merged.loc[merged["subreddit"].isin(BANNED_SUBREDDITS), "author"]
    )
    merged["from_banned_sub"] = merged["author"].isin(banned_authors)

    # Drop internal columns not needed downstream
    merged = merged.drop(columns=["score_sum", "C_sum", "C_removed_sum", "C_removed_n"])

    merged.to_parquet(out_path, index=False)
    log.info(f"  Saved user_thresholds.parquet: {len(merged):,} rows")

    tc = merged["user_type"].value_counts()
    total = len(merged)
    for t, n in tc.items():
        log.info(f"    {t:<12}  {n:>8,}  ({100*n/total:.1f}%)")

    return merged


# ---------------------------------------------------------------------------
# Step 4: Panel — vectorized
# ---------------------------------------------------------------------------

def build_panel(out_dir: Path, daily_metrics: pd.DataFrame,
                user_thresholds: pd.DataFrame) -> pd.DataFrame:
    out_path = out_dir / "panel.parquet"
    if out_path.exists():
        log.info("panel.parquet already exists — loading.")
        return pd.read_parquet(out_path)

    log.info("Building panel (vectorized)...")
    c_path = out_dir / "comments_filtered.csv"

    # Weekly subreddit C and T
    # Vectorized week-start: floor to Monday by subtracting dayofweek days
    dm = daily_metrics.copy()
    dm["day"] = pd.to_datetime(dm["day"], utc=True)
    dm["week"] = (dm["day"] - pd.to_timedelta(dm["day"].dt.dayofweek, unit="D")).dt.normalize()
    weekly_ct = (
        dm.groupby(["subreddit", "week"])
        .agg(C=("C_smooth", "mean"), T=("T_norm", "mean"),
             x_global=("C_raw", "mean"))
        .reset_index()
    )

    # Accumulate user×subreddit×week aggregates
    week_agg_list = []
    n_chunks = 0

    for chunk in pd.read_csv(
        c_path, chunksize=CHUNK,
        usecols=["author", "subreddit", "created_utc", "is_removed", "score"]
    ):
        chunk = chunk[~chunk["author"].isin(EXCLUDE)].copy()
        chunk["score"] = pd.to_numeric(chunk["score"], errors="coerce").fillna(0)
        # Fully vectorized week-start — no apply(), no Python loops
        dt_utc = pd.to_datetime(chunk["created_utc"], unit="s", utc=True)
        chunk["week"] = (dt_utc - pd.to_timedelta(dt_utc.dt.dayofweek, unit="D")).dt.normalize()

        agg = (
            chunk.groupby(["author", "subreddit", "week"], sort=False)
            .agg(
                n_posts_week = ("is_removed", "count"),
                n_resisted   = ("is_removed", "sum"),
                score_sum    = ("score",      "sum"),
            )
            .reset_index()
        )
        week_agg_list.append(agg)

        n_chunks += 1
        del chunk, agg
        gc.collect()

        if n_chunks % 10 == 0:
            log.info(f"  ... panel pass {n_chunks * CHUNK / 1e6:.0f}M rows")

    log.info(f"  Panel chunk pass complete. Consolidating...")
    week_df = pd.concat(week_agg_list, ignore_index=True)
    del week_agg_list
    gc.collect()

    panel = (
        week_df.groupby(["author", "subreddit", "week"], sort=False)
        .agg(
            n_posts_week = ("n_posts_week", "sum"),
            n_resisted   = ("n_resisted",   "sum"),
            score_sum    = ("score_sum",    "sum"),
        )
        .reset_index()
    )
    del week_df
    gc.collect()
    log.info(f"  Panel consolidated: {len(panel):,} user×sub×week rows")

    panel["week"]       = pd.to_datetime(panel["week"], utc=True)
    panel["resisted"]   = (panel["n_resisted"] > 0).astype(int)
    panel["mean_score"] = panel["score_sum"] / panel["n_posts_week"]
    panel = panel.drop(columns=["n_resisted", "score_sum"])

    # Merge weekly C/T
    panel = panel.merge(weekly_ct, on=["subreddit", "week"], how="left")

    # Merge user-level attributes
    panel = panel.merge(
        user_thresholds[["author", "subreddit", "theta_proxy",
                         "user_type", "removal_rate", "from_banned_sub"]],
        on=["author", "subreddit"], how="left"
    )

    panel["post_ban"]      = panel["week"] >= BAN_DATE
    panel["is_banned_sub"] = panel["subreddit"].isin(BANNED_SUBREDDITS)
    panel["days_from_ban"] = (panel["week"] - BAN_DATE).dt.days

    panel.to_parquet(out_path, index=False)
    log.info(f"  Saved panel.parquet: {len(panel):,} rows, "
             f"{panel['author'].nunique():,} unique users")
    return panel


# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------

def print_summary(daily_metrics, user_thresholds, panel):
    print("\n" + "="*60)
    print("STEPS 3 & 4 COMPLETE")
    print("="*60)

    dm = daily_metrics.copy()
    dm["day"] = pd.to_datetime(dm["day"], utc=True)

    print("\n--- Comments per subreddit ---")
    sub_counts = daily_metrics.groupby("subreddit")["n_comments"].sum() \
                              .sort_values(ascending=False)
    for sub, n in sub_counts.items():
        tag = "[BANNED]" if sub in BANNED_SUBREDDITS else "       "
        print(f"  {tag}  r/{sub:<25}  {n:>10,.0f}")

    print("\n--- User type distribution ---")
    tc    = user_thresholds["user_type"].value_counts()
    total = len(user_thresholds)
    for t, n in tc.items():
        print(f"  {t:<12}  {n:>8,}  ({100*n/total:.1f}%)")

    print("\n--- Pre vs. Post-ban enforcement (C_smooth) ---")
    dm["post_ban"] = dm["day"] >= BAN_DATE
    for label, is_banned in [("Banned subs", True), ("Control subs", False)]:
        pre  = dm[(dm["subreddit"].isin(BANNED_SUBREDDITS) == is_banned) & ~dm["post_ban"]]["C_smooth"].mean()
        post = dm[(dm["subreddit"].isin(BANNED_SUBREDDITS) == is_banned) &  dm["post_ban"]]["C_smooth"].mean()
        print(f"  {label}:  pre={pre:.4f}  post={post:.4f}  Δ={post-pre:+.4f}")

    print("\n--- Top spillover destinations (users from banned subs) ---")
    spill = (
        panel[panel["from_banned_sub"] & ~panel["is_banned_sub"]]
        .groupby("subreddit")["author"].nunique()
        .sort_values(ascending=False).head(8)
    )
    for sub, n in spill.items():
        print(f"  r/{sub:<25}  {n:>6,} migrant users")

    print("\nNext steps:")
    print("  python model_calibration.py  --out_dir ./output")
    print("  python stage2_analysis.py    --out_dir ./output")
    print("  python ../Confidence-Score/confidence_score.py --out_dir ./output")
    print("="*60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./output")
    args    = parser.parse_args()
    out_dir = Path(args.out_dir)

    if not (out_dir / "daily_metrics.parquet").exists():
        raise FileNotFoundError(
            "daily_metrics.parquet not found. "
            "Run reddit_pipeline_safe.py first (Steps 1–2 must complete)."
        )

    log.info("Loading daily_metrics.parquet...")
    daily_metrics = pd.read_parquet(out_dir / "daily_metrics.parquet")
    daily_metrics["day"] = pd.to_datetime(daily_metrics["day"], utc=True)
    log.info(f"  {len(daily_metrics):,} subreddit-day rows, "
             f"{daily_metrics['subreddit'].nunique()} subreddits")

    log.info("\n--- Step 3: User Thresholds ---")
    user_thresholds = build_user_thresholds(out_dir, daily_metrics)

    log.info("\n--- Step 4: Panel ---")
    panel = build_panel(out_dir, daily_metrics, user_thresholds)

    print_summary(daily_metrics, user_thresholds, panel)


if __name__ == "__main__":
    main()
