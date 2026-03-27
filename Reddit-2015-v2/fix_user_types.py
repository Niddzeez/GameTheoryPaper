"""
fix_user_types.py
=================
Repairs the user_thresholds.parquet file by reclassifying user types
using subreddit-relative thresholds instead of absolute ones.

The original thresholds (30% / 5%) were too strict given community-wide
removal rates of only 6-13%. This version classifies relative to each
subreddit's median removal rate.

Classification logic:
  INITIATOR : user removal_rate > 2× subreddit median  AND  n_removed >= 2
  JOINER    : user removal_rate > subreddit median      AND  n_removed >= 1
  COMPLIER  : at or below subreddit median

Run from your project directory:
    python fix_user_types.py --out_dir ./output
"""

import argparse
import pandas as pd
from pathlib import Path


def fix_user_types(out_dir: str):
    out_dir  = Path(out_dir)
    in_path  = out_dir / "user_thresholds.parquet"
    bak_path = out_dir / "user_thresholds_original.parquet"

    print(f"Loading {in_path} ...")
    df = pd.read_parquet(in_path)
    print(f"  {len(df):,} user × subreddit pairs")

    # Back up original
    df.to_parquet(bak_path, index=False)
    print(f"  Original backed up to user_thresholds_original.parquet")

    # Compute per-subreddit median removal rate
    sub_stats = (
        df.groupby("subreddit")["removal_rate"]
        .agg(sub_median="median", sub_mean="mean", sub_p75=lambda x: x.quantile(0.75))
        .reset_index()
    )
    df = df.merge(sub_stats, on="subreddit", how="left")

    # Relative classification
    df["user_type"] = "COMPLIER"

    joiner_mask = (
        (df["removal_rate"] > df["sub_median"]) &
        (df["n_removed"] >= 1)
    )
    initiator_mask = (
        (df["removal_rate"] > 2 * df["sub_median"]) &
        (df["n_removed"] >= 2)
    )

    df.loc[joiner_mask,    "user_type"] = "JOINER"
    df.loc[initiator_mask, "user_type"] = "INITIATOR"

    # Print breakdown
    print("\n--- Repaired user type distribution ---")
    tc = df["user_type"].value_counts()
    total = len(df)
    for t, n in tc.items():
        print(f"  {t:<12}  {n:>7,}  ({100*n/total:.1f}%)")

    # Breakdown by subreddit type
    BANNED = {"fatpeoplehate", "CoonTown", "transfags", "neofag", "hamplanet"}
    HIGH_LEG = {"science", "AskHistorians", "askscience"}

    print("\n--- Initiator % by subreddit ---")
    sub_initiators = (
        df.groupby("subreddit")
        .apply(lambda g: (g["user_type"] == "INITIATOR").mean() * 100)
        .sort_values(ascending=False)
    )
    for sub, pct in sub_initiators.items():
        tag = "[BANNED]" if sub in BANNED else "[hi-leg]" if sub in HIGH_LEG else "        "
        bar = "█" * int(pct / 2)
        print(f"  {tag} r/{sub:<25}  {pct:5.1f}%  {bar}")

    # Save fixed version
    df.drop(columns=["sub_median", "sub_mean", "sub_p75"], inplace=True)
    df.to_parquet(in_path, index=False)
    print(f"\nSaved fixed user_thresholds.parquet")

    # Also update the panel with new user types
    panel_path = out_dir / "panel.parquet"
    if panel_path.exists():
        print("Updating panel.parquet with new user types...")
        panel = pd.read_parquet(panel_path)
        panel = panel.drop(columns=["user_type"], errors="ignore")
        panel = panel.merge(
            df[["author", "subreddit", "user_type"]],
            on=["author", "subreddit"],
            how="left"
        )
        panel["user_type"] = panel["user_type"].fillna("COMPLIER")
        panel.to_parquet(panel_path, index=False)
        print(f"Panel updated: {len(panel):,} rows")

    print("\nDone. Now re-run model_calibration.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./output")
    args = parser.parse_args()
    fix_user_types(args.out_dir)