"""
reddit_pipeline_safe.py  — May + June + July + August 2015
===========================================================
Key change from previous version: MONTHS now includes "2015-05"
Pre-ban window extends from 9 days to ~40 days.

Before running, clear the old cache:
    rm ./output/.processed_files.txt
    rm ./output/comments_filtered.csv
    rm ./output/submissions_filtered.csv
    rm ./output/daily_metrics.parquet
    rm ./output/user_thresholds.parquet
    rm ./output/panel.parquet

Then run:
    python3 reddit_pipeline_safe.py \
        --data_dir /home/nidhi/Downloads/reddit/all \
        --out_dir  ./output \
        --skip_networks
"""

import json
import argparse
import logging
import gc
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import zstandard as zstd
import pandas as pd
import numpy as np
from tqdm import tqdm

BANNED_SUBREDDITS = {
    "fatpeoplehate", "CoonTown", "transfags", "neofag", "hamplanet",
}
CONTROL_SUBREDDITS = {
    "science", "AskHistorians", "askscience",
    "politics", "worldnews", "news",
    "loseit", "progresspics", "obesity",
    "KotakuInAction", "MensRights", "TumblrInAction",
    "AskReddit", "funny", "todayilearned",
}
TARGET_SUBREDDITS = BANNED_SUBREDDITS | CONTROL_SUBREDDITS
BAN_WAVE_TS = int(datetime(2015, 6, 10, tzinfo=timezone.utc).timestamp())

# *** UPDATED: May added as first month ***
MONTHS = ["2015-05", "2015-06", "2015-07", "2015-08"]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def stream_zst(filepath, max_rows=None):
    dctx = zstd.ZstdDecompressor(max_window_size=2**31)
    with open(filepath, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            buffer = b""
            count  = 0
            while True:
                chunk = reader.read(512 * 1024)
                if not chunk:
                    break
                buffer += chunk
                lines   = buffer.split(b"\n")
                buffer  = lines[-1]
                for line in lines[:-1]:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                        count += 1
                        if max_rows and count >= max_rows:
                            return
                    except json.JSONDecodeError:
                        continue
            if buffer.strip():
                try:
                    yield json.loads(buffer)
                except json.JSONDecodeError:
                    pass


# Helper functions for processing records and building datasets
def is_removed(r, kind):
    if kind == "comment":
        return r.get("body", "") in ("[removed]", "[deleted]")
    return bool(r.get("removed_by_category")) or \
           r.get("selftext", "") in ("[removed]", "[deleted]")


def extract_comment(r):
    return {
        "id": r.get("id"), "author": r.get("author"),
        "subreddit": r.get("subreddit"), "subreddit_id": r.get("subreddit_id"),
        "parent_id": r.get("parent_id"), "link_id": r.get("link_id"),
        "score": r.get("score", 0), "created_utc": r.get("created_utc", 0),
        "gilded": r.get("gilded", 0),
        "controversiality": r.get("controversiality", 0),
        "is_removed": is_removed(r, "comment"),
    }


def extract_submission(r):
    return {
        "id": r.get("id"), "author": r.get("author"),
        "subreddit": r.get("subreddit"), "subreddit_id": r.get("subreddit_id"),
        "score": r.get("score", 0), "upvote_ratio": r.get("upvote_ratio"),
        "num_comments": r.get("num_comments", 0),
        "created_utc": r.get("created_utc", 0),
        "is_self": r.get("is_self", False), "over_18": r.get("over_18", False),
        "quarantine": r.get("quarantine", False),
        "stickied": r.get("stickied", False), "locked": r.get("locked", False),
        "is_removed": is_removed(r, "submission"),
    }


def stream_filter_write(data_dir, out_dir, chunk_size=10_000, test_mode=False):
    max_rows = 500_000 if test_mode else None
    c_path   = Path(out_dir) / "comments_filtered.csv"
    s_path   = Path(out_dir) / "submissions_filtered.csv"
    done_path = Path(out_dir) / ".processed_files.txt"
    done_files = set(done_path.read_text().splitlines()) \
                 if done_path.exists() else set()

    for month in MONTHS:
        for prefix, out_path, extractor in [
            ("RC", c_path, extract_comment),
            ("RS", s_path, extract_submission),
        ]:
            fname = f"{prefix}_{month}.zst"
            fpath = Path(data_dir) / fname

            if fname in done_files:
                log.info(f"Skipping {fname} (already processed)")
                continue
            if not fpath.exists():
                log.warning(f"Not found: {fpath}")
                continue

            log.info(f"Processing {fname} ...")
            first_write   = not out_path.exists()
            batch         = []
            total_matched = 0

            for record in tqdm(stream_zst(str(fpath), max_rows=max_rows),
                               desc=fname, unit=" rec", mininterval=2.0):
                if record.get("subreddit") not in TARGET_SUBREDDITS:
                    continue
                batch.append(extractor(record))
                if len(batch) >= chunk_size:
                    pd.DataFrame(batch).to_csv(
                        out_path, mode="a", header=first_write, index=False)
                    total_matched += len(batch)
                    first_write    = False
                    batch          = []
                    gc.collect()

            if batch:
                pd.DataFrame(batch).to_csv(
                    out_path, mode="a", header=first_write, index=False)
                total_matched += len(batch)
                gc.collect()

            log.info(f"  {fname}: {total_matched:,} matched records")
            with open(done_path, "a") as f:
                f.write(fname + "\n")

    log.info("All files processed.")


def build_daily_metrics(out_dir):
    out_path = Path(out_dir) / "daily_metrics.parquet"
    if out_path.exists():
        log.info("Daily metrics already built — loading.")
        return pd.read_parquet(out_path)

    log.info("Computing daily metrics...")
    c_path    = Path(out_dir) / "comments_filtered.csv"
    s_path    = Path(out_dir) / "submissions_filtered.csv"
    daily_agg = defaultdict(lambda: {"n": 0, "removed": 0, "score_sum": 0.0})
    CHUNK     = 200_000

    for chunk in pd.read_csv(c_path, chunksize=CHUNK,
                              usecols=["subreddit", "created_utc",
                                       "is_removed", "score"]):
        chunk["day"] = pd.to_datetime(
            chunk["created_utc"], unit="s", utc=True).dt.floor("D")
        for (sub, day), g in chunk.groupby(["subreddit", "day"]):
            key = (sub, str(day))
            daily_agg[key]["n"]         += len(g)
            daily_agg[key]["removed"]   += g["is_removed"].sum()
            daily_agg[key]["score_sum"] += g["score"].fillna(0).sum()
        del chunk
        gc.collect()

    rows = [{"subreddit": s, "day": pd.Timestamp(d),
              "n_comments": v["n"], "n_removed": v["removed"],
              "C_raw": v["removed"] / max(v["n"], 1),
              "mean_score": v["score_sum"] / max(v["n"], 1)}
             for (s, d), v in daily_agg.items()]

    metrics = pd.DataFrame(rows).sort_values(["subreddit", "day"])

    sub_daily = defaultdict(lambda: {"n": 0, "score_sum": 0.0})
    for chunk in pd.read_csv(s_path, chunksize=CHUNK,
                              usecols=["subreddit", "created_utc",
                                       "score", "is_removed"]):
        chunk = chunk[~chunk["is_removed"]]
        chunk["day"] = pd.to_datetime(
            chunk["created_utc"], unit="s", utc=True).dt.floor("D")
        for (sub, day), g in chunk.groupby(["subreddit", "day"]):
            key = (sub, str(day))
            sub_daily[key]["n"]         += len(g)
            sub_daily[key]["score_sum"] += g["score"].fillna(0).sum()
        del chunk
        gc.collect()

    sub_rows = [{"subreddit": s, "day": pd.Timestamp(d),
                  "mean_sub_score": v["score_sum"] / max(v["n"], 1)}
                 for (s, d), v in sub_daily.items()]
    metrics = metrics.merge(pd.DataFrame(sub_rows),
                            on=["subreddit", "day"], how="left")
    metrics = metrics.sort_values(["subreddit", "day"])

    metrics["C_smooth"] = metrics.groupby("subreddit")["C_raw"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    metrics["T_proxy"]  = metrics.groupby("subreddit")["mean_sub_score"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    metrics["T_norm"]   = metrics.groupby("subreddit")["T_proxy"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9))

    metrics["post_ban"]      = metrics["day"] >= pd.Timestamp("2015-06-10", tz="UTC")
    metrics["is_banned_sub"] = metrics["subreddit"].isin(BANNED_SUBREDDITS)

    metrics.to_parquet(out_path, index=False)
    log.info(f"Daily metrics: {len(metrics):,} rows")
    return metrics


def estimate_user_thresholds(out_dir, daily_metrics):
    out_path = Path(out_dir) / "user_thresholds.parquet"
    if out_path.exists():
        log.info("User thresholds already built — loading.")
        return pd.read_parquet(out_path)

    log.info("Estimating user thresholds...")
    c_path = Path(out_dir) / "comments_filtered.csv"

    dm = daily_metrics.copy()
    dm["day_str"] = dm["day"].astype(str).str[:10]
    ct_lookup = dm.set_index(["subreddit", "day_str"])[
        ["C_smooth", "T_norm"]].to_dict("index")

    user_agg = defaultdict(lambda: {
        "n": 0, "removed": 0, "score_sum": 0.0,
        "C_sum": 0.0, "C_removed_sum": 0.0, "C_removed_n": 0,
    })
    EXCLUDE = {"[deleted]", "AutoModerator", None}

    for chunk in pd.read_csv(c_path, chunksize=200_000,
                              usecols=["author", "subreddit", "created_utc",
                                       "is_removed", "score"]):
        chunk = chunk[~chunk["author"].isin(EXCLUDE)]
        chunk["day_str"] = pd.to_datetime(
            chunk["created_utc"], unit="s", utc=True).dt.strftime("%Y-%m-%d")

        for _, row in chunk.iterrows():
            key = (row["author"], row["subreddit"])
            ct  = ct_lookup.get((row["subreddit"], row["day_str"]), {})
            C   = ct.get("C_smooth", 0) or 0
            user_agg[key]["n"]         += 1
            user_agg[key]["removed"]   += int(row["is_removed"])
            user_agg[key]["score_sum"] += float(row["score"] or 0)
            user_agg[key]["C_sum"]     += C
            if row["is_removed"]:
                user_agg[key]["C_removed_sum"] += C
                user_agg[key]["C_removed_n"]   += 1
        del chunk
        gc.collect()

    rows = []
    for (author, subreddit), v in user_agg.items():
        if v["n"] < 5:
            continue
        rr  = v["removed"] / v["n"]
        avg_C = v["C_sum"] / v["n"]
        avg_C_rem = (v["C_removed_sum"] / v["C_removed_n"]
                     if v["C_removed_n"] > 0 else 0)
        rows.append({
            "author": author, "subreddit": subreddit,
            "n_posts": v["n"], "n_removed": v["removed"],
            "removal_rate": rr, "avg_C": avg_C,
            "avg_C_on_removed": avg_C_rem,
            "theta_proxy": (1 - rr) * avg_C_rem,
            "user_type": ("INITIATOR" if rr >= 0.30 and avg_C >= 0.1
                          else "JOINER" if rr >= 0.05 else "COMPLIER"),
        })

    banned_users = {a for (a, s) in user_agg if s in BANNED_SUBREDDITS}
    df = pd.DataFrame(rows)
    df["from_banned_sub"] = df["author"].isin(banned_users)
    df.to_parquet(out_path, index=False)
    log.info(f"User thresholds: {len(df):,} pairs")
    return df


def build_panel(out_dir, daily_metrics, user_thresholds):
    out_path = Path(out_dir) / "panel.parquet"
    if out_path.exists():
        log.info("Panel already built — loading.")
        return pd.read_parquet(out_path)

    log.info("Building panel...")
    c_path = Path(out_dir) / "comments_filtered.csv"

    dm = daily_metrics.copy()
    dm["week"] = pd.to_datetime(dm["day"]).dt.to_period("W").apply(
        lambda p: p.start_time.tz_localize("UTC"))
    weekly_ct = (dm.groupby(["subreddit", "week"])
                   .agg(C=("C_smooth", "mean"), T=("T_norm", "mean"),
                        x_global=("C_raw", "mean"))
                   .reset_index())

    EXCLUDE   = {"[deleted]", "AutoModerator"}
    week_agg  = defaultdict(lambda: {"resisted": 0, "n": 0, "score_sum": 0.0})

    for chunk in pd.read_csv(c_path, chunksize=200_000,
                              usecols=["author", "subreddit", "created_utc",
                                       "is_removed", "score"]):
        chunk = chunk[~chunk["author"].isin(EXCLUDE)]
        chunk["week"] = pd.to_datetime(
            chunk["created_utc"], unit="s", utc=True
        ).dt.to_period("W").apply(lambda p: p.start_time.tz_localize("UTC"))

        for _, row in chunk.iterrows():
            key = (row["author"], row["subreddit"],
                   row["week"].isoformat() if hasattr(row["week"], "isoformat")
                   else str(row["week"]))
            week_agg[key]["n"]         += 1
            week_agg[key]["resisted"]  += int(row["is_removed"])
            week_agg[key]["score_sum"] += float(row["score"] or 0)
        del chunk
        gc.collect()

    rows = [{"author": a, "subreddit": s, "week": pd.Timestamp(w),
              "resisted": v["resisted"] > 0, "n_posts_week": v["n"],
              "mean_score": v["score_sum"] / v["n"]}
             for (a, s, w), v in week_agg.items()]

    panel = pd.DataFrame(rows)
    panel["week"] = pd.to_datetime(panel["week"], utc=True)
    panel = panel.merge(weekly_ct, on=["subreddit", "week"], how="left")
    panel = panel.merge(
        user_thresholds[["author", "subreddit", "theta_proxy",
                          "user_type", "removal_rate", "from_banned_sub"]],
        on=["author", "subreddit"], how="left")

    panel["post_ban"]      = panel["week"] >= pd.Timestamp("2015-06-10", tz="UTC")
    panel["is_banned_sub"] = panel["subreddit"].isin(BANNED_SUBREDDITS)
    panel["days_from_ban"] = (
        panel["week"] - pd.Timestamp("2015-06-10", tz="UTC")).dt.days
    panel["resisted"] = panel["resisted"].astype(int)

    panel.to_parquet(out_path, index=False)
    log.info(f"Panel: {len(panel):,} rows, {panel['author'].nunique():,} users")
    return panel


def print_summary(daily_metrics, user_thresholds, panel):
    print("\n" + "="*60)
    print("PIPELINE COMPLETE — May + June + July + August 2015")
    print("="*60)

    dm = daily_metrics.copy()
    dm["day"] = pd.to_datetime(dm["day"], utc=True)
    pre = dm[~dm["post_ban"]]
    print(f"\nDate range:     {dm['day'].min().date()} → {dm['day'].max().date()}")
    print(f"Pre-ban window: {pre['day'].min().date()} → {pre['day'].max().date()}")
    n_subs = dm["subreddit"].nunique()
    n_days = len(pre) // n_subs if n_subs else 0
    print(f"Pre-ban days per subreddit (avg): {n_days}")

    print("\n--- Comments per subreddit ---")
    sub_counts = (dm.groupby("subreddit")["n_comments"]
                    .sum().sort_values(ascending=False))
    for sub, n in sub_counts.items():
        tag = "[BANNED]" if sub in BANNED_SUBREDDITS else "       "
        print(f"  {tag}  r/{sub:<25}  {n:>12,.0f}")

    print("\n--- User types ---")
    tc = user_thresholds["user_type"].value_counts()
    total = len(user_thresholds)
    for t, n in tc.items():
        print(f"  {t:<12}  {n:>8,}  ({100*n/total:.1f}%)")

    print("\n--- Pre vs. Post-ban enforcement ---")
    for label, flag in [("Banned subs", True), ("Control subs", False)]:
        pre_c  = dm[(dm["is_banned_sub"] == flag) & ~dm["post_ban"]]["C_smooth"].mean()
        post_c = dm[(dm["is_banned_sub"] == flag) &  dm["post_ban"]]["C_smooth"].mean()
        print(f"  {label}:  pre={pre_c:.3f}  post={post_c:.3f}  Δ={post_c-pre_c:+.3f}")

    print("\n--- Top spillover destinations ---")
    spill = (panel[panel["from_banned_sub"] & ~panel["is_banned_sub"]]
             .groupby("subreddit")["author"].nunique()
             .sort_values(ascending=False).head(8))
    for sub, n in spill.items():
        print(f"  r/{sub:<25}  {n:>6,} users from banned subs")

    print("\nNext steps:")
    print("  python3 model_calibration.py   --out_dir ./output")
    print("  python3 stage2_analysis.py     --out_dir ./output")
    print("  python3 fix_trajectories_v3.py --out_dir ./output")
    print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",      required=True)
    parser.add_argument("--out_dir",       default="./output")
    parser.add_argument("--chunk_size",    type=int, default=10_000)
    parser.add_argument("--skip_networks", action="store_true")
    parser.add_argument("--test_mode",     action="store_true")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    log.info("Step 1/4: Streaming May–Aug 2015 dumps...")
    stream_filter_write(args.data_dir, args.out_dir,
                         chunk_size=args.chunk_size,
                         test_mode=args.test_mode)

    log.info("Step 2/4: Daily metrics...")
    daily_metrics = build_daily_metrics(args.out_dir)

    log.info("Step 3/4: User thresholds...")
    user_thresholds = estimate_user_thresholds(args.out_dir, daily_metrics)

    log.info("Step 4/4: Panel...")
    panel = build_panel(args.out_dir, daily_metrics, user_thresholds)

    print_summary(daily_metrics, user_thresholds, panel)


if __name__ == "__main__":
    main()