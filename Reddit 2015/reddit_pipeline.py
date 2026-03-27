"""
Reddit 2015 Pipeline — Memory-Safe Version
===========================================
Fixes the RAM freeze by writing to disk in small batches (never accumulates
the full dataset in memory). Each .zst file is streamed record-by-record;
matched records are written to CSV in 10,000-row chunks and flushed immediately.

Usage:
    python reddit_pipeline_safe.py --data_dir /path/to/dumps --out_dir ./output

Optional flags:
    --chunk_size 5000       rows per write batch (lower = less RAM, default 10000)
    --skip_networks         skip reply-network building (saves RAM + time)
    --test_mode             only process first 500,000 records per file (fast sanity check)
"""

import os
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

# ---------------------------------------------------------------------------
# Target subreddits  (same as before)
# ---------------------------------------------------------------------------

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
MONTHS = ["2015-06", "2015-07", "2015-08"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Streaming reader — yields one record at a time, never buffers the full file
# ---------------------------------------------------------------------------

def stream_zst(filepath: str, max_rows: int = None):
    """
    Yields parsed JSON dicts one at a time from a .zst ndjson file.
    Uses a 512KB read buffer — small enough to be safe on any machine.
    """
    dctx = zstd.ZstdDecompressor(max_window_size=2**31)
    with open(filepath, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            buffer = b""
            count  = 0
            while True:
                chunk = reader.read(512 * 1024)   # 512 KB at a time
                if not chunk:
                    break
                buffer += chunk
                lines   = buffer.split(b"\n")
                buffer  = lines[-1]               # keep incomplete trailing line
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


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------

def is_removed(r: dict, kind: str) -> bool:
    if kind == "comment":
        return r.get("body", "") in ("[removed]", "[deleted]")
    removed_by = r.get("removed_by_category")
    if removed_by:
        return True
    return r.get("selftext", "") in ("[removed]", "[deleted]")


COMMENT_FIELDS = [
    "id", "author", "subreddit", "subreddit_id",
    "parent_id", "link_id", "score", "created_utc",
    "gilded", "controversiality",
]

SUBMISSION_FIELDS = [
    "id", "author", "subreddit", "subreddit_id",
    "score", "upvote_ratio", "num_comments", "created_utc",
    "is_self", "over_18", "quarantine", "stickied", "locked",
]


def extract_comment(r: dict) -> dict:
    row = {f: r.get(f) for f in COMMENT_FIELDS}
    row["is_removed"] = is_removed(r, "comment")
    return row


def extract_submission(r: dict) -> dict:
    row = {f: r.get(f) for f in SUBMISSION_FIELDS}
    row["is_removed"] = is_removed(r, "submission")
    return row


# ---------------------------------------------------------------------------
# Step 1: Stream → filter → write CSV in chunks  (NEVER accumulates in RAM)
# ---------------------------------------------------------------------------

def stream_filter_write(data_dir: str, out_dir: str,
                         chunk_size: int = 10_000,
                         test_mode: bool = False):
    """
    Streams each .zst file, filters to TARGET_SUBREDDITS, and appends
    matched records to CSV files in `chunk_size` batches.

    Writes two files:
        comments_filtered.csv
        submissions_filtered.csv

    Each is written incrementally — peak RAM usage = chunk_size rows ≈ a few MB.
    """
    max_rows = 500_000 if test_mode else None

    c_path = Path(out_dir) / "comments_filtered.csv"
    s_path = Path(out_dir) / "submissions_filtered.csv"

    # Track which files have already been processed (resume support)
    done_path = Path(out_dir) / ".processed_files.txt"
    done_files = set()
    if done_path.exists():
        done_files = set(done_path.read_text().splitlines())

    for month in MONTHS:
        for prefix, out_path, extractor, kind in [
            ("RC", c_path, extract_comment,    "comment"),
            ("RS", s_path, extract_submission, "submission"),
        ]:
            fname  = f"{prefix}_{month}.zst"
            fpath  = Path(data_dir) / fname

            if fname in done_files:
                log.info(f"Skipping {fname} (already processed)")
                continue

            if not fpath.exists():
                log.warning(f"Not found: {fpath} — skipping")
                continue

            log.info(f"Processing {fname} ...")

            first_write = not out_path.exists()
            batch       = []
            total_matched = 0

            for record in tqdm(stream_zst(str(fpath), max_rows=max_rows),
                               desc=fname, unit=" rec", mininterval=2.0):

                if record.get("subreddit") not in TARGET_SUBREDDITS:
                    continue

                batch.append(extractor(record))

                if len(batch) >= chunk_size:
                    df = pd.DataFrame(batch)
                    df.to_csv(out_path, mode="a",
                              header=first_write, index=False)
                    total_matched += len(batch)
                    first_write    = False
                    batch          = []
                    del df
                    gc.collect()

            # Flush remainder
            if batch:
                df = pd.DataFrame(batch)
                df.to_csv(out_path, mode="a",
                          header=first_write, index=False)
                total_matched += len(batch)
                del df
                gc.collect()

            log.info(f"  {fname}: {total_matched:,} matched records written")

            # Mark file as done so re-runs skip it
            with open(done_path, "a") as f:
                f.write(fname + "\n")

    log.info("All .zst files processed.")


# ---------------------------------------------------------------------------
# Step 2: Read filtered CSVs in chunks → compute daily metrics
# ---------------------------------------------------------------------------

def build_daily_metrics(out_dir: str) -> pd.DataFrame:
    out_path = Path(out_dir) / "daily_metrics.parquet"
    if out_path.exists():
        log.info("Daily metrics already built — loading.")
        return pd.read_parquet(out_path)

    log.info("Computing daily metrics (chunked read)...")

    c_path = Path(out_dir) / "comments_filtered.csv"
    s_path = Path(out_dir) / "submissions_filtered.csv"

    # Aggregate comments in chunks — accumulate only daily counts, not raw rows
    daily_agg = defaultdict(lambda: {"n": 0, "removed": 0, "score_sum": 0.0,
                                      "controversial": 0})

    CHUNK = 200_000
    for chunk in pd.read_csv(c_path, chunksize=CHUNK,
                              usecols=["subreddit", "created_utc",
                                       "is_removed", "score", "controversiality"]):
        chunk["day"] = pd.to_datetime(chunk["created_utc"], unit="s", utc=True).dt.floor("D")
        for (sub, day), g in chunk.groupby(["subreddit", "day"]):
            key = (sub, str(day))
            daily_agg[key]["n"]            += len(g)
            daily_agg[key]["removed"]      += g["is_removed"].sum()
            daily_agg[key]["score_sum"]    += g["score"].fillna(0).sum()
            daily_agg[key]["controversial"]+= g["controversiality"].fillna(0).sum()
        del chunk
        gc.collect()

    rows = []
    for (sub, day), v in daily_agg.items():
        rows.append({
            "subreddit":    sub,
            "day":          pd.Timestamp(day),
            "n_comments":   v["n"],
            "n_removed":    v["removed"],
            "C_raw":        v["removed"] / max(v["n"], 1),
            "mean_score":   v["score_sum"] / max(v["n"], 1),
        })

    metrics = pd.DataFrame(rows).sort_values(["subreddit", "day"])

    # Submission-based legitimacy proxy (upvote_ratio + score)
    sub_daily = defaultdict(lambda: {"n": 0, "score_sum": 0.0, "ratio_sum": 0.0})

    for chunk in pd.read_csv(s_path, chunksize=CHUNK,
                              usecols=["subreddit", "created_utc",
                                       "score", "upvote_ratio", "is_removed"]):
        chunk = chunk[~chunk["is_removed"]]
        chunk["day"] = pd.to_datetime(chunk["created_utc"], unit="s", utc=True).dt.floor("D")
        for (sub, day), g in chunk.groupby(["subreddit", "day"]):
            key = (sub, str(day))
            sub_daily[key]["n"]         += len(g)
            sub_daily[key]["score_sum"] += g["score"].fillna(0).sum()
            sub_daily[key]["ratio_sum"] += g["upvote_ratio"].fillna(0.5).sum()
        del chunk
        gc.collect()

    sub_rows = [{"subreddit": s, "day": pd.Timestamp(d),
                 "n_submissions": v["n"],
                 "mean_sub_score": v["score_sum"] / max(v["n"], 1),
                 "mean_upvote_ratio": v["ratio_sum"] / max(v["n"], 1)}
                for (s, d), v in sub_daily.items()]

    sub_df = pd.DataFrame(sub_rows)
    metrics = metrics.merge(sub_df, on=["subreddit", "day"], how="left")
    metrics = metrics.sort_values(["subreddit", "day"])

    # Rolling proxies (per subreddit)
    def rolling_per_sub(df, col, window, label):
        return (df.groupby("subreddit")[col]
                  .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean()))

    metrics["C_smooth"] = rolling_per_sub(metrics, "C_raw", 3, "C")
    metrics["T_proxy"]  = rolling_per_sub(metrics, "mean_sub_score", 7, "T")
    metrics["T_norm"]   = metrics.groupby("subreddit")["T_proxy"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9)
    )

    metrics["post_ban"]      = metrics["day"] >= pd.Timestamp("2015-06-10", tz="UTC")
    metrics["is_banned_sub"] = metrics["subreddit"].isin(BANNED_SUBREDDITS)

    metrics.to_parquet(out_path, index=False)
    log.info(f"Daily metrics: {len(metrics):,} subreddit-day rows → daily_metrics.parquet")
    return metrics


# ---------------------------------------------------------------------------
# Step 3: Reply networks  (optional, most memory-intensive)
# ---------------------------------------------------------------------------

def build_reply_networks(out_dir: str):
    net_dir = Path(out_dir) / "reply_networks"
    net_dir.mkdir(exist_ok=True)
    c_path = Path(out_dir) / "comments_filtered.csv"

    log.info("Building reply networks (chunked)...")

    # Pass 1: build id → author lookup in chunks (only keep id + author)
    id_to_author = {}
    for chunk in pd.read_csv(c_path, chunksize=200_000,
                              usecols=["id", "author"]):
        chunk = chunk[~chunk["author"].isin({"[deleted]", "AutoModerator"})]
        id_to_author.update(dict(zip(chunk["id"].astype(str),
                                      chunk["author"])))
        del chunk
        gc.collect()
    log.info(f"  id→author map: {len(id_to_author):,} entries")

    # Pass 2: resolve parent_id → edge (author → parent_author)
    edge_counts = defaultdict(lambda: defaultdict(int))

    for chunk in pd.read_csv(c_path, chunksize=200_000,
                              usecols=["author", "subreddit", "parent_id",
                                       "created_utc"]):
        chunk = chunk[~chunk["author"].isin({"[deleted]", "AutoModerator"})]
        chunk["month"] = pd.to_datetime(
            chunk["created_utc"], unit="s", utc=True
        ).dt.to_period("M").astype(str)

        # Only t1_ parents are comment replies; t3_ = top-level (skip)
        is_reply = chunk["parent_id"].str.startswith("t1_", na=False)
        replies  = chunk[is_reply].copy()
        replies["parent_comment_id"] = replies["parent_id"].str[3:]
        replies["parent_author"]     = replies["parent_comment_id"].map(id_to_author)
        replies = replies.dropna(subset=["parent_author"])
        replies = replies[~replies["parent_author"].isin({"[deleted]", "AutoModerator"})]

        for _, row in replies.iterrows():
            key = (row["subreddit"], row["month"])
            edge_counts[key][(row["author"], row["parent_author"])] += 1

        del chunk, replies
        gc.collect()

    # Write edge lists
    stats = []
    for (sub, month), edges in edge_counts.items():
        rows = [{"author": a, "parent_author": b, "weight": w}
                for (a, b), w in edges.items()]
        df = pd.DataFrame(rows)
        df.to_csv(net_dir / f"{sub}_{month}_edges.csv", index=False)
        stats.append({"subreddit": sub, "month": month,
                       "n_edges": len(rows),
                       "n_nodes": len(set(df["author"]) | set(df["parent_author"]))})
        del df
        gc.collect()

    stats_df = pd.DataFrame(stats)
    stats_df.to_parquet(Path(out_dir) / "network_stats.parquet", index=False)
    log.info(f"Reply networks done: {len(stats_df)} subreddit-month graphs.")


# ---------------------------------------------------------------------------
# Step 4: User thresholds  (chunked)
# ---------------------------------------------------------------------------

def estimate_user_thresholds(out_dir: str, daily_metrics: pd.DataFrame) -> pd.DataFrame:
    out_path = Path(out_dir) / "user_thresholds.parquet"
    if out_path.exists():
        log.info("User thresholds already built — loading.")
        return pd.read_parquet(out_path)

    log.info("Estimating user thresholds (chunked)...")
    c_path = Path(out_dir) / "comments_filtered.csv"

    # Build a day → (subreddit → C, T) lookup from daily_metrics
    daily_metrics = daily_metrics.copy()
    daily_metrics["day_str"] = daily_metrics["day"].astype(str).str[:10]
    ct_lookup = daily_metrics.set_index(["subreddit", "day_str"])[
        ["C_smooth", "T_norm"]
    ].to_dict("index")

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
            chunk["created_utc"], unit="s", utc=True
        ).dt.strftime("%Y-%m-%d")

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
        removal_rate = v["removed"] / v["n"]
        avg_C        = v["C_sum"] / v["n"]
        avg_C_removed = (v["C_removed_sum"] / v["C_removed_n"]
                         if v["C_removed_n"] > 0 else 0)
        theta_proxy  = (1 - removal_rate) * avg_C_removed

        if removal_rate >= 0.30 and avg_C >= 0.1:
            user_type = "INITIATOR"
        elif removal_rate >= 0.05:
            user_type = "JOINER"
        else:
            user_type = "COMPLIER"

        rows.append({
            "author": author, "subreddit": subreddit,
            "n_posts": v["n"], "n_removed": v["removed"],
            "removal_rate": removal_rate,
            "avg_score": v["score_sum"] / v["n"],
            "avg_C": avg_C,
            "avg_C_on_removed": avg_C_removed,
            "theta_proxy": theta_proxy,
            "user_type": user_type,
        })

    # Tag users from banned subs
    banned_users = {a for (a, s) in user_agg if s in BANNED_SUBREDDITS}
    df = pd.DataFrame(rows)
    df["from_banned_sub"] = df["author"].isin(banned_users)

    df.to_parquet(out_path, index=False)
    log.info(f"User thresholds: {len(df):,} user × subreddit pairs.")
    return df


# ---------------------------------------------------------------------------
# Step 5: Build panel  (chunked)
# ---------------------------------------------------------------------------

def build_panel(out_dir: str, daily_metrics: pd.DataFrame,
                user_thresholds: pd.DataFrame) -> pd.DataFrame:
    out_path = Path(out_dir) / "panel.parquet"
    if out_path.exists():
        log.info("Panel already built — loading.")
        return pd.read_parquet(out_path)

    log.info("Building final panel (chunked)...")
    c_path = Path(out_dir) / "comments_filtered.csv"

    # Weekly subreddit C and T
    daily_metrics = daily_metrics.copy()
    daily_metrics["week"] = pd.to_datetime(daily_metrics["day"]).dt.to_period("W").apply(
        lambda p: p.start_time.tz_localize("UTC")
    )
    weekly_ct = (
        daily_metrics.groupby(["subreddit", "week"])
        .agg(C=("C_smooth", "mean"), T=("T_norm", "mean"),
             x_global=("C_raw", "mean"))
        .reset_index()
    )

    # User × subreddit × week aggregation
    EXCLUDE = {"[deleted]", "AutoModerator"}
    user_week_agg = defaultdict(lambda: {"resisted": 0, "n": 0, "score_sum": 0.0})

    for chunk in pd.read_csv(c_path, chunksize=200_000,
                              usecols=["author", "subreddit", "created_utc",
                                       "is_removed", "score"]):
        chunk = chunk[~chunk["author"].isin(EXCLUDE)]
        chunk["week"] = pd.to_datetime(
            chunk["created_utc"], unit="s", utc=True
        ).dt.to_period("W").apply(lambda p: p.start_time.tz_localize("UTC"))

        for _, row in chunk.iterrows():
            key = (row["author"], row["subreddit"],
                   row["week"].isoformat() if hasattr(row["week"], "isoformat") else str(row["week"]))
            user_week_agg[key]["n"]          += 1
            user_week_agg[key]["resisted"]   += int(row["is_removed"])
            user_week_agg[key]["score_sum"]  += float(row["score"] or 0)
        del chunk
        gc.collect()

    rows = [{"author": a, "subreddit": s, "week": pd.Timestamp(w),
              "resisted": v["resisted"] > 0,
              "n_posts_week": v["n"],
              "mean_score": v["score_sum"] / v["n"]}
             for (a, s, w), v in user_week_agg.items()]

    panel = pd.DataFrame(rows)
    panel["week"] = pd.to_datetime(panel["week"], utc=True)

    panel = panel.merge(weekly_ct, on=["subreddit", "week"], how="left")
    panel = panel.merge(
        user_thresholds[["author", "subreddit", "theta_proxy",
                          "user_type", "removal_rate", "from_banned_sub"]],
        on=["author", "subreddit"], how="left"
    )

    panel["post_ban"]      = panel["week"] >= pd.Timestamp("2015-06-10", tz="UTC")
    panel["is_banned_sub"] = panel["subreddit"].isin(BANNED_SUBREDDITS)
    panel["days_from_ban"] = (panel["week"] - pd.Timestamp("2015-06-10", tz="UTC")).dt.days
    panel["resisted"]      = panel["resisted"].astype(int)

    panel.to_parquet(out_path, index=False)
    log.info(f"Panel: {len(panel):,} rows, {panel['author'].nunique():,} users.")
    return panel


# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------

def print_summary(daily_metrics, user_thresholds, panel):
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)

    print("\n--- Comments per subreddit ---")
    sub_counts = daily_metrics.groupby("subreddit")["n_comments"].sum().sort_values(ascending=False)
    for sub, n in sub_counts.items():
        tag = "[BANNED]" if sub in BANNED_SUBREDDITS else "       "
        print(f"  {tag}  r/{sub:<25}  {n:>10,.0f} comments")

    print("\n--- User types ---")
    tc = user_thresholds["user_type"].value_counts()
    total = len(user_thresholds)
    for t, n in tc.items():
        print(f"  {t:<12}  {n:>7,}  ({100*n/total:.1f}%)")

    print("\n--- Top spillover destinations (from banned subs) ---")
    spill = (panel[panel["from_banned_sub"] & ~panel["is_banned_sub"]]
             .groupby("subreddit")["author"].nunique()
             .sort_values(ascending=False).head(8))
    for sub, n in spill.items():
        print(f"  r/{sub:<25}  {n:>5,} users")

    print("\n--- Pre vs. Post-ban enforcement ---")
    for label, flag in [("Banned subs", True), ("Control subs", False)]:
        pre  = daily_metrics[(daily_metrics["is_banned_sub"]==flag) & ~daily_metrics["post_ban"]]["C_smooth"].mean()
        post = daily_metrics[(daily_metrics["is_banned_sub"]==flag) &  daily_metrics["post_ban"]]["C_smooth"].mean()
        print(f"  {label}:  pre={pre:.3f}  post={post:.3f}  Δ={post-pre:+.3f}")

    print("\nOutputs ready in ./output/")
    print("Next: python model_calibration.py --out_dir ./output")
    print("="*60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--out_dir",    default="./output")
    parser.add_argument("--chunk_size", type=int, default=10_000)
    parser.add_argument("--skip_networks", action="store_true")
    parser.add_argument("--test_mode",  action="store_true",
                        help="Only read first 500k records per file — fast sanity check")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    log.info("Step 1/5: Streaming and filtering .zst dumps...")
    stream_filter_write(args.data_dir, args.out_dir,
                         chunk_size=args.chunk_size,
                         test_mode=args.test_mode)

    log.info("Step 2/5: Computing daily metrics...")
    daily_metrics = build_daily_metrics(args.out_dir)

    if not args.skip_networks:
        log.info("Step 3/5: Building reply networks...")
        build_reply_networks(args.out_dir)
    else:
        log.info("Step 3/5: Skipping reply networks.")

    log.info("Step 4/5: Estimating user thresholds...")
    user_thresholds = estimate_user_thresholds(args.out_dir, daily_metrics)

    log.info("Step 5/5: Building panel...")
    panel = build_panel(args.out_dir, daily_metrics, user_thresholds)

    print_summary(daily_metrics, user_thresholds, panel)


if __name__ == "__main__":
    main()