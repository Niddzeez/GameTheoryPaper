"""
patch_fixes.py
==============
Fixes two remaining issues:

1. theta_proxy collapses to zero for almost all users.
   Root cause: (1 - removal_rate) × avg_C_on_removed = 0 when
   avg_C_on_removed = 0 (user never had a post removed).

   New proxy: posting PERSISTENCE under enforcement pressure.
   A user who keeps posting in a subreddit even when community C is high
   is revealed to have a low threshold (they resist despite deterrence).
   A user who stops posting when C rises has a high threshold (they comply).

   theta_proxy_v2 = -corr(user_weekly_posts, weekly_C)
   Negative correlation = posts fall when C rises = high threshold (complier)
   Positive correlation = posts rise when C rises = low threshold (initiator)

2. Spillover heatmap crash: ValueError on pd.Timestamp(w, tz="UTC")
   when w already has timezone info.
   Fix: use pd.Timestamp(w).tz_convert("UTC") safely.

Run:
    python3 patch_fixes.py --out_dir ./output
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

BANNED_SUBREDDITS = {
    "fatpeoplehate", "CoonTown", "transfags", "neofag", "hamplanet",
}
BAN_DATE = pd.Timestamp("2015-06-10", tz="UTC")


# ---------------------------------------------------------------------------
# Fix 1: Rebuild theta_proxy using posting-persistence correlation
# ---------------------------------------------------------------------------

def rebuild_theta_proxy(out_dir: Path):
    """
    For each user × subreddit, computes the correlation between
    the user's weekly posting volume and the subreddit's weekly C.

    Interpretation:
      corr < -0.3  → posts fall when enforcement rises → COMPLIER (high θ)
      corr in [-0.3, 0.3] → no clear relationship → JOINER (mid θ)
      corr > 0.3   → posts rise when enforcement rises → INITIATOR (low θ)
      (initiators may be provoked to post MORE when enforcement increases)

    Also uses removal_rate as a secondary signal.
    """
    print("Rebuilding theta_proxy using posting-persistence method...")

    panel = pd.read_parquet(out_dir / "panel.parquet")
    panel["week"] = pd.to_datetime(panel["week"], utc=True)

    # Weekly subreddit C — already in panel
    # Compute correlation between user's n_posts_week and subreddit C
    # Only for users with >= 5 active weeks

    records = []
    EXCLUDE = {"[deleted]", "AutoModerator"}

    for (author, subreddit), grp in panel.groupby(["author", "subreddit"]):
        if author in EXCLUDE:
            continue
        grp = grp.sort_values("week").dropna(subset=["C", "n_posts_week"])
        if len(grp) < 5:
            continue

        # Persistence correlation
        if grp["C"].std() > 1e-6 and grp["n_posts_week"].std() > 1e-6:
            corr = grp["C"].corr(grp["n_posts_week"])
        else:
            corr = 0.0

        removal_rate = grp["resisted"].mean()
        n_posts      = grp["n_posts_week"].sum()
        n_removed    = grp["resisted"].sum()

        # Combined theta proxy: persistence correlation + removal signal
        # High corr + high removal_rate = initiator
        # Low corr + low removal_rate = complier
        theta_v2 = corr * 0.6 + (removal_rate - 0.1) * 0.4

        # Classify
        if corr > 0.25 and removal_rate > 0.05:
            user_type = "INITIATOR"
        elif corr > 0.1 or removal_rate > 0.03:
            user_type = "JOINER"
        else:
            user_type = "COMPLIER"

        records.append({
            "author":         author,
            "subreddit":      subreddit,
            "n_posts":        int(n_posts),
            "n_removed":      int(n_removed),
            "removal_rate":   float(removal_rate),
            "persistence_corr": float(corr) if np.isfinite(corr) else 0.0,
            "theta_proxy":    float(theta_v2) if np.isfinite(theta_v2) else 0.0,
            "user_type":      user_type,
            "from_banned_sub": subreddit in BANNED_SUBREDDITS,
        })

    df = pd.DataFrame(records)

    # Backup old file
    old_path = out_dir / "user_thresholds.parquet"
    bak_path = out_dir / "user_thresholds_v1.parquet"
    if old_path.exists():
        import shutil
        shutil.copy(old_path, bak_path)

    df.to_parquet(old_path, index=False)

    tc    = df["user_type"].value_counts()
    total = len(df)
    print(f"\n  Rebuilt {len(df):,} user × subreddit pairs")
    print("  --- New user type distribution ---")
    for t, n in tc.items():
        print(f"    {t:<12}  {n:>7,}  ({100*n/total:.1f}%)")

    return df


# ---------------------------------------------------------------------------
# Fix 2: Spillover heatmap with safe timezone handling
# ---------------------------------------------------------------------------

def safe_ts(w):
    """Convert any timestamp-like to UTC Timestamp safely."""
    ts = pd.Timestamp(w)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def plot_spillover_heatmap(panel: pd.DataFrame, plot_dir: Path):
    ctrl = panel[~panel["is_banned_sub"]].copy()
    ctrl["week"] = pd.to_datetime(ctrl["week"], utc=True)

    weekly = (
        ctrl.groupby(["subreddit", "week"])
        .agg(total_users =("author", "nunique"),
             banned_users=("from_banned_sub", "sum"))
        .reset_index()
    )
    weekly["pct_from_banned"] = (
        weekly["banned_users"] / weekly["total_users"].clip(lower=1) * 100
    )

    pivot = weekly.pivot(
        index="subreddit", columns="week", values="pct_from_banned"
    ).fillna(0)

    if pivot.empty:
        print("  Skipping spillover_heatmap.png — no data")
        return

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.5)))
    vmax = max(pivot.values.max(), 0.1)
    im   = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label="% users from banned subs")

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"r/{s}" for s in pivot.index], fontsize=9)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        [str(w)[:10] for w in pivot.columns],
        rotation=45, ha="right", fontsize=7
    )

    # Safe timezone comparison
    ban_cols = [
        i for i, w in enumerate(pivot.columns)
        if safe_ts(w) >= BAN_DATE and safe_ts(w) < BAN_DATE + pd.Timedelta(days=7)
    ]
    for bc in ban_cols:
        ax.axvline(bc - 0.5, color="blue", linewidth=2)

    ax.set_title(
        "Spillover: % of weekly users arriving from banned subreddits\n"
        "Blue line = Jun 10 ban wave",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(plot_dir / "spillover_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: spillover_heatmap.png")


# ---------------------------------------------------------------------------
# Replot threshold distributions and user types with new proxy
# ---------------------------------------------------------------------------

def plot_threshold_distributions_v2(user_thresholds: pd.DataFrame, plot_dir: Path):
    HIGH_LEG = {"science", "AskHistorians", "askscience"}
    MED_LEG  = {"politics", "worldnews", "news"}
    FRINGE   = {"KotakuInAction", "MensRights", "TumblrInAction"}

    cats = {
        "Banned (collapsed)": user_thresholds[
            user_thresholds["subreddit"].isin(BANNED_SUBREDDITS)],
        "Fringe (survived)":  user_thresholds[
            user_thresholds["subreddit"].isin(FRINGE)],
        "Medium legitimacy":  user_thresholds[
            user_thresholds["subreddit"].isin(MED_LEG)],
        "High legitimacy":    user_thresholds[
            user_thresholds["subreddit"].isin(HIGH_LEG)],
    }

    colors = ["#E24B4A", "#EF9F27", "#378ADD", "#1D9E75"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: persistence_corr distribution
    ax = axes[0]
    any_plotted = False
    for (label, sub_df), color in zip(cats.items(), colors):
        vals = sub_df["persistence_corr"].dropna()
        if len(vals) < 5:
            continue
        ax.hist(vals, bins=40, alpha=0.55, color=color,
                label=f"{label} (n={len(vals):,})", density=True)
        any_plotted = True
    if any_plotted:
        ax.axvline(0, color="black", linestyle=":", linewidth=1.2)
        ax.axvline(0.25, color="gray", linestyle="--", linewidth=1,
                   label="Initiator threshold")
        ax.set_xlabel("Posting-persistence correlation with C\n"
                      "(positive = posts MORE when enforcement rises = initiator)",
                      fontsize=10)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title("Persistence correlation\nby community type", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)

    # Right: theta_proxy_v2 distribution
    ax = axes[1]
    any_plotted = False
    for (label, sub_df), color in zip(cats.items(), colors):
        vals = sub_df["theta_proxy"].dropna()
        if len(vals) < 5:
            continue
        ax.hist(vals.clip(-0.5, 0.5), bins=40, alpha=0.55, color=color,
                label=f"{label} (n={len(vals):,})", density=True)
        any_plotted = True
    if any_plotted:
        ax.axvline(0, color="black", linestyle=":", linewidth=1.2, label="θ=0 boundary")
        ax.set_xlabel("θ proxy v2  (left = initiator, right = complier)", fontsize=10)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title("Combined θ proxy\nby community type", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)

    plt.suptitle("User threshold distributions (v2 — persistence-based)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(plot_dir / "threshold_distributions_v2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: threshold_distributions_v2.png")


def plot_user_types_v2(user_thresholds: pd.DataFrame, plot_dir: Path):
    counts = (
        user_thresholds.groupby(["subreddit", "user_type"])
        .size()
        .unstack(fill_value=0)
    )
    for col in ["INITIATOR", "JOINER", "COMPLIER"]:
        if col not in counts.columns:
            counts[col] = 0

    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    pct = pct.sort_values("INITIATOR", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(pct) * 0.5)))
    y = np.arange(len(pct))

    ax.barh(y, pct["COMPLIER"],  color="#1D9E75", label="Complier")
    ax.barh(y, pct["JOINER"],    left=pct["COMPLIER"],
            color="#EF9F27", label="Joiner")
    ax.barh(y, pct["INITIATOR"], left=pct["COMPLIER"] + pct["JOINER"],
            color="#E24B4A", label="Initiator")

    labels = [
        f"r/{s}  {'[BANNED]' if s in BANNED_SUBREDDITS else ''}"
        for s in pct.index
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("% of users", fontsize=11)
    ax.set_title(
        "User type composition — v2 (persistence-based θ)\n"
        "Prediction: banned/fringe subs have more initiators (red)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlim(0, 100)

    # Print initiator ranking
    print("\n  --- Initiator % by subreddit (v2) ---")
    for sub in pct.sort_values("INITIATOR", ascending=False).index:
        pct_init = pct.loc[sub, "INITIATOR"]
        tag = "[BANNED]" if sub in BANNED_SUBREDDITS else "       "
        bar = "█" * int(pct_init / 2)
        print(f"    {tag} r/{sub:<25}  {pct_init:5.1f}%  {bar}")

    plt.tight_layout()
    fig.savefig(plot_dir / "user_types_by_subreddit_v2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: user_types_by_subreddit_v2.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./output")
    parser.add_argument("--skip_theta_rebuild", action="store_true",
                        help="Skip theta rebuild (use if already done)")
    args   = parser.parse_args()
    out_dir   = Path(args.out_dir)
    plot_dir  = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    print("Loading data...")
    panel = pd.read_parquet(out_dir / "panel.parquet")

    # Fix 1: Rebuild theta
    if not args.skip_theta_rebuild:
        user_thresholds = rebuild_theta_proxy(out_dir)
    else:
        user_thresholds = pd.read_parquet(out_dir / "user_thresholds.parquet")

    # Fix 2: Spillover heatmap
    print("\nGenerating spillover heatmap (timezone-fixed)...")
    plot_spillover_heatmap(panel, plot_dir)

    # Replot threshold and user-type with v2 proxy
    print("\nRegenerating threshold and user-type plots...")
    plot_threshold_distributions_v2(user_thresholds, plot_dir)
    plot_user_types_v2(user_thresholds, plot_dir)

    print("\nDone. New files in output/plots/:")
    for p in sorted(plot_dir.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()