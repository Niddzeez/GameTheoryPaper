"""
distribution_analysis.py
=========================
Produces publication-quality plots of the threshold distribution results,
using statistical tests rather than hard classification thresholds.

The core claim: community type predicts the SHAPE of the persistence
correlation distribution, not just a binary user type label.

Outputs:
  plots/distribution_ks_test.png   — distributions + KS test p-values
  plots/mean_corr_by_subreddit.png — mean persistence corr per subreddit
                                     (cleaner than stacked bar)
  distribution_stats.csv           — summary stats + KS test results

Run:
    python3 distribution_analysis.py --out_dir ./output
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

BANNED = {"fatpeoplehate", "CoonTown", "transfags", "neofag", "hamplanet"}
HIGH_LEG = {"science", "AskHistorians", "askscience"}
MED_LEG  = {"politics", "worldnews", "news"}
FRINGE   = {"KotakuInAction", "MensRights", "TumblrInAction"}
GENERAL  = {"AskReddit", "funny", "todayilearned"}
SPILLOVER= {"loseit", "progresspics", "obesity"}

def get_category(sub):
    if sub in BANNED:     return "Banned (collapsed)"
    if sub in FRINGE:     return "Fringe (survived)"
    if sub in MED_LEG:    return "Medium legitimacy"
    if sub in HIGH_LEG:   return "High legitimacy"
    if sub in SPILLOVER:  return "Spillover destinations"
    return "General"

CATEGORY_COLORS = {
    "Banned (collapsed)":    "#E24B4A",
    "Fringe (survived)":     "#EF9F27",
    "Spillover destinations":"#D4537E",
    "Medium legitimacy":     "#378ADD",
    "High legitimacy":       "#1D9E75",
    "General":               "#888780",
}


def load_data(out_dir: Path):
    ut = pd.read_parquet(out_dir / "user_thresholds.parquet")
    ut["category"] = ut["subreddit"].apply(get_category)
    return ut


def run_ks_tests(ut: pd.DataFrame) -> pd.DataFrame:
    """
    KS test: for each pair of community categories, test whether their
    persistence_corr distributions are significantly different.
    Also compute mean, median, std per category.
    """
    cats = ["Banned (collapsed)", "Fringe (survived)",
            "Medium legitimacy", "High legitimacy"]

    # Summary stats per category
    stats_rows = []
    for cat in cats:
        vals = ut[ut["category"] == cat]["persistence_corr"].dropna()
        if len(vals) < 10:
            continue
        stats_rows.append({
            "category": cat,
            "n": len(vals),
            "mean": vals.mean(),
            "median": vals.median(),
            "std": vals.std(),
            "pct_positive": (vals > 0.1).mean() * 100,   # % initiator-like
            "pct_negative": (vals < -0.1).mean() * 100,  # % complier-like
        })
    stats_df = pd.DataFrame(stats_rows)

    # KS tests: each category vs high-legitimacy baseline
    baseline = ut[ut["category"] == "High legitimacy"]["persistence_corr"].dropna()
    ks_rows = []
    for cat in cats:
        if cat == "High legitimacy":
            continue
        vals = ut[ut["category"] == cat]["persistence_corr"].dropna()
        if len(vals) < 10:
            continue
        ks_stat, p_val = stats.ks_2samp(vals, baseline)
        ks_rows.append({
            "comparison": f"{cat} vs High legitimacy",
            "ks_statistic": ks_stat,
            "p_value": p_val,
            "significant": p_val < 0.05,
        })
    ks_df = pd.DataFrame(ks_rows)

    return stats_df, ks_df


def plot_distributions_with_ks(ut: pd.DataFrame, ks_df: pd.DataFrame,
                                 stats_df: pd.DataFrame, plot_dir: Path):
    """
    Clean 4-panel plot: one histogram per community category,
    with KS test result annotated, and vertical lines for mean.
    """
    cats = ["Banned (collapsed)", "Fringe (survived)",
            "Medium legitimacy", "High legitimacy"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=False)
    axes = axes.flatten()

    # Shared reference: high-legitimacy distribution
    baseline = ut[ut["category"] == "High legitimacy"]["persistence_corr"].dropna()

    for i, cat in enumerate(cats):
        ax   = axes[i]
        vals = ut[ut["category"] == cat]["persistence_corr"].dropna()
        color = CATEGORY_COLORS[cat]

        if len(vals) < 10:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(cat, fontsize=11)
            continue

        # Histogram
        ax.hist(vals, bins=40, color=color, alpha=0.7, density=True, label=cat)

        # Overlay baseline in gray for comparison
        if cat != "High legitimacy":
            ax.hist(baseline, bins=40, color="#888780", alpha=0.25,
                    density=True, label="High legitimacy (ref)")

        # Vertical lines
        ax.axvline(vals.mean(), color=color, linestyle="-", linewidth=2,
                   label=f"Mean = {vals.mean():.3f}")
        ax.axvline(0, color="black", linestyle=":", linewidth=1.2, alpha=0.7)
        ax.axvline(0.25, color="gray", linestyle="--", linewidth=1, alpha=0.5)

        # KS test annotation
        ks_row = ks_df[ks_df["comparison"].str.startswith(cat)]
        if not ks_row.empty:
            ks_stat = ks_row.iloc[0]["ks_statistic"]
            p_val   = ks_row.iloc[0]["p_value"]
            sig     = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            ax.text(0.97, 0.95,
                    f"KS vs hi-leg: D={ks_stat:.3f}, p={p_val:.2e} {sig}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

        # Stats annotation
        s_row = stats_df[stats_df["category"] == cat]
        if not s_row.empty:
            pct_pos = s_row.iloc[0]["pct_positive"]
            pct_neg = s_row.iloc[0]["pct_negative"]
            ax.text(0.03, 0.95,
                    f"n={len(vals):,}\n+initiator-like: {pct_pos:.1f}%\n−complier-like: {pct_neg:.1f}%",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

        ax.set_title(cat, fontsize=12, fontweight="bold", color=color)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.2)

    for ax in axes:
        ax.set_xlabel("Posting-persistence correlation with C\n"
                      "(right = posts more when enforcement rises = initiator)",
                      fontsize=9)

    fig.suptitle(
        "Persistence correlation distributions by community type\n"
        "KS test vs high-legitimacy baseline — do community types differ?",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(plot_dir / "distribution_ks_test.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: distribution_ks_test.png")


def plot_mean_corr_by_subreddit(ut: pd.DataFrame, plot_dir: Path):
    """
    Dot plot: mean persistence correlation per subreddit, coloured by category.
    Cleaner and more honest than a stacked bar with near-zero initiators.
    Error bars = 95% CI.
    """
    rows = []
    for sub, grp in ut.groupby("subreddit"):
        vals = grp["persistence_corr"].dropna()
        if len(vals) < 20:
            continue
        se = vals.std() / np.sqrt(len(vals))
        rows.append({
            "subreddit": sub,
            "category": get_category(sub),
            "mean_corr": vals.mean(),
            "ci95": 1.96 * se,
            "n": len(vals),
            "pct_initiator_like": (vals > 0.25).mean() * 100,
        })

    df = pd.DataFrame(rows).sort_values("mean_corr", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.5)))

    for i, (_, row) in enumerate(df.iterrows()):
        color = CATEGORY_COLORS.get(row["category"], "#888780")
        ax.errorbar(row["mean_corr"], i,
                    xerr=row["ci95"],
                    fmt="o", color=color, markersize=8,
                    elinewidth=1.5, capsize=4)
        ax.text(row["mean_corr"] + row["ci95"] + 0.01, i,
                f"{row['pct_initiator_like']:.1f}%",
                va="center", fontsize=8, color=color)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([
        f"r/{r['subreddit']}  {'[BANNED]' if r['subreddit'] in BANNED else ''}"
        for _, r in df.iterrows()
    ], fontsize=9)

    ax.axvline(0, color="black", linestyle=":", linewidth=1.2, alpha=0.7, label="No correlation")
    ax.axvline(0.25, color="gray", linestyle="--", linewidth=1, alpha=0.6,
               label="Initiator threshold (0.25)")

    # Legend patches
    patches = [mpatches.Patch(color=v, label=k)
               for k, v in CATEGORY_COLORS.items()
               if k in df["category"].values]
    ax.legend(handles=patches, fontsize=9, loc="lower right")

    ax.set_xlabel(
        "Mean posting-persistence correlation with enforcement C\n"
        "Right of zero = community's users post MORE when enforcement rises",
        fontsize=10
    )
    ax.set_title(
        "Mean persistence correlation by subreddit\n"
        "% labels = fraction of users with corr > 0.25 (initiator-like behaviour)",
        fontsize=12, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(plot_dir / "mean_corr_by_subreddit.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: mean_corr_by_subreddit.png")


def print_and_save_results(stats_df, ks_df, out_dir):
    print("\n" + "="*60)
    print("DISTRIBUTION ANALYSIS RESULTS")
    print("="*60)

    print("\n--- Summary statistics by community category ---")
    print(f"  {'Category':<26} {'n':>7}  {'mean':>7}  {'median':>7}  "
          f"{'% init-like':>12}  {'% compl-like':>12}")
    print("  " + "-"*75)
    for _, r in stats_df.iterrows():
        print(f"  {r['category']:<26} {r['n']:>7,}  {r['mean']:>7.3f}  "
              f"{r['median']:>7.3f}  {r['pct_positive']:>11.1f}%  "
              f"{r['pct_negative']:>11.1f}%")

    print("\n--- KS tests vs high-legitimacy baseline ---")
    print(f"  {'Comparison':<45} {'D':>6}  {'p-value':>10}  {'Sig':>5}")
    print("  " + "-"*70)
    for _, r in ks_df.iterrows():
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 \
              else "*" if r["p_value"] < 0.05 else "ns"
        print(f"  {r['comparison']:<45} {r['ks_statistic']:>6.3f}  "
              f"{r['p_value']:>10.2e}  {sig:>5}")

    print("\n--- Interpretation ---")
    print("  KS statistic D: how different the distributions are (0=identical, 1=completely different)")
    print("  Significant p (< 0.05): community type predicts threshold distribution shape")
    print("  % initiator-like: users with corr > 0.25 (post more when enforcement rises)")
    print("  % complier-like:  users with corr < -0.1 (post less when enforcement rises)")

    combined = pd.concat([
        stats_df.assign(type="stats"),
        ks_df.assign(type="ks")
    ], ignore_index=True)
    combined.to_csv(out_dir / "distribution_stats.csv", index=False)
    print("\n  Saved: distribution_stats.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./output")
    args    = parser.parse_args()
    out_dir  = Path(args.out_dir)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    print("Loading user thresholds...")
    ut = load_data(out_dir)
    print(f"  {len(ut):,} user × subreddit pairs loaded")

    print("\nRunning KS tests...")
    stats_df, ks_df = run_ks_tests(ut)

    print_and_save_results(stats_df, ks_df, out_dir)

    print("\nGenerating plots...")
    plot_distributions_with_ks(ut, ks_df, stats_df, plot_dir)
    plot_mean_corr_by_subreddit(ut, plot_dir)

    print(f"\nDone. Check output/plots/ and output/distribution_stats.csv")


if __name__ == "__main__":
    main()