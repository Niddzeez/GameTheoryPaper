"""
spearman_test.py
================
Primary empirical proof for Prediction 1 (Post-Shock Divergence):

  Communities operating closer to their enforcement backfire threshold (g*)
  before an external shock exhibit worse post-shock behavioural trajectories.

Test
----
Spearman rank correlation between:
  X = C_pre / g*  (pre-intervention enforcement as fraction of backfire threshold)
  Y = post_slope  (OLS slope of C_raw_clean in the post-intervention window;
                   positive slope = worsening violations)

Prediction: ρ_s > 0 and p < 0.05
  (higher X → closer to backfire → worse post-shock trajectory)

Three cohorts
-------------
  A — 2015 Reddit pushshift cohort (primary; used for Q_backfire in MCI)
  B — TBBT quarantine cohort (supplementary; volume-based fragility proxy)
  C — Combined (primary public-facing result)

Outputs
-------
  spearman_results.json          — all ρ_s, p-values, CIs, N
  plots/spearman_scatter.png     — scatter C/g* vs post-slope (cohort A)
  plots/spearman_combined.png    — both cohorts on normalised axes

Usage
-----
    python spearman_test.py --pushshift_out ../Reddit-2015-v2/output \
                            --out_dir ./output
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BAN_DATE   = pd.Timestamp("2015-06-10", tz="UTC")
BANNED     = {"fatpeoplehate", "CoonTown", "transfags", "neofag", "hamplanet"}

# Confidence gate for pushshift cohort
# Q_fit is NOT used as a hard gate — poor ODE fit is expected and is reported
# honestly in the MCI. Excluding on Q_fit would remove all subreddits.
# Gate only on data availability (n_preban) and g* validity (non-NaN, positive).
GATE_MIN_PREBAN_DAYS = 21
GATE_GSTAR_LOW       = 1e-4   # only exclude literal zero / numerical failure
GATE_GSTAR_HIGH      = 10.0   # exclude only wildly implausible estimates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ols_slope(y: np.ndarray) -> tuple[float, float]:
    """Returns (slope, p_value) of OLS regression of y on integer time index."""
    n = len(y)
    if n < 5:
        return np.nan, np.nan
    x = np.arange(n, dtype=float)
    res = stats.linregress(x, y)
    return float(res.slope), float(res.pvalue)


def bootstrap_spearman(x: np.ndarray, y: np.ndarray,
                        n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """Bootstrap 95% CI for Spearman ρ."""
    rng = np.random.default_rng(seed)
    rhos = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xs, ys = x[idx], y[idx]
        if len(np.unique(xs)) < 2 or len(np.unique(ys)) < 2:
            continue
        r, _ = stats.spearmanr(xs, ys)
        rhos.append(r)
    if not rhos:
        return np.nan, np.nan
    lo = float(np.percentile(rhos, 2.5))
    hi = float(np.percentile(rhos, 97.5))
    return lo, hi


# ---------------------------------------------------------------------------
# Cohort A: 2015 pushshift
# ---------------------------------------------------------------------------

def build_pushshift_cohort(pushshift_out: Path) -> pd.DataFrame:
    """
    Build the X / Y pairs for the pushshift cohort.
      X = C_pre_mean / g_star   (pre-ban enforcement as fraction of backfire threshold)
      Y = post_slope             (OLS slope of C_raw_clean post-ban)
    """
    # Required files
    required = {
        "daily":    pushshift_out / "daily_metrics.parquet",
        "calibrated": pushshift_out / "calibrated_params.csv",
        "stage2":   pushshift_out / "stage2_results.csv",
    }
    missing = [k for k, p in required.items() if not p.exists()]
    if missing:
        log.error(f"Missing pushshift files: {missing} — run the pipeline first.")
        return pd.DataFrame()

    daily      = pd.read_parquet(required["daily"])
    calibrated = pd.read_csv(required["calibrated"])
    stage2     = pd.read_csv(required["stage2"])

    daily["day"] = pd.to_datetime(daily["day"], utc=True)

    # Use C_raw_clean if available, fall back to C_raw
    c_col = "C_raw_clean" if "C_raw_clean" in daily.columns else "C_raw"
    if c_col == "C_raw":
        log.warning("C_raw_clean not in daily_metrics — using C_raw (includes self-deletions). "
                    "Re-run reddit_pipeline_safe.py for the clean signal.")

    rows = []
    for _, cal in calibrated.iterrows():
        sub    = cal["subreddit"]
        g_star = cal.get("g_star", np.nan)
        n_preban = cal.get("n_preban_days", 0)

        sub_daily = daily[daily["subreddit"] == sub].sort_values("day")
        pre  = sub_daily[sub_daily["day"] < BAN_DATE]
        post = sub_daily[sub_daily["day"] >= BAN_DATE]

        # ------- Confidence gate -------
        # Only gate on data quantity and g* numerical validity.
        # Q_fit is intentionally excluded from the gate: negative Q_fit values
        # reflect genuine poor ODE fit (reported in MCI as a limitation) but do
        # not invalidate the g* estimate as a structural ranking variable.
        gate_reason = None
        if n_preban < GATE_MIN_PREBAN_DAYS:
            gate_reason = f"n_preban={n_preban} < {GATE_MIN_PREBAN_DAYS}"
        elif np.isnan(g_star):
            gate_reason = "g_star=NaN"
        elif g_star < GATE_GSTAR_LOW or g_star > GATE_GSTAR_HIGH:
            gate_reason = f"g_star={g_star:.6f} outside [{GATE_GSTAR_LOW}, {GATE_GSTAR_HIGH}]"

        passed_gate = gate_reason is None

        # ------- X: pre-ban enforcement / g* -------
        C_pre_mean = float(pre[c_col].mean()) if not pre.empty else np.nan
        C_g_star_ratio = C_pre_mean / g_star if (g_star and g_star > 0) else np.nan

        # ------- Y: post-ban violation slope -------
        if len(post) >= 7:
            slope, slope_p = ols_slope(post[c_col].fillna(0).values)
        else:
            slope, slope_p = np.nan, np.nan

        # ------- Stability margin from stage2 -------
        s2row = stage2[stage2["subreddit"] == sub]
        regime  = s2row["regime"].iloc[0]  if not s2row.empty else "Indeterminate"
        C_eq    = s2row["C_equilibrium"].iloc[0] if not s2row.empty else np.nan
        Xi      = s2row["Xi"].iloc[0]             if not s2row.empty else np.nan
        alpha_gov = s2row["alpha_gov"].iloc[0]    if not s2row.empty else np.nan
        alpha_gov_min = s2row["alpha_gov_min"].iloc[0] if not s2row.empty else np.nan

        rows.append({
            "cohort":            "pushshift_2015",
            "subreddit":         sub,
            "is_banned":         sub in BANNED,
            "g_star":            g_star,
            "C_pre_mean":        C_pre_mean,
            "C_g_star_ratio":    C_g_star_ratio,
            "post_slope":        slope,
            "post_slope_p":      slope_p,
            "regime":            regime,
            "C_equilibrium":     C_eq,
            "Xi":                Xi,
            "alpha_gov":         alpha_gov,
            "alpha_gov_min":     alpha_gov_min,
            "n_preban_days":     n_preban,
            "passed_gate":       passed_gate,
            "gate_reason":       gate_reason,
        })

    df = pd.DataFrame(rows)
    passed = df[df["passed_gate"]]
    log.info(f"Pushshift cohort: {len(df)} subreddits total, "
             f"{len(passed)} passed gate, "
             f"{len(df) - len(passed)} excluded")
    for _, r in df[~df["passed_gate"]].iterrows():
        log.info(f"  EXCLUDED r/{r['subreddit']}: {r['gate_reason']}")
    return df


# ---------------------------------------------------------------------------
# Cohort B: TBBT quarantine
# ---------------------------------------------------------------------------

def build_tbbt_cohort(out_dir: Path) -> pd.DataFrame:
    """
    Load the TBBT quarantine fragility output from tbbt_calibration.py.
    Rename columns to match pushshift cohort format for combined test.
    """
    frag_path = out_dir / "tbbt_fragility.csv"
    if not frag_path.exists():
        log.warning(f"TBBT fragility file not found: {frag_path} "
                    f"— run tbbt_calibration.py first.")
        return pd.DataFrame()

    df = pd.read_csv(frag_path)

    # Map to shared column names for combined test
    df = df.rename(columns={
        "fragility_index":   "C_g_star_ratio",   # normalised [0,1], higher = more fragile
        "post_deterioration":"post_slope",        # normalised [0,1], higher = worse outcome
    })
    df["cohort"]     = "tbbt_quarantine"
    df["is_banned"]  = False
    df["g_star"]     = np.nan           # not computed for TBBT
    df["C_pre_mean"] = np.nan
    df["regime"]     = "quarantine"

    # Already filtered by tbbt_calibration.py gate
    df["passed_gate"] = True

    log.info(f"TBBT quarantine cohort: {len(df)} events")
    return df


# ---------------------------------------------------------------------------
# Spearman test runner
# ---------------------------------------------------------------------------

def run_spearman(df: pd.DataFrame, label: str) -> dict:
    """Run Spearman test on passed-gate rows, return result dict."""
    passed = df[df["passed_gate"] & df["C_g_star_ratio"].notna()
                & df["post_slope"].notna()]
    n = len(passed)
    if n < 4:
        log.warning(f"{label}: only {n} valid observations — test unreliable.")
        return {"label": label, "n": n, "rho": np.nan, "p": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan, "interpretation": "insufficient_data"}

    rho, p = stats.spearmanr(passed["C_g_star_ratio"], passed["post_slope"])
    ci_lo, ci_hi = bootstrap_spearman(
        passed["C_g_star_ratio"].values, passed["post_slope"].values)

    # Interpretation
    if p < 0.05 and rho > 0:
        interp = "CONFIRMED: higher fragility → worse post-shock outcome (p<0.05)"
    elif p < 0.10 and rho > 0:
        interp = "MARGINAL: directionally consistent but p<0.10 only"
    elif p >= 0.10 and rho > 0:
        interp = "DIRECTIONAL but not significant: consistent with model, insufficient power"
    elif rho <= 0:
        interp = "INCONSISTENT with prediction (ρ≤0)"
    else:
        interp = "INCONCLUSIVE"

    log.info(f"\n{'='*55}")
    log.info(f"Cohort: {label}")
    log.info(f"  N         = {n}")
    log.info(f"  ρ_s       = {rho:+.4f}")
    log.info(f"  p-value   = {p:.4f}")
    log.info(f"  95% CI    = [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    log.info(f"  Result    → {interp}")
    log.info(f"{'='*55}")

    return {
        "label": label, "n": n,
        "rho": round(float(rho), 4),
        "p":   round(float(p), 4),
        "ci_lo": round(float(ci_lo), 4),
        "ci_hi": round(float(ci_hi), 4),
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# Kruskal-Wallis: post_slope by regime (Prediction 2)
# ---------------------------------------------------------------------------

def run_kruskal(df_pushshift: pd.DataFrame) -> dict:
    """
    Test whether post-ban slope differs across stability regimes.
    Prediction 2: structural instability regime should have highest slopes.
    """
    passed = df_pushshift[df_pushshift["passed_gate"] & df_pushshift["post_slope"].notna()]
    groups = {r: grp["post_slope"].values
              for r, grp in passed.groupby("regime")
              if len(grp) >= 2}

    if len(groups) < 2:
        log.info("Kruskal-Wallis: not enough regime groups (need ≥2 with n≥2)")
        return {}

    stat, p = stats.kruskal(*groups.values())
    result = {
        "statistic": round(float(stat), 4),
        "p": round(float(p), 4),
        "groups": {r: {"n": len(v), "median_slope": round(float(np.median(v)), 5)}
                   for r, v in groups.items()},
    }
    log.info(f"\nKruskal-Wallis (post-slope by regime): H={stat:.3f}, p={p:.4f}")
    for r, info in result["groups"].items():
        log.info(f"  {r:<28}: n={info['n']}, median slope={info['median_slope']:+.5f}")
    return result


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_scatter(df: pd.DataFrame, out_dir: Path):
    """Scatter C/g* vs post_slope for pushshift cohort."""
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    passed = df[df["passed_gate"] & df["C_g_star_ratio"].notna()
                & df["post_slope"].notna()].copy()
    if passed.empty:
        log.warning("No data for scatter plot.")
        return

    regime_colors = {
        "Stable":                "#1D9E75",
        "Governance trap":       "#EF9F27",
        "Structural instability":"#E24B4A",
        "Indeterminate":         "#888780",
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for regime, grp in passed.groupby("regime"):
        color = regime_colors.get(regime, "#888780")
        ax.scatter(grp["C_g_star_ratio"], grp["post_slope"],
                   c=color, s=90, alpha=0.85, label=regime, zorder=3,
                   edgecolors="white", linewidth=0.5)
        for _, r in grp.iterrows():
            ax.annotate(
                r["subreddit"],
                (r["C_g_star_ratio"], r["post_slope"]),
                fontsize=7.5, alpha=0.8,
                xytext=(5, 3), textcoords="offset points",
            )

    # Regression line
    x_all = passed["C_g_star_ratio"].values
    y_all = passed["post_slope"].values
    if len(x_all) >= 3:
        z = np.polyfit(x_all, y_all, 1)
        x_line = np.linspace(x_all.min(), x_all.max(), 100)
        ax.plot(x_line, np.poly1d(z)(x_line), "k--", linewidth=1.2, alpha=0.5)

    # Spearman annotation
    rho, p = stats.spearmanr(x_all, y_all)
    ax.text(0.97, 0.05,
            f"Spearman ρ = {rho:+.3f}\np = {p:.3f}  (N = {len(x_all)})",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.axvline(1, color="red", linewidth=0.8, linestyle=":", alpha=0.6,
               label="g* threshold (C/g*=1)")
    ax.set_xlabel("Pre-ban Enforcement Intensity  (C̄_pre / g*)", fontsize=11)
    ax.set_ylabel("Post-ban Violation Slope  (OLS β on C_raw_clean)", fontsize=11)
    ax.set_title("Prediction 1: Structural Fragility → Post-Shock Divergence\n"
                 "2015 Reddit Pushshift Cohort", fontsize=12)
    ax.legend(fontsize=8, framealpha=0.8)
    fig.tight_layout()

    out_path = plot_dir / "spearman_scatter.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {out_path}")


def plot_combined(ps_df: pd.DataFrame, tbbt_df: pd.DataFrame, out_dir: Path):
    """Combined normalised scatter: both cohorts on the same axes."""
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    frames = []
    for df, label in [(ps_df, "Pushshift 2015"), (tbbt_df, "TBBT Quarantine")]:
        if df.empty:
            continue
        passed = df[df["passed_gate"] & df["C_g_star_ratio"].notna()
                    & df["post_slope"].notna()].copy()
        if passed.empty:
            continue
        # Normalise each cohort's axes to [0,1] for comparability
        for col in ["C_g_star_ratio", "post_slope"]:
            lo, hi = passed[col].min(), passed[col].max()
            passed[f"{col}_norm"] = (passed[col] - lo) / (hi - lo + 1e-9)
        passed["cohort_label"] = label
        frames.append(passed)

    if not frames:
        log.warning("No data for combined scatter.")
        return

    combined = pd.concat(frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    cohort_colors = {"Pushshift 2015": "#2166ac", "TBBT Quarantine": "#d6604d"}

    for cohort_label, grp in combined.groupby("cohort_label"):
        color = cohort_colors.get(cohort_label, "grey")
        ax.scatter(grp["C_g_star_ratio_norm"], grp["post_slope_norm"],
                   c=color, s=80, alpha=0.8, label=cohort_label,
                   edgecolors="white", linewidth=0.5)

    # Overall trend
    x_all = combined["C_g_star_ratio_norm"].values
    y_all = combined["post_slope_norm"].values
    if len(x_all) >= 4:
        z = np.polyfit(x_all, y_all, 1)
        x_line = np.linspace(0, 1, 100)
        ax.plot(x_line, np.poly1d(z)(x_line), "k--", linewidth=1.2,
                alpha=0.5, label="Combined trend")
        rho, p = stats.spearmanr(x_all, y_all)
        ax.text(0.97, 0.05,
                f"Combined ρ = {rho:+.3f}\np = {p:.3f}  (N = {len(x_all)})",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    ax.set_xlabel("Normalised Fragility Index  (within cohort)", fontsize=11)
    ax.set_ylabel("Normalised Post-Shock Deterioration  (within cohort)", fontsize=11)
    ax.set_title("Prediction 1 — Combined Cohort Validation\n"
                 "Pushshift 2015 (N={}) + TBBT Quarantine (N={})".format(
                     len(frames[0]) if frames else 0,
                     len(frames[1]) if len(frames) > 1 else 0),
                 fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out_path = plot_dir / "spearman_combined.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pushshift_out", required=True,
                        help="Path to Reddit-2015-v2/output/ folder")
    parser.add_argument("--out_dir", default="./output")
    args = parser.parse_args()

    pushshift_out = Path(args.pushshift_out)
    out_dir       = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Building cohorts...")
    ps_df   = build_pushshift_cohort(pushshift_out)
    tbbt_df = build_tbbt_cohort(out_dir)

    # Save per-subreddit data for MCI use
    if not ps_df.empty:
        ps_df.to_csv(out_dir / "pushshift_spearman_input.csv", index=False)
        log.info(f"Saved: pushshift_spearman_input.csv")

    # --- Run Spearman tests ---
    results = {}

    if not ps_df.empty:
        results["cohort_A_pushshift"] = run_spearman(ps_df, "Cohort A — 2015 Pushshift")

    if not tbbt_df.empty:
        results["cohort_B_tbbt"] = run_spearman(tbbt_df, "Cohort B — TBBT Quarantine")

    if not ps_df.empty and not tbbt_df.empty:
        combined = pd.concat([ps_df, tbbt_df], ignore_index=True)
        results["cohort_C_combined"] = run_spearman(combined, "Cohort C — Combined")
    elif not ps_df.empty:
        results["cohort_C_combined"] = results.get("cohort_A_pushshift", {})

    # --- Kruskal-Wallis (Prediction 2) ---
    if not ps_df.empty:
        results["kruskal_wallis_prediction2"] = run_kruskal(ps_df)

    # --- Save JSON results ---
    out_json = out_dir / "spearman_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nSaved: {out_json}")

    # --- Plots ---
    if not ps_df.empty:
        plot_scatter(ps_df, out_dir)
    plot_combined(ps_df, tbbt_df, out_dir)

    # --- Capsule summary for paper ---
    log.info("\n" + "="*55)
    log.info("PAPER RESULT CAPSULE")
    log.info("="*55)
    for key, res in results.items():
        if "rho" in res:
            log.info(f"  {key}")
            log.info(f"    ρ_s = {res.get('rho', 'n/a'):>8}  "
                     f"p = {res.get('p', 'n/a'):>6}  "
                     f"N = {res.get('n', 'n/a')}  "
                     f"95%CI [{res.get('ci_lo', '?')}, {res.get('ci_hi', '?')}]")
            log.info(f"    → {res.get('interpretation', '')}")
    log.info("="*55)


if __name__ == "__main__":
    main()
