"""
tbbt_calibration.py
===================
Builds a cross-intervention fragility-vs-outcome dataset from TBBT data for
use in the Spearman rank correlation test (spearman_test.py).

Why this file exists
--------------------
TBBT does not contain removal rates, so the 3-variable ODE system (x, T, C)
cannot be calibrated on TBBT communities directly. Instead, we use an
engagement-based fragility proxy:

  Fragility proxy (X):
    A composite of pre-intervention trends in volume and engagement score.
    Communities with declining volume or score in the 30 days before the
    intervention are treated as "more fragile" — they are closer to a tipping
    point regardless of whether we can compute g* explicitly.

  Post-intervention outcome (Y):
    Volume and engagement retention 30 days after the intervention, expressed
    as a ratio relative to the pre-intervention baseline.

This operationalisation is consistent with the paper's Prediction 1 (systems
closer to the threshold before an external shock exhibit worse post-shock
trajectories) but uses observable engagement signals rather than model-derived
parameters. It supplements — not replaces — the ODE-based Spearman test on
the 2015 pushshift cohort.

Quarantine events are used because they are the only TBBT intervention type
that provides BOTH in-before AND in-after data from the SAME community.

Outputs
-------
  tbbt_fragility.csv  — per-intervention fragility proxy, outcome, and metadata

Usage
-----
    python tbbt_calibration.py --out_dir ./output
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def ols_slope(y: np.ndarray) -> float:
    """OLS slope of y on integer time index, centred."""
    n = len(y)
    if n < 3:
        return np.nan
    x = np.arange(n, dtype=float) - np.arange(n, dtype=float).mean()
    ss_xx = np.sum(x ** 2)
    if ss_xx < 1e-12:
        return 0.0
    return float(np.sum(x * (y - y.mean())) / ss_xx)


def compute_fragility(out_dir: Path):
    quar_path = out_dir / "tbbt_quarantine_outcomes.csv"
    if not quar_path.exists():
        log.error(f"Missing: {quar_path} — run tbbt_parser.py first.")
        return

    quar = pd.read_csv(quar_path)
    log.info(f"Loaded {len(quar)} quarantine events from {quar_path.name}")

    # ------------------------------------------------------------------
    # Confidence gate — same philosophy as the pushshift gate
    # ------------------------------------------------------------------
    # Require at least 21 days of pre-intervention data
    initial_n = len(quar)
    quar = quar[quar["n_pre_days"] >= 21].copy()
    # Require post-intervention data present
    quar = quar[quar["n_post_days"] >= 14].copy()
    # Require non-NaN outcome
    quar = quar[quar["post_vol_ratio_30d"].notna()].copy()
    log.info(f"After gate: {len(quar)} / {initial_n} quarantine events pass "
             f"(≥21 pre-days, ≥14 post-days, non-NaN outcome)")

    if quar.empty:
        log.warning("No quarantine events passed the confidence gate.")
        return

    # ------------------------------------------------------------------
    # Fragility index
    # ------------------------------------------------------------------
    # We combine two pre-intervention signals into a single fragility score
    # scaled to [0, 1].  Higher = more fragile / closer to instability.
    #
    #   F1: normalised negative volume slope
    #       (declining volume → approaching collapse → more fragile)
    #   F2: normalised negative score slope
    #       (declining engagement → eroding legitimacy → more fragile)
    #
    # Both components are normalised to [0, 1] via min-max across the cohort.

    def minmax(series: pd.Series) -> pd.Series:
        lo, hi = series.min(), series.max()
        if hi == lo:
            return pd.Series(0.5, index=series.index)
        return (series - lo) / (hi - lo)

    # Invert: more negative slope → higher fragility
    quar["F1"] = minmax(-quar["pre_volume_slope"].fillna(0))
    quar["F2"] = minmax(-quar["pre_score_slope"].fillna(0))
    quar["fragility_index"] = 0.5 * quar["F1"] + 0.5 * quar["F2"]

    # ------------------------------------------------------------------
    # Post-intervention outcome: worse outcome = lower volume retention
    # Invert: lower vol_ratio → worse outcome → higher "post_deterioration"
    # ------------------------------------------------------------------
    quar["post_deterioration"] = 1.0 - quar["post_vol_ratio_30d"].clip(0, 2) / 2.0

    # ------------------------------------------------------------------
    # Spearman test within TBBT quarantine cohort
    # ------------------------------------------------------------------
    if len(quar) >= 4:
        rho, p = stats.spearmanr(
            quar["fragility_index"], quar["post_deterioration"])
        log.info(f"TBBT quarantine Spearman: ρ_s = {rho:.3f},  p = {p:.3f}  "
                 f"(N = {len(quar)})")
    else:
        rho, p = np.nan, np.nan
        log.warning(f"Only {len(quar)} quarantine events after gating — "
                    f"Spearman test not reliable at this N.")

    # ------------------------------------------------------------------
    # Flag interventions for attention system compatibility
    # ------------------------------------------------------------------
    # Map to the same output format as pushshift stage2_results for use
    # in the combined Spearman test.
    quar["cohort"]          = "tbbt_quarantine"
    quar["spearman_rho"]    = rho
    quar["spearman_p"]      = p
    quar["n_cohort"]        = len(quar)
    quar["passed_gate"]     = True

    out_path = out_dir / "tbbt_fragility.csv"
    quar.to_csv(out_path, index=False)
    log.info(f"Saved: {out_path}")

    # ------------------------------------------------------------------
    # Also compute post-ban score trends for OUT-AFTER data
    # (supports extended spillover analysis)
    # ------------------------------------------------------------------
    daily_path = out_dir / "tbbt_daily.parquet"
    if not daily_path.exists():
        log.warning("tbbt_daily.parquet not found — skipping out-after analysis.")
        return

    daily = pd.read_parquet(daily_path)
    out_after = daily[daily["slice"] == "out_after"].copy()
    if out_after.empty:
        log.info("No out-after data found in tbbt_daily.parquet.")
        return

    # Compute per-intervention aggregate post-ban volume and score
    spillover_agg = (out_after
                     .groupby("intervention_id")
                     .agg(
                         n_post_comments=("n_comments", "sum"),
                         mean_post_score=("mean_score", "mean"),
                         n_post_days=("day", "count"),
                     )
                     .reset_index())

    out_spill = out_dir / "tbbt_ban_postban_agg.csv"
    spillover_agg.to_csv(out_spill, index=False)
    log.info(f"Saved post-ban out-after aggregate: {out_spill} "
             f"({len(spillover_agg)} ban events)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./output")
    args = parser.parse_args()
    compute_fragility(Path(args.out_dir))


if __name__ == "__main__":
    main()
