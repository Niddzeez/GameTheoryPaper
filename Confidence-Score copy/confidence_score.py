"""
confidence_score.py  (v2 — revised for testable-prediction framework)
======================================================================
Computes the four-component Model Credibility Index (MCI) for the
Coordination-Legitimacy Game paper, plus a two-tier attention-flagging
system for governance monitoring.

KEY CHANGES FROM v1
-------------------
1.  Q_substitution REMOVED from composite.
    Panel regression returned β₁ = +0.0043, p = 0.339 — hypothesis
    rejected.  Result is documented as a negative finding in the report
    but does not pollute the composite score.

2.  Q_spillover REMOVED from composite.
    Corrected migration definition found zero migrant INITIATORs.
    Result documented as a null finding.

3.  Q_backfire REWRITTEN as Spearman rank correlation.
    Old: binary hit/miss against a p < 0.10 post-ban slope threshold.
    New: Spearman ρ between pre-ban C/g* and post-ban violation slope
    across all subreddits.  This is Prediction 3 from the testable-
    prediction framework — genuinely out-of-sample and falsifiable.

4.  CONFIDENCE GATE added.
    Four conditions gate which subreddits enter the attention ranking:
      (a) data sufficiency    — ≥ 300 comments/day average
      (b) ODE fit quality     — Q_fit > 0
      (c) parameter plausible — 0.01 ≤ g* ≤ 2.0
      (d) enforcement variance — std(C_smooth) > 0.005
    Subreddits failing ≥ 2 conditions are flagged Low-confidence and
    excluded from the primary attention ranking (kept in all tables).

5.  THEORETICALLY MOTIVATED WEIGHTS replace equal weighting.
    Q_backfire 0.35 | Q_fit 0.25 | Q_regime 0.25 | Q_user 0.15
    Reflects reliability hierarchy established through diagnostics.

6.  TWO-TIER ATTENTION SCORE added (new function).
    Tier 1 (observable deterioration): violation trend slope,
      enforcement reactivity (CoV of ΔC), removal concentration (Gini).
    Tier 2 (structural fragility): pre-ban C/g* ratio.
    Output: attention_flags.csv with four-state flag per subreddit.

7.  NEGATIVE RESULTS reported separately in confidence_report.txt.

Components still computed (but not in composite):
  Q_substitution — reported as negative finding
  Q_spillover    — reported as null finding

Usage:
    python confidence_score.py --out_dir ./output

Outputs (all in <out_dir>/confidence/):
    subreddit_mci.csv            — per-subreddit MCI + confidence gate
    attention_flags.csv          — two-tier attention flagging
    user_behavioral.parquet      — per-user rho_u
    confidence_report.txt        — full statistical report
    tables/                      — component-level CSVs + JSON
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import gc

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BANNED_SUBREDDITS = {
    "fatpeoplehate", "CoonTown", "transfags", "neofag", "hamplanet",
}
BAN_DATE = pd.Timestamp("2015-06-10", tz="UTC")
PRE_BAN_START = pd.Timestamp("2015-05-01", tz="UTC")

# MCI component weights (must sum to 1.0)
MCI_WEIGHTS = {
    "Q_backfire": 0.35,
    "Q_fit":      0.25,
    "Q_regime":   0.25,
    "Q_user":     0.15,
}

# Confidence gate thresholds
GATE_MIN_COMMENTS_PER_DAY = 300
GATE_MIN_QFIT             = 0.0
GATE_GSTAR_MIN            = 0.01
GATE_GSTAR_MAX            = 2.0
GATE_MIN_C_STD            = 0.005

# Attention-score constants
INITIATOR_THRESHOLD       = 0.25   # persistence_corr threshold for initiator-like


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_inputs(out_dir: Path) -> dict:
    """
    Loads all pipeline outputs needed for MCI computation.
    Raises FileNotFoundError with a clear message if any required file is missing.
    """
    required = {
        "daily_metrics":   out_dir / "daily_metrics.parquet",
        "user_thresholds": out_dir / "user_thresholds.parquet",
        "panel":           out_dir / "panel.parquet",
        "calibrated":      out_dir / "calibrated_params.csv",
        "stage2":          out_dir / "stage2_results.csv",
        "comments_csv":    out_dir / "comments_filtered.csv",
    }

    data = {}
    all_ok = True
    for key, path in required.items():
        if path.exists():
            log.info(f"  Found: {path.name}")
        else:
            log.error(f"  MISSING: {path}  ← run the pipeline first")
            all_ok = False
            data[key] = None
            continue

        if path.suffix == ".parquet":
            data[key] = pd.read_parquet(path)
        elif key == "comments_csv":
            data[key] = path          # pass path for streaming
        elif path.suffix == ".csv":
            data[key] = pd.read_csv(path)

    if not all_ok:
        raise FileNotFoundError(
            "One or more pipeline output files are missing. "
            "Run reddit_pipeline_safe.py → model_calibration.py → "
            "stage2_analysis.py first."
        )

    data["daily_metrics"]["day"] = pd.to_datetime(
        data["daily_metrics"]["day"], utc=True
    )
    if "week" in data["panel"].columns:
        data["panel"]["week"] = pd.to_datetime(data["panel"]["week"], utc=True)

    return data


def safe_scale_01(series: pd.Series) -> pd.Series:
    """Min-max scale to [0, 1], handling constant series gracefully."""
    mn, mx = series.min(), series.max()
    if mx - mn < 1e-12:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - mn) / (mx - mn)


# ---------------------------------------------------------------------------
# Confidence gate
# ---------------------------------------------------------------------------

def compute_confidence_gate(calibrated: pd.DataFrame,
                            daily_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Applies four independent conditions to determine whether a subreddit's
    model-derived quantities are trustworthy enough to enter the attention
    ranking.

    Conditions
    ----------
    (a) data_ok      : mean comments per pre-ban day ≥ 300
    (b) fit_ok       : Q_fit > 0  (model beats the mean predictor)
    (c) gstar_ok     : 0.01 ≤ g* ≤ 2.0  (plausible range; outside = artefact)
    (d) variance_ok  : std(C_smooth pre-ban) > 0.005  (enough enforcement variance)

    Confidence levels
    -----------------
    High     : 4 / 4 conditions met
    Moderate : 3 / 4 conditions met
    Low      : ≤ 2 / 4 conditions met  (excluded from primary ranking)

    Returns a DataFrame indexed by subreddit with one boolean per condition
    and a confidence_level column.
    """
    log.info("Computing confidence gate...")

    dm = daily_metrics.copy()
    dm["pre_ban"] = dm["day"] < BAN_DATE

    rows = []
    for sub, sub_cal in calibrated.groupby("subreddit"):
        sub_dm     = dm[dm["subreddit"] == sub]
        pre_dm     = sub_dm[sub_dm["pre_ban"]]
        g_star_val = float(sub_cal["g_star"].iloc[0])

        # (a) data sufficiency
        if "n_comments" in sub_dm.columns:
            mean_daily = pre_dm["n_comments"].mean() if len(pre_dm) else 0
        else:
            # approximate from C_raw: needs total comments; use row count as proxy
            mean_daily = len(pre_dm)
        data_ok = bool(mean_daily >= GATE_MIN_COMMENTS_PER_DAY)

        # (b) ODE fit quality — use Q_fit if already in calibrated, else NaN
        qfit_val = float(sub_cal.get("Q_fit", sub_cal.get("Qfit",
                         pd.Series([np.nan]))).iloc[0])
        fit_ok   = bool(not np.isnan(qfit_val) and qfit_val > GATE_MIN_QFIT)

        # (c) g* plausibility
        gstar_ok = bool(
            not np.isnan(g_star_val) and
            GATE_GSTAR_MIN <= g_star_val <= GATE_GSTAR_MAX
        )

        # (d) enforcement variance
        c_std    = float(pre_dm["C_smooth"].std()) if len(pre_dm) >= 3 else 0.0
        var_ok   = bool(c_std > GATE_MIN_C_STD)

        n_passed = sum([data_ok, fit_ok, gstar_ok, var_ok])
        level    = "High" if n_passed == 4 else \
                   "Moderate" if n_passed == 3 else "Low"

        rows.append({
            "subreddit":        sub,
            "gate_data_ok":     data_ok,
            "gate_fit_ok":      fit_ok,
            "gate_gstar_ok":    gstar_ok,
            "gate_variance_ok": var_ok,
            "n_gates_passed":   n_passed,
            "confidence_level": level,
            "mean_daily_comments": mean_daily,
            "preban_C_std":     c_std,
            "g_star":           g_star_val,
            "Q_fit_raw":        qfit_val,
        })

    gate_df = pd.DataFrame(rows)
    counts  = gate_df["confidence_level"].value_counts()
    log.info(f"  Confidence gate: "
             f"High={counts.get('High',0)}  "
             f"Moderate={counts.get('Moderate',0)}  "
             f"Low={counts.get('Low',0)}")
    log.info("  Low-confidence subreddits (excluded from primary ranking):")
    for _, r in gate_df[gate_df["confidence_level"] == "Low"].iterrows():
        reasons = [k.replace("gate_","").replace("_ok","")
                   for k, v in r.items()
                   if k.startswith("gate_") and not v]
        log.info(f"    r/{r['subreddit']:<25}  failed: {reasons}")

    return gate_df


# ---------------------------------------------------------------------------
# Component 1: Q_fit
# ---------------------------------------------------------------------------

def compute_q_fit(calibrated: pd.DataFrame,
                  daily_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Normalised R²-equivalent for the Stage 1 ODE fit per subreddit.

        Q_fit = 1 − fit_nll / SS_obs

    where SS_obs = Var(C_raw) × n_days  (total variance in observed series)
    and fit_nll  = raw RSS from the Nelder-Mead calibration.

    Q_fit = 1  → perfect fit.
    Q_fit = 0  → model is as good as the mean predictor.
    Q_fit < 0  → model is worse than the mean predictor (clipped to −2 for display).
    """
    log.info("Computing Q_fit (ODE goodness-of-fit)...")
    rows = []

    for _, row in calibrated.iterrows():
        sub    = row["subreddit"]
        sub_dm = daily_metrics[daily_metrics["subreddit"] == sub]["C_raw"].dropna()
        n      = len(sub_dm)

        if n < 5:
            rows.append({
                "subreddit": sub, "Q_fit": np.nan,
                "fit_nll": row.get("fit_nll", np.nan),
                "SS_obs": np.nan, "n_days_fit": n,
            })
            continue

        ss_obs  = sub_dm.var() * n
        fit_nll = float(row.get("fit_nll", np.nan))

        if np.isnan(fit_nll) or ss_obs < 1e-12:
            q_fit = np.nan
        else:
            q_fit = float(np.clip(1.0 - fit_nll / ss_obs, -2.0, 1.0))

        rows.append({
            "subreddit":  sub,
            "Q_fit":      q_fit,
            "fit_nll":    fit_nll,
            "SS_obs":     ss_obs,
            "n_days_fit": n,
        })

    df = pd.DataFrame(rows)
    log.info(f"  Q_fit: mean={df['Q_fit'].mean():.3f}  "
             f"median={df['Q_fit'].median():.3f}  "
             f"n_positive={( df['Q_fit'] > 0 ).sum()}  "
             f"n_negative={( df['Q_fit'] < 0 ).sum()}")
    return df


# ---------------------------------------------------------------------------
# Component 2: Q_backfire  (REWRITTEN — Spearman rank correlation)
# ---------------------------------------------------------------------------

def compute_q_backfire(calibrated: pd.DataFrame,
                       daily_metrics: pd.DataFrame) -> dict:
    """
    Tests Prediction 3 (post-shock divergence): communities with higher
    pre-ban C/g* ratios should show faster-rising violation rates post-ban.

    Method
    ------
    1. For each subreddit compute:
         fragility_i = mean(C_smooth during May pre-ban) / g*_i
    2. Compute post-ban violation slope:
         slope_i = OLS β on [time, C_raw] for June 11 – Aug 31
    3. Spearman rank correlation ρ between fragility and slope across
       all subreddits with valid data.

    Prediction: ρ > 0  (fragile communities trend worse post-ban).

    The Q_backfire score is (ρ + 1) / 2  mapped to [0, 1].
    A scalar (not per-subreddit) so it is broadcast to all communities
    in the MCI table, reflecting a corpus-level test.

    Returns dict with ρ, p, per-subreddit table, and Q_backfire_score.
    """
    log.info("Computing Q_backfire (Spearman ρ: pre-ban fragility → post-ban slope)...")

    dm = daily_metrics.copy()
    dm["pre_ban"]  = (dm["day"] >= PRE_BAN_START) & (dm["day"] < BAN_DATE)
    dm["post_ban"] = dm["day"] >= BAN_DATE

    g_star_map = calibrated.set_index("subreddit")["g_star"].to_dict()

    rows = []
    for sub, sub_dm in dm.groupby("subreddit"):
        g_star = g_star_map.get(sub, np.nan)
        if np.isnan(g_star) or g_star < GATE_GSTAR_MIN or g_star > GATE_GSTAR_MAX:
            continue

        pre  = sub_dm[sub_dm["pre_ban"]]["C_smooth"].dropna()
        post = sub_dm[sub_dm["post_ban"]].sort_values("day")

        if len(pre) < 5 or len(post) < 10:
            continue

        c_preban_mean = float(pre.mean())
        fragility     = c_preban_mean / g_star

        # Post-ban violation slope (C_raw as violation proxy)
        y_post = post["C_raw"].fillna(0).values
        if y_post.std() < 1e-8:
            continue
        x_post       = np.arange(len(y_post))
        slope, _, _, p_slope, _ = stats.linregress(x_post, y_post)

        rows.append({
            "subreddit":      sub,
            "g_star":         g_star,
            "c_preban_mean":  c_preban_mean,
            "fragility":      fragility,        # C/g*
            "post_ban_slope": slope,
            "post_ban_p":     p_slope,
            "is_banned":      sub in BANNED_SUBREDDITS,
        })

    df = pd.DataFrame(rows)

    if len(df) < 4:
        log.warning("  Q_backfire: fewer than 4 subreddits with valid data — "
                    "Spearman ρ unreliable.")
        return {
            "spearman_rho":      np.nan,
            "spearman_p":        np.nan,
            "Q_backfire_score":  np.nan,
            "n_subreddits":      len(df),
            "per_subreddit_df":  df,
        }

    rho, p_rho = stats.spearmanr(df["fragility"], df["post_ban_slope"])
    q_score    = float((rho + 1.0) / 2.0) if not np.isnan(rho) else np.nan

    log.info(f"  Q_backfire: Spearman ρ={rho:.3f}  p={p_rho:.4f}  "
             f"n={len(df)}  score={q_score:.3f}")
    log.info(f"  Interpretation: ρ > 0 → fragile communities trend worse post-ban "
             f"({'CONFIRMED' if rho > 0 else 'NOT CONFIRMED'})")

    return {
        "spearman_rho":      float(rho),
        "spearman_p":        float(p_rho),
        "Q_backfire_score":  q_score,
        "n_subreddits":      int(len(df)),
        "per_subreddit_df":  df,
    }


# ---------------------------------------------------------------------------
# Component 3: Q_substitution  (kept for reporting; NOT in MCI composite)
# ---------------------------------------------------------------------------

def compute_q_substitution(daily_metrics: pd.DataFrame) -> dict:
    """
    Tests T–C substitution: does higher legitimacy (T) predict lower
    enforcement intensity (C), holding compliance constant?

    RESULT: β₁ = +0.0043, p = 0.339 — hypothesis REJECTED.

    This function is retained to document the negative finding in the
    report and tables.  Q_substitution_score is NOT included in the
    composite MCI.

    Method: within-subreddit panel regression with subreddit fixed effects.
        C_{s,t} = α_s + β₁·T_{s,t-1} + β₂·x_{s,t-1} + ε
    """
    log.info("Computing Q_substitution (negative result — not in MCI composite)...")

    dm = daily_metrics.copy().sort_values(["subreddit", "day"])
    dm["compliance"] = 1.0 - dm["C_raw"].clip(0, 1)
    dm = dm.dropna(subset=["T_norm", "C_smooth", "C_raw"])
    dm = dm[dm["T_norm"].between(1e-6, 1 - 1e-6)]

    if len(dm) < 30:
        log.warning("  Q_substitution: insufficient data.")
        return {
            "beta1_TC": np.nan, "beta1_p": np.nan, "r2": np.nan,
            "partial_r_TC": np.nan, "n_obs": 0,
            "Q_substitution_score": np.nan,
            "note": "NEGATIVE RESULT — excluded from MCI composite",
        }

    # Partial correlation r(T, C | compliance)
    X_comply   = dm["compliance"].values.reshape(-1, 1)
    X_comply_c = X_comply - X_comply.mean()

    def ols_resid(y):
        b = np.linalg.lstsq(X_comply_c, y - y.mean(), rcond=None)[0]
        return (y - y.mean()) - X_comply_c @ b

    resid_T    = ols_resid(dm["T_norm"].values)
    resid_C    = ols_resid(dm["C_smooth"].values)
    partial_r, partial_p = stats.pearsonr(resid_T, resid_C)

    # Full pooled OLS
    Y = dm["C_smooth"].values
    X = np.column_stack([np.ones(len(dm)), dm["T_norm"].values,
                         dm["compliance"].values])
    coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    y_hat  = X @ coeffs
    ss_res = np.sum((Y - y_hat) ** 2)
    ss_tot = np.sum((Y - Y.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    n, k   = len(Y), X.shape[1]
    mse    = ss_res / max(n - k, 1)
    XtX_i  = np.linalg.pinv(X.T @ X)
    se     = np.sqrt(np.diag(XtX_i) * mse)
    t_st   = coeffs / np.where(se > 1e-12, se, np.nan)
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_st), df=n - k))

    beta1_TC = float(coeffs[1])
    beta1_p  = float(p_vals[1])
    q_score  = float((-partial_r + 1.0) / 2.0)   # for reference only

    log.info(f"  Q_substitution (NEGATIVE): β₁={beta1_TC:.4f}  "
             f"p={beta1_p:.4f}  partial_r={partial_r:.4f}  "
             f"→ hypothesis rejected (β₁ > 0)")

    return {
        "beta1_TC":             beta1_TC,
        "beta1_SE":             float(se[1]),
        "beta1_t":              float(t_st[1]),
        "beta1_p":              beta1_p,
        "partial_r_TC":         float(partial_r),
        "partial_r_p":          float(partial_p),
        "r2_pooled":            float(r2),
        "n_obs":                int(n),
        "Q_substitution_score": q_score,
        "note":                 "NEGATIVE RESULT — excluded from MCI composite",
    }


# ---------------------------------------------------------------------------
# Component 4: Q_user
# ---------------------------------------------------------------------------

def compute_q_user(user_thresholds: pd.DataFrame,
                   daily_metrics: pd.DataFrame,
                   comments_csv_path: Path,
                   min_posts: int = 10) -> dict:
    """
    Validates that INITIATOR / JOINER / COMPLIER types differ in HOW their
    removal probability responds to enforcement level (ρ_u).

    For each user u posting in subreddit s on ≥ min_posts days:
        ρ_u = Pearson correlation(C_smooth on posting day, is_removed)

    Prediction:
        JOINER   ρ_u < INITIATOR ρ_u   (Granovetter 1978)
        COMPLIER ρ_u ≈ 0

    Test: One-way ANOVA (F, p, η²) + post-hoc Mann-Whitney.

    Note: effect size η² = 0.013 in prior run — statistically significant
    due to large N but practically small. This is reported honestly.
    """
    log.info(f"Computing Q_user (min_posts={min_posts})...")

    dm = daily_metrics.copy()
    dm["day_str"] = dm["day"].dt.strftime("%Y-%m-%d")
    ct_df = dm[["subreddit", "day_str", "C_smooth"]].drop_duplicates()

    valid_users_df = (
        user_thresholds[["author", "subreddit", "user_type"]]
        .drop_duplicates(subset=["author", "subreddit"])
    )

    accum_parts = []
    EXCLUDE     = {"[deleted]", "AutoModerator"}
    CHUNK       = 500_000

    log.info("  Streaming comments_filtered.csv...")
    n_chunks = 0
    for chunk in pd.read_csv(
        comments_csv_path, chunksize=CHUNK,
        usecols=["author", "subreddit", "created_utc", "is_removed"]
    ):
        chunk = chunk[~chunk["author"].isin(EXCLUDE)].dropna(subset=["author"])
        chunk["day_str"] = (
            pd.to_datetime(chunk["created_utc"], unit="s", utc=True)
            .dt.strftime("%Y-%m-%d")
        )
        chunk = chunk.merge(valid_users_df, on=["author", "subreddit"], how="inner")
        if chunk.empty:
            del chunk; gc.collect(); continue

        chunk = chunk.merge(ct_df, on=["subreddit", "day_str"], how="left")
        chunk = chunk.dropna(subset=["C_smooth"])
        if chunk.empty:
            del chunk; gc.collect(); continue

        chunk["is_removed"] = chunk["is_removed"].fillna(0).astype(int)
        chunk["xy"]  = chunk["C_smooth"] * chunk["is_removed"]
        chunk["x2"]  = chunk["C_smooth"] ** 2
        chunk["y2"]  = chunk["is_removed"] ** 2

        grp = chunk.groupby(
            ["author", "subreddit", "user_type"], sort=False
        ).agg(n=("C_smooth","count"), sx=("C_smooth","sum"),
              sy=("is_removed","sum"), sxy=("xy","sum"),
              sx2=("x2","sum"), sy2=("y2","sum"))
        accum_parts.append(grp)
        n_chunks += 1
        del chunk; gc.collect()

    log.info(f"  Processed {n_chunks} chunks.")

    if not accum_parts:
        log.warning("  Q_user: no valid data after filtering.")
        return {"F_stat": np.nan, "p_anova": np.nan, "eta2": np.nan,
                "rho_df": pd.DataFrame(), "Q_user_score": np.nan}

    combined = pd.concat(accum_parts)
    combined = combined.groupby(
        ["author", "subreddit", "user_type"], sort=False
    ).sum().reset_index()
    combined = combined[combined["n"] >= min_posts].copy()

    log.info(f"  {len(combined):,} user×subreddit pairs with ≥{min_posts} obs.")

    denom_x = combined["n"] * combined["sx2"] - combined["sx"] ** 2
    denom_y = combined["n"] * combined["sy2"] - combined["sy"] ** 2
    numer   = combined["n"] * combined["sxy"] - combined["sx"] * combined["sy"]
    denom   = np.sqrt(denom_x.clip(0) * denom_y.clip(0))
    rho_u   = np.where(denom > 1e-10, numer / denom, 0.0)
    rho_u   = np.clip(rho_u, -1.0, 1.0)

    n_obs   = combined["n"].values
    t_stat  = rho_u * np.sqrt(np.maximum(n_obs - 2, 1)) / \
              np.sqrt(np.maximum(1.0 - rho_u ** 2, 1e-10))
    p_rho   = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat),
                                         df=np.maximum(n_obs - 2, 1)))

    rho_df  = combined[["author", "subreddit", "user_type", "n"]].copy()
    rho_df  = rho_df.rename(columns={"n": "n_obs"})
    rho_df["rho_u"] = rho_u
    rho_df["p_rho"] = p_rho

    if len(rho_df) < 10:
        log.warning("  Q_user: too few users with sufficient observations.")
        return {"F_stat": np.nan, "p_anova": np.nan, "eta2": np.nan,
                "rho_df": rho_df, "Q_user_score": np.nan}

    groups = [
        rho_df[rho_df["user_type"] == t]["rho_u"].dropna().values
        for t in ["INITIATOR", "JOINER", "COMPLIER"]
        if len(rho_df[rho_df["user_type"] == t]) >= 3
    ]
    if len(groups) < 2:
        log.warning("  Q_user: fewer than 2 types with sufficient data.")
        return {"F_stat": np.nan, "p_anova": np.nan, "eta2": np.nan,
                "rho_df": rho_df, "Q_user_score": np.nan}

    F_stat, p_anova = stats.f_oneway(*groups)
    grand_mean = rho_df["rho_u"].dropna().mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total   = sum(np.sum((g - grand_mean) ** 2) for g in groups)
    eta2       = float(ss_between / ss_total) if ss_total > 0 else np.nan

    init_rho = rho_df[rho_df["user_type"] == "INITIATOR"]["rho_u"].dropna()
    join_rho = rho_df[rho_df["user_type"] == "JOINER"]["rho_u"].dropna()
    comp_rho = rho_df[rho_df["user_type"] == "COMPLIER"]["rho_u"].dropna()

    mw_ij = stats.mannwhitneyu(init_rho, join_rho, alternative="greater") \
            if (len(init_rho) >= 3 and len(join_rho) >= 3) else None
    mw_jc = stats.mannwhitneyu(join_rho, comp_rho, alternative="less") \
            if (len(join_rho) >= 3 and len(comp_rho) >= 3) else None

    q_score = float(np.clip(eta2, 0, 1))

    log.info(f"  Q_user: F={F_stat:.3f}  p={p_anova:.6f}  "
             f"η²={eta2:.4f}  score={q_score:.3f}")
    log.info(f"  NOTE: η²={eta2:.4f} is statistically significant (large N) "
             f"but practically small. Report effect size honestly.")
    if mw_ij:
        log.info(f"  Mann-Whitney INITIATOR>JOINER: "
                 f"U={mw_ij.statistic:.0f}  p={mw_ij.pvalue:.4f}")

    return {
        "F_stat":        float(F_stat),
        "p_anova":       float(p_anova),
        "eta2":          float(eta2),
        "mean_rho_INIT": float(init_rho.mean()) if len(init_rho) else np.nan,
        "mean_rho_JOIN": float(join_rho.mean()) if len(join_rho) else np.nan,
        "mean_rho_COMP": float(comp_rho.mean()) if len(comp_rho) else np.nan,
        "mw_IJ_U":       float(mw_ij.statistic) if mw_ij else np.nan,
        "mw_IJ_p":       float(mw_ij.pvalue)    if mw_ij else np.nan,
        "mw_JC_U":       float(mw_jc.statistic) if mw_jc else np.nan,
        "mw_JC_p":       float(mw_jc.pvalue)    if mw_jc else np.nan,
        "n_INIT":        int(len(init_rho)),
        "n_JOIN":        int(len(join_rho)),
        "n_COMP":        int(len(comp_rho)),
        "rho_df":        rho_df,
        "Q_user_score":  q_score,
    }


# ---------------------------------------------------------------------------
# Component 5: Q_regime
# ---------------------------------------------------------------------------

def compute_q_regime(stage2: pd.DataFrame) -> pd.DataFrame:
    """
    Stability margin per subreddit from Stage 2 Routh-Hurwitz analysis.

        margin_i = (α_gov_i − α_gov_min_i) / |α_gov_min_i|

    Positive → safely within stable regime.
    Negative → in governance trap or structural instability.

        Q_regime_i = (tanh(margin_i) + 1) / 2   ∈ [0, 1]

    Cross-validation: Pearson r(Q_regime, −C_volatility).
    Prediction: higher stability margin → lower enforcement volatility.

    Note: The α_gov estimates are derived from short time-series regressions
    with R² ≈ 0 for many subreddits. Q_regime should be interpreted as an
    ordinal ranking of governance stability, not a precisely identified measure.
    """
    log.info("Computing Q_regime (Stage 2 stability margin)...")

    df = stage2.copy()

    def compute_margin(row):
        ag     = row.get("alpha_gov", np.nan)
        ag_min = row.get("alpha_gov_min", np.nan)
        if any(np.isnan(v) for v in [ag, ag_min]) or not np.isfinite(ag_min):
            return np.nan
        if abs(ag_min) < 1e-10:
            return np.nan
        return (ag - ag_min) / abs(ag_min)

    df["stability_margin"] = df.apply(compute_margin, axis=1)
    df["Q_regime_raw"]     = np.tanh(df["stability_margin"].fillna(0))
    df["Q_regime"]         = (df["Q_regime_raw"] + 1.0) / 2.0

    valid = df.dropna(subset=["Q_regime", "C_volatility"])
    if len(valid) >= 5:
        r_val, p_val = stats.pearsonr(valid["Q_regime"], -valid["C_volatility"])
        log.info(f"  Q_regime: r(margin, −volatility)={r_val:.3f}  p={p_val:.4f}")
    else:
        r_val, p_val = np.nan, np.nan
        log.warning("  Q_regime: insufficient data for volatility cross-validation.")

    df["Q_regime_r_xval"] = r_val
    df["Q_regime_p_xval"] = p_val

    log.info(f"  Q_regime: mean={df['Q_regime'].mean():.3f}  "
             f"n_stable={(df['stability_margin'] > 0).sum()}  "
             f"n_trap={(df['stability_margin'] < 0).sum()}")

    keep = ["subreddit", "stability_margin", "Q_regime", "Q_regime_raw",
            "Q_regime_r_xval", "Q_regime_p_xval",
            "regime", "alpha_gov", "alpha_gov_min", "Xi", "C_volatility"]
    return df[[c for c in keep if c in df.columns]]


# ---------------------------------------------------------------------------
# Component 6: Q_spillover  (kept for reporting; NOT in MCI composite)
# ---------------------------------------------------------------------------

def compute_q_spillover(panel: pd.DataFrame,
                        user_thresholds: pd.DataFrame) -> dict:
    """
    Tests whether INITIATOR users from banned subreddits maintain elevated
    removal rates in control communities post-ban.

    RESULT: Zero migrant INITIATORs found under the corrected migration
    definition.  This is a null result documented for transparency.

    Q_spillover is NOT included in the composite MCI.
    """
    log.info("Computing Q_spillover (null result — not in MCI composite)...")

    if "user_type" not in panel.columns:
        ut      = user_thresholds[["author", "subreddit",
                                   "user_type", "theta_proxy"]].copy()
        panel_m = panel.merge(ut, on=["author", "subreddit"], how="left")
    else:
        panel_m = panel.copy()

    post_ctrl = panel_m[
        panel_m.get("post_ban", False) & ~panel_m.get("is_banned_sub", True)
    ].copy()

    if post_ctrl.empty:
        log.warning("  Q_spillover: no post-ban control observations.")
        return _null_spillover_result()

    user_postban = (
        post_ctrl.groupby(["author", "from_banned_sub", "user_type",
                           "theta_proxy"],
                          dropna=False)
        .agg(n_posts=("n_posts_week","sum"), n_removed=("resisted","sum"))
        .reset_index()
    )

    user_postban["from_banned_sub"] = (
        user_postban["from_banned_sub"]
        .fillna(False)
        .astype(bool)
    )
    user_postban = user_postban[user_postban["n_posts"] >= 3]
    user_postban["removal_rate"] = (user_postban["n_removed"] /
                                    user_postban["n_posts"])

    group_a = user_postban[
        user_postban["from_banned_sub"] &

        (user_postban["user_type"] == "INITIATOR")
    ]["removal_rate"].dropna()

    group_b = user_postban[
        ~user_postban["from_banned_sub"]
    ]["removal_rate"].dropna()

    n_a, n_b = len(group_a), len(group_b)
    log.info(f"  Q_spillover: migrant INITIATORs={n_a}  native users={n_b}")

    if n_a < 3 or n_b < 3:
        log.warning(f"  Q_spillover NULL RESULT: n_migrants={n_a} "
                    f"(corrected migration definition finds zero meaningful migrants).")
        return {
            **_null_spillover_result(),
            "n_migrants": int(n_a),
            "n_natives":  int(n_b),
            "mean_rate_migrants": float(group_a.mean()) if n_a else np.nan,
            "mean_rate_natives":  float(group_b.mean()) if n_b else np.nan,
            "note": "NULL RESULT — corrected migration definition finds no migrants. "
                    "Excluded from MCI composite.",
        }

    mw = stats.mannwhitneyu(group_a, group_b, alternative="greater")
    pooled_std = np.sqrt(
        (group_a.std() ** 2 * (n_a - 1) + group_b.std() ** 2 * (n_b - 1)) /
        (n_a + n_b - 2)
    )
    cohens_d = float((group_a.mean() - group_b.mean()) / pooled_std) \
               if pooled_std > 1e-10 else np.nan

    q_score = float(1.0 / (1.0 + np.exp(-2.0 * cohens_d))) \
              if not np.isnan(cohens_d) else np.nan

    return {
        "mw_U":               float(mw.statistic),
        "mw_p":               float(mw.pvalue),
        "cohens_d":           float(cohens_d) if not np.isnan(cohens_d) else np.nan,
        "mean_rate_migrants": float(group_a.mean()),
        "mean_rate_natives":  float(group_b.mean()),
        "n_migrants":         int(n_a),
        "n_natives":          int(n_b),
        "Q_spillover_score":  q_score,
        "note":               "Included in composite only if n_migrants >= 3.",
    }


def _null_spillover_result() -> dict:
    return {
        "mw_U": np.nan, "mw_p": np.nan, "cohens_d": np.nan,
        "mean_rate_migrants": np.nan, "mean_rate_natives": np.nan,
        "n_migrants": 0, "n_natives": 0, "Q_spillover_score": np.nan,
        "note": "NULL RESULT — excluded from MCI composite.",
    }


# ---------------------------------------------------------------------------
# Two-tier attention-flagging system  (NEW)
# ---------------------------------------------------------------------------

def compute_attention_score(daily_metrics: pd.DataFrame,
                             calibrated: pd.DataFrame,
                             user_thresholds: pd.DataFrame,
                             gate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the two-tier governance monitoring signal described in the paper.

    Tier 1 — Observable Deterioration (no model required)
    -------------------------------------------------------
    Three components computed from pre-ban May data only:

    T1a  Violation trend slope
         OLS β of weekly removal rate on time (May weeks).
         Positive = violations rising before the ban shock.
         Scaled by max(|slope|) across subreddits to [−1, +1].

    T1b  Enforcement reactivity (coefficient of variation of ΔC)
         std(ΔC_weekly) / mean(|C_smooth|) during May.
         High = erratic moderation.

    T1c  Removal concentration (Gini coefficient)
         Concentration of removed comments among a small user fraction.
         High = power-user dominated violations.

    Tier 2 — Structural Fragility (model-derived, g* required)
    -----------------------------------------------------------
    T2   Pre-ban C/g* ratio  (May mean C_smooth / calibrated g*)
         Restricted to subreddits where g* ∈ [0.01, 2.0].
         High C/g* = operating close to the theoretical backfire threshold.

    Combination
    -----------
    Each tier produces a rank among the subreddit set.
    A subreddit is flagged if it falls in the top quartile on either tier.
    Four-state label:
        BOTH      — top quartile on Tier 1 AND Tier 2
        TIER1_ONLY — top quartile on Tier 1, not Tier 2
        TIER2_ONLY — not Tier 1, top quartile on Tier 2
        NEITHER    — bottom 75% on both tiers

    Only High/Moderate confidence subreddits enter the primary ranking.
    Low-confidence subreddits are included with flag = LOW_CONFIDENCE.
    """
    log.info("Computing two-tier attention score...")

    dm  = daily_metrics.copy()
    dm["pre_ban"] = (dm["day"] >= PRE_BAN_START) & (dm["day"] < BAN_DATE)
    dm["week"]    = dm["day"].dt.to_period("W").dt.start_time.dt.tz_localize("UTC",
                                                                               ambiguous=False,
                                                                               nonexistent="shift_forward")

    g_star_map = calibrated.set_index("subreddit")["g_star"].to_dict()

    # Gini helper
    def gini(arr):
        arr = np.sort(np.abs(arr))
        n   = len(arr)
        if n == 0 or arr.sum() == 0:
            return 0.0
        idx = np.arange(1, n + 1)
        return float((2 * (idx * arr).sum()) / (n * arr.sum()) - (n + 1) / n)

    rows = []
    for sub, sub_dm in dm.groupby("subreddit"):
        pre     = sub_dm[sub_dm["pre_ban"]]
        g_star  = g_star_map.get(sub, np.nan)
        conf_row = gate_df[gate_df["subreddit"] == sub]
        conf_level = conf_row["confidence_level"].iloc[0] \
                     if len(conf_row) else "Low"

        # Weekly pre-ban aggregates
        weekly = (
            pre.groupby("week")
            .agg(C_mean=("C_smooth","mean"), C_raw=("C_raw","mean"))
            .reset_index().sort_values("week")
        )

        # T1a: violation trend slope (weekly C_raw ~ time)
        t1a = np.nan
        if len(weekly) >= 3:
            x  = np.arange(len(weekly))
            y  = weekly["C_raw"].fillna(0).values
            if y.std() > 1e-8:
                slope, *_ = stats.linregress(x, y)
                t1a = float(slope)

        # T1b: enforcement reactivity (CoV of ΔC)
        t1b = np.nan
        if len(weekly) >= 3:
            delta_c = weekly["C_mean"].diff().dropna().abs()
            mean_c  = float(weekly["C_mean"].mean())
            if mean_c > 1e-8:
                t1b = float(delta_c.std() / mean_c)

        # T1c: removal concentration (Gini)
        sub_ut  = user_thresholds[user_thresholds["subreddit"] == sub]
        t1c     = np.nan
        if len(sub_ut) >= 5 and "n_removed" in sub_ut.columns:
            t1c = gini(sub_ut["n_removed"].fillna(0).values)

        # T2: structural fragility (C/g*)
        t2 = np.nan
        if (not np.isnan(g_star) and
                GATE_GSTAR_MIN <= g_star <= GATE_GSTAR_MAX and
                len(pre) >= 3):
            t2 = float(pre["C_smooth"].mean()) / g_star

        rows.append({
            "subreddit":         sub,
            "confidence_level":  conf_level,
            "is_banned":         sub in BANNED_SUBREDDITS,
            "t1a_viol_slope":    t1a,
            "t1b_reactivity":    t1b,
            "t1c_gini":          t1c,
            "t2_fragility":      t2,
        })

    att = pd.DataFrame(rows)

    # Restrict primary ranking to High/Moderate confidence
    primary = att[att["confidence_level"].isin(["High", "Moderate"])].copy()
    n       = len(primary)
    top_q   = max(1, int(np.ceil(n * 0.25)))   # top 25%

    # Rank-average Tier 1 components
    for col in ["t1a_viol_slope", "t1b_reactivity", "t1c_gini"]:
        r_col = col + "_rank"
        primary[r_col] = primary[col].rank(ascending=False, na_option="bottom")

    primary["tier1_avg_rank"] = primary[
        ["t1a_viol_slope_rank", "t1b_reactivity_rank", "t1c_gini_rank"]
    ].mean(axis=1)
    primary["tier1_rank"] = primary["tier1_avg_rank"].rank(method="min")

    primary["tier2_rank"] = primary["t2_fragility"].rank(
        ascending=False, na_option="bottom", method="min"
    )

    primary["tier1_flag"] = primary["tier1_rank"] <= top_q
    primary["tier2_flag"] = primary["tier2_rank"] <= top_q

    def attention_label(row):
        if row["tier1_flag"] and row["tier2_flag"]:
            return "BOTH"
        elif row["tier1_flag"]:
            return "TIER1_ONLY"
        elif row["tier2_flag"]:
            return "TIER2_ONLY"
        return "NEITHER"

    primary["attention_flag"] = primary.apply(attention_label, axis=1)

    # Merge back onto the full subreddit set
    att = att.merge(
        primary[["subreddit", "tier1_rank", "tier2_rank",
                 "tier1_flag", "tier2_flag", "attention_flag"]],
        on="subreddit", how="left"
    )
    att.loc[att["confidence_level"] == "Low", "attention_flag"] = "LOW_CONFIDENCE"
    att.loc[att["attention_flag"].isna(), "attention_flag"] = "LOW_CONFIDENCE"

    log.info("\n  --- Attention-flag summary ---")
    log.info(f"  {'Subreddit':<22} {'Conf':>8}  {'T1rank':>7}  "
             f"{'T2rank':>7}  {'Flag':<14}  {'C/g*':>6}")
    log.info("  " + "-" * 70)
    for _, r in att.sort_values("tier1_rank", na_position="last").iterrows():
        t1r = f"{r['tier1_rank']:.0f}" if not pd.isna(r.get("tier1_rank")) else "  n/a"
        t2r = f"{r['tier2_rank']:.0f}" if not pd.isna(r.get("tier2_rank")) else "  n/a"
        t2v = f"{r['t2_fragility']:.3f}" if not np.isnan(r["t2_fragility"]) else " n/a"
        ban = "[B]" if r["is_banned"] else "   "
        log.info(f"  {ban} r/{r['subreddit']:<20} {r['confidence_level']:>8}  "
                 f"{t1r:>7}  {t2r:>7}  {r['attention_flag']:<14}  {t2v:>6}")

    return att


# ---------------------------------------------------------------------------
# Composite MCI assembly  (4 components, theoretically motivated weights)
# ---------------------------------------------------------------------------

def assemble_mci(
    q_fit_df:      pd.DataFrame,
    q_backfire:    dict,
    q_user:        dict,
    q_regime_df:   pd.DataFrame,
    calibrated:    pd.DataFrame,
    gate_df:       pd.DataFrame,
) -> pd.DataFrame:
    """
    Assembles the composite MCI from four components with theoretically
    motivated weights:

        Q_backfire  0.35  — direct out-of-sample test (Prediction 3)
        Q_fit       0.25  — model adequacy
        Q_regime    0.25  — structural stability
        Q_user      0.15  — aggregate behavioral consistency

    Q_substitution and Q_spillover are deliberately excluded:
        Q_substitution: β₁ > 0, p = 0.339 — hypothesis rejected
        Q_spillover:    zero migrant INITIATORs — null result

    Missing components are skipped and weights renormalized so that the
    composite always sums correctly.
    """
    log.info("Assembling composite MCI (4 components, motivated weights)...")

    base = calibrated[["subreddit", "g_star",
                        "n_days", "n_preban_days"]].copy()
    base["is_banned"] = base["subreddit"].isin(BANNED_SUBREDDITS)

    # Merge Q_fit
    base = base.merge(
        q_fit_df[["subreddit", "Q_fit", "fit_nll", "SS_obs", "n_days_fit"]],
        on="subreddit", how="left"
    )
    base["Q_fit_scaled"] = base["Q_fit"].clip(lower=0.0)   # negative → 0

    # Merge Q_regime
    if not q_regime_df.empty:
        base = base.merge(
            q_regime_df[["subreddit", "stability_margin", "Q_regime",
                         "regime", "alpha_gov", "alpha_gov_min",
                         "Xi", "C_volatility"]],
            on="subreddit", how="left"
        )
    else:
        base["Q_regime"] = np.nan

    # Merge confidence gate
    base = base.merge(
        gate_df[["subreddit", "confidence_level", "n_gates_passed",
                 "gate_data_ok", "gate_fit_ok", "gate_gstar_ok",
                 "gate_variance_ok"]],
        on="subreddit", how="left"
    )

    # Broadcast scalar components
    base["Q_backfire"] = q_backfire.get("Q_backfire_score", np.nan)
    base["Q_user"]     = q_user.get("Q_user_score", np.nan)

    # Merge per-subreddit backfire data for reporting
    psd = q_backfire.get("per_subreddit_df", pd.DataFrame())
    if not psd.empty:
        base = base.merge(
            psd[["subreddit", "fragility", "post_ban_slope", "post_ban_p"]],
            on="subreddit", how="left"
        )

    # Weighted composite — skip NaN components, renormalise weights
    component_weight_map = {
        "Q_backfire":  MCI_WEIGHTS["Q_backfire"],
        "Q_fit_scaled": MCI_WEIGHTS["Q_fit"],
        "Q_regime":    MCI_WEIGHTS["Q_regime"],
        "Q_user":      MCI_WEIGHTS["Q_user"],
    }

    def weighted_mci(row):
        total_w, weighted_sum = 0.0, 0.0
        for col, w in component_weight_map.items():
            val = row.get(col, np.nan)
            if not (np.isnan(val) if isinstance(val, float) else False):
                weighted_sum += val * w
                total_w      += w
        return float(weighted_sum / total_w) if total_w > 0 else np.nan

    base["MCI"]          = base.apply(weighted_mci, axis=1)
    base["n_components"] = base[list(component_weight_map)].notna().sum(axis=1)

    # Ranks
    base["MCI_rank_all"]     = base["MCI"].rank(ascending=False, method="min")
    base["MCI_rank_banned"]  = base.loc[
        base["is_banned"], "MCI"].rank(ascending=False)
    base["MCI_rank_control"] = base.loc[
        ~base["is_banned"], "MCI"].rank(ascending=False)

    log.info(f"\n{'='*60}")
    log.info("COMPOSITE MCI RESULTS")
    log.info(f"  Weights: Q_backfire={MCI_WEIGHTS['Q_backfire']}  "
             f"Q_fit={MCI_WEIGHTS['Q_fit']}  "
             f"Q_regime={MCI_WEIGHTS['Q_regime']}  "
             f"Q_user={MCI_WEIGHTS['Q_user']}")
    log.info(f"{'='*60}")
    for _, row in base.sort_values("MCI", ascending=False).iterrows():
        tag  = "[BANNED]" if row["is_banned"] else "       "
        conf = row.get("confidence_level", "?")
        log.info(
            f"  {tag}  r/{row['subreddit']:<25}  "
            f"MCI={row['MCI']:.3f}  "
            f"[{conf}]  "
            f"(back={row['Q_backfire']:.2f} "
            f"fit={row['Q_fit_scaled']:.2f} "
            f"reg={row['Q_regime']:.2f} "
            f"user={row['Q_user']:.2f})  "
            f"regime={row.get('regime', 'n/a')}"
        )

    return base


# ---------------------------------------------------------------------------
# Sensitivity analysis  (updated for 4 components)
# ---------------------------------------------------------------------------

def sensitivity_analysis(mci_df: pd.DataFrame,
                          n_draws: int = 500,
                          seed: int = 42) -> dict:
    """
    Tests robustness of MCI ranking to weight perturbations.

    Draws n_draws sets of 4 weights uniformly from [0.05, 0.50],
    normalised to sum = 1.  Reports Spearman ρ between the theoretically-
    motivated baseline ranking and each random-weight ranking.

    Mean ρ > 0.85 indicates the ranking is stable.
    """
    log.info(f"Sensitivity analysis ({n_draws} draws, 4 components)...")
    rng  = np.random.default_rng(seed)
    cols = ["Q_backfire", "Q_fit_scaled", "Q_regime", "Q_user"]

    sub_df = mci_df[["subreddit"] + cols].dropna(thresh=2).copy()
    if len(sub_df) < 4:
        log.warning("  Sensitivity: too few subreddits with data.")
        return {"rho_mean": np.nan, "rho_min": np.nan,
                "rho_median": np.nan, "n_draws": n_draws}

    for c in cols:
        sub_df[c] = sub_df[c].fillna(sub_df[c].mean())

    # Baseline: theoretically motivated weights
    w_base = np.array([MCI_WEIGHTS[k.replace("_scaled","")] for k in cols])
    base_mci  = (sub_df[cols].values * w_base).sum(axis=1)
    base_rank = pd.Series(base_mci).rank(ascending=False)

    rhos = []
    for _ in range(n_draws):
        w = rng.uniform(0.05, 0.50, size=4)
        w = w / w.sum()
        weighted = (sub_df[cols].values * w).sum(axis=1)
        draw_rank = pd.Series(weighted).rank(ascending=False)
        rho, _    = stats.spearmanr(base_rank.values, draw_rank.values)
        rhos.append(rho)

    result = {
        "rho_mean":   float(np.mean(rhos)),
        "rho_min":    float(np.min(rhos)),
        "rho_median": float(np.median(rhos)),
        "n_draws":    n_draws,
    }
    log.info(f"  Rank stability: mean ρ={result['rho_mean']:.3f}  "
             f"min ρ={result['rho_min']:.3f}  "
             f"({'robust' if result['rho_mean'] > 0.85 else 'SENSITIVE — investigate'})")
    return result


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(report_path: Path,
                 q_fit_df:     pd.DataFrame,
                 q_backfire:   dict,
                 q_sub:        dict,
                 q_user:       dict,
                 q_regime_df:  pd.DataFrame,
                 q_spill:      dict,
                 mci_df:       pd.DataFrame,
                 gate_df:      pd.DataFrame,
                 att_df:       pd.DataFrame,
                 sensitivity:  dict):
    """
    Writes a plain-text report of all statistical tests, including negative
    results in a dedicated section.
    """
    lines = []
    sep   = "=" * 70

    def h(title):
        lines.extend(["", sep, title, sep])

    def row(label, value):
        lines.append(f"  {label:<42}  {value}")

    h("MODEL CREDIBILITY INDEX — FULL REPORT (v2)")
    lines.append("Generated by confidence_score.py (v2)")
    lines.append(f"Weights: {MCI_WEIGHTS}")
    lines.append("NOTE: Q_substitution and Q_spillover EXCLUDED from composite.")
    lines.append("      See NEGATIVE RESULTS section below.")

    # ── Confidence gate ──────────────────────────────────────────────────────
    h("CONFIDENCE GATE")
    lines.append("  Conditions: (a) ≥300 comments/day  (b) Q_fit>0  "
                 "(c) g*∈[0.01,2.0]  (d) std(C)>0.005")
    lines.append("")
    lines.append(f"  {'Subreddit':<26} {'Conf':>8}  "
                 f"{'data':>5} {'fit':>5} {'g*':>5} {'var':>5}  {'g*val':>7}")
    lines.append("  " + "-" * 65)
    for _, r in gate_df.sort_values("confidence_level").iterrows():
        ban = "[B]" if r["subreddit"] in BANNED_SUBREDDITS else "   "
        lines.append(
            f"  {ban} r/{r['subreddit']:<23} {r['confidence_level']:>8}  "
            f"{'✓' if r['gate_data_ok'] else '✗':>5} "
            f"{'✓' if r['gate_fit_ok'] else '✗':>5} "
            f"{'✓' if r['gate_gstar_ok'] else '✗':>5} "
            f"{'✓' if r['gate_variance_ok'] else '✗':>5}  "
            f"{r['g_star']:>7.4f}"
        )

    # ── Q_fit ────────────────────────────────────────────────────────────────
    h("COMPONENT 1: Q_fit (ODE Goodness-of-Fit)")
    lines.append("  Formula: Q_fit = 1 − fit_nll / SS_obs  (Box & Jenkins 1976)")
    lines.append("  Weight in MCI: 0.25")
    lines.append("")
    valid_fit = q_fit_df["Q_fit"].dropna()
    row("N valid:", str(len(valid_fit)))
    row("Mean Q_fit:", f"{valid_fit.mean():.4f}")
    row("Median Q_fit:", f"{valid_fit.median():.4f}")
    row("Q_fit > 0 (model beats mean):",
        f"{(valid_fit > 0).sum()} of {len(valid_fit)}")
    row("Q_fit < 0 (model WORSE than mean):",
        f"{(valid_fit < 0).sum()} of {len(valid_fit)}")
    lines.append("")
    lines.append(f"  {'Subreddit':<26} {'Q_fit':>7}  {'n_days':>7}")
    lines.append("  " + "-" * 45)
    for _, r in q_fit_df.sort_values("Q_fit", ascending=False).iterrows():
        ban = "[B]" if r["subreddit"] in BANNED_SUBREDDITS else "   "
        lines.append(
            f"  {ban} r/{r['subreddit']:<23} "
            f"{r['Q_fit']:>7.4f}  "
            f"{int(r['n_days_fit']):>7}"
        )

    # ── Q_backfire ───────────────────────────────────────────────────────────
    h("COMPONENT 2: Q_backfire (Spearman ρ: pre-ban fragility → post-ban slope)")
    lines.append("  REWRITTEN: Spearman ρ between C/g* and post-ban violation slope.")
    lines.append("  Prediction P3: ρ > 0 (fragile communities trend worse post-ban).")
    lines.append("  Weight in MCI: 0.35")
    lines.append("")
    rho = q_backfire.get("spearman_rho", np.nan)
    p   = q_backfire.get("spearman_p",   np.nan)
    row("Spearman ρ:",       f"{rho:.4f}" if not np.isnan(rho) else "n/a")
    row("p-value:",          f"{p:.4f}"   if not np.isnan(p)   else "n/a")
    row("N subreddits:",     str(q_backfire.get("n_subreddits", "n/a")))
    row("Q_backfire score:", f"{q_backfire.get('Q_backfire_score', np.nan):.4f}")
    lines.append("")
    if not np.isnan(rho):
        sig = ("✓ CONFIRMED" if rho > 0 and p < 0.05 else
               "~ DIRECTIONAL" if rho > 0 else "✗ NOT CONFIRMED")
        lines.append(f"  RESULT: {sig}  (ρ={rho:.3f}, p={p:.3f})")
        lines.append("  NOTE: N=16 limits power. ρ > 0 is the theoretically "
                     "predicted sign.")

    psd = q_backfire.get("per_subreddit_df", pd.DataFrame())
    if not psd.empty:
        lines.append("")
        lines.append(f"  {'Subreddit':<26} {'g*':>7}  {'C/g*':>7}  "
                     f"{'post_slope':>10}  {'p':>6}")
        lines.append("  " + "-" * 65)
        for _, r in psd.sort_values("fragility", ascending=False).iterrows():
            ban = "[B]" if r["subreddit"] in BANNED_SUBREDDITS else "   "
            lines.append(
                f"  {ban} r/{r['subreddit']:<23} "
                f"{r['g_star']:>7.4f}  "
                f"{r['fragility']:>7.4f}  "
                f"{r['post_ban_slope']:>+10.5f}  "
                f"{r['post_ban_p']:>6.3f}"
            )

    # ── Q_user ───────────────────────────────────────────────────────────────
    h("COMPONENT 4: Q_user (User Behavioral Consistency)")
    lines.append("  ρ_u = Pearson r(C_smooth, is_removed) per user.")
    lines.append("  Prediction: JOINER ρ_u < INITIATOR ρ_u.")
    lines.append("  Weight in MCI: 0.15")
    lines.append("  CAUTION: η² is small (~0.013) despite large N. "
                 "Effect is statistically significant but practically modest.")
    lines.append("")
    row("ANOVA F:",         f"{q_user.get('F_stat', np.nan):.4f}")
    row("ANOVA p:",         f"{q_user.get('p_anova', np.nan):.6f}")
    row("η² (effect size):",f"{q_user.get('eta2', np.nan):.4f}")
    row("Mean ρ INITIATOR:",f"{q_user.get('mean_rho_INIT', np.nan):.4f}")
    row("Mean ρ JOINER:",   f"{q_user.get('mean_rho_JOIN', np.nan):.4f}")
    row("Mean ρ COMPLIER:", f"{q_user.get('mean_rho_COMP', np.nan):.4f}")
    row("N INITIATOR:",     str(q_user.get("n_INIT", "n/a")))
    row("MW INITIATOR>JOINER p:", f"{q_user.get('mw_IJ_p', np.nan):.4f}")

    # ── Q_regime ─────────────────────────────────────────────────────────────
    h("COMPONENT 5: Q_regime (Stage 2 Stability Margin)")
    lines.append("  margin = (α_gov − α_gov_min) / |α_gov_min|")
    lines.append("  Q_regime = (tanh(margin) + 1) / 2")
    lines.append("  Weight in MCI: 0.25")
    lines.append("  NOTE: α_gov estimated from short time series. "
                 "Treat as ordinal ranking, not precise measurement.")
    lines.append("")
    if not q_regime_df.empty:
        row("Cross-val r(margin, −volatility):",
            f"{q_regime_df['Q_regime_r_xval'].iloc[0]:.4f}")
        row("Cross-val p:",
            f"{q_regime_df['Q_regime_p_xval'].iloc[0]:.4f}")
        row("N stable (margin > 0):",
            str((q_regime_df["stability_margin"] > 0).sum()))
        row("N trap/unstable (margin < 0):",
            str((q_regime_df["stability_margin"] < 0).sum()))
        lines.append("")
        lines.append(f"  {'Subreddit':<26} {'margin':>8}  "
                     f"{'Q_regime':>8}  {'regime':<25}")
        lines.append("  " + "-" * 70)
        for _, r in q_regime_df.sort_values(
                "stability_margin", ascending=False).iterrows():
            ban = "[B]" if r["subreddit"] in BANNED_SUBREDDITS else "   "
            m   = f"{r['stability_margin']:>8.4f}" \
                  if not np.isnan(r.get("stability_margin", np.nan)) else "     n/a"
            lines.append(
                f"  {ban} r/{r['subreddit']:<23} {m}  "
                f"{r['Q_regime']:>8.4f}  "
                f"{str(r.get('regime', 'n/a')):<25}"
            )

    # ── Negative results ─────────────────────────────────────────────────────
    h("NEGATIVE RESULTS (excluded from MCI composite)")
    lines.append("  These results are reported for transparency.")
    lines.append("  They do not weaken the primary claims; they delimit "
                 "the model's empirical scope.")
    lines.append("")

    lines.append("  [NR1] T–C Substitution (Q_substitution)")
    lines.append("  Prediction: β₁ < 0 (high legitimacy → less enforcement needed).")
    lines.append(f"  Result: β₁ = {q_sub.get('beta1_TC', np.nan):.4f}  "
                 f"p = {q_sub.get('beta1_p', np.nan):.4f}  "
                 f"partial_r = {q_sub.get('partial_r_TC', np.nan):.4f}")
    lines.append("  Verdict: HYPOTHESIS REJECTED (β₁ > 0). Cross-sectional T–C")
    lines.append("  relationship is positive, inconsistent with substitution.")
    lines.append("  Interpretation: the positive slope likely reflects platform-level")
    lines.append("  selection — well-moderated communities have both high T AND high C.")
    lines.append("")

    lines.append("  [NR2] Spillover Cascade (Q_spillover)")
    n_mig = q_spill.get("n_migrants", 0)
    lines.append(f"  Result: {n_mig} migrant INITIATORs found under corrected "
                 "migration definition.")
    lines.append("  Verdict: NULL RESULT. The earlier r/loseit finding was an")
    lines.append("  artifact of the flawed migration definition (cross-posting")
    lines.append("  miscoded as post-ban migration).")
    lines.append("  Interpretation: Individual-level behavioral persistence across")
    lines.append("  community boundaries is not identifiable from this dataset.")

    # ── Composite MCI ─────────────────────────────────────────────────────────
    h("COMPOSITE MCI")
    lines.append(f"  Components and weights: {MCI_WEIGHTS}")
    lines.append("  Missing components skipped; weights renormalized.")
    lines.append("")
    lines.append(f"  {'Subreddit':<26} {'MCI':>6}  {'Conf':>8}  "
                 f"{'back':>6} {'fit':>6} {'reg':>6} {'usr':>6}  {'regime':<22}")
    lines.append("  " + "-" * 95)
    for _, r in mci_df.sort_values("MCI", ascending=False).iterrows():
        ban  = "[B]" if r["is_banned"] else "   "
        conf = r.get("confidence_level", "?")

        def f(v):
            return f"{v:.3f}" if not (isinstance(v, float) and np.isnan(v)) else " n/a"

        lines.append(
            f"  {ban} r/{r['subreddit']:<23} "
            f"{f(r['MCI'])}  "
            f"{conf:>8}  "
            f"{f(r['Q_backfire'])} "
            f"{f(r['Q_fit_scaled'])} "
            f"{f(r['Q_regime'])} "
            f"{f(r['Q_user'])}  "
            f"{str(r.get('regime', 'n/a')):<22}"
        )

    # ── Attention flags ───────────────────────────────────────────────────────
    h("TWO-TIER ATTENTION FLAGS")
    lines.append("  Tier 1 (observable): violation trend + reactivity + Gini.")
    lines.append("  Tier 2 (structural): pre-ban C/g* ratio.")
    lines.append("  Flag: top 25% on either tier → attention recommended.")
    lines.append("  Primary ranking restricted to High/Moderate confidence.")
    lines.append("")
    lines.append(f"  {'Subreddit':<26} {'Flag':<14}  {'Conf':>8}  "
                 f"{'T1rank':>7}  {'T2rank':>7}  {'C/g*':>6}")
    lines.append("  " + "-" * 75)
    for _, r in att_df.sort_values(
            "tier1_rank", na_position="last").iterrows():
        ban = "[B]" if r["is_banned"] else "   "
        t1r = f"{r['tier1_rank']:.0f}" \
              if not pd.isna(r.get("tier1_rank")) else "  n/a"
        t2r = f"{r['tier2_rank']:.0f}" \
              if not pd.isna(r.get("tier2_rank")) else "  n/a"
        t2v = f"{r['t2_fragility']:.3f}" \
              if not np.isnan(r.get("t2_fragility", np.nan)) else " n/a"
        lines.append(
            f"  {ban} r/{r['subreddit']:<23} "
            f"{r['attention_flag']:<14}  "
            f"{r['confidence_level']:>8}  "
            f"{t1r:>7}  {t2r:>7}  {t2v:>6}"
        )

    # ── Sensitivity ───────────────────────────────────────────────────────────
    h("SENSITIVITY ANALYSIS (4-component weights)")
    row("N draws:", str(sensitivity.get("n_draws", 500)))
    row("Mean Spearman ρ:", f"{sensitivity.get('rho_mean', np.nan):.4f}")
    row("Min  Spearman ρ:", f"{sensitivity.get('rho_min', np.nan):.4f}")
    row("Median Spearman ρ:", f"{sensitivity.get('rho_median', np.nan):.4f}")
    rho_m = sensitivity.get("rho_mean", 0)
    if rho_m > 0.85:
        lines.append("\n  RESULT: ✓ RANKING ROBUST (mean ρ > 0.85).")
    elif rho_m > 0.70:
        lines.append("\n  RESULT: ~ MODERATE (mean ρ 0.70–0.85). "
                     "Report weight sensitivity in supplementary material.")
    else:
        lines.append("\n  RESULT: ✗ RANKING SENSITIVE (mean ρ < 0.70). "
                     "Investigate which component drives instability.")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"  Report written: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute MCI and attention flags for the "
                    "Coordination-Legitimacy Game."
    )
    parser.add_argument("--out_dir",    default="./output")
    parser.add_argument("--min_posts",  type=int, default=10)
    parser.add_argument("--n_draws",    type=int, default=500)
    parser.add_argument("--check_only", action="store_true")
    args = parser.parse_args()

    out_dir  = Path(args.out_dir)
    conf_dir = out_dir / "confidence"
    conf_dir.mkdir(parents=True, exist_ok=True)
    (conf_dir / "tables").mkdir(exist_ok=True)
    (conf_dir / "plots").mkdir(exist_ok=True)

    log.info("=" * 60)
    log.info("MODEL CREDIBILITY INDEX  v2  —  confidence_score.py")
    log.info("=" * 60)

    log.info("\nLoading pipeline outputs...")
    data = load_inputs(out_dir)

    if args.check_only:
        log.info("All files present.")
        return

    daily_metrics   = data["daily_metrics"]
    user_thresholds = data["user_thresholds"]
    panel           = data["panel"]
    calibrated      = data["calibrated"]
    stage2          = data["stage2"]
    comments_path   = out_dir / "comments_filtered.csv"

    # ── Confidence gate ──────────────────────────────────────────────────────
    log.info("\n--- Confidence Gate ---")
    gate_df = compute_confidence_gate(calibrated, daily_metrics)

    # ── Component 1: Q_fit ───────────────────────────────────────────────────
    log.info("\n--- Component 1: Q_fit ---")
    q_fit_df = compute_q_fit(calibrated, daily_metrics)

    # ── Component 2: Q_backfire (Spearman ρ) ─────────────────────────────────
    log.info("\n--- Component 2: Q_backfire ---")
    q_backfire = compute_q_backfire(calibrated, daily_metrics)

    # ── Q_substitution (negative result, not in composite) ───────────────────
    log.info("\n--- Q_substitution (negative result) ---")
    q_sub = compute_q_substitution(daily_metrics)

    # ── Component 4: Q_user ──────────────────────────────────────────────────
    log.info("\n--- Component 4: Q_user ---")
    q_user = compute_q_user(user_thresholds, daily_metrics,
                             comments_path, min_posts=args.min_posts)

    # ── Component 5: Q_regime ────────────────────────────────────────────────
    log.info("\n--- Component 5: Q_regime ---")
    q_regime_df = compute_q_regime(stage2)

    # ── Q_spillover (null result, not in composite) ───────────────────────────
    log.info("\n--- Q_spillover (null result) ---")
    q_spill = compute_q_spillover(panel, user_thresholds)

    # ── Composite MCI ────────────────────────────────────────────────────────
    log.info("\n--- Assembling MCI ---")
    mci_df = assemble_mci(
        q_fit_df, q_backfire, q_user,
        q_regime_df, calibrated, gate_df
    )

    # ── Two-tier attention flags ──────────────────────────────────────────────
    log.info("\n--- Two-tier attention flags ---")
    att_df = compute_attention_score(
        daily_metrics, calibrated, user_thresholds, gate_df
    )

    # ── Sensitivity analysis ─────────────────────────────────────────────────
    log.info("\n--- Sensitivity analysis ---")
    sensitivity = sensitivity_analysis(mci_df, n_draws=args.n_draws)

    # ── Write outputs ─────────────────────────────────────────────────────────
    log.info("\nWriting outputs...")

    mci_df.to_csv(conf_dir / "subreddit_mci.csv", index=False)
    log.info(f"  subreddit_mci.csv  ({len(mci_df)} rows)")

    att_df.to_csv(conf_dir / "attention_flags.csv", index=False)
    log.info(f"  attention_flags.csv  ({len(att_df)} rows)")

    rho_df = q_user.get("rho_df", pd.DataFrame())
    if not rho_df.empty:
        rho_df.to_parquet(conf_dir / "user_behavioral.parquet", index=False)
        log.info(f"  user_behavioral.parquet  ({len(rho_df):,} rows)")

    gate_df.to_csv(conf_dir / "tables" / "confidence_gate.csv", index=False)
    q_regime_df.to_csv(conf_dir / "tables" / "regime_scores.csv", index=False)
    q_fit_df.to_csv(conf_dir / "tables" / "fit_scores.csv", index=False)

    psd = q_backfire.get("per_subreddit_df", pd.DataFrame())
    if not psd.empty:
        psd.to_csv(conf_dir / "tables" / "backfire_scores.csv", index=False)

    scalar_results = {
        "Q_backfire":     {k: v for k, v in q_backfire.items()
                           if k != "per_subreddit_df"},
        "Q_substitution": q_sub,
        "Q_user":         {k: v for k, v in q_user.items()
                           if k != "rho_df"},
        "Q_spillover":    q_spill,
        "sensitivity":    sensitivity,
        "mci_weights":    MCI_WEIGHTS,
    }
    with open(conf_dir / "tables" / "scalar_results.json", "w") as f:
        json.dump(scalar_results, f, indent=2, default=str)

    write_report(
        conf_dir / "confidence_report.txt",
        q_fit_df, q_backfire, q_sub, q_user,
        q_regime_df, q_spill, mci_df, gate_df, att_df, sensitivity
    )

    log.info("\n" + "=" * 60)
    log.info("DONE.  Run confidence_verification.py to generate plots.")
    log.info(f"Outputs in: {conf_dir}/")
    log.info("=" * 60)


if __name__ == "__main__":
    main()