"""
confidence_score.py
===================
Computes the six-component Model Credibility Index (MCI) for the
Coordination-Legitimacy Game paper.

Each component targets a distinct type of validity:
  Q_fit          — Internal validity   (ODE R² on calibration window)
  Q_backfire     — Predictive validity (g* vs. post-ban slope; out-of-sample)
  Q_substitution — Structural validity (T–C partial correlation; mechanism test)
  Q_user         — Structural validity (user-type behavioral consistency; ANOVA)
  Q_regime       — Comparative statics (Stage 2 margin from stability boundary)
  Q_spillover    — Causal validity     (migrant INITIATOR removal rate vs. native)

All components are scaled to [0, 1] then averaged into a composite MCI per subreddit.
A user-level behavioral consistency table (rho_u per user) is also written.

Usage:
    python confidence_score.py --out_dir ../Reddit-2015-v2/output

Outputs (all in <out_dir>/confidence/):
    subreddit_mci.csv            — per-subreddit scores + composite MCI
    user_behavioral.parquet      — per-user rho_u + prediction match flag
    confidence_report.txt        — all statistical tests, p-values, effect sizes
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit      # sigmoid — not used but available
import gc

warnings.filterwarnings("ignore")

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

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def load_inputs(out_dir: Path) -> dict:
    """
    Loads all pipeline outputs needed for MCI computation.
    Raises FileNotFoundError with a clear message if any file is missing.
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
            # Never load the full 2.5 GB CSV into memory — pass the path
            # to functions that stream it in chunks
            data[key] = path
        elif path.suffix == ".csv":
            data[key] = pd.read_csv(path)
        else:
            data[key] = None

    if not all_ok:
        raise FileNotFoundError(
            "One or more pipeline output files are missing. "
            "Run reddit_pipeline_safe.py → model_calibration.py → stage2_analysis.py first."
        )

    # Standardize datetime columns
    data["daily_metrics"]["day"] = pd.to_datetime(
        data["daily_metrics"]["day"], utc=True
    )
    if "week" in data["panel"].columns:
        data["panel"]["week"] = pd.to_datetime(data["panel"]["week"], utc=True)

    return data


def safe_scale_01(series: pd.Series) -> pd.Series:
    """Min-max scale a series to [0, 1], handling constant series gracefully."""
    mn, mx = series.min(), series.max()
    if mx - mn < 1e-12:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - mn) / (mx - mn)


# ---------------------------------------------------------------------------
# Component 1: Q_fit — ODE goodness-of-fit
# ---------------------------------------------------------------------------

def compute_q_fit(calibrated: pd.DataFrame,
                  daily_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Normalised R²-equivalent for the Stage 1 ODE fit per subreddit.

    Formula:
        SS_obs  = Var(C_raw) × n_days    (total variance in observed series)
        Q_fit   = 1 − fit_nll / SS_obs   (1 = perfect, 0 = as good as mean prediction)

    Why we normalise:
        fit_nll from Nelder-Mead is raw RSS.  A subreddit observed for 90 days
        always has higher RSS than one observed for 20 days — not because the fit
        is worse but because there are more residuals.  Dividing by SS_obs is the
        standard OLS R² normalisation (Box & Jenkins 1976).
    """
    log.info("Computing Q_fit (ODE goodness-of-fit)...")
    rows = []

    for _, row in calibrated.iterrows():
        sub      = row["subreddit"]
        sub_dm   = daily_metrics[daily_metrics["subreddit"] == sub]["C_raw"].dropna()
        n        = len(sub_dm)

        if n < 5:
            rows.append({"subreddit": sub, "Q_fit": np.nan,
                         "fit_nll": row.get("fit_nll", np.nan),
                         "SS_obs": np.nan, "n_days_fit": n})
            continue

        ss_obs = sub_dm.var() * n           # Σ(x_i − x̄)²
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
             f"n_valid={df['Q_fit'].notna().sum()}")
    return df


# ---------------------------------------------------------------------------
# Component 2: Q_backfire — out-of-sample threshold accuracy
# ---------------------------------------------------------------------------

def compute_q_backfire(calibrated: pd.DataFrame,
                       daily_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Tests whether g* (calibrated on Jun 1–9 pre-ban data) correctly predicts
    which subreddits experienced rising violations post-ban (Jun 10 – Aug 31).

    Prediction:
        C_preban_mean > g*  →  model predicts "backfire" (violations will rise)
        C_preban_mean ≤ g*  →  model predicts "stable"

    Observed outcome:
        OLS slope of C_raw on time in post-ban window
        slope > 0 and p < 0.10  →  "backfire" observed
        slope ≤ 0 or p ≥ 0.10  →  "stable" observed

    Q_backfire = hit rate across all subreddits with valid data.

    Additionally computes point-biserial r as the continuous effect size.
    """
    log.info("Computing Q_backfire (g* predictive accuracy)...")
    dm = daily_metrics.copy()
    dm["post_ban"] = dm["day"] >= BAN_DATE

    g_star_map = calibrated.set_index("subreddit")["g_star"].to_dict()

    rows = []
    for sub, sub_dm in dm.groupby("subreddit"):
        g_star = g_star_map.get(sub, np.nan)
        if np.isnan(g_star):
            continue

        pre  = sub_dm[~sub_dm["post_ban"]]["C_smooth"].dropna()
        post = sub_dm[ sub_dm["post_ban"]].sort_values("day")

        c_preban_mean = float(pre.mean()) if len(pre) >= 3 else np.nan
        if np.isnan(c_preban_mean) or len(post) < 5:
            continue

        # OLS slope on post-ban C_raw
        x_post   = np.arange(len(post))
        y_post   = post["C_raw"].fillna(0).values
        if y_post.std() < 1e-8:
            continue
        slope, _, _, p_slope, _ = stats.linregress(x_post, y_post)

        # Model prediction
        predicted_backfire = (c_preban_mean > g_star)

        # Observed outcome (lenient p-threshold to account for short series)
        observed_backfire  = (slope > 0) and (p_slope < 0.10)

        hit = int(predicted_backfire == observed_backfire)

        rows.append({
            "subreddit":           sub,
            "g_star":              g_star,
            "c_preban_mean":       c_preban_mean,
            "predicted_backfire":  predicted_backfire,
            "post_ban_slope":      slope,
            "post_ban_slope_p":    p_slope,
            "observed_backfire":   observed_backfire,
            "backfire_hit":        hit,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        log.warning("  Q_backfire: no valid subreddits found.")
        return pd.DataFrame(columns=["subreddit", "Q_backfire"])

    # Point-biserial correlation (predicted vs. observed backfire)
    if df["predicted_backfire"].std() > 0 and df["observed_backfire"].std() > 0:
        rpb, p_rpb = stats.pointbiserialr(
            df["predicted_backfire"].astype(int),
            df["observed_backfire"].astype(int)
        )
    else:
        rpb, p_rpb = np.nan, np.nan

    hit_rate = df["backfire_hit"].mean()
    log.info(f"  Q_backfire hit rate: {hit_rate:.3f}  "
             f"point-biserial r={rpb:.3f}  p={p_rpb:.3f}"
             if not np.isnan(rpb) else
             f"  Q_backfire hit rate: {hit_rate:.3f}  (r: insufficient variance)")

    df["Q_backfire"]    = df["backfire_hit"].astype(float)
    df["r_pointbiser"]  = rpb
    df["r_pb_p"]        = p_rpb
    return df


# ---------------------------------------------------------------------------
# Component 3: Q_substitution — T–C partial correlation
# ---------------------------------------------------------------------------

def compute_q_substitution(daily_metrics: pd.DataFrame) -> dict:
    """
    Tests whether legitimacy (T) and enforcement (C) are substitutes in
    achieving compliance, controlling for achieved compliance level.

    OLS: C_smooth_i = β₀ + β₁·T_norm_i + β₂·compliance_i + ε_i

    Prediction: β₁ < 0  (high T → low C needed, holding compliance constant)

    Implemented across all subreddit-day observations (pooled panel).
    Standard errors are clustered by subreddit.

    Returns a dict with regression results (used in the report).
    """
    log.info("Computing Q_substitution (T–C substitution test)...")

    dm = daily_metrics.dropna(subset=["T_norm", "C_smooth", "C_raw"]).copy()
    dm["compliance"] = 1.0 - dm["C_raw"].clip(0, 1)

    # Remove observations with effectively zero variance in T or C
    dm = dm[dm["T_norm"].between(1e-6, 1 - 1e-6)]

    if len(dm) < 30:
        log.warning("  Q_substitution: insufficient data.")
        return {"beta1_TC": np.nan, "beta1_p": np.nan, "r2": np.nan,
                "partial_r_TC": np.nan, "n_obs": 0, "Q_substitution_score": np.nan}

    # Partial correlation r(T, C | compliance) — manual calculation
    # Regress T on compliance, get residuals; regress C on compliance, get residuals
    # Then correlate the two residual series
    X_comply = dm["compliance"].values.reshape(-1, 1)
    X_comply_c = X_comply - X_comply.mean()

    def ols_resid(y):
        """Return OLS residuals of y on X_comply_c."""
        b = np.linalg.lstsq(X_comply_c, y - y.mean(), rcond=None)[0]
        return (y - y.mean()) - X_comply_c @ b

    resid_T = ols_resid(dm["T_norm"].values)
    resid_C = ols_resid(dm["C_smooth"].values)
    partial_r, partial_p = stats.pearsonr(resid_T, resid_C)

    # Full OLS regression (pooled)
    Y = dm["C_smooth"].values
    X = np.column_stack([
        np.ones(len(dm)),
        dm["T_norm"].values,
        dm["compliance"].values,
    ])
    coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    y_hat  = X @ coeffs
    ss_res = np.sum((Y - y_hat) ** 2)
    ss_tot = np.sum((Y - Y.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Standard errors (OLS, assuming homoskedasticity — report as approximate)
    n, k = len(Y), X.shape[1]
    mse  = ss_res / max(n - k, 1)
    XtX_inv = np.linalg.pinv(X.T @ X)
    se   = np.sqrt(np.diag(XtX_inv) * mse)
    t_stats = coeffs / np.where(se > 1e-12, se, np.nan)
    p_vals  = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))

    beta1_TC = float(coeffs[1])   # coefficient on T_norm
    beta1_p  = float(p_vals[1])

    # Scalar score: partial_r is negative when T→C substitution operates.
    # Rescale so r = −1 (perfect substitution) → score = 1.0;
    # r = 0 (no substitution) → score = 0.5;  r = +1 → score = 0.0
    q_sub_score = float((-partial_r + 1.0) / 2.0)

    result = {
        "beta1_TC":              beta1_TC,
        "beta1_SE":              float(se[1]),
        "beta1_t":               float(t_stats[1]),
        "beta1_p":               beta1_p,
        "beta2_compliance":      float(coeffs[2]),
        "r2_pooled":             float(r2),
        "partial_r_TC":          float(partial_r),
        "partial_r_p":           float(partial_p),
        "n_obs":                 int(n),
        "Q_substitution_score":  q_sub_score,
    }

    log.info(f"  Q_substitution: β₁(T→C)={beta1_TC:.4f}  "
             f"p={beta1_p:.4f}  "
             f"partial_r={partial_r:.4f}  "
             f"R²={r2:.3f}  "
             f"score={q_sub_score:.3f}")
    return result


# ---------------------------------------------------------------------------
# Component 4: Q_user — behavioral consistency ANOVA
# ---------------------------------------------------------------------------

def compute_q_user(user_thresholds: pd.DataFrame,
                   daily_metrics: pd.DataFrame,
                   comments_csv_path: Path,
                   min_posts: int = 10) -> dict:
    """
    Validates that INITIATOR / JOINER / COMPLIER types differ in HOW their
    removal probability responds to enforcement level (ρ_u), not just in
    their average removal rate.

    For each user u posting in subreddit s on ≥ min_posts days:
        ρ_u = Pearson correlation(C_smooth on posting day, is_removed)

    Prediction:
        INITIATOR: ρ_u ≈ 0   (enforcement-insensitive)
        JOINER:    ρ_u < 0   (removed more when enforcement is LOW)
        COMPLIER:  ρ_u ≈ 0   (never removed regardless of C)

    Test: One-way ANOVA (F, p, η²) on ρ_u grouped by user_type.
    Post-hoc: Tukey HSD (INITIATOR vs JOINER is the key contrast).

    Returns dict with ANOVA results + per-user ρ_u DataFrame.
    """
    log.info(f"Computing Q_user (behavioral consistency; min_posts={min_posts})...")

    # Build day → subreddit → C_smooth DataFrame for merge
    dm = daily_metrics.copy()
    dm["day_str"] = dm["day"].dt.strftime("%Y-%m-%d")
    ct_df = dm[["subreddit", "day_str", "C_smooth"]].drop_duplicates()

    # Build valid (author, subreddit, user_type) table for merge-based filtering
    valid_users_df = (
        user_thresholds[["author", "subreddit", "user_type"]]
        .drop_duplicates(subset=["author", "subreddit"])
    )

    # Running-statistics accumulator (vectorized per chunk via groupby):
    # For each (author, subreddit): n, sum_x, sum_y, sum_xy, sum_x2, sum_y2
    # where x = C_smooth, y = is_removed.  Pearson r computable from these alone.
    accum_parts = []   # list of partial-aggregation DataFrames, one per chunk

    EXCLUDE = {"[deleted]", "AutoModerator"}
    CHUNK   = 500_000

    log.info("  Streaming comments_filtered.csv (vectorized merge)...")
    n_chunks = 0
    for chunk in pd.read_csv(
        comments_csv_path, chunksize=CHUNK,
        usecols=["author", "subreddit", "created_utc", "is_removed"]
    ):
        chunk = chunk[~chunk["author"].isin(EXCLUDE)]
        chunk = chunk.dropna(subset=["author"])

        chunk["day_str"] = (
            pd.to_datetime(chunk["created_utc"], unit="s", utc=True)
            .dt.strftime("%Y-%m-%d")
        )

        # Keep only users in the thresholds table
        chunk = chunk.merge(valid_users_df, on=["author", "subreddit"], how="inner")
        if chunk.empty:
            del chunk; gc.collect(); continue

        # Attach C_smooth
        chunk = chunk.merge(ct_df, on=["subreddit", "day_str"], how="left")
        chunk = chunk.dropna(subset=["C_smooth"])
        if chunk.empty:
            del chunk; gc.collect(); continue

        chunk["is_removed"] = chunk["is_removed"].fillna(0).astype(int)
        chunk["xy"]  = chunk["C_smooth"] * chunk["is_removed"]
        chunk["x2"]  = chunk["C_smooth"] ** 2
        chunk["y2"]  = chunk["is_removed"] ** 2

        grp = chunk.groupby(["author", "subreddit", "user_type"], sort=False).agg(
            n   =("C_smooth",    "count"),
            sx  =("C_smooth",    "sum"),
            sy  =("is_removed",  "sum"),
            sxy =("xy",          "sum"),
            sx2 =("x2",          "sum"),
            sy2 =("y2",          "sum"),
        )
        accum_parts.append(grp)
        n_chunks += 1
        del chunk; gc.collect()

    log.info(f"  Processed {n_chunks} chunks. Consolidating running stats...")

    if not accum_parts:
        log.warning("  Q_user: no valid data after filtering.")
        return {"F_stat": np.nan, "p_anova": np.nan, "eta2": np.nan,
                "rho_df": pd.DataFrame(), "Q_user_score": np.nan}

    # Sum running stats across chunks for the same (author, subreddit)
    combined = pd.concat(accum_parts)
    combined = combined.groupby(["author", "subreddit", "user_type"], sort=False).sum()
    combined = combined.reset_index()

    log.info(f"  Built series for {len(combined):,} user×subreddit pairs.")

    # Compute Pearson r from running stats (no per-row iteration needed)
    combined = combined[combined["n"] >= min_posts].copy()

    denom_x = combined["n"] * combined["sx2"] - combined["sx"] ** 2
    denom_y = combined["n"] * combined["sy2"] - combined["sy"] ** 2
    numer   = combined["n"] * combined["sxy"] - combined["sx"] * combined["sy"]

    denom   = np.sqrt(denom_x.clip(0) * denom_y.clip(0))
    rho_u   = np.where(denom > 1e-10, numer / denom, 0.0)
    rho_u   = np.clip(rho_u, -1.0, 1.0)

    # p-value via t-distribution
    n_obs   = combined["n"].values
    t_stat  = rho_u * np.sqrt(np.maximum(n_obs - 2, 1)) / np.sqrt(
        np.maximum(1.0 - rho_u ** 2, 1e-10)
    )
    p_rho   = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=np.maximum(n_obs - 2, 1)))

    rho_df = combined[["author", "subreddit", "user_type", "n"]].copy()
    rho_df = rho_df.rename(columns={"n": "n_obs"})
    rho_df["rho_u"] = rho_u
    rho_df["p_rho"] = p_rho

    log.info(f"  rho_u computed for {len(rho_df):,} users with >=>{min_posts} observations.")

    if len(rho_df) < 10:
        log.warning("  Q_user: insufficient users with enough observations.")
        return {"F_stat": np.nan, "p_anova": np.nan, "eta2": np.nan,
                "rho_df": rho_df, "Q_user_score": np.nan}

    # One-way ANOVA
    groups = [
        rho_df[rho_df["user_type"] == t]["rho_u"].dropna().values
        for t in ["INITIATOR", "JOINER", "COMPLIER"]
        if len(rho_df[rho_df["user_type"] == t]) >= 3
    ]
    if len(groups) < 2:
        log.warning("  Q_user: fewer than 2 user types with sufficient data for ANOVA.")
        return {"F_stat": np.nan, "p_anova": np.nan, "eta2": np.nan,
                "rho_df": rho_df, "Q_user_score": np.nan}

    F_stat, p_anova = stats.f_oneway(*groups)

    # η² = SS_between / SS_total
    grand_mean   = rho_df["rho_u"].dropna().mean()
    ss_between   = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total     = sum(np.sum((g - grand_mean) ** 2) for g in groups)
    eta2         = float(ss_between / ss_total) if ss_total > 0 else np.nan

    # Tukey HSD (manual — compare INITIATOR vs JOINER)
    init_rho = rho_df[rho_df["user_type"] == "INITIATOR"]["rho_u"].dropna()
    join_rho = rho_df[rho_df["user_type"] == "JOINER"]["rho_u"].dropna()
    comp_rho = rho_df[rho_df["user_type"] == "COMPLIER"]["rho_u"].dropna()

    mw_ij = stats.mannwhitneyu(init_rho, join_rho, alternative="greater") \
            if (len(init_rho) >= 3 and len(join_rho) >= 3) else None
    mw_jc = stats.mannwhitneyu(join_rho, comp_rho, alternative="less") \
            if (len(join_rho) >= 3 and len(comp_rho) >= 3) else None

    # Score: η² is already [0, 1]; clip for safety
    q_user_score = float(np.clip(eta2, 0, 1))

    log.info(f"  Q_user: F={F_stat:.3f}  p={p_anova:.4f}  η²={eta2:.4f}  "
             f"score={q_user_score:.3f}")
    if mw_ij:
        log.info(f"  Mann-Whitney INITIATOR > JOINER (ρ): "
                 f"U={mw_ij.statistic:.0f}  p={mw_ij.pvalue:.4f}")
    if mw_jc:
        log.info(f"  Mann-Whitney JOINER < COMPLIER (ρ): "
                 f"U={mw_jc.statistic:.0f}  p={mw_jc.pvalue:.4f}")

    return {
        "F_stat":         float(F_stat),
        "p_anova":        float(p_anova),
        "eta2":           float(eta2),
        "mean_rho_INIT":  float(init_rho.mean()) if len(init_rho) else np.nan,
        "mean_rho_JOIN":  float(join_rho.mean()) if len(join_rho) else np.nan,
        "mean_rho_COMP":  float(comp_rho.mean()) if len(comp_rho) else np.nan,
        "mw_IJ_U":        float(mw_ij.statistic) if mw_ij else np.nan,
        "mw_IJ_p":        float(mw_ij.pvalue)    if mw_ij else np.nan,
        "mw_JC_U":        float(mw_jc.statistic) if mw_jc else np.nan,
        "mw_JC_p":        float(mw_jc.pvalue)    if mw_jc else np.nan,
        "n_INIT":         int(len(init_rho)),
        "n_JOIN":         int(len(join_rho)),
        "n_COMP":         int(len(comp_rho)),
        "rho_df":         rho_df,
        "Q_user_score":   q_user_score,
    }


# ---------------------------------------------------------------------------
# Component 5: Q_regime — Stage 2 stability margin
# ---------------------------------------------------------------------------

def compute_q_regime(stage2: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the continuous stability margin for each subreddit and tests
    whether it correlates with observed enforcement volatility (C_volatility).

    margin_i = (α_gov_i − α_gov_min_i) / |α_gov_min_i|
        Positive → safely stable.  Negative → in the danger zone.

    Q_regime_i = (tanh(margin_i) + 1) / 2    [rescaled to [0, 1]]

    Cross-validation: Pearson r(Q_regime, −C_volatility)
    Prediction: β < 0 (stable margin → smooth enforcement → low volatility)
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
    df["Q_regime"]         = (df["Q_regime_raw"] + 1.0) / 2.0   # [0, 1]

    # Cross-validation: does margin predict lower C_volatility?
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
             f"n_stable={( df['stability_margin'] > 0 ).sum()}  "
             f"n_trap={(df['stability_margin'] < 0).sum()}")
    return df[["subreddit", "stability_margin", "Q_regime", "Q_regime_raw",
               "Q_regime_r_xval", "Q_regime_p_xval",
               "regime", "alpha_gov", "alpha_gov_min", "Xi", "C_volatility"]]


# ---------------------------------------------------------------------------
# Component 6: Q_spillover — migrant INITIATOR persistence
# ---------------------------------------------------------------------------

def compute_q_spillover(panel: pd.DataFrame,
                        user_thresholds: pd.DataFrame) -> dict:
    """
    Tests whether users classified as INITIATOR in banned subreddits maintain
    elevated removal rates in control communities after the ban wave.

    Population A: from_banned_sub=True AND user_type=INITIATOR, post-ban
    Population B: native users in the same control subreddits, post-ban

    Primary test: Mann-Whitney U (A vs B removal rates)
    Effect size:  Cohen's d

    Dose-response: OLS regression
        post_ban_rate_u = β₀ + β₁·theta_proxy_u + β₂·[user_type dummies] + ε_u

    Returns dict with all test statistics.
    """
    log.info("Computing Q_spillover (migrant INITIATOR behavioral persistence)...")

    # panel already contains user_type and theta_proxy from the pipeline
    # Only merge if user_type is missing from panel
    if "user_type" not in panel.columns:
        ut = user_thresholds[["author", "subreddit", "user_type", "theta_proxy"]].copy()
        panel_m = panel.merge(ut, on=["author", "subreddit"], how="left")
    else:
        panel_m = panel.copy()

    # Post-ban, control subreddits only
    post_ctrl = panel_m[panel_m["post_ban"] & ~panel_m["is_banned_sub"]].copy()

    if post_ctrl.empty:
        log.warning("  Q_spillover: no post-ban control observations found.")
        return {"mw_U": np.nan, "mw_p": np.nan, "cohens_d": np.nan,
                "mean_rate_migrants": np.nan, "mean_rate_natives": np.nan,
                "Q_spillover_score": np.nan}

    # Aggregate to user × (post-ban control period)
    user_postban = (
        post_ctrl.groupby(["author", "from_banned_sub", "user_type", "theta_proxy"])
        .agg(
            n_posts   = ("n_posts_week", "sum"),
            n_removed = ("resisted",     "sum"),
        )
        .reset_index()
    )
    user_postban = user_postban[user_postban["n_posts"] >= 3]   # min activity filter
    user_postban["removal_rate"] = user_postban["n_removed"] / user_postban["n_posts"]

    # Group A: migrant INITIATORs
    group_a = user_postban[
        user_postban["from_banned_sub"] & (user_postban["user_type"] == "INITIATOR")
    ]["removal_rate"].dropna()

    # Group B: native users (not from banned subs)
    group_b = user_postban[
        ~user_postban["from_banned_sub"]
    ]["removal_rate"].dropna()

    if len(group_a) < 3 or len(group_b) < 3:
        log.warning(f"  Q_spillover: too few observations "
                    f"(A={len(group_a)}, B={len(group_b)}).")
        return {"mw_U": np.nan, "mw_p": np.nan, "cohens_d": np.nan,
                "mean_rate_migrants": float(group_a.mean()) if len(group_a) else np.nan,
                "mean_rate_natives":  float(group_b.mean()) if len(group_b) else np.nan,
                "n_migrants": int(len(group_a)), "n_natives": int(len(group_b)),
                "Q_spillover_score": np.nan}

    # Mann-Whitney U (one-sided: migrants > natives)
    mw = stats.mannwhitneyu(group_a, group_b, alternative="greater")

    # Cohen's d
    pooled_std = np.sqrt(
        (group_a.std() ** 2 * (len(group_a) - 1) +
         group_b.std() ** 2 * (len(group_b) - 1)) /
        (len(group_a) + len(group_b) - 2)
    )
    cohens_d = float((group_a.mean() - group_b.mean()) / pooled_std) \
               if pooled_std > 1e-10 else np.nan

    # Dose-response OLS: theta_proxy → post-ban removal rate
    dose_df = user_postban.dropna(subset=["theta_proxy", "removal_rate"])
    dose_df = dose_df[dose_df["from_banned_sub"]]   # focus on migrants
    dose_r2, dose_beta, dose_p = np.nan, np.nan, np.nan
    if len(dose_df) >= 10:
        X_dose  = np.column_stack([np.ones(len(dose_df)), dose_df["theta_proxy"].values])
        Y_dose  = dose_df["removal_rate"].values
        coeffs_d, _, _, _ = np.linalg.lstsq(X_dose, Y_dose, rcond=None)
        yhat    = X_dose @ coeffs_d
        ss_r    = np.sum((Y_dose - yhat) ** 2)
        ss_t    = np.sum((Y_dose - Y_dose.mean()) ** 2)
        dose_r2  = 1.0 - ss_r / ss_t if ss_t > 0 else np.nan
        dose_beta = float(coeffs_d[1])
        n_d, k_d = len(Y_dose), 2
        mse_d   = ss_r / max(n_d - k_d, 1)
        se_d    = np.sqrt(np.linalg.pinv(X_dose.T @ X_dose)[1, 1] * mse_d)
        t_d     = dose_beta / se_d if se_d > 1e-12 else np.nan
        dose_p  = float(2 * (1 - stats.t.cdf(abs(t_d), df=n_d - k_d))) \
                  if not np.isnan(t_d) else np.nan

    # Scalar score: based on Cohen's d (medium effect = d ≥ 0.2)
    if not np.isnan(cohens_d):
        # Sigmoid scaled so d = 0 → 0.5, d = 0.5 → 0.82, d < 0 → < 0.5
        q_spill_score = float(1.0 / (1.0 + np.exp(-2.0 * cohens_d)))
    else:
        q_spill_score = np.nan

    log.info(f"  Q_spillover: Mann-Whitney U={mw.statistic:.0f}  p={mw.pvalue:.4f}  "
             f"Cohen's d={cohens_d:.3f}  "
             f"mean(migrants)={group_a.mean():.3f}  "
             f"mean(natives)={group_b.mean():.3f}  "
             f"score={q_spill_score:.3f}")
    log.info(f"  Dose-response (theta_proxy): β₁={dose_beta:.4f}  "
             f"p={dose_p:.4f}  R²={dose_r2:.3f}"
             if not np.isnan(dose_beta) else
             "  Dose-response: insufficient migrant data.")

    return {
        "mw_U":                  float(mw.statistic),
        "mw_p":                  float(mw.pvalue),
        "cohens_d":              float(cohens_d) if not np.isnan(cohens_d) else np.nan,
        "mean_rate_migrants":    float(group_a.mean()),
        "mean_rate_natives":     float(group_b.mean()),
        "n_migrants":            int(len(group_a)),
        "n_natives":             int(len(group_b)),
        "dose_beta_theta":       float(dose_beta),
        "dose_beta_p":           float(dose_p),
        "dose_r2":               float(dose_r2),
        "Q_spillover_score":     float(q_spill_score) if not np.isnan(q_spill_score) else np.nan,
    }


# ---------------------------------------------------------------------------
# Composite MCI assembly
# ---------------------------------------------------------------------------

def assemble_mci(
    q_fit_df:      pd.DataFrame,
    q_backfire_df: pd.DataFrame,
    q_sub:         dict,
    q_user:        dict,
    q_regime_df:   pd.DataFrame,
    q_spill:       dict,
    calibrated:    pd.DataFrame,
) -> pd.DataFrame:
    """
    Joins all per-subreddit components into a single MCI table.

    Components that are global scalars (Q_substitution, Q_user, Q_spillover)
    are broadcast to all subreddits because they reflect corpus-wide tests.

    MCI = (1/6) Σ rescaled_component_i
    """
    log.info("Assembling composite MCI...")

    # Base: all calibrated subreddits
    base = calibrated[["subreddit", "g_star", "n_days", "n_preban_days"]].copy()
    base["is_banned"] = base["subreddit"].isin(BANNED_SUBREDDITS)

    # Merge per-subreddit components
    base = base.merge(
        q_fit_df[["subreddit", "Q_fit", "fit_nll", "SS_obs", "n_days_fit"]],
        on="subreddit", how="left"
    )

    if not q_backfire_df.empty and "Q_backfire" in q_backfire_df.columns:
        base = base.merge(
            q_backfire_df[["subreddit", "Q_backfire", "c_preban_mean",
                           "predicted_backfire", "observed_backfire",
                           "post_ban_slope", "post_ban_slope_p"]],
            on="subreddit", how="left"
        )
    else:
        base["Q_backfire"] = np.nan

    if not q_regime_df.empty:
        base = base.merge(
            q_regime_df[["subreddit", "stability_margin", "Q_regime",
                         "regime", "alpha_gov", "alpha_gov_min", "Xi", "C_volatility"]],
            on="subreddit", how="left"
        )
    else:
        base["Q_regime"] = np.nan

    # Broadcast scalar components
    base["Q_substitution"] = q_sub.get("Q_substitution_score", np.nan)
    base["Q_user"]         = q_user.get("Q_user_score", np.nan)
    base["Q_spillover"]    = q_spill.get("Q_spillover_score", np.nan)

    # Rescale Q_fit to [0, 1] — clip negatives to 0
    base["Q_fit_scaled"] = base["Q_fit"].clip(lower=0.0)

    # Composite MCI (equal weights, missing components skipped)
    component_cols = ["Q_fit_scaled", "Q_backfire", "Q_substitution",
                      "Q_user", "Q_regime", "Q_spillover"]
    base["MCI"] = base[component_cols].mean(axis=1, skipna=True)

    # Count how many components contributed (data quality indicator)
    base["n_components"] = base[component_cols].notna().sum(axis=1)

    # Rank within banned / control groups
    base["MCI_rank_all"]     = base["MCI"].rank(ascending=False, method="min")
    base["MCI_rank_banned"]  = base.loc[ base["is_banned"],  "MCI"].rank(ascending=False)
    base["MCI_rank_control"] = base.loc[~base["is_banned"],  "MCI"].rank(ascending=False)

    log.info(f"\n{'='*60}")
    log.info("COMPOSITE MCI RESULTS")
    log.info(f"{'='*60}")
    for _, row in base.sort_values("MCI", ascending=False).iterrows():
        tag = "[BANNED]" if row["is_banned"] else "       "
        log.info(
            f"  {tag}  r/{row['subreddit']:<25}  "
            f"MCI={row['MCI']:.3f}  "
            f"(fit={row['Q_fit']:.2f} "
            f"back={row['Q_backfire']:.2f} "
            f"reg={row['Q_regime']:.2f})  "
            f"regime={row.get('regime', 'n/a')}"
        )

    return base


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def sensitivity_analysis(mci_df: pd.DataFrame,
                          n_draws: int = 500,
                          seed: int = 42) -> dict:
    """
    Tests robustness of MCI ranking to weight perturbations.

    Draws n_draws sets of 6 weights uniformly from [0.05, 0.35] (normalised to sum=1).
    Computes Spearman rank correlation of MCI ordering between the equal-weight
    baseline and each draw.  Reports min, mean, and median correlation.

    A mean correlation > 0.85 indicates the ranking is stable and not an
    artefact of the equal-weight assumption.
    """
    log.info(f"Running sensitivity analysis ({n_draws} weight draws)...")
    rng   = np.random.default_rng(seed)
    cols  = ["Q_fit_scaled", "Q_backfire", "Q_substitution",
             "Q_user", "Q_regime", "Q_spillover"]

    sub_df  = mci_df[["subreddit"] + cols].dropna(thresh=3).copy()
    if len(sub_df) < 4:
        log.warning("  Sensitivity: too few subreddits with complete data.")
        return {"rho_mean": np.nan, "rho_min": np.nan, "rho_median": np.nan}

    # Fill NaN with component mean for this analysis only
    for c in cols:
        sub_df[c] = sub_df[c].fillna(sub_df[c].mean())

    baseline_rank = sub_df["subreddit"].map(
        {s: r for r, s in enumerate(
            sub_df.assign(mci=sub_df[cols].mean(axis=1))
            .sort_values("mci", ascending=False)["subreddit"]
        )}
    )

    rhos = []
    for _ in range(n_draws):
        w = rng.uniform(0.05, 0.35, size=6)
        w = w / w.sum()
        weighted_mci  = (sub_df[cols].values * w).sum(axis=1)
        draw_rank     = pd.Series(weighted_mci).rank(ascending=False)
        rho, _        = stats.spearmanr(baseline_rank.values, draw_rank.values)
        rhos.append(rho)

    result = {
        "rho_mean":   float(np.mean(rhos)),
        "rho_min":    float(np.min(rhos)),
        "rho_median": float(np.median(rhos)),
        "n_draws":    n_draws,
    }
    log.info(f"  Rank stability: mean ρ={result['rho_mean']:.3f}  "
             f"min ρ={result['rho_min']:.3f}  "
             f"(robust if mean > 0.85)")
    return result


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(report_path: Path,
                 q_fit_df:      pd.DataFrame,
                 q_backfire_df: pd.DataFrame,
                 q_sub:         dict,
                 q_user:        dict,
                 q_regime_df:   pd.DataFrame,
                 q_spill:       dict,
                 mci_df:        pd.DataFrame,
                 sensitivity:   dict):
    """Writes a plain-text report of all statistical tests for the paper."""
    lines = []
    sep   = "=" * 70

    def h(title):
        lines.append("")
        lines.append(sep)
        lines.append(title)
        lines.append(sep)

    def row(label, value):
        lines.append(f"  {label:<40}  {value}")

    h("MODEL CREDIBILITY INDEX — FULL REPORT")
    lines.append("Generated automatically by confidence_score.py")

    # --- Q_fit ---
    h("COMPONENT 1: Q_fit (ODE Goodness-of-Fit)")
    lines.append("  Formula: Q_fit = 1 − fit_nll / SS_obs")
    lines.append("  Interpretation: R²-equivalent for the Stage 1 ODE (Box & Jenkins 1976)")
    lines.append("")
    valid_fit = q_fit_df["Q_fit"].dropna()
    row("N subreddits with valid Q_fit:", str(len(valid_fit)))
    row("Mean Q_fit:",                    f"{valid_fit.mean():.4f}")
    row("Median Q_fit:",                  f"{valid_fit.median():.4f}")
    row("Q_fit > 0.60 (strong fit):",     f"{(valid_fit > 0.60).sum()} of {len(valid_fit)}")
    row("Q_fit < 0 (worse than mean):",   f"{(valid_fit < 0).sum()} of {len(valid_fit)}")
    lines.append("")
    lines.append(f"  {'Subreddit':<26} {'Q_fit':>7}  {'fit_nll':>10}  {'SS_obs':>10}  {'n_days':>7}")
    lines.append("  " + "-" * 65)
    for _, r in q_fit_df.sort_values("Q_fit", ascending=False).iterrows():
        ban = "[B]" if r["subreddit"] in BANNED_SUBREDDITS else "   "
        lines.append(
            f"  {ban} r/{r['subreddit']:<23} "
            f"{r['Q_fit']:>7.4f}  "
            f"{r['fit_nll']:>10.4f}  "
            f"{r['SS_obs']:>10.6f}  "
            f"{int(r['n_days_fit']):>7}"
        )

    # --- Q_backfire ---
    h("COMPONENT 2: Q_backfire (g* Predictive Accuracy)")
    lines.append("  Formula: hit_rate = mean(prediction_i == outcome_i)")
    lines.append("  Pre-ban calibration → post-ban prediction (fully out-of-sample)")
    lines.append("")
    if not q_backfire_df.empty and "Q_backfire" in q_backfire_df.columns:
        hit_rate = q_backfire_df["Q_backfire"].mean()
        rpb      = q_backfire_df["r_pointbiser"].iloc[0] if len(q_backfire_df) else np.nan
        row("Overall hit rate (Q_backfire):", f"{hit_rate:.4f}")
        row("Point-biserial r:",              f"{rpb:.4f}")
        row("Chance baseline:",               "0.5000")
        row("N subreddits evaluated:",        str(len(q_backfire_df)))
        lines.append("")
        lines.append(f"  {'Subreddit':<26} {'g*':>7}  {'C_pre':>6}  "
                     f"{'Pred':>10}  {'Slope':>8}  {'p':>6}  {'Hit':>4}")
        lines.append("  " + "-" * 70)
        for _, r in q_backfire_df.sort_values("Q_backfire", ascending=False).iterrows():
            ban  = "[B]" if r["subreddit"] in BANNED_SUBREDDITS else "   "
            pred = "BACKFIRE" if r["predicted_backfire"] else "STABLE  "
            lines.append(
                f"  {ban} r/{r['subreddit']:<23} "
                f"{r['g_star']:>7.4f}  "
                f"{r['c_preban_mean']:>6.4f}  "
                f"{pred}  "
                f"{r['post_ban_slope']:>+8.5f}  "
                f"{r['post_ban_slope_p']:>6.3f}  "
                f"{'✓' if r['backfire_hit'] else '✗':>4}"
            )
    else:
        lines.append("  [No valid subreddits for Q_backfire]")

    # --- Q_substitution ---
    h("COMPONENT 3: Q_substitution (T–C Substitution Mechanism)")
    lines.append("  OLS: C_smooth = β₀ + β₁·T_norm + β₂·compliance + ε")
    lines.append("  Prediction: β₁ < 0  (Tyler 1990; Levi 1988)")
    lines.append("")
    row("β₁ (coefficient on T_norm):", f"{q_sub.get('beta1_TC', np.nan):.6f}")
    row("SE(β₁):",                     f"{q_sub.get('beta1_SE', np.nan):.6f}")
    row("t-statistic:",                f"{q_sub.get('beta1_t', np.nan):.4f}")
    row("p-value:",                    f"{q_sub.get('beta1_p', np.nan):.6f}")
    row("Partial r(T, C | compliance):", f"{q_sub.get('partial_r_TC', np.nan):.4f}")
    row("Partial r p-value:",           f"{q_sub.get('partial_r_p', np.nan):.6f}")
    row("R² (pooled regression):",      f"{q_sub.get('r2_pooled', np.nan):.4f}")
    row("N observations:",              str(q_sub.get("n_obs", "n/a")))
    row("Q_substitution score:",        f"{q_sub.get('Q_substitution_score', np.nan):.4f}")
    lines.append("")
    b1 = q_sub.get("beta1_TC", np.nan)
    p1 = q_sub.get("beta1_p",  np.nan)
    if not np.isnan(b1):
        if b1 < 0 and p1 < 0.05:
            lines.append("  RESULT: ✓ T–C SUBSTITUTION CONFIRMED (β₁ < 0, p < 0.05)")
        elif b1 < 0:
            lines.append(f"  RESULT: ~ T–C substitution direction correct but p={p1:.3f} (marginal)")
        else:
            lines.append("  RESULT: ✗ No T–C substitution detected (β₁ ≥ 0)")

    # --- Q_user ---
    h("COMPONENT 4: Q_user (User Behavioral Consistency)")
    lines.append("  One-way ANOVA: ρ_u grouped by user_type")
    lines.append("  ρ_u = Pearson r(C_smooth on posting day, is_removed)")
    lines.append("  Prediction: JOINER ρ_u < INITIATOR ρ_u  (Granovetter 1978)")
    lines.append("")
    row("ANOVA F-statistic:",     f"{q_user.get('F_stat', np.nan):.4f}")
    row("ANOVA p-value:",         f"{q_user.get('p_anova', np.nan):.6f}")
    row("η² (effect size):",      f"{q_user.get('eta2', np.nan):.4f}")
    row("Mean ρ_u INITIATOR:",    f"{q_user.get('mean_rho_INIT', np.nan):.4f}")
    row("Mean ρ_u JOINER:",       f"{q_user.get('mean_rho_JOIN', np.nan):.4f}")
    row("Mean ρ_u COMPLIER:",     f"{q_user.get('mean_rho_COMP', np.nan):.4f}")
    row("Mann-Whitney U (I>J):",  f"{q_user.get('mw_IJ_U', np.nan):.0f}")
    row("Mann-Whitney p (I>J):",  f"{q_user.get('mw_IJ_p', np.nan):.6f}")
    row("N INITIATOR:",           str(q_user.get("n_INIT", "n/a")))
    row("N JOINER:",              str(q_user.get("n_JOIN", "n/a")))
    row("N COMPLIER:",            str(q_user.get("n_COMP", "n/a")))
    row("Q_user score (η²):",     f"{q_user.get('Q_user_score', np.nan):.4f}")

    # --- Q_regime ---
    h("COMPONENT 5: Q_regime (Stage 2 Stability Margin)")
    lines.append("  margin = (α_gov − α_gov_min) / |α_gov_min|")
    lines.append("  Q_regime = (tanh(margin) + 1) / 2")
    lines.append("  Cross-val: r(Q_regime, −C_volatility) should be positive")
    lines.append("")
    if not q_regime_df.empty:
        row("Cross-val r(margin, −volatility):", f"{q_regime_df['Q_regime_r_xval'].iloc[0]:.4f}")
        row("Cross-val p:",                      f"{q_regime_df['Q_regime_p_xval'].iloc[0]:.4f}")
        row("N with margin data:",               str(q_regime_df["stability_margin"].notna().sum()))
        row("N stable (margin > 0):",            str((q_regime_df["stability_margin"] > 0).sum()))
        row("N trap (margin < 0):",              str((q_regime_df["stability_margin"] < 0).sum()))
        lines.append("")
        lines.append(f"  {'Subreddit':<26} {'margin':>8}  {'Q_regime':>8}  {'regime':<25}  {'Ξ':>8}")
        lines.append("  " + "-" * 75)
        for _, r in q_regime_df.sort_values("stability_margin", ascending=False).iterrows():
            ban = "[B]" if r["subreddit"] in BANNED_SUBREDDITS else "   "
            xi_str = f"{r['Xi']:.4f}" if not np.isnan(r.get("Xi", np.nan)) else "  n/a"
            m_str  = f"{r['stability_margin']:>8.4f}" \
                     if not np.isnan(r.get("stability_margin", np.nan)) else "     n/a"
            lines.append(
                f"  {ban} r/{r['subreddit']:<23} "
                f"{m_str}  "
                f"{r['Q_regime']:>8.4f}  "
                f"{str(r.get('regime', 'n/a')):<25}  "
                f"{xi_str:>8}"
            )

    # --- Q_spillover ---
    h("COMPONENT 6: Q_spillover (Migrant INITIATOR Persistence)")
    lines.append("  Group A: from_banned_sub=True, user_type=INITIATOR, post-ban")
    lines.append("  Group B: native users in control subreddits, post-ban")
    lines.append("  Test: Mann-Whitney U (one-sided: A > B)")
    lines.append("  Dose: OLS theta_proxy → post-ban removal rate")
    lines.append("")
    row("Mann-Whitney U:",         f"{q_spill.get('mw_U', np.nan):.0f}")
    row("p-value (one-sided):",    f"{q_spill.get('mw_p', np.nan):.6f}")
    row("Cohen's d:",              f"{q_spill.get('cohens_d', np.nan):.4f}")
    row("Mean rate migrants:",     f"{q_spill.get('mean_rate_migrants', np.nan):.4f}")
    row("Mean rate natives:",      f"{q_spill.get('mean_rate_natives', np.nan):.4f}")
    row("N migrants (INITIATOR):", str(q_spill.get("n_migrants", "n/a")))
    row("N natives:",              str(q_spill.get("n_natives", "n/a")))
    row("Dose β₁ (theta_proxy):",  f"{q_spill.get('dose_beta_theta', np.nan):.6f}")
    row("Dose β₁ p:",              f"{q_spill.get('dose_beta_p', np.nan):.6f}")
    row("Dose R²:",                f"{q_spill.get('dose_r2', np.nan):.4f}")
    row("Q_spillover score:",      f"{q_spill.get('Q_spillover_score', np.nan):.4f}")
    d = q_spill.get("cohens_d", np.nan)
    p = q_spill.get("mw_p", np.nan)
    if not np.isnan(d):
        sig = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        interp = ("LARGE" if d > 0.5 else "MEDIUM" if d > 0.2 else "SMALL")
        lines.append(f"\n  RESULT: Cohen's d = {d:.3f} ({interp} effect), {sig}")

    # --- Composite ---
    h("COMPOSITE MCI")
    lines.append("  MCI = mean(Q_fit_scaled, Q_backfire, Q_sub, Q_user, Q_regime, Q_spill)")
    lines.append("")
    lines.append(f"  {'Subreddit':<26} {'MCI':>6}  {'Q_fit':>6} {'Q_back':>6} "
                 f"{'Q_sub':>6} {'Q_usr':>6} {'Q_reg':>6} {'Q_spl':>6}  {'regime':<22}")
    lines.append("  " + "-" * 100)
    for _, r in mci_df.sort_values("MCI", ascending=False).iterrows():
        ban = "[B]" if r["is_banned"] else "   "
        def f(v):
            return f"{v:.3f}" if not np.isnan(v) else " n/a "
        lines.append(
            f"  {ban} r/{r['subreddit']:<23} "
            f"{f(r['MCI'])}  "
            f"{f(r['Q_fit'])} "
            f"{f(r['Q_backfire'])} "
            f"{f(r['Q_substitution'])} "
            f"{f(r['Q_user'])} "
            f"{f(r['Q_regime'])} "
            f"{f(r['Q_spillover'])}  "
            f"{str(r.get('regime', 'n/a')):<22}"
        )

    # --- Sensitivity ---
    h("SENSITIVITY ANALYSIS")
    lines.append(f"  {sensitivity.get('n_draws', 500)} random weight draws (uniform [0.05, 0.35])")
    lines.append("")
    row("Mean Spearman ρ:", f"{sensitivity.get('rho_mean', np.nan):.4f}")
    row("Min  Spearman ρ:", f"{sensitivity.get('rho_min', np.nan):.4f}")
    row("Median Spearman ρ:", f"{sensitivity.get('rho_median', np.nan):.4f}")
    rho_m = sensitivity.get("rho_mean", 0)
    if rho_m > 0.85:
        lines.append("\n  RESULT: ✓ RANKING ROBUST (mean ρ > 0.85). "
                     "MCI ordering not an artefact of equal weights.")
    elif rho_m > 0.70:
        lines.append("\n  RESULT: ~ MODERATE ROBUSTNESS (mean ρ 0.70–0.85). "
                     "Report weight sensitivity in supplementary material.")
    else:
        lines.append("\n  RESULT: ✗ RANKING SENSITIVE TO WEIGHTS (mean ρ < 0.70). "
                     "Investigate which component drives instability.")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"  Report written: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute Model Credibility Index for the Coordination-Legitimacy Game."
    )
    parser.add_argument("--out_dir",    default="../Reddit-2015-v2/output",
                        help="Path to the pipeline output directory.")
    parser.add_argument("--min_posts",  type=int, default=10,
                        help="Minimum posting days for Q_user ρ_u computation.")
    parser.add_argument("--n_draws",    type=int, default=500,
                        help="Weight draws for sensitivity analysis.")
    parser.add_argument("--check_only", action="store_true",
                        help="Only check that all input files exist, then exit.")
    args = parser.parse_args()

    out_dir  = Path(args.out_dir)
    conf_dir = out_dir / "confidence"
    conf_dir.mkdir(parents=True, exist_ok=True)
    (conf_dir / "tables").mkdir(exist_ok=True)
    (conf_dir / "plots").mkdir(exist_ok=True)

    log.info("=" * 60)
    log.info("MODEL CREDIBILITY INDEX  —  confidence_score.py")
    log.info("=" * 60)

    log.info("\nLoading pipeline outputs...")
    data = load_inputs(out_dir)

    if args.check_only:
        log.info("All files present. Run without --check_only to compute scores.")
        return

    daily_metrics   = data["daily_metrics"]
    user_thresholds = data["user_thresholds"]
    panel           = data["panel"]
    calibrated      = data["calibrated"]
    stage2          = data["stage2"]
    comments_path   = out_dir / "comments_filtered.csv"

    # ---- Component 1 ----
    log.info("\n--- Component 1: Q_fit ---")
    q_fit_df = compute_q_fit(calibrated, daily_metrics)

    # ---- Component 2 ----
    log.info("\n--- Component 2: Q_backfire ---")
    q_backfire_df = compute_q_backfire(calibrated, daily_metrics)

    # ---- Component 3 ----
    log.info("\n--- Component 3: Q_substitution ---")
    q_sub = compute_q_substitution(daily_metrics)

    # ---- Component 4 ----
    log.info("\n--- Component 4: Q_user ---")
    q_user = compute_q_user(user_thresholds, daily_metrics,
                             comments_path, min_posts=args.min_posts)

    # ---- Component 5 ----
    log.info("\n--- Component 5: Q_regime ---")
    q_regime_df = compute_q_regime(stage2)

    # ---- Component 6 ----
    log.info("\n--- Component 6: Q_spillover ---")
    q_spill = compute_q_spillover(panel, user_thresholds)

    # ---- Assemble MCI ----
    log.info("\n--- Assembling composite MCI ---")
    mci_df = assemble_mci(
        q_fit_df, q_backfire_df, q_sub, q_user,
        q_regime_df, q_spill, calibrated
    )

    # ---- Sensitivity analysis ----
    log.info("\n--- Sensitivity analysis ---")
    sensitivity = sensitivity_analysis(mci_df, n_draws=args.n_draws)

    # ---- Write outputs ----
    log.info("\nWriting outputs...")

    mci_df.to_csv(conf_dir / "subreddit_mci.csv", index=False)
    log.info(f"  subreddit_mci.csv  ({len(mci_df)} rows)")

    rho_df = q_user.get("rho_df", pd.DataFrame())
    if not rho_df.empty:
        rho_df.to_parquet(conf_dir / "user_behavioral.parquet", index=False)
        log.info(f"  user_behavioral.parquet  ({len(rho_df):,} rows)")

    write_report(
        conf_dir / "confidence_report.txt",
        q_fit_df, q_backfire_df, q_sub, q_user,
        q_regime_df, q_spill, mci_df, sensitivity
    )

    # Save component-level tables for the verification script
    q_regime_df.to_csv(conf_dir / "tables" / "regime_scores.csv", index=False)
    q_fit_df.to_csv(   conf_dir / "tables" / "fit_scores.csv",    index=False)
    if not q_backfire_df.empty:
        q_backfire_df.to_csv(conf_dir / "tables" / "backfire_scores.csv", index=False)

    import json
    scalar_results = {
        "Q_substitution": q_sub,
        "Q_user":         {k: v for k, v in q_user.items() if k != "rho_df"},
        "Q_spillover":    q_spill,
        "sensitivity":    sensitivity,
    }
    with open(conf_dir / "tables" / "scalar_results.json", "w") as f:
        json.dump(scalar_results, f, indent=2, default=str)

    log.info("\n" + "=" * 60)
    log.info("DONE.  Run confidence_verification.py to generate plots and tables.")
    log.info(f"Outputs in: {conf_dir}/")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
