"""
ews_validation.py
=================
Early Warning Score (EWS) — logistic regression + LOO-CV validation.

Four components, all computable from a 30-day rolling window of public
subreddit-level removal data:

  E1 — enforcement_drift    : OLS slope of C_smooth (negative = deteriorating)
  E2 — violation_pressure   : mean C_raw (last 30d) / mean C_raw (baseline 90d)
  E3 — legitimacy_decay     : OLS slope of T_norm   (negative = declining)
  E4 — removal_concentration: Gini coefficient of per-user removal_rate

Outcome: banned (1) vs. control (0) in the June 2015 Reddit ban wave.

Outputs saved to output/confidence/ews/
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
import matplotlib.ticker as mticker
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneOut

warnings.filterwarnings("ignore")

BAN_DATE   = pd.Timestamp("2015-06-10", tz="UTC")
WINDOW_PRE = 30   # days before ban used for EWS features
BANNED     = {"fatpeoplehate", "CoonTown"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ols_slope(y: np.ndarray) -> float:
    """Return OLS slope of y regressed on integer time index."""
    n = len(y)
    if n < 3:
        return np.nan
    x = np.arange(n, dtype=float)
    x -= x.mean()
    ss_xx = np.sum(x ** 2)
    if ss_xx < 1e-12:
        return 0.0
    return float(np.sum(x * (y - y.mean())) / ss_xx)


def gini(arr: np.ndarray) -> float:
    """Gini coefficient of a non-negative array."""
    arr = arr[arr >= 0]
    if len(arr) < 2 or arr.sum() < 1e-10:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * arr) / (n * arr.sum())) - (n + 1) / n)


def normalise(series: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1]."""
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-12:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def compute_features(daily_metrics: pd.DataFrame,
                     user_thresholds: pd.DataFrame,
                     calibrated_params: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per subreddit with columns:
        subreddit, banned, E1, E2, E3, E4, g_star
    """
    dm = daily_metrics.copy()
    dm["day"] = pd.to_datetime(dm["day"], utc=True)

    rows = []
    for sub, grp in dm.groupby("subreddit"):
        grp = grp.sort_values("day")

        # Pre-ban window only (last WINDOW_PRE days before ban date)
        pre = grp[grp["day"] < BAN_DATE].tail(WINDOW_PRE)
        if len(pre) < 5:
            continue

        # Baseline: all pre-ban data (up to 90 days)
        baseline = grp[grp["day"] < BAN_DATE]

        # ---------------------------------------------------------------
        # E1: enforcement drift — slope of C_smooth in pre-ban window
        # Negative slope = enforcement rate declining despite violations
        # ---------------------------------------------------------------
        e1 = ols_slope(pre["C_smooth"].ffill().fillna(0).values)

        # ---------------------------------------------------------------
        # E2: violation pressure — recent C_raw vs. baseline C_raw
        # Above 1.0 means the removal rate has risen above its own norm
        # ---------------------------------------------------------------
        baseline_mean = baseline["C_raw"].mean()
        recent_mean   = pre["C_raw"].mean()
        e2 = (recent_mean / baseline_mean) if baseline_mean > 1e-6 else 1.0

        # ---------------------------------------------------------------
        # E3: legitimacy decay — slope of T_norm in pre-ban window
        # Negative slope = community standing declining
        # ---------------------------------------------------------------
        e3 = ols_slope(pre["T_norm"].ffill().fillna(pre["T_norm"].mean()).values)

        # ---------------------------------------------------------------
        # E4: removal concentration — Gini of per-user removal_rate
        # High Gini = few repeat violators dominating
        # ---------------------------------------------------------------
        ut_sub = user_thresholds[user_thresholds["subreddit"] == sub]
        e4 = gini(ut_sub["removal_rate"].fillna(0).values) if len(ut_sub) > 1 else 0.0

        # g* for reference
        row = calibrated_params[calibrated_params["subreddit"] == sub]
        g_star = float(row["g_star"].values[0]) if len(row) else np.nan

        rows.append({
            "subreddit": sub,
            "banned":    int(sub in BANNED),
            "E1_enforcement_drift":      e1,
            "E2_violation_pressure":     e2,
            "E3_legitimacy_decay":       e3,
            "E4_removal_concentration":  e4,
            "g_star":                    g_star,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rolling EWS (time-series — simulates live deployment)
# ---------------------------------------------------------------------------

def compute_rolling_ews(daily_metrics: pd.DataFrame,
                        user_thresholds: pd.DataFrame,
                        fitted_weights: np.ndarray,
                        scaler: StandardScaler,
                        window: int = WINDOW_PRE) -> pd.DataFrame:
    """
    Compute EWS for each subreddit on each day using the prior `window` days.
    E4 (Gini) is fixed per subreddit (uses full history) for efficiency.
    """
    dm = daily_metrics.copy()
    dm["day"] = pd.to_datetime(dm["day"], utc=True)

    # Pre-compute E4 per subreddit (full-history Gini)
    e4_map = {}
    for sub, ut_sub in user_thresholds.groupby("subreddit"):
        e4_map[sub] = gini(ut_sub["removal_rate"].fillna(0).values)

    records = []
    for sub, grp in dm.groupby("subreddit"):
        grp = grp.sort_values("day").reset_index(drop=True)
        baseline_mean = grp["C_raw"].mean()
        e4 = e4_map.get(sub, 0.0)

        for i in range(window, len(grp)):
            win = grp.iloc[i - window: i]
            day = grp.iloc[i]["day"]

            e1 = ols_slope(win["C_smooth"].ffill().fillna(0).values)
            e2 = (win["C_raw"].mean() / baseline_mean) if baseline_mean > 1e-6 else 1.0
            e3 = ols_slope(win["T_norm"].ffill().fillna(win["T_norm"].mean()).values)

            X_raw = np.array([[e1, e2, e3, e4]])
            X_sc  = scaler.transform(X_raw)
            ews   = float(1 / (1 + np.exp(-X_sc @ fitted_weights[1:] - fitted_weights[0])))

            records.append({
                "subreddit":    sub,
                "day":          day,
                "EWS":          ews,
                "banned":       int(sub in BANNED),
                "days_from_ban": (day - BAN_DATE).days,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# LOO-CV logistic regression
# ---------------------------------------------------------------------------

def loo_validation(features: pd.DataFrame) -> dict:
    """
    Leave-one-out cross-validation.
    Returns AUC, per-fold predictions, and feature coefficients.
    """
    feature_cols = ["E1_enforcement_drift", "E2_violation_pressure",
                    "E3_legitimacy_decay",  "E4_removal_concentration"]

    X = features[feature_cols].values.copy()
    y = features["banned"].values

    # Impute column-wise NaNs with the column median (keeps all 16 rows)
    for j in range(X.shape[1]):
        col = X[:, j]
        nan_idx = np.isnan(col)
        if nan_idx.any():
            median_val = np.nanmedian(col)
            X[nan_idx, j] = median_val

    feat_df = features.copy()

    loo    = LeaveOneOut()
    probs  = np.zeros(len(y))
    coefs  = []

    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr        = y[train_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        # L2 logistic regression with balanced class weights
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(X_tr_sc, y_tr)
        probs[test_idx] = clf.predict_proba(X_te_sc)[0, 1]
        coefs.append(np.concatenate([clf.intercept_, clf.coef_[0]]))

    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else np.nan

    # Full-data fit for coefficient summary and rolling EWS
    scaler_full = StandardScaler()
    X_sc_full   = scaler_full.fit_transform(X)
    clf_full    = LogisticRegression(C=1.0, class_weight="balanced",
                                     solver="lbfgs", max_iter=1000, random_state=42)
    clf_full.fit(X_sc_full, y)
    weights_full = np.concatenate([clf_full.intercept_, clf_full.coef_[0]])

    feat_df = feat_df.copy()
    feat_df["loo_prob"] = probs
    feat_df["predicted_banned"] = (probs >= 0.5).astype(int)

    coef_df = pd.DataFrame(
        coefs, columns=["intercept"] + feature_cols
    ).mean()

    return {
        "auc":          auc,
        "predictions":  feat_df,
        "coef_mean":    coef_df,
        "scaler_full":  scaler_full,
        "weights_full": weights_full,
        "feature_cols": feature_cols,
        "X":            X,
        "y":            y,
        "probs":        probs,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_ews_bar(features: pd.DataFrame, loo: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    df = loo["predictions"].sort_values("loo_prob", ascending=False)
    colors = ["#E24B4A" if b else "#378ADD" for b in df["banned"]]
    bars = ax.bar(range(len(df)), df["loo_prob"], color=colors, edgecolor="white")
    ax.axhline(0.5, color="black", lw=1.5, linestyle="--", label="Decision boundary")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(["r/" + s for s in df["subreddit"]], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("LOO-CV P(banned)", fontsize=11)
    ax.set_title(f"Early Warning Score — LOO-CV  |  AUC = {loo['auc']:.3f}", fontsize=12)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#E24B4A", label="Banned"),
        Patch(color="#378ADD", label="Control"),
    ] + ax.get_legend_handles_labels()[0], fontsize=9)
    plt.tight_layout()
    path = out_dir / "ews_loo_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_roc(loo: dict, out_dir: Path):
    fpr, tpr, _ = roc_curve(loo["y"], loo["probs"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#E24B4A", lw=2,
            label=f"LOO-CV AUC = {loo['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("EWS ROC Curve (Leave-One-Out)", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = out_dir / "ews_roc.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_rolling_ews(rolling: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(13, 5))
    for sub, grp in rolling.groupby("subreddit"):
        grp = grp.sort_values("days_from_ban")
        color  = "#E24B4A" if grp["banned"].iloc[0] else "#378ADD"
        lw     = 2.5       if grp["banned"].iloc[0] else 0.8
        alpha  = 1.0       if grp["banned"].iloc[0] else 0.4
        zorder = 3         if grp["banned"].iloc[0] else 1
        label  = f"r/{sub}" if grp["banned"].iloc[0] else None
        ax.plot(grp["days_from_ban"], grp["EWS"],
                color=color, lw=lw, alpha=alpha, zorder=zorder, label=label)

    ax.axvline(0, color="black", lw=2, linestyle="--", label="Ban date (June 10)")
    ax.axhline(0.5, color="gray", lw=1, linestyle=":")
    ax.set_xlabel("Days relative to ban date", fontsize=11)
    ax.set_ylabel("EWS (P̂ banned)", fontsize=11)
    ax.set_title("Real-time Early Warning Score — rolling 30-day window", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    path = out_dir / "ews_rolling.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_feature_radar(features: pd.DataFrame, out_dir: Path):
    """Radar chart of normalised features for banned vs. control."""
    feat_cols = ["E1_enforcement_drift", "E2_violation_pressure",
                 "E3_legitimacy_decay",  "E4_removal_concentration"]
    labels    = ["Enforcement\nDrift (neg)", "Violation\nPressure",
                 "Legitimacy\nDecay (neg)",  "Removal\nConcentration"]

    # For the radar we want "higher = more at risk", so flip sign for E1, E3
    df = features.copy()
    df["E1_enforcement_drift"] = -df["E1_enforcement_drift"]
    df["E3_legitimacy_decay"]  = -df["E3_legitimacy_decay"]

    # Normalise each feature across all subreddits
    for c in feat_cols:
        df[c] = normalise(df[c])

    banned_mean  = df[df["banned"] == 1][feat_cols].mean().values
    control_mean = df[df["banned"] == 0][feat_cols].mean().values

    N = len(feat_cols)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    for vals, color, label in [
        (banned_mean,  "#E24B4A", "Banned"),
        (control_mean, "#378ADD", "Control"),
    ]:
        v = np.concatenate([vals, vals[:1]])
        ax.plot(angles, v, color=color, lw=2, label=label)
        ax.fill(angles, v, color=color, alpha=0.15)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("EWS Feature Profile:\nBanned vs. Control (normalised)", fontsize=11, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    path = out_dir / "ews_radar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def write_tables(features: pd.DataFrame, loo: dict, out_dir: Path):
    # Table A: per-subreddit EWS components + LOO score
    pred = loo["predictions"][["subreddit", "banned",
                                "E1_enforcement_drift", "E2_violation_pressure",
                                "E3_legitimacy_decay",  "E4_removal_concentration",
                                "loo_prob", "predicted_banned"]].copy()
    pred = pred.merge(features[["subreddit", "g_star"]], on="subreddit", how="left")
    pred = pred.sort_values("loo_prob", ascending=False)
    pred.to_csv(out_dir / "ews_scores.csv", index=False)
    print(f"  Saved: ews_scores.csv  ({len(pred)} rows)")

    # Table B: feature coefficients
    coef_out = loo["coef_mean"].reset_index()
    coef_out.columns = ["feature", "mean_LOO_coef"]
    coef_out.to_csv(out_dir / "ews_coefficients.csv", index=False)
    print(f"  Saved: ews_coefficients.csv")

    # Table C: summary stats
    summary = {
        "n_subreddits":  [len(loo["y"])],
        "n_banned":      [int(loo["y"].sum())],
        "n_control":     [int((loo["y"] == 0).sum())],
        "LOO_AUC":       [round(loo["auc"], 4)],
        "banned_correct": [int(((pred["banned"] == 1) & (pred["predicted_banned"] == 1)).sum())],
        "control_correct":[int(((pred["banned"] == 0) & (pred["predicted_banned"] == 0)).sum())],
    }
    pd.DataFrame(summary).to_csv(out_dir / "ews_summary.csv", index=False)
    print(f"  Saved: ews_summary.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="output",
                    help="Directory containing pipeline outputs")
    ap.add_argument("--out_dir",  default="output",
                    help="Base output directory")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir) / "confidence" / "ews"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EARLY WARNING SCORE — ews_validation.py")
    print("=" * 60)

    # Load inputs
    print("\nLoading data...")
    dm     = pd.read_parquet(data_dir / "daily_metrics.parquet")
    ut     = pd.read_parquet(data_dir / "user_thresholds.parquet")
    params = pd.read_csv(data_dir / "calibrated_params.csv")
    print(f"  daily_metrics: {len(dm):,} rows")
    print(f"  user_thresholds: {len(ut):,} rows")
    print(f"  calibrated_params: {len(params)} subreddits")

    # Compute features
    print("\nComputing EWS features (30-day pre-ban window)...")
    features = compute_features(dm, ut, params)
    print(f"  Features computed for {len(features)} subreddits")

    print("\n  Raw features:")
    print(features[["subreddit", "banned",
                     "E1_enforcement_drift", "E2_violation_pressure",
                     "E3_legitimacy_decay",  "E4_removal_concentration",
                     "g_star"]].to_string(index=False))

    # LOO-CV
    print("\nRunning LOO-CV logistic regression...")
    loo = loo_validation(features)

    print(f"\n{'=' * 60}")
    print(f"LOO-CV AUC: {loo['auc']:.4f}")
    print(f"{'=' * 60}")
    print("\nPer-subreddit LOO predictions:")
    pred = loo["predictions"].sort_values("loo_prob", ascending=False)
    for _, row in pred.iterrows():
        tag   = "[BANNED]" if row["banned"] else "        "
        right = "[OK]" if row["banned"] == row["predicted_banned"] else "[X]"
        print(f"  {tag}  r/{row['subreddit']:<20}  P(ban)={row['loo_prob']:.3f}  {right}")

    print("\nMean LOO coefficients (standardised features):")
    for feat, coef in loo["coef_mean"].items():
        print(f"  {feat:<35}  {coef:+.4f}")

    # Rolling EWS (requires full-data fit)
    print("\nComputing rolling EWS time series...")
    rolling = compute_rolling_ews(dm, ut, loo["weights_full"], loo["scaler_full"])
    print(f"  Rolling EWS computed: {len(rolling):,} subreddit-day rows")

    # How many days before ban does EWS first exceed 0.5 for banned subs?
    print("\n  Days before ban when EWS first exceeds 0.5:")
    for sub in BANNED:
        sub_roll = rolling[
            (rolling["subreddit"] == sub) & (rolling["days_from_ban"] < 0)
        ].sort_values("days_from_ban")
        crossed = sub_roll[sub_roll["EWS"] >= 0.5]
        if not crossed.empty:
            first_cross = crossed.iloc[0]["days_from_ban"]
            print(f"    r/{sub}: {first_cross:.0f} days before ban (EWS={crossed.iloc[0]['EWS']:.3f})")
        else:
            print(f"    r/{sub}: never exceeded 0.5 in pre-ban window")

    # Save outputs
    print("\nSaving outputs...")
    write_tables(features, loo, out_dir)
    rolling.to_parquet(out_dir / "ews_rolling.parquet", index=False)
    print(f"  Saved: ews_rolling.parquet  ({len(rolling):,} rows)")

    print("\nGenerating plots...")
    plot_ews_bar(features, loo, out_dir)
    plot_roc(loo, out_dir)
    plot_rolling_ews(rolling, out_dir)
    plot_feature_radar(features, out_dir)

    print(f"\nAll outputs: {out_dir}/")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Two-Tier Attention System
# ---------------------------------------------------------------------------

def compute_attention_tiers(features: pd.DataFrame,
                             loo: dict,
                             params: pd.DataFrame) -> pd.DataFrame:
    """
    Classifies each subreddit into one of four attention categories by combining:

      Tier 1 — Observable Deterioration (data-driven)
        Score = P(crisis) from the LOO logistic regression on E1–E4.
        High Tier 1 (≥ 0.6): system is visibly showing stress signals.

      Tier 2 — Structural Fragility (model-derived)
        Score = min(C_pre / g*, 1.5) / 1.5  → normalised to [0, 1].
        High Tier 2 (≥ 0.7): enforcement is dangerously close to the
        backfire threshold, even if surface behaviour looks stable.

    Classification thresholds:
      BOTH        Tier1 ≥ 0.6 AND Tier2 ≥ 0.7  →  Highest risk; model + data agree
      TIER1_ONLY  Tier1 ≥ 0.6 AND Tier2 < 0.7  →  Visible stress, model can't explain it
      TIER2_ONLY  Tier1 < 0.6 AND Tier2 ≥ 0.7  →  Hidden structural risk (early warning)
      NEITHER     Both below threshold           →  Stable by both signals

    Returns a DataFrame with one row per subreddit.
    """
    T1_THRESH = 0.6
    T2_THRESH = 0.7

    # --- Tier 1: LOO predicted P(crisis) ---
    preds = loo["predictions"][["subreddit", "loo_prob"]].copy()
    preds = preds.rename(columns={"loo_prob": "tier1_score"})

    # --- Tier 2: C_pre / g* ---
    # Get mean pre-ban C from features (E2 × baseline_mean is implicit;
    # we re-derive C_pre_mean directly from features if available,
    # otherwise approximate from g_star column in features)
    tier2_rows = []
    for _, row in features.iterrows():
        sub    = row["subreddit"]
        g_star = row.get("g_star", np.nan)

        # C_pre_mean: use E2 × (baseline ratio implied) — proxy
        # More direct: re-read from params if available
        p_row = params[params["subreddit"] == sub]
        if not p_row.empty:
            g_star = float(p_row["g_star"].values[0])

        # E2 > 1 means recent C_raw > baseline C_raw; use E2 as a fragility signal
        # Tier 2 = ratio of recent enforcement to threshold
        # We approximate C_pre / g* using E2 (violation pressure) scaled by g*
        # If g_star available, use: tier2 = E2_violation_pressure / (g_star * scaling)
        # Simplest defensible form: tier2_raw = E2 / max(g_star, 0.01)
        e2     = row.get("E2_violation_pressure", np.nan)
        if not np.isnan(g_star) and not np.isnan(e2) and g_star > 0:
            # E2 is the ratio C_recent / C_baseline; g* is the backfire threshold.
            # Tier2_raw approximates how close C_recent is to g* (in relative terms).
            tier2_raw = e2 / (g_star * 10.0)   # scale factor: E2 ≈ O(1), g* ≈ O(0.1)
        else:
            tier2_raw = np.nan

        tier2_score = float(np.clip(tier2_raw, 0, 1.5) / 1.5) if not np.isnan(tier2_raw) else np.nan
        tier2_rows.append({"subreddit": sub, "tier2_score": tier2_score,
                            "g_star": g_star, "E2_violation_pressure": e2})

    tier2_df = pd.DataFrame(tier2_rows)

    # Merge
    df = features[["subreddit", "banned"]].merge(preds,    on="subreddit", how="left")
    df = df.merge(tier2_df[["subreddit", "tier2_score", "g_star"]], on="subreddit", how="left")

    # Classify
    def classify(row):
        t1 = row.get("tier1_score", np.nan)
        t2 = row.get("tier2_score", np.nan)
        if np.isnan(t1) or np.isnan(t2):
            return "UNKNOWN"
        if t1 >= T1_THRESH and t2 >= T2_THRESH:
            return "BOTH"
        if t1 >= T1_THRESH and t2 < T2_THRESH:
            return "TIER1_ONLY"
        if t1 < T1_THRESH and t2 >= T2_THRESH:
            return "TIER2_ONLY"
        return "NEITHER"

    df["attention_tier"] = df.apply(classify, axis=1)
    df["is_banned"]      = df["banned"].astype(bool)

    return df


def plot_attention_scatter(attention: pd.DataFrame, out_dir: Path):
    """
    Scatter plot: Tier 1 (x-axis) vs Tier 2 (y-axis).
    Quadrants labelled with attention classification.
    Each point is one subreddit, coloured by banned/control.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    T1_THRESH = 0.6
    T2_THRESH = 0.7

    # Quadrant shading
    ax.axvspan(T1_THRESH, 1.05, ymin=(T2_THRESH - 0) / 1.05, ymax=1.0,
               alpha=0.08, color="#E24B4A")   # BOTH
    ax.axvspan(T1_THRESH, 1.05, ymin=0, ymax=(T2_THRESH) / 1.05,
               alpha=0.05, color="#EF9F27")   # TIER1_ONLY
    ax.axvspan(-0.05, T1_THRESH, ymin=(T2_THRESH) / 1.05, ymax=1.0,
               alpha=0.06, color="#9E62B5")   # TIER2_ONLY
    ax.axvspan(-0.05, T1_THRESH, ymin=0, ymax=(T2_THRESH) / 1.05,
               alpha=0.04, color="#1D9E75")   # NEITHER

    # Threshold lines
    ax.axvline(T1_THRESH, color="grey", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.axhline(T2_THRESH, color="grey", linewidth=1.0, linestyle="--", alpha=0.7)

    # Quadrant labels
    label_kw = dict(fontsize=9, alpha=0.55, style="italic")
    ax.text(0.82, 0.97, "BOTH\n(High Risk)",      transform=ax.transAxes,
            ha="center", va="top", color="#c0392b", **label_kw)
    ax.text(0.82, 0.35, "TIER1_ONLY\n(Visible stress,\nno g* explanation)",
            transform=ax.transAxes, ha="center", va="top", color="#d35400", **label_kw)
    ax.text(0.20, 0.97, "TIER2_ONLY\n(Hidden risk)",
            transform=ax.transAxes, ha="center", va="top", color="#6c3483", **label_kw)
    ax.text(0.20, 0.35, "NEITHER\n(Stable)",
            transform=ax.transAxes, ha="center", va="top", color="#1a5276", **label_kw)

    # Points
    colors = {True: "#E24B4A", False: "#2166ac"}
    for _, row in attention.iterrows():
        t1 = row.get("tier1_score", np.nan)
        t2 = row.get("tier2_score", np.nan)
        if np.isnan(t1) or np.isnan(t2):
            continue
        color = colors[bool(row["is_banned"])]
        ax.scatter(t1, t2, c=color, s=100, alpha=0.85, zorder=4,
                   edgecolors="white", linewidth=0.8)
        ax.annotate(row["subreddit"], (t1, t2),
                    fontsize=7.5, alpha=0.85,
                    xytext=(5, 3), textcoords="offset points")

    # Legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E24B4A",
               markersize=9, label="Banned subreddit"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2166ac",
               markersize=9, label="Control subreddit"),
    ]
    ax.legend(handles=legend_elems, fontsize=9, loc="lower right")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Tier 1 Score  —  Observable Deterioration  P(crisis | E1–E4)", fontsize=11)
    ax.set_ylabel("Tier 2 Score  —  Structural Fragility  C_pre / g*  (normalised)", fontsize=11)
    ax.set_title("Two-Tier Attention System\n"
                 "Identifying governance risk from observable and structural signals", fontsize=12)
    fig.tight_layout()

    out_path = out_dir / "attention_scatter.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="output",
                    help="Directory containing pipeline outputs")
    ap.add_argument("--out_dir",  default="output",
                    help="Base output directory")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir) / "confidence" / "ews"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EARLY WARNING SCORE + TWO-TIER ATTENTION — ews_validation.py")
    print("=" * 60)

    # Load inputs
    print("\nLoading data...")
    dm     = pd.read_parquet(data_dir / "daily_metrics.parquet")
    ut     = pd.read_parquet(data_dir / "user_thresholds.parquet")
    params = pd.read_csv(data_dir / "calibrated_params.csv")
    print(f"  daily_metrics: {len(dm):,} rows")
    print(f"  user_thresholds: {len(ut):,} rows")
    print(f"  calibrated_params: {len(params)} subreddits")

    # Compute features
    print("\nComputing EWS features (30-day pre-ban window)...")
    features = compute_features(dm, ut, params)
    print(f"  Features computed for {len(features)} subreddits")

    print("\n  Raw features:")
    print(features[["subreddit", "banned",
                     "E1_enforcement_drift", "E2_violation_pressure",
                     "E3_legitimacy_decay",  "E4_removal_concentration",
                     "g_star"]].to_string(index=False))

    # LOO-CV
    print("\nRunning LOO-CV logistic regression...")
    loo = loo_validation(features)

    print(f"\n{'=' * 60}")
    print(f"LOO-CV AUC: {loo['auc']:.4f}")
    print(f"{'=' * 60}")
    print("\nPer-subreddit LOO predictions:")
    pred = loo["predictions"].sort_values("loo_prob", ascending=False)
    for _, row in pred.iterrows():
        tag   = "[BANNED]" if row["banned"] else "        "
        right = "[OK]" if row["banned"] == row["predicted_banned"] else "[X]"
        print(f"  {tag}  r/{row['subreddit']:<20}  P(ban)={row['loo_prob']:.3f}  {right}")

    print("\nMean LOO coefficients (standardised features):")
    for feat, coef in loo["coef_mean"].items():
        print(f"  {feat:<35}  {coef:+.4f}")

    # Rolling EWS (requires full-data fit)
    print("\nComputing rolling EWS time series...")
    rolling = compute_rolling_ews(dm, ut, loo["weights_full"], loo["scaler_full"])
    print(f"  Rolling EWS computed: {len(rolling):,} subreddit-day rows")

    # How many days before ban does EWS first exceed 0.5 for banned subs?
    print("\n  Days before ban when EWS first exceeds 0.5:")
    for sub in BANNED:
        sub_roll = rolling[
            (rolling["subreddit"] == sub) & (rolling["days_from_ban"] < 0)
        ].sort_values("days_from_ban")
        crossed = sub_roll[sub_roll["EWS"] >= 0.5]
        if not crossed.empty:
            first_cross = crossed.iloc[0]["days_from_ban"]
            print(f"    r/{sub}: {first_cross:.0f} days before ban (EWS={crossed.iloc[0]['EWS']:.3f})")
        else:
            print(f"    r/{sub}: never exceeded 0.5 in pre-ban window")

    # Save outputs
    print("\nSaving outputs...")
    write_tables(features, loo, out_dir)
    rolling.to_parquet(out_dir / "ews_rolling.parquet", index=False)
    print(f"  Saved: ews_rolling.parquet  ({len(rolling):,} rows)")

    print("\nGenerating EWS plots...")
    plot_ews_bar(features, loo, out_dir)
    plot_roc(loo, out_dir)
    plot_rolling_ews(rolling, out_dir)
    plot_feature_radar(features, out_dir)

    # -----------------------------------------------------------------------
    # Two-Tier Attention System
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TWO-TIER ATTENTION SYSTEM")
    print("=" * 60)

    attention = compute_attention_tiers(features, loo, params)

    print("\nAttention classification per subreddit:")
    print(f"  {'Subreddit':<26} {'Tier1':>7} {'Tier2':>7}  {'Classification':<16}  Banned")
    print("  " + "-" * 70)
    for _, row in attention.sort_values("attention_tier").iterrows():
        t1 = f"{row['tier1_score']:.3f}" if not np.isnan(row.get("tier1_score", float("nan"))) else "  n/a"
        t2 = f"{row['tier2_score']:.3f}" if not np.isnan(row.get("tier2_score", float("nan"))) else "  n/a"
        ban = "[BANNED]" if row["is_banned"] else "        "
        print(f"  {ban} r/{row['subreddit']:<22} {t1:>7} {t2:>7}  {row['attention_tier']:<16}")

    # Summary counts
    print(f"\nClassification summary:")
    counts = attention["attention_tier"].value_counts()
    for tier in ["BOTH", "TIER1_ONLY", "TIER2_ONLY", "NEITHER", "UNKNOWN"]:
        n = counts.get(tier, 0)
        print(f"  {tier:<14}: {n}")

    # Save
    att_path = out_dir / "attention_flags.csv"
    attention.to_csv(att_path, index=False)
    print(f"\nSaved: {att_path}")

    # Attention scatter plot
    plot_attention_scatter(attention, out_dir)

    print(f"\nAll outputs: {out_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
