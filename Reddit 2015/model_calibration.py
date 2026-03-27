"""
model_calibration.py  (fixed)
==============================
Fixes two bugs from the original:
  1. Calibration used tail(30) pre-ban days but we only have June 1-9 = 9 days.
     Fix: use the full Jun-Aug window; pre-ban = Jun 1-9, post-ban = Jun 10+.
     Minimum pre-ban rows lowered to 3.
  2. UnboundLocalError crash when calibrated DataFrame is empty.
     Fix: guard all plot functions against empty input.

Also adds two new plots:
  - user_types_by_subreddit.png  (stacked bar per subreddit)
  - spillover_heatmap.png        (migration wave from banned subs)

Usage:
    python3 model_calibration.py --out_dir ./output
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
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
# Stage 1 ODE
# ---------------------------------------------------------------------------

def stage1_odes(t, state, params, g_func):
    x, T = state
    sigma, theta, alpha, beta, gamma, T0, kappa = params
    g = float(g_func(t))
    T = max(float(T), 1e-6)
    x = float(np.clip(x, 1e-6, 1 - 1e-6))
    dx = sigma * x * (1 - x) * (theta + alpha * x - g * (T ** kappa))
    dT = -beta * g + gamma * (T0 - T)
    return [dx, dT]


def simulate_stage1(params, g_series, x0, T0_init):
    n = len(g_series)
    if n < 2:
        return np.array([0.0]), np.array([x0]), np.array([T0_init])

    def g_func(t):
        idx = int(np.clip(t, 0, n - 1))
        return g_series[idx]

    sol = solve_ivp(
        stage1_odes,
        (0, n - 1),
        [x0, T0_init],
        args=(params, g_func),
        t_eval=np.arange(n, dtype=float),
        method="RK45",
        rtol=1e-5,
        atol=1e-7,
        max_step=1.0,
    )
    return sol.t, sol.y[0], sol.y[1]


def nll(free_params, g_series, x_obs, x0, T0_init):
    try:
        ls, th, la, lb, lg, lk = free_params
        params = (
            np.exp(np.clip(ls, -5, 3)),
            th,
            np.exp(np.clip(la, -5, 3)),
            np.exp(np.clip(lb, -5, 3)),
            np.exp(np.clip(lg, -5, 3)),
            T0_init,
            np.exp(np.clip(lk, -2, 2)),
        )
        _, x_sim, _ = simulate_stage1(params, g_series, x0, T0_init)
        n = min(len(x_sim), len(x_obs))
        resid = x_sim[:n] - x_obs[:n]
        return float(np.sum(resid ** 2))
    except Exception:
        return 1e10


def calibrate_subreddit(sub_data: pd.DataFrame, subreddit: str):
    """
    Uses the FULL Jun-Aug window for calibration.
    Pre-ban = Jun 1-9 (at least 3 days required).
    The ODE is fit over the entire 92-day period.
    """
    sub_data = sub_data.sort_values("day").copy()
    sub_data["day"] = pd.to_datetime(sub_data["day"], utc=True)

    pre_ban = sub_data[sub_data["day"] < BAN_DATE]

    if len(pre_ban) < 3:
        return None

    full     = sub_data.copy()
    g_series = full["C_smooth"].fillna(method="ffill").fillna(0).values
    x_obs    = full["C_raw"].fillna(method="ffill").fillna(0).values

    if len(g_series) < 5:
        return None

    x0     = float(x_obs[0]) if x_obs[0] > 0 else 0.01
    T0_val = float(pre_ban["T_norm"].mean()) if not pre_ban["T_norm"].isna().all() else 0.5
    T0_val = max(T0_val, 0.01)

    p0 = [0.0, -0.5, np.log(0.3), np.log(0.8), np.log(0.5), np.log(2.0)]

    result = minimize(
        nll,
        p0,
        args=(g_series, x_obs, x0, T0_val),
        method="Nelder-Mead",
        options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-7},
    )

    lp    = result.x
    sigma = np.exp(np.clip(lp[0], -5, 3))
    theta = lp[1]
    alpha = np.exp(np.clip(lp[2], -5, 3))
    beta  = np.exp(np.clip(lp[3], -5, 3))
    gamma = np.exp(np.clip(lp[4], -5, 3))
    kappa = np.exp(np.clip(lp[5], -2, 2))
    g_star = compute_g_star(beta, gamma, T0_val, kappa)

    return {
        "subreddit": subreddit,
        "sigma": sigma, "theta": theta, "alpha": alpha,
        "beta": beta,   "gamma": gamma, "T0": T0_val,
        "kappa": kappa, "g_star": g_star,
        "fit_nll": float(result.fun),
        "n_days": len(full),
        "n_preban_days": len(pre_ban),
    }


def compute_g_star(beta, gamma, T0, kappa):
    return T0 * gamma / (beta * (1 + kappa))


def compute_F(g_range, beta, gamma, T0, kappa):
    T_star = np.clip(T0 - (beta / gamma) * g_range, 1e-9, None)
    return g_range * (T_star ** kappa)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_backfire_curves(calibrated: pd.DataFrame, plot_dir: Path):
    if calibrated.empty:
        print("  Skipping backfire_curves.png — no calibrated subreddits")
        return

    n    = len(calibrated)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).flatten()

    for i, (_, row) in enumerate(calibrated.iterrows()):
        ax      = axes[i]
        g_max   = min(5.0, row["T0"] * row["gamma"] / row["beta"] * 3)
        g_range = np.linspace(0, g_max, 300)
        F       = compute_F(g_range, row["beta"], row["gamma"], row["T0"], row["kappa"])
        g_star  = row["g_star"]

        ax.plot(g_range, F, color="#1D9E75", linewidth=2)
        if 0 < g_star < g_max:
            ax.axvline(g_star, color="#D85A30", linestyle="--", linewidth=1.2,
                       label=f"g*={g_star:.3f}")
        ax.fill_between(g_range, F, alpha=0.08, color="#1D9E75")
        banned = row["subreddit"] in BANNED_SUBREDDITS
        ax.set_title(f"r/{row['subreddit']}", fontsize=9, fontweight="bold",
                     color="#A32D2D" if banned else "black")
        ax.set_xlabel("Enforcement g", fontsize=8)
        ax.set_ylabel("F(g)", fontsize=8)
        ax.legend(fontsize=7)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Effective deterrence F(g) — peak at g* = backfire point\n"
        "(red title = banned subreddit)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(plot_dir / "backfire_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: backfire_curves.png")


def plot_tc_isoquants(daily_metrics: pd.DataFrame, plot_dir: Path):
    ctrl = daily_metrics[~daily_metrics["subreddit"].isin(BANNED_SUBREDDITS)].copy()
    ctrl = ctrl.dropna(subset=["T_norm", "C_smooth"])
    if len(ctrl) < 10:
        print("  Skipping tc_isoquants.png — insufficient data")
        return

    compliance = 1 - ctrl["C_smooth"]
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(ctrl["C_smooth"], ctrl["T_norm"],
                    c=compliance, cmap="RdYlGn", alpha=0.5, s=20,
                    vmin=0.6, vmax=1.0)
    plt.colorbar(sc, ax=ax, label="Compliance rate (1 - C)")

    z = np.polyfit(ctrl["C_smooth"], ctrl["T_norm"], 1)
    g_line = np.linspace(ctrl["C_smooth"].min(), ctrl["C_smooth"].max(), 100)
    ax.plot(g_line, np.poly1d(z)(g_line), "k--", linewidth=1.5,
            label=f"Trend (slope={z[0]:.2f})")

    for sub, g in ctrl.groupby("subreddit"):
        ax.annotate(sub, (g["C_smooth"].mean(), g["T_norm"].mean()),
                    fontsize=7, alpha=0.7)

    ax.set_xlabel("Enforcement intensity C (removal rate)", fontsize=12)
    ax.set_ylabel("Legitimacy T (normalised score)", fontsize=12)
    ax.set_title("T–C substitution space\nNegative slope = legitimacy reduces enforcement need",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(plot_dir / "tc_isoquants.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: tc_isoquants.png")


def plot_threshold_distributions(user_thresholds: pd.DataFrame, plot_dir: Path):
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

    colors      = ["#E24B4A", "#EF9F27", "#378ADD", "#1D9E75"]
    fig, ax     = plt.subplots(figsize=(10, 6))
    any_plotted = False

    for (label, sub_df), color in zip(cats.items(), colors):
        vals = sub_df["theta_proxy"].dropna()
        if len(vals) < 5:
            continue
        ax.hist(vals.clip(-0.5, 0.5), bins=60, alpha=0.5, color=color,
                label=f"{label} (n={len(vals):,})", density=True)
        any_plotted = True

    if not any_plotted:
        print("  Skipping threshold_distributions.png — insufficient data")
        plt.close()
        return

    ax.axvline(0, color="black", linestyle=":", linewidth=1.2, label="θ=0 boundary")
    ax.set_xlabel("θ proxy  (left = initiator, right = complier)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("User threshold distributions by community type\n"
                 "Prediction: banned subs shifted left vs. high-legitimacy subs",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(plot_dir / "threshold_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: threshold_distributions.png")


def plot_event_study(daily_metrics: pd.DataFrame, plot_dir: Path):
    dm = daily_metrics.copy()
    dm["day"] = pd.to_datetime(dm["day"], utc=True)
    dm["days_from_ban"] = (dm["day"] - BAN_DATE).dt.days

    windowed = dm[(dm["days_from_ban"] >= -9) & (dm["days_from_ban"] <= 60)]
    if windowed.empty:
        print("  Skipping event_study.png — no data in window")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for is_banned, label, color in [
        (True,  "Banned subreddits",  "#E24B4A"),
        (False, "Control subreddits", "#378ADD"),
    ]:
        grp = windowed[windowed["is_banned_sub"] == is_banned]
        if grp.empty:
            continue
        dm_mean = grp.groupby("days_from_ban")[["C_smooth", "T_norm"]].mean()
        ax1.plot(dm_mean.index, dm_mean["C_smooth"],
                 label=label, color=color, linewidth=2)
        ax2.plot(dm_mean.index, dm_mean["T_norm"],
                 label=label, color=color, linewidth=2)

    for ax in (ax1, ax2):
        ax.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Ban (Jun 10)")
        ax.axvspan(-9, 0, alpha=0.06, color="gray")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    ax1.set_ylabel("Enforcement C\n(removal rate)", fontsize=11)
    ax2.set_ylabel("Legitimacy T\n(norm. score)", fontsize=11)
    ax2.set_xlabel("Days relative to ban wave", fontsize=11)
    ax1.set_title("Event study: enforcement and legitimacy around Jun 10 2015 ban wave",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(plot_dir / "event_study.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: event_study.png")


def plot_simulation_vs_observed(daily_metrics, calibrated, plot_dir):
    if calibrated.empty:
        print("  Skipping simulation_vs_observed.png — no calibrated subreddits")
        return

    n    = len(calibrated)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).flatten()

    dm = daily_metrics.copy()
    dm["day"] = pd.to_datetime(dm["day"], utc=True)

    for i, (_, row) in enumerate(calibrated.iterrows()):
        sub_data = dm[dm["subreddit"] == row["subreddit"]].sort_values("day")
        if sub_data.empty:
            continue

        g_series = sub_data["C_smooth"].fillna(0).values
        x_obs    = sub_data["C_raw"].fillna(0).values
        x0       = float(x_obs[0]) if x_obs[0] > 0 else 0.01

        params = (row["sigma"], row["theta"], row["alpha"],
                  row["beta"],  row["gamma"], row["T0"], row["kappa"])
        try:
            t_out, x_sim, _ = simulate_stage1(params, g_series, x0, row["T0"])
        except Exception:
            continue

        ax = axes[i]
        ax.plot(range(len(x_obs)), x_obs, alpha=0.6, color="#378ADD",
                linewidth=1.5, label="Observed")
        ax.plot(t_out, x_sim, color="#D85A30", linewidth=2, label="Model")

        ban_day = (BAN_DATE - sub_data["day"].min()).days
        if 0 < ban_day < len(g_series):
            ax.axvline(ban_day, color="black", linestyle="--", linewidth=1, label="Ban")

        banned = row["subreddit"] in BANNED_SUBREDDITS
        ax.set_title(f"r/{row['subreddit']}", fontsize=9, fontweight="bold",
                     color="#A32D2D" if banned else "black")
        ax.set_xlabel("Day", fontsize=8)
        ax.set_ylabel("Removal rate", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Stage 1 model vs. observed removal rate", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(plot_dir / "simulation_vs_observed.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: simulation_vs_observed.png")


def plot_user_type_by_subreddit(user_thresholds: pd.DataFrame, plot_dir: Path):
    if user_thresholds.empty:
        return

    counts = (
        user_thresholds.groupby(["subreddit", "user_type"])
        .size()
        .unstack(fill_value=0)
    )
    for col in ["INITIATOR", "JOINER", "COMPLIER"]:
        if col not in counts.columns:
            counts[col] = 0

    totals = counts.sum(axis=1)
    pct    = counts.div(totals, axis=0) * 100
    pct    = pct.sort_values("INITIATOR", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, len(pct) * 0.45)))
    y = np.arange(len(pct))

    ax.barh(y, pct["COMPLIER"],  color="#1D9E75", label="Complier")
    ax.barh(y, pct["JOINER"],    left=pct["COMPLIER"],
            color="#EF9F27", label="Joiner")
    ax.barh(y, pct["INITIATOR"], left=pct["COMPLIER"] + pct["JOINER"],
            color="#E24B4A", label="Initiator")

    labels = [f"r/{s}  {'[BANNED]' if s in BANNED_SUBREDDITS else ''}"
              for s in pct.index]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("% of users", fontsize=11)
    ax.set_title("User type composition by subreddit\n"
                 "Prediction: banned/fringe subs have more initiators (red)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlim(0, 100)
    plt.tight_layout()
    fig.savefig(plot_dir / "user_types_by_subreddit.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: user_types_by_subreddit.png")


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
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=max(pivot.values.max(), 0.1))
    plt.colorbar(im, ax=ax, label="% users from banned subs")

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"r/{s}" for s in pivot.index], fontsize=9)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        [str(w)[:10] for w in pivot.columns],
        rotation=45, ha="right", fontsize=7
    )

    ban_cols = [i for i, w in enumerate(pivot.columns)
                if pd.Timestamp(w, tz="UTC") >= BAN_DATE and
                   pd.Timestamp(w, tz="UTC") < BAN_DATE + pd.Timedelta(days=7)]
    for bc in ban_cols:
        ax.axvline(bc - 0.5, color="blue", linewidth=2)

    ax.set_title("Spillover: % of weekly users arriving from banned subreddits\n"
                 "Blue line = Jun 10 ban wave",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(plot_dir / "spillover_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: spillover_heatmap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./output")
    args = parser.parse_args()

    out_dir  = Path(args.out_dir)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    print("Loading data...")
    panel           = pd.read_parquet(out_dir / "panel.parquet")
    daily_metrics   = pd.read_parquet(out_dir / "daily_metrics.parquet")
    user_thresholds = pd.read_parquet(out_dir / "user_thresholds.parquet")

    daily_metrics["day"] = pd.to_datetime(daily_metrics["day"], utc=True)

    # Auto-fix all-COMPLIER bug if present
    if (user_thresholds["user_type"] == "COMPLIER").mean() > 0.99:
        print("\nDetected all-COMPLIER issue — applying relative threshold fix...")
        sub_medians = (
            user_thresholds.groupby("subreddit")["removal_rate"]
            .median().rename("sub_median").reset_index()
        )
        user_thresholds = user_thresholds.merge(sub_medians, on="subreddit", how="left")
        user_thresholds["user_type"] = "COMPLIER"
        user_thresholds.loc[
            (user_thresholds["removal_rate"] > user_thresholds["sub_median"]) &
            (user_thresholds["n_removed"] >= 1), "user_type"
        ] = "JOINER"
        user_thresholds.loc[
            (user_thresholds["removal_rate"] > 2 * user_thresholds["sub_median"]) &
            (user_thresholds["n_removed"] >= 2), "user_type"
        ] = "INITIATOR"
        user_thresholds.drop(columns=["sub_median"], inplace=True)
        user_thresholds.to_parquet(out_dir / "user_thresholds.parquet", index=False)
        print("  Fixed and saved.")

    tc    = user_thresholds["user_type"].value_counts()
    total = len(user_thresholds)
    print("\n--- User type distribution ---")
    for t, n in tc.items():
        print(f"  {t:<12}  {n:>7,}  ({100*n/total:.1f}%)")

    # Calibrate
    print("\nCalibrating Stage 1 parameters...")
    calibrated_rows = []
    for subreddit, sub_data in daily_metrics.groupby("subreddit"):
        result = calibrate_subreddit(sub_data.copy(), subreddit)
        if result:
            calibrated_rows.append(result)
            print(f"  r/{subreddit:<25}  g*={result['g_star']:.4f}  "
                  f"β={result['beta']:.3f}  κ={result['kappa']:.3f}  "
                  f"days={result['n_days']} (pre-ban={result['n_preban_days']})")
        else:
            print(f"  r/{subreddit:<25}  [failed — insufficient data]")

    calibrated = pd.DataFrame(calibrated_rows)
    calibrated.to_csv(out_dir / "calibrated_params.csv", index=False)
    print(f"\nCalibrated {len(calibrated)} subreddits → calibrated_params.csv")

    if not calibrated.empty:
        print("\n--- Backfire tolerance ranking (lower g* = more fragile) ---")
        print(f"  {'Subreddit':<26} {'g*':>7}  {'Banned?':>8}  {'β':>6}  {'κ':>6}")
        print("  " + "-" * 58)
        for _, row in calibrated.sort_values("g_star").iterrows():
            banned = "YES" if row["subreddit"] in BANNED_SUBREDDITS else "no"
            print(f"  r/{row['subreddit']:<24} {row['g_star']:>7.4f}  "
                  f"{banned:>8}  {row['beta']:>6.3f}  {row['kappa']:>6.3f}")

    print("\nGenerating plots...")
    plot_event_study(daily_metrics, plot_dir)
    plot_backfire_curves(calibrated, plot_dir)
    plot_tc_isoquants(daily_metrics, plot_dir)
    plot_threshold_distributions(user_thresholds, plot_dir)
    plot_simulation_vs_observed(daily_metrics, calibrated, plot_dir)
    plot_user_type_by_subreddit(user_thresholds, plot_dir)
    plot_spillover_heatmap(panel, plot_dir)

    print(f"\nAll plots saved to: {plot_dir}/")
    for p in sorted(plot_dir.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()