"""
model_calibration.py  — updated for May + June + July + August 2015
=====================================================================
Key changes from previous version:
  - Pre-ban window now uses full May data (~40 days) for calibration
  - Minimum pre-ban rows lowered to 5 (was 3) — more reliable with 40 days
  - Event study window extended: -40 to +83 days (full May pre-period)
  - fatpeoplehate now has enough data to calibrate (~31 days in May)
  - Pre-ban escalation test now covers the full May window

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
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

BANNED_SUBREDDITS = {
    "fatpeoplehate", "CoonTown", "transfags", "neofag", "hamplanet",
}
BAN_DATE = pd.Timestamp("2015-06-10", tz="UTC")
DATA_START = pd.Timestamp("2015-05-01", tz="UTC")


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
        return g_series[int(np.clip(t, 0, n - 1))]

    sol = solve_ivp(
        stage1_odes, (0, n - 1), [x0, T0_init],
        args=(params, g_func),
        t_eval=np.arange(n, dtype=float),
        method="RK45", rtol=1e-5, atol=1e-7, max_step=1.0,
    )
    return sol.t, sol.y[0], sol.y[1]


def nll(free_params, g_series, x_obs, x0, T0_init):
    try:
        ls, th, la, lb, lg, lk = free_params
        params = (
            np.exp(np.clip(ls, -5, 3)), th,
            np.exp(np.clip(la, -5, 3)), np.exp(np.clip(lb, -5, 3)),
            np.exp(np.clip(lg, -5, 3)), T0_init,
            np.exp(np.clip(lk, -2, 2)),
        )
        _, x_sim, _ = simulate_stage1(params, g_series, x0, T0_init)
        n = min(len(x_sim), len(x_obs))
        return float(np.sum((x_sim[:n] - x_obs[:n]) ** 2))
    except Exception:
        return 1e10


def calibrate_subreddit(sub_data, subreddit):
    """
    Uses the full pre-ban window (May 1 – June 9) for T0 initialisation
    and fits the ODE over the entire observation period.
    With May data, pre_ban now has ~40 days for most subreddits and
    ~31 days for fatpeoplehate (banned June 10, so May is its only window).
    """
    sub_data = sub_data.sort_values("day").copy()
    sub_data["day"] = pd.to_datetime(sub_data["day"], utc=True)

    pre_ban = sub_data[sub_data["day"] < BAN_DATE]

    # Now require 5 pre-ban days (was 3) — reliable with ~40 day window
    if len(pre_ban) < 5:
        return None

    g_series = sub_data["C_smooth"].ffill().fillna(0).values
    x_obs    = sub_data["C_raw"].ffill().fillna(0).values

    if len(g_series) < 10:
        return None

    x0     = float(x_obs[0]) if x_obs[0] > 0 else 0.01
    # T0 now uses full pre-ban window mean — much more stable than 9-day estimate
    T0_val = float(pre_ban["T_norm"].mean()) \
             if not pre_ban["T_norm"].isna().all() else 0.5
    T0_val = max(T0_val, 0.01)

    p0 = [0.0, -0.5, np.log(0.3), np.log(0.8), np.log(0.5), np.log(2.0)]
    result = minimize(nll, p0, args=(g_series, x_obs, x0, T0_val),
                      method="Nelder-Mead",
                      options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-7})

    lp    = result.x
    sigma = np.exp(np.clip(lp[0], -5, 3))
    theta = lp[1]
    alpha = np.exp(np.clip(lp[2], -5, 3))
    beta  = np.exp(np.clip(lp[3], -5, 3))
    gamma = np.exp(np.clip(lp[4], -5, 3))
    kappa = np.exp(np.clip(lp[5], -2, 2))
    g_star = T0_val * gamma / (beta * (1 + kappa))

    return {
        "subreddit": subreddit,
        "sigma": sigma, "theta": theta, "alpha": alpha,
        "beta": beta,   "gamma": gamma, "T0": T0_val,
        "kappa": kappa, "g_star": g_star,
        "fit_nll": float(result.fun),
        "n_days": len(sub_data),
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

def plot_event_study(daily_metrics, plot_dir):
    """Extended event study: -40 to +83 days (full May pre-period visible)."""
    dm = daily_metrics.copy()
    dm["day"] = pd.to_datetime(dm["day"], utc=True)
    dm["days_from_ban"] = (dm["day"] - BAN_DATE).dt.days

    # Full window: May 1 = day -40, Aug 31 = day +82
    windowed = dm[(dm["days_from_ban"] >= -40) & (dm["days_from_ban"] <= 83)]
    if windowed.empty:
        print("  Skipping event_study.png")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

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
        ax.axvline(0, color="black", linestyle="--",
                   linewidth=1.5, label="Ban (Jun 10)")
        ax.axvspan(-40, 0, alpha=0.05, color="gray", label="Pre-ban (May)")
        ax.axvspan(0, 83, alpha=0.03, color="red")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    ax1.set_ylabel("Enforcement C\n(removal rate)", fontsize=11)
    ax2.set_ylabel("Legitimacy T\n(norm. score)", fontsize=11)
    ax2.set_xlabel("Days relative to ban wave (Day 0 = June 10 2015)", fontsize=11)
    ax1.set_title(
        "Event study — enforcement and legitimacy: May 1 through August 31 2015\n"
        "Grey shading = pre-ban May window  |  Red shading = post-ban period",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(plot_dir / "event_study.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: event_study.png")


def plot_backfire_curves(calibrated, plot_dir):
    if calibrated.empty:
        print("  Skipping backfire_curves.png")
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
        F       = compute_F(g_range, row["beta"], row["gamma"],
                            row["T0"], row["kappa"])
        g_star  = row["g_star"]

        ax.plot(g_range, F, color="#1D9E75", linewidth=2)
        if 0 < g_star < g_max:
            ax.axvline(g_star, color="#D85A30", linestyle="--",
                       linewidth=1.2, label=f"g*={g_star:.3f}")
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
        "Effective deterrence F(g) — calibrated on May–Aug 2015\n"
        "Peak at g* = backfire point  |  Red title = banned subreddit",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(plot_dir / "backfire_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: backfire_curves.png")


def plot_tc_isoquants(daily_metrics, plot_dir):
    ctrl = daily_metrics[~daily_metrics["subreddit"].isin(BANNED_SUBREDDITS)].copy()
    ctrl = ctrl.dropna(subset=["T_norm", "C_smooth"])
    if len(ctrl) < 10:
        return

    compliance = 1 - ctrl["C_smooth"]
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(ctrl["C_smooth"], ctrl["T_norm"],
                    c=compliance, cmap="RdYlGn",
                    alpha=0.5, s=20, vmin=0.6, vmax=1.0)
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
    ax.set_title("T–C substitution space — May–Aug 2015\n"
                 "Negative slope = legitimacy reduces enforcement need",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(plot_dir / "tc_isoquants.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: tc_isoquants.png")


def plot_threshold_distributions(user_thresholds, plot_dir):
    HIGH_LEG = {"science", "AskHistorians", "askscience"}
    MED_LEG  = {"politics", "worldnews", "news"}
    FRINGE   = {"KotakuInAction", "MensRights", "TumblrInAction"}

    cats   = {
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
    fig, ax = plt.subplots(figsize=(10, 6))
    any_plotted = False

    for (label, sub_df), color in zip(cats.items(), colors):
        vals = sub_df["theta_proxy"].dropna()
        if len(vals) < 5:
            continue
        ax.hist(vals.clip(-0.5, 0.5), bins=60, alpha=0.5, color=color,
                label=f"{label} (n={len(vals):,})", density=True)
        any_plotted = True

    if not any_plotted:
        plt.close()
        return

    ax.axvline(0, color="black", linestyle=":", linewidth=1.2)
    ax.set_xlabel("θ proxy  (left = initiator, right = complier)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("User threshold distributions — May–Aug 2015\n"
                 "Prediction: banned subs shifted left vs. high-legitimacy subs",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(plot_dir / "threshold_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: threshold_distributions.png")


def plot_user_type_by_subreddit(user_thresholds, plot_dir):
    counts = (user_thresholds.groupby(["subreddit", "user_type"])
                .size().unstack(fill_value=0))
    for col in ["INITIATOR", "JOINER", "COMPLIER"]:
        if col not in counts.columns:
            counts[col] = 0

    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    pct = pct.sort_values("INITIATOR", ascending=True)

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
    ax.set_title("User type composition — May–Aug 2015\n"
                 "Prediction: banned/fringe subs have more initiators (red)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlim(0, 100)
    plt.tight_layout()
    fig.savefig(plot_dir / "user_types_by_subreddit.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: user_types_by_subreddit.png")


def plot_simulation_vs_observed(daily_metrics, calibrated, plot_dir):
    if calibrated.empty:
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
        params   = (row["sigma"], row["theta"], row["alpha"],
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
            ax.axvline(ban_day, color="black", linestyle="--",
                       linewidth=1, label="Ban")

        banned = row["subreddit"] in BANNED_SUBREDDITS
        ax.set_title(f"r/{row['subreddit']}", fontsize=9, fontweight="bold",
                     color="#A32D2D" if banned else "black")
        ax.set_xlabel("Day", fontsize=8)
        ax.set_ylabel("Removal rate", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Stage 1 model vs. observed — May–Aug 2015",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(plot_dir / "simulation_vs_observed.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: simulation_vs_observed.png")


def plot_preban_trends(daily_metrics, calibrated, plot_dir):
    """
    New plot: full May pre-ban trend per subreddit.
    Tests whether enforcement was rising for the entire month before the ban,
    not just the final 9 days.
    """
    dm = daily_metrics.copy()
    dm["day"] = pd.to_datetime(dm["day"], utc=True)
    pre = dm[dm["day"] < BAN_DATE].copy()
    pre["days_from_ban"] = (pre["day"] - BAN_DATE).dt.days

    g_stars = calibrated.set_index("subreddit")["g_star"].to_dict() \
              if not calibrated.empty else {}

    subs = pre["subreddit"].unique()
    n    = len(subs)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows),
                             sharex=False)
    axes = np.array(axes).flatten()

    print("\n--- Full pre-ban enforcement trend (May 1 – June 9) ---")
    print(f"  {'Subreddit':<25} {'slope':>9}  {'p':>7}  {'sig':>4}  "
          f"{'n_days':>6}  regime")
    print("  " + "-"*65)

    for i, sub in enumerate(sorted(subs)):
        ax       = axes[i]
        sub_pre  = pre[pre["subreddit"] == sub].sort_values("days_from_ban")
        g_star   = g_stars.get(sub, np.nan)
        banned   = sub in BANNED_SUBREDDITS
        color    = "#E24B4A" if banned else "#378ADD"

        if len(sub_pre) < 4:
            ax.set_title(f"r/{sub}", fontsize=9)
            continue

        days   = np.arange(len(sub_pre))
        C_vals = sub_pre["C_smooth"].fillna(0).values

        slope, intercept, r, p, se = stats.linregress(days, C_vals)
        sig  = "*" if p < 0.05 else ""
        tag  = "[BAN]" if banned else "     "
        print(f"  {tag} r/{sub:<23} {slope:>+9.5f}  {p:>7.3f}  "
              f"{sig:>4}  {len(sub_pre):>6}")

        ax.plot(sub_pre["days_from_ban"], C_vals, color=color,
                linewidth=1.8, label="C (enforcement)")
        # Trend line
        trend = intercept + slope * days
        ax.plot(sub_pre["days_from_ban"], trend, color=color,
                linewidth=1, linestyle="--", alpha=0.6)

        if not np.isnan(g_star) and 0 < g_star < 0.6:
            ax.axhline(g_star, color="#EF9F27", linestyle="--",
                       linewidth=1.5, label=f"g*={g_star:.3f}")
            ax.fill_between(sub_pre["days_from_ban"], C_vals, g_star,
                            where=C_vals >= g_star,
                            alpha=0.2, color="#E24B4A",
                            label="C > g*")

        title_col = "#A32D2D" if banned else "black"
        sig_str   = f"  p={p:.3f}{'*' if p < 0.05 else ''}"
        ax.set_title(f"r/{sub}{'  [BAN]' if banned else ''}{sig_str}",
                     fontsize=8, fontweight="bold", color=title_col)
        ax.set_xlabel("Days before ban", fontsize=7)
        ax.set_ylabel("C", fontsize=7)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Full pre-ban enforcement trend: May 1 – June 9 2015\n"
        "Dashed = OLS trend  |  Orange = g* threshold  |  Red shading = backfire zone",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(plot_dir / "preban_full_trend.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: preban_full_trend.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./output")
    args     = parser.parse_args()
    out_dir  = Path(args.out_dir)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    print("Loading data...")
    panel           = pd.read_parquet(out_dir / "panel.parquet")
    daily_metrics   = pd.read_parquet(out_dir / "daily_metrics.parquet")
    user_thresholds = pd.read_parquet(out_dir / "user_thresholds.parquet")
    daily_metrics["day"] = pd.to_datetime(daily_metrics["day"], utc=True)

    # Auto-fix all-COMPLIER if present
    if (user_thresholds["user_type"] == "COMPLIER").mean() > 0.99:
        print("Applying relative threshold fix...")
        sub_med = (user_thresholds.groupby("subreddit")["removal_rate"]
                   .median().rename("sub_median").reset_index())
        user_thresholds = user_thresholds.merge(sub_med, on="subreddit", how="left")
        user_thresholds["user_type"] = "COMPLIER"
        user_thresholds.loc[
            (user_thresholds["removal_rate"] > user_thresholds["sub_median"]) &
            (user_thresholds["n_removed"] >= 1), "user_type"] = "JOINER"
        user_thresholds.loc[
            (user_thresholds["removal_rate"] > 2 * user_thresholds["sub_median"]) &
            (user_thresholds["n_removed"] >= 2), "user_type"] = "INITIATOR"
        user_thresholds.drop(columns=["sub_median"], inplace=True)
        user_thresholds.to_parquet(out_dir / "user_thresholds.parquet", index=False)

    tc = user_thresholds["user_type"].value_counts()
    total = len(user_thresholds)
    print("\n--- User type distribution ---")
    for t, n in tc.items():
        print(f"  {t:<12}  {n:>8,}  ({100*n/total:.1f}%)")

    # Calibrate
    print("\nCalibrating Stage 1 parameters (using full May pre-ban window)...")
    calibrated_rows = []
    for subreddit, sub_data in daily_metrics.groupby("subreddit"):
        result = calibrate_subreddit(sub_data.copy(), subreddit)
        if result:
            calibrated_rows.append(result)
            print(f"  r/{subreddit:<25}  g*={result['g_star']:.4f}  "
                  f"β={result['beta']:.3f}  κ={result['kappa']:.3f}  "
                  f"days={result['n_days']} (pre-ban={result['n_preban_days']})")
        else:
            print(f"  r/{subreddit:<25}  [failed]")

    calibrated = pd.DataFrame(calibrated_rows)
    calibrated.to_csv(out_dir / "calibrated_params.csv", index=False)
    print(f"\nCalibrated {len(calibrated)} subreddits")

    if not calibrated.empty:
        print("\n--- Backfire tolerance ranking ---")
        print(f"  {'Subreddit':<26} {'g*':>7}  {'Banned?':>8}  "
              f"{'β':>6}  {'κ':>6}  {'pre-ban days':>12}")
        print("  " + "-"*70)
        for _, row in calibrated.sort_values("g_star").iterrows():
            banned = "YES" if row["subreddit"] in BANNED_SUBREDDITS else "no"
            print(f"  r/{row['subreddit']:<24} {row['g_star']:>7.4f}  "
                  f"{banned:>8}  {row['beta']:>6.3f}  {row['kappa']:>6.3f}  "
                  f"{row['n_preban_days']:>12}")

    print("\nGenerating plots...")
    plot_event_study(daily_metrics, plot_dir)
    plot_backfire_curves(calibrated, plot_dir)
    plot_tc_isoquants(daily_metrics, plot_dir)
    plot_threshold_distributions(user_thresholds, plot_dir)
    plot_simulation_vs_observed(daily_metrics, calibrated, plot_dir)
    plot_user_type_by_subreddit(user_thresholds, plot_dir)
    plot_preban_trends(daily_metrics, calibrated, plot_dir)

    print(f"\nAll plots saved to: {plot_dir}/")
    for p in sorted(plot_dir.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()