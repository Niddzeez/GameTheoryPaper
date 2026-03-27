"""
stage2_analysis.py
==================
Adds the Stage 2 Adaptive Governance Model on top of the Stage 1 results.

Stage 2 extends Stage 1 by making enforcement C endogenous:
  ẋ = σ·x(1−x)·[θ + α·x − C·T^κ]          (behaviour)
  Ṫ = −β·C + γ·(T₀ − T)                     (trust)
  Ċ = α_gov·(x − x_target) − δ·C            (enforcement — NEW)

Key quantities derived:
  g*          = T₀γ / β(1+κ)               [Stage 1 backfire point]
  C* equilibrium enforcement level
  Ξ = (T*)^(κ-1) · [γT₀ − β·C*·(1+κ)]    [stability indicator]
  α_gov,min  = α·γ·δ / Ξ                   [minimum responsiveness for stability]

Stability regimes:
  Regime 1 — Safe: C* < g*, α_gov > α_gov,min → stable equilibrium
  Regime 2 — Governance trap: C* < g*, α_gov < α_gov,min → too slow to stabilise
  Regime 3 — Structural instability: C* ≥ g* → IMPOSSIBLE to stabilise
              (enforcement already past backfire point, no α_gov helps)

Outputs:
  plots/stage2_regime_map.png          — which regime each subreddit is in
  plots/stage2_trajectories.png        — 3D ODE simulation (x, T, C over time)
  plots/stage2_governance_trap.png     — α_gov vs enforcement volatility
  plots/stage2_preban_escalation.png   — was C rising before the ban? (key question)
  plots/stage2_equilibrium_surface.png — C* vs T* across subreddits
  stage2_results.csv                   — all estimated parameters and regime classifications

Run:
    python3 stage2_analysis.py --out_dir ./output
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, brentq
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

BANNED = {"fatpeoplehate", "CoonTown", "transfags", "neofag", "hamplanet"}
BAN_DATE = pd.Timestamp("2015-06-10", tz="UTC")

REGIME_COLORS = {
    "Stable":                "#1D9E75",
    "Governance trap":       "#EF9F27",
    "Structural instability":"#E24B4A",
    "Indeterminate":         "#888780",
}


# ---------------------------------------------------------------------------
# Stage 2 ODE system
# ---------------------------------------------------------------------------

def stage2_odes(t, state, params):
    x, T, C = state
    sigma, theta, alpha, beta, gamma, T0, kappa, alpha_gov, delta, x_target = params

    x = float(np.clip(x, 1e-6, 1 - 1e-6))
    T = max(float(T), 1e-6)
    C = max(float(C), 0.0)

    dx = sigma * x * (1 - x) * (theta + alpha * x - C * (T ** kappa))
    dT = -beta * C + gamma * (T0 - T)
    dC = alpha_gov * (x - x_target) - delta * C

    return [dx, dT, dC]


def simulate_stage2(params, t_span, x0, T0_init, C0, n_points=200):
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        stage2_odes,
        t_span,
        [x0, T0_init, C0],
        args=(params,),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
        max_step=0.5,
    )
    return sol.t, sol.y[0], sol.y[1], sol.y[2]


# ---------------------------------------------------------------------------
# Step 1: Estimate α_gov and δ from observed C time series
# ---------------------------------------------------------------------------

def estimate_governance_params(sub_data: pd.DataFrame, x_target: float):
    """
    From Ċ = α_gov·(x − x_target) − δ·C we estimate α_gov and δ via OLS.

    ΔC_t ≈ α_gov·(x_{t-1} − x_target) − δ·C_{t-1}

    Regress: ΔC ~ (x - x_target) + C_lagged
    Coefficients: α_gov = coef on (x - x_target), δ = -coef on C_lagged
    """
    df = sub_data.sort_values("day").copy()
    df["C"]       = df["C_smooth"].fillna(method="ffill").fillna(0)
    df["x"]       = df["C_raw"].fillna(method="ffill").fillna(0)
    df["dC"]      = df["C"].diff()
    df["x_dev"]   = df["x"].shift(1) - x_target
    df["C_lag"]   = df["C"].shift(1)

    df = df.dropna(subset=["dC", "x_dev", "C_lag"])
    if len(df) < 10:
        return None, None, None

    X = np.column_stack([df["x_dev"], df["C_lag"]])
    y = df["dC"].values

    try:
        result = np.linalg.lstsq(X, y, rcond=None)
        coeffs = result[0]
        alpha_gov = max(coeffs[0], 1e-4)
        delta     = max(-coeffs[1], 1e-4)

        # R² of the regression
        y_pred = X @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return alpha_gov, delta, r2
    except Exception:
        return None, None, None


# ---------------------------------------------------------------------------
# Step 2: Classify each subreddit into a stability regime
# ---------------------------------------------------------------------------

def classify_regime(row: dict) -> str:
    """
    Uses the Routh-Hurwitz conditions from Stage 2 to classify the regime.

    Regime 3 — Structural instability: C* >= g*  (Ξ ≤ 0)
        → Stabilisation is IMPOSSIBLE regardless of α_gov
    Regime 2 — Governance trap: C* < g* but α_gov < α_gov_min
        → System could be stable but governance too slow to achieve it
    Regime 1 — Stable: C* < g* and α_gov >= α_gov_min
        → System converges to safe equilibrium
    """
    g_star    = row.get("g_star", np.nan)
    C_eq      = row.get("C_equilibrium", np.nan)
    alpha_gov = row.get("alpha_gov", np.nan)
    alpha_gov_min = row.get("alpha_gov_min", np.nan)

    if any(np.isnan(v) for v in [g_star, C_eq, alpha_gov]):
        return "Indeterminate"

    if C_eq >= g_star:
        return "Structural instability"

    if np.isnan(alpha_gov_min) or alpha_gov_min <= 0:
        return "Indeterminate"

    if alpha_gov < alpha_gov_min:
        return "Governance trap"

    return "Stable"


def compute_equilibrium(row: dict):
    """
    Solves for C* using the Stage 2 equilibrium equation:
    C*·[T₀ − (β/γ)·C*]^κ = θ + α·x_target + (α·δ/α_gov)·C*

    i.e., find root of Φ(C) = F(C) − (α·δ/α_gov)·C − (θ + α·x_target)
    where F(C) = C·[T₀ − (β/γ)·C]^κ

    Returns C*, T*, and Ξ
    """
    try:
        beta      = row["beta"]
        gamma     = row["gamma"]
        T0        = row["T0"]
        kappa     = row["kappa"]
        alpha     = row["alpha"]
        theta     = row["theta"]
        alpha_gov = row["alpha_gov"]
        delta     = row["delta"]
        x_target  = row["x_target"]

        c_ratio = beta / gamma
        u_max = min(T0 / c_ratio * 0.99,
                    alpha_gov / delta * (1 - x_target) * 0.99)

        if u_max <= 0:
            return np.nan, np.nan, np.nan

        def phi(u):
            T_star = T0 - c_ratio * u
            if T_star <= 0:
                return -1e10
            F = u * (T_star ** kappa)
            rhs_slope = (alpha * delta / alpha_gov) * u
            rhs_const = theta + alpha * x_target
            return F - rhs_slope - rhs_const

        # Check if interior root exists
        if phi(1e-6) >= 0 or phi(u_max) >= 0:
            return np.nan, np.nan, np.nan

        # Find the peak of phi (where interior root might exist)
        u_test = np.linspace(1e-6, u_max, 500)
        phi_vals = [phi(u) for u in u_test]

        if max(phi_vals) <= 0:
            return np.nan, np.nan, np.nan

        # Find root on descending branch (high-enforcement equilibrium)
        peak_idx = np.argmax(phi_vals)
        if peak_idx >= len(u_test) - 1:
            return np.nan, np.nan, np.nan

        try:
            C_star = brentq(phi, u_test[peak_idx], u_max * 0.99, xtol=1e-8)
        except Exception:
            return np.nan, np.nan, np.nan

        T_star = T0 - c_ratio * C_star
        if T_star <= 0:
            return np.nan, np.nan, np.nan

        # Ξ = (T*)^(κ-1) · [γ·T₀ − β·C*·(1+κ)]
        Xi = (T_star ** (kappa - 1)) * (gamma * T0 - beta * C_star * (1 + kappa))

        return C_star, T_star, Xi

    except Exception:
        return np.nan, np.nan, np.nan


# ---------------------------------------------------------------------------
# Step 3: Compute α_gov,min (minimum responsiveness for stability)
# ---------------------------------------------------------------------------

def compute_alpha_gov_min(row: dict) -> float:
    """
    From RH2 condition (Ξ > 0 case):
    α_gov,min = α·γ·δ / Ξ

    If Ξ ≤ 0, stabilisation is impossible (return inf).
    """
    try:
        Xi    = row.get("Xi", np.nan)
        alpha = row["alpha"]
        gamma = row["gamma"]
        delta = row["delta"]

        if np.isnan(Xi) or Xi <= 0:
            return np.inf

        return (alpha * gamma * delta) / Xi
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_regime_map(results: pd.DataFrame, plot_dir: Path):
    """
    Horizontal bar chart showing each subreddit's regime,
    with α_gov on x-axis and α_gov,min as a threshold marker.
    """
    df = results.dropna(subset=["alpha_gov"]).copy()
    df = df.sort_values("alpha_gov", ascending=True)

    fig, ax = plt.subplots(figsize=(11, max(6, len(df) * 0.55)))
    y = np.arange(len(df))

    for i, (_, row) in enumerate(df.iterrows()):
        color = REGIME_COLORS.get(row["regime"], "#888780")
        ax.barh(i, row["alpha_gov"], color=color, alpha=0.8, height=0.6)

        # Mark α_gov,min threshold
        if np.isfinite(row.get("alpha_gov_min", np.inf)):
            ax.axvline(row["alpha_gov_min"], ymin=(i - 0.3) / len(df),
                       ymax=(i + 0.3) / len(df),
                       color="black", linewidth=1.5, linestyle="--")

        # Regime label
        ax.text(row["alpha_gov"] + 0.001, i,
                f"{row['regime']}  (g*={row.get('g_star', 0):.3f})",
                va="center", fontsize=8)

    labels = [
        f"r/{r['subreddit']}  {'[BANNED]' if r['subreddit'] in BANNED else ''}"
        for _, r in df.iterrows()
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Estimated α_gov (moderator response speed)", fontsize=11)
    ax.set_title(
        "Stage 2 regime classification by subreddit\n"
        "Dashed line = α_gov,min threshold for stability\n"
        "Green=Stable  |  Orange=Governance trap  |  Red=Structural instability",
        fontsize=11, fontweight="bold"
    )

    patches = [mpatches.Patch(color=v, label=k) for k, v in REGIME_COLORS.items()
               if k in df["regime"].values]
    ax.legend(handles=patches, fontsize=9, loc="lower right")
    ax.set_xlim(left=0)
    plt.tight_layout()
    fig.savefig(plot_dir / "stage2_regime_map.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: stage2_regime_map.png")


def plot_trajectories(results: pd.DataFrame, daily_metrics: pd.DataFrame,
                      plot_dir: Path):
    """
    For each calibrated subreddit, simulate Stage 2 ODE and plot
    x(t), T(t), C(t) on three panels. Shows how the three variables
    co-evolve — the key visual difference from Stage 1.
    """
    valid = results.dropna(subset=["alpha_gov", "alpha"]).head(12)
    if valid.empty:
        print("  Skipping stage2_trajectories.png — no valid calibrations")
        return

    n    = len(valid)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(5 * cols, 4 * rows))

    dm = daily_metrics.copy()
    dm["day"] = pd.to_datetime(dm["day"], utc=True)

    for idx, (_, row) in enumerate(valid.iterrows()):
        sub_data = dm[dm["subreddit"] == row["subreddit"]].sort_values("day")
        if sub_data.empty:
            continue

        n_days  = len(sub_data)
        x0      = float(sub_data["C_raw"].iloc[0]) or 0.02
        T0_init = row["T0"]
        C0      = float(sub_data["C_smooth"].iloc[0]) or 0.05

        params = (
            row["sigma"], row["theta"], row["alpha"],
            row["beta"],  row["gamma"], row["T0"], row["kappa"],
            row["alpha_gov"], row["delta"], row["x_target"],
        )

        try:
            t_out, x_sim, T_sim, C_sim = simulate_stage2(
                params, (0, n_days - 1), x0, T0_init, C0, n_points=n_days
            )
        except Exception:
            continue

        ax = fig.add_subplot(rows, cols, idx + 1)
        ax.plot(t_out, x_sim, color="#E24B4A", linewidth=1.5, label="x (violations)")
        ax.plot(t_out, T_sim, color="#1D9E75", linewidth=1.5, label="T (trust)")
        ax.plot(t_out, C_sim, color="#378ADD", linewidth=1.5, label="C (enforcement)")

        # Overlay observed x
        ax.plot(range(n_days), sub_data["C_raw"].fillna(0).values,
                color="#E24B4A", alpha=0.3, linewidth=1, linestyle="--",
                label="x observed")

        # Mark ban date
        ban_day = (BAN_DATE - sub_data["day"].min()).days
        if 0 < ban_day < n_days:
            ax.axvline(ban_day, color="black", linestyle=":", linewidth=1)

        regime = row.get("regime", "")
        color  = REGIME_COLORS.get(regime, "black")
        banned = row["subreddit"] in BANNED
        ax.set_title(f"r/{row['subreddit']}{'  [BANNED]' if banned else ''}\n{regime}",
                     fontsize=9, fontweight="bold", color=color)
        ax.set_xlabel("Day", fontsize=8)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Stage 2 ODE: x (violations), T (trust), C (enforcement) co-evolving",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(plot_dir / "stage2_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: stage2_trajectories.png")


def plot_governance_trap(results: pd.DataFrame, plot_dir: Path):
    """
    Scatter: α_gov (x) vs enforcement volatility σ(C) (y).
    Prediction: governance trap communities (high α_gov but C > g*)
    should show HIGH volatility.
    Stable communities should show LOW volatility.
    """
    df = results.dropna(subset=["alpha_gov", "C_volatility"]).copy()
    if df.empty:
        print("  Skipping stage2_governance_trap.png — no data")
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    for regime, grp in df.groupby("regime"):
        color = REGIME_COLORS.get(regime, "#888780")
        ax.scatter(grp["alpha_gov"], grp["C_volatility"],
                   color=color, s=80, alpha=0.85, label=regime, zorder=3)
        for _, r in grp.iterrows():
            ax.annotate(r["subreddit"],
                        (r["alpha_gov"], r["C_volatility"]),
                        fontsize=8, alpha=0.8,
                        xytext=(4, 4), textcoords="offset points")

    # Trend line
    if len(df) > 3:
        z = np.polyfit(df["alpha_gov"], df["C_volatility"], 1)
        g_line = np.linspace(df["alpha_gov"].min(), df["alpha_gov"].max(), 100)
        ax.plot(g_line, np.poly1d(z)(g_line), "k--", linewidth=1.2, alpha=0.5,
                label=f"Trend (slope={z[0]:.4f})")

    ax.set_xlabel("α_gov — moderator response speed", fontsize=12)
    ax.set_ylabel("C volatility (std of weekly removal rate)", fontsize=12)
    ax.set_title(
        "Governance trap: does reactive moderation produce enforcement volatility?\n"
        "Stage 2 prediction: governance-trap communities have higher C volatility",
        fontsize=11, fontweight="bold"
    )
    patches = [mpatches.Patch(color=v, label=k) for k, v in REGIME_COLORS.items()
               if k in df["regime"].values]
    ax.legend(handles=patches + [plt.Line2D([0], [0], color="k", linestyle="--",
              label="Trend")], fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(plot_dir / "stage2_governance_trap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: stage2_governance_trap.png")


def plot_preban_escalation(daily_metrics: pd.DataFrame,
                            results: pd.DataFrame, plot_dir: Path):
    """
    The key Stage 2 question: was enforcement escalating BEFORE June 10,
    and did that escalation push communities past their g*?

    Shows: daily C for each subreddit in the pre-ban window (Jun 1-9),
    with g* marked as a horizontal threshold.
    """
    dm = daily_metrics.copy()
    dm["day"] = pd.to_datetime(dm["day"], utc=True)

    pre_ban = dm[dm["day"] < BAN_DATE].copy()
    pre_ban["days_from_ban"] = (pre_ban["day"] - BAN_DATE).dt.days

    # Get g* per subreddit from results
    g_stars = results.set_index("subreddit")["g_star"].to_dict()

    subs_to_plot = [s for s in pre_ban["subreddit"].unique()
                    if s in g_stars and not np.isnan(g_stars[s])][:12]

    cols = min(4, len(subs_to_plot))
    rows = (len(subs_to_plot) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(4 * cols, 3 * rows), sharex=True)
    axes = np.array(axes).flatten()

    for i, sub in enumerate(subs_to_plot):
        ax  = axes[i]
        sub_pre = pre_ban[pre_ban["subreddit"] == sub].sort_values("days_from_ban")
        g_star  = g_stars[sub]
        regime  = results[results["subreddit"] == sub]["regime"].iloc[0] \
                  if sub in results["subreddit"].values else "Unknown"
        color   = REGIME_COLORS.get(regime, "#888780")
        banned  = sub in BANNED

        ax.plot(sub_pre["days_from_ban"], sub_pre["C_smooth"],
                color=color, linewidth=2, label="C (enforcement)")
        ax.axhline(g_star, color="#E24B4A", linestyle="--", linewidth=1.5,
                   label=f"g* = {g_star:.3f}")
        ax.fill_between(sub_pre["days_from_ban"],
                        sub_pre["C_smooth"], g_star,
                        where=sub_pre["C_smooth"] >= g_star,
                        alpha=0.25, color="#E24B4A",
                        label="C > g* (backfire zone)")

        ax.set_title(f"r/{sub}{'  [BANNED]' if banned else ''}\n{regime}",
                     fontsize=9, fontweight="bold", color=color)
        ax.set_xlabel("Days before ban", fontsize=8)
        ax.set_ylabel("C", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Pre-ban enforcement escalation: did C exceed g* before June 10?\n"
        "Red shading = enforcement in backfire zone",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(plot_dir / "stage2_preban_escalation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: stage2_preban_escalation.png")


def plot_equilibrium_surface(results: pd.DataFrame, plot_dir: Path):
    """
    Scatter of C* (equilibrium enforcement) vs T* (equilibrium trust)
    coloured by regime. Shows the governance trade-off surface.
    """
    df = results.dropna(subset=["C_equilibrium", "T_equilibrium"]).copy()
    if df.empty:
        print("  Skipping stage2_equilibrium_surface.png — no equilibria found")
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    for regime, grp in df.groupby("regime"):
        color = REGIME_COLORS.get(regime, "#888780")
        ax.scatter(grp["C_equilibrium"], grp["T_equilibrium"],
                   color=color, s=100, alpha=0.85, label=regime, zorder=3)
        for _, r in grp.iterrows():
            ax.annotate(r["subreddit"],
                        (r["C_equilibrium"], r["T_equilibrium"]),
                        fontsize=8, alpha=0.8,
                        xytext=(4, 4), textcoords="offset points")

    # Mark g* boundary as a vertical reference
    if "g_star" in df.columns:
        for _, r in df.iterrows():
            if not np.isnan(r.get("g_star", np.nan)):
                ax.axvline(r["g_star"], color=REGIME_COLORS.get(r["regime"], "gray"),
                           alpha=0.15, linewidth=0.8)

    ax.set_xlabel("C* — equilibrium enforcement intensity", fontsize=12)
    ax.set_ylabel("T* — equilibrium trust level", fontsize=12)
    ax.set_title(
        "Stage 2 equilibrium surface: where each community settles\n"
        "Higher T* and lower C* = better governance outcome",
        fontsize=11, fontweight="bold"
    )
    patches = [mpatches.Patch(color=v, label=k) for k, v in REGIME_COLORS.items()
               if k in df["regime"].values]
    ax.legend(handles=patches, fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(plot_dir / "stage2_equilibrium_surface.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: stage2_equilibrium_surface.png")


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
    daily_metrics = pd.read_parquet(out_dir / "daily_metrics.parquet")
    stage1_params = pd.read_csv(out_dir / "calibrated_params.csv")

    daily_metrics["day"] = pd.to_datetime(daily_metrics["day"], utc=True)

    print(f"  {len(stage1_params)} subreddits with Stage 1 parameters")
    print(f"  {len(daily_metrics)} subreddit-day rows\n")

    # --------------- Step 1: Estimate α_gov and δ per subreddit ---------------
    print("Estimating governance parameters (α_gov, δ) per subreddit...")
    results = []

    for _, s1 in stage1_params.iterrows():
        sub      = s1["subreddit"]
        sub_data = daily_metrics[daily_metrics["subreddit"] == sub].copy()

        # x_target = mean pre-ban violation rate
        pre_ban = sub_data[sub_data["day"] < BAN_DATE]
        x_target = float(pre_ban["C_raw"].mean()) if len(pre_ban) >= 3 else \
                   float(sub_data["C_raw"].mean())
        x_target = max(x_target, 0.01)

        alpha_gov, delta, r2 = estimate_governance_params(sub_data, x_target)

        # C volatility = std of weekly removal rate
        sub_data["week"] = sub_data["day"].dt.to_period("W").apply(
            lambda p: p.start_time
        )
        C_volatility = sub_data.groupby("week")["C_smooth"].mean().std()

        row = {
            "subreddit":   sub,
            "is_banned":   sub in BANNED,
            "x_target":    x_target,
            "alpha_gov":   alpha_gov,
            "delta":       delta,
            "gov_r2":      r2,
            "C_volatility": C_volatility,
            # Stage 1 parameters
            "sigma":  s1["sigma"],  "theta":  s1["theta"],
            "alpha":  s1["alpha"],  "beta":   s1["beta"],
            "gamma":  s1["gamma"],  "T0":     s1["T0"],
            "kappa":  s1["kappa"],  "g_star": s1["g_star"],
        }

        if alpha_gov is not None:
            print(f"  r/{sub:<25}  α_gov={alpha_gov:.4f}  δ={delta:.4f}"
                  f"  R²={r2:.3f}  x_target={x_target:.3f}")
        else:
            print(f"  r/{sub:<25}  [governance estimation failed]")

        results.append(row)

    results_df = pd.DataFrame(results)

    # --------------- Step 2: Compute Stage 2 equilibria ----------------------
    print("\nComputing Stage 2 equilibria (C*, T*, Ξ)...")
    C_stars, T_stars, Xis = [], [], []

    for _, row in results_df.iterrows():
        if row["alpha_gov"] is None or np.isnan(row.get("alpha_gov", np.nan)):
            C_stars.append(np.nan)
            T_stars.append(np.nan)
            Xis.append(np.nan)
            continue
        C_star, T_star, Xi = compute_equilibrium(row.to_dict())
        C_stars.append(C_star)
        T_stars.append(T_star)
        Xis.append(Xi)

    results_df["C_equilibrium"] = C_stars
    results_df["T_equilibrium"] = T_stars
    results_df["Xi"]            = Xis

    # --------------- Step 3: Compute α_gov,min and classify regimes ----------
    print("\nClassifying stability regimes...")
    results_df["alpha_gov_min"] = results_df.apply(
        lambda r: compute_alpha_gov_min(r.to_dict()), axis=1
    )
    results_df["regime"] = results_df.apply(
        lambda r: classify_regime(r.to_dict()), axis=1
    )

    # --------------- Step 4: Print results table -----------------------------
    print("\n" + "="*75)
    print("STAGE 2 RESULTS")
    print("="*75)
    print(f"\n{'Subreddit':<22} {'Regime':<25} {'α_gov':>7}  "
          f"{'α_gov,min':>10}  {'C*':>6}  {'g*':>6}  {'Ξ':>8}")
    print("-"*90)

    for _, r in results_df.sort_values("regime").iterrows():
        banned = "[BAN]" if r["is_banned"] else "     "
        alpha_gov_min_str = f"{r['alpha_gov_min']:.4f}" \
                            if np.isfinite(r.get("alpha_gov_min", np.inf)) else "∞"
        C_eq_str  = f"{r['C_equilibrium']:.4f}" \
                    if not np.isnan(r.get("C_equilibrium", np.nan)) else "n/a"
        Xi_str    = f"{r['Xi']:.4f}" \
                    if not np.isnan(r.get("Xi", np.nan)) else "n/a"
        ag_str    = f"{r['alpha_gov']:.4f}" \
                    if r["alpha_gov"] is not None and not np.isnan(r.get("alpha_gov", np.nan)) \
                    else "n/a"

        print(f"  {banned} r/{r['subreddit']:<20} {r['regime']:<25} "
              f"{ag_str:>7}  {alpha_gov_min_str:>10}  "
              f"{C_eq_str:>6}  {r['g_star']:>6.4f}  {Xi_str:>8}")

    # Summary
    print("\n--- Regime counts ---")
    for regime, grp in results_df.groupby("regime"):
        print(f"  {regime:<25}  {len(grp):>2} subreddits")

    # --------------- Step 5: Key finding — governance trap test --------------
    print("\n--- Governance trap test ---")
    print("Prediction: governance-trap communities have higher C volatility\n"
          "than stable communities.\n")

    for regime in ["Stable", "Governance trap", "Structural instability"]:
        grp = results_df[results_df["regime"] == regime]
        if grp.empty:
            continue
        vol = grp["C_volatility"].dropna()
        print(f"  {regime:<25}  mean C volatility = {vol.mean():.4f}  "
              f"(n={len(vol)})")

    # KS test between regimes if enough data
    stable_vol = results_df[results_df["regime"] == "Stable"]["C_volatility"].dropna()
    trap_vol   = results_df[results_df["regime"] == "Governance trap"]["C_volatility"].dropna()
    if len(stable_vol) >= 3 and len(trap_vol) >= 3:
        ks_stat, p_val = stats.ks_2samp(stable_vol, trap_vol)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 \
              else "*" if p_val < 0.05 else "ns"
        print(f"\n  KS test (stable vs governance trap): "
              f"D={ks_stat:.3f}, p={p_val:.3f} {sig}")

    # --------------- Step 6: Pre-ban escalation test -------------------------
    print("\n--- Pre-ban enforcement escalation test ---")
    print("Was C rising significantly in the 9 days before June 10?\n")

    for sub in results_df["subreddit"].values:
        sub_data = daily_metrics[
            (daily_metrics["subreddit"] == sub) &
            (daily_metrics["day"] < BAN_DATE)
        ].sort_values("day")

        if len(sub_data) < 4:
            continue

        days  = np.arange(len(sub_data))
        C_vals = sub_data["C_smooth"].fillna(0).values

        if C_vals.std() < 1e-6:
            continue

        slope, intercept, r, p, se = stats.linregress(days, C_vals)
        g_star = results_df[results_df["subreddit"] == sub]["g_star"].iloc[0]
        regime = results_df[results_df["subreddit"] == sub]["regime"].iloc[0]
        sig    = "*" if p < 0.05 else ""
        banned = "[BAN]" if sub in BANNED else "     "

        print(f"  {banned} r/{sub:<22}  slope={slope:+.5f}  "
              f"p={p:.3f}{sig}  regime={regime}")

    # Save
    results_df.to_csv(out_dir / "stage2_results.csv", index=False)
    print(f"\nSaved: stage2_results.csv")

    # --------------- Step 7: Generate plots ----------------------------------
    print("\nGenerating plots...")
    plot_regime_map(results_df, plot_dir)
    plot_trajectories(results_df, daily_metrics, plot_dir)
    plot_governance_trap(results_df, plot_dir)
    plot_preban_escalation(daily_metrics, results_df, plot_dir)
    plot_equilibrium_surface(results_df, plot_dir)

    print(f"\nAll Stage 2 outputs saved.")
    print("Plots:")
    for p in sorted(plot_dir.glob("stage2_*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()