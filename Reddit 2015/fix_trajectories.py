"""
fix_trajectories_v3.py
======================
Root cause of NaN: stage2_results.csv has NaN in Stage 1 columns
(sigma, theta, alpha etc.) because the merge in stage2_analysis.py
silently dropped them for most rows.

Fix: read calibrated_params.csv and stage2_results.csv separately,
join on subreddit, then simulate.

Run:
    python3 fix_trajectories_v3.py --out_dir ./output
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BANNED   = {"fatpeoplehate", "CoonTown", "transfags", "neofag", "hamplanet"}
BAN_DATE = pd.Timestamp("2015-06-10", tz="UTC")
REGIME_COLORS = {
    "Structural instability": "#E24B4A",
    "Indeterminate":          "#888780",
    "Stable":                 "#1D9E75",
    "Governance trap":        "#EF9F27",
}


def euler_stage2(p, x0, T0, C0, n_days, dt=0.05):
    """Forward Euler with conservative caps. Returns (t, x, T, C) daily arrays."""
    sigma     = min(abs(p["sigma"]),     2.0)
    theta     = float(p["theta"])
    alpha     = min(abs(p["alpha"]),     1.0)
    beta      = min(abs(p["beta"]),      2.0)
    gamma     = min(abs(p["gamma"]),     2.0)
    T0_par    = min(abs(p["T0"]),        1.0)
    kappa     = min(abs(p["kappa"]),     4.0)
    alpha_gov = min(abs(p["alpha_gov"]), 0.5)
    delta     = max(abs(p["delta"]),     0.05)  # floor at 0.05
    x_target  = float(np.clip(p["x_target"], 0.01, 0.50))

    x = float(np.clip(x0, 0.01, 0.99))
    T = float(np.clip(T0, 0.01, 1.00))
    C = float(np.clip(C0, 0.001, 0.50))

    steps = int(n_days / dt)
    record_every = max(1, int(1.0 / dt))

    t_arr, x_arr, T_arr, C_arr = [], [], [], []

    for step in range(steps):
        if step % record_every == 0:
            t_arr.append(step * dt)
            x_arr.append(x)
            T_arr.append(T)
            C_arr.append(C)

        Tk = max(T, 0.001) ** kappa
        dx = sigma * x * (1 - x) * (theta + alpha * x - C * Tk)
        dT = -beta * C + gamma * (T0_par - T)
        dC = alpha_gov * (x - x_target) - delta * C

        dx = float(np.clip(dx, -0.3, 0.3))
        dT = float(np.clip(dT, -0.3, 0.3))
        dC = float(np.clip(dC, -0.2, 0.2))

        x = float(np.clip(x + dx * dt, 0.001, 0.999))
        T = float(np.clip(T + dT * dt, 0.001, 1.5))
        C = float(np.clip(C + dC * dt, 0.000, 1.000))

    return np.array(t_arr), np.array(x_arr), np.array(T_arr), np.array(C_arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./output")
    args     = parser.parse_args()
    out_dir  = Path(args.out_dir)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # --- Load from SEPARATE files and join ---
    s1 = pd.read_csv(out_dir / "calibrated_params.csv")
    s2 = pd.read_csv(out_dir / "stage2_results.csv")[
        ["subreddit", "alpha_gov", "delta", "x_target", "regime",
         "g_star", "C_volatility", "is_banned"]
    ]
    params = s1.merge(s2, on="subreddit", how="inner")

    dm = pd.read_parquet(out_dir / "daily_metrics.parquet")
    dm["day"] = pd.to_datetime(dm["day"], utc=True)

    # --- Diagnostic: print all parameter values ---
    print(f"{'Subreddit':<22} {'σ':>6} {'θ':>7} {'α':>6} "
          f"{'β':>6} {'γ':>6} {'κ':>5} "
          f"{'α_gov':>7} {'δ':>7} {'x_tgt':>7}")
    print("-" * 90)
    for _, r in params.iterrows():
        print(f"  r/{r['subreddit']:<20} "
              f"{r['sigma']:>6.3f} {r['theta']:>7.3f} {r['alpha']:>6.3f} "
              f"{r['beta']:>6.3f} {r['gamma']:>6.3f} {r['kappa']:>5.2f} "
              f"{r['alpha_gov']:>7.4f} {r['delta']:>7.4f} {r['x_target']:>7.3f}")

    # --- Plot ---
    n    = len(params)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(5 * cols, 4 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, (_, row) in enumerate(params.iterrows()):
        sub      = row["subreddit"]
        ax       = axes_flat[idx]
        sub_data = dm[dm["subreddit"] == sub].sort_values("day")

        if sub_data.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        n_days = len(sub_data)
        x_obs  = sub_data["C_raw"].fillna(0).values
        x0     = float(np.clip(x_obs[0] if x_obs[0] > 0 else 0.05, 0.01, 0.99))
        T0     = float(np.clip(row["T0"], 0.01, 1.0))
        C0     = float(np.clip(sub_data["C_smooth"].fillna(0).iloc[0], 0.001, 0.5))

        try:
            t_out, x_sim, T_sim, C_sim = euler_stage2(
                row.to_dict(), x0, T0, C0, n_days, dt=0.05
            )
            ok = (len(t_out) > 2 and
                  not np.isnan(x_sim).any() and
                  not np.isnan(T_sim).any() and
                  not np.isnan(C_sim).any())
        except Exception as e:
            print(f"  r/{sub}: {e}")
            ok = False

        if not ok:
            ax.text(0.5, 0.5, f"r/{sub}\nfailed",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            ax.set_title(f"r/{sub}", fontsize=9)
            continue

        print(f"  r/{sub:<22} x∈[{x_sim.min():.3f},{x_sim.max():.3f}]  "
              f"T∈[{T_sim.min():.3f},{T_sim.max():.3f}]  "
              f"C∈[{C_sim.min():.3f},{C_sim.max():.3f}]")

        # Observed
        ax.plot(range(len(x_obs)), x_obs,
                color="#E24B4A", alpha=0.2, linewidth=1,
                linestyle="--", label="x observed")

        # Simulated
        ax.plot(t_out, x_sim, color="#E24B4A", linewidth=2,   label="x violations")
        ax.plot(t_out, T_sim, color="#1D9E75", linewidth=1.8, label="T trust")
        ax.plot(t_out, C_sim, color="#378ADD", linewidth=1.8, label="C enforcement")

        # Ban line
        ban_day = (BAN_DATE - sub_data["day"].min()).days
        if 0 < ban_day < n_days:
            ax.axvline(ban_day, color="black", linestyle=":",
                       linewidth=1.2, alpha=0.7)

        # g* line
        g_star = float(row.get("g_star", np.nan))
        if not np.isnan(g_star) and 0 < g_star < 0.6:
            ax.axhline(g_star, color="#EF9F27", linestyle="--",
                       linewidth=1.2, alpha=0.9,
                       label=f"g*={g_star:.3f}")

        regime = str(row.get("regime", "Indeterminate"))
        color  = REGIME_COLORS.get(regime, "#444441")
        ax.set_title(
            f"r/{sub}{'  [BAN]' if sub in BANNED else ''}\n{regime}",
            fontsize=9, fontweight="bold", color=color
        )
        ax.set_xlabel("Day", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        ax.set_xlim(0, n_days)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Stage 2: x (violations), T (trust), C (enforcement) co-evolving\n"
        "Dashed red=observed x  |  Orange=g*  |  Dotted=ban wave",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(plot_dir / "stage2_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved: stage2_trajectories.png")


if __name__ == "__main__":
    main()