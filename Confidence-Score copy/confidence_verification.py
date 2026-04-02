"""
confidence_verification.py
===========================
Generates all publication-quality tables and figures for the MCI validation
section of the Coordination-Legitimacy Game paper.

Reads from <out_dir>/confidence/ (produced by confidence_score.py).

Outputs (all in <out_dir>/confidence/):
  tables/
    table1_mci_decomposition.csv      ← Main results table for the paper
    table2_regressions.csv            ← T→C substitution + theta_proxy regression
    table3_statistical_tests.csv      ← All hypothesis tests in one summary
  plots/
    mci_bar.png                       ← MCI composite bar chart by subreddit
    mci_radar.png                     ← Spider/radar chart: 6 components per subreddit
    mci_vs_gstar.png                  ← MCI vs g* scatter, coloured by regime
    tc_substitution.png               ← T vs C scatter with partial regression line
    user_consistency_violin.png       ← ρ_u distributions by user type
    spillover_rates.png               ← Migrant vs native post-ban removal rates
    mci_sensitivity_heatmap.png       ← MCI rank stability under 500 weight draws
    regime_margin_volatility.png      ← Q_regime margin vs C_volatility

Usage:
    python confidence_verification.py --out_dir ../Reddit-2015-v2/output

Run AFTER confidence_score.py has completed.
"""

import argparse
import json
import logging
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour / style constants (matching the existing pipeline plots)
# ---------------------------------------------------------------------------

PALETTE = {
    "banned":     "#E24B4A",
    "stable":     "#1D9E75",
    "trap":       "#EF9F27",
    "structural": "#E24B4A",
    "control":    "#378ADD",
    "neutral":    "#888780",
    "initiator":  "#E24B4A",
    "joiner":     "#EF9F27",
    "complier":   "#1D9E75",
}

REGIME_COLORS = {
    "Stable":                 PALETTE["stable"],
    "Governance trap":        PALETTE["trap"],
    "Structural instability": PALETTE["structural"],
    "Indeterminate":          PALETTE["neutral"],
}

BANNED_SUBREDDITS = {
    "fatpeoplehate", "CoonTown", "transfags", "neofag", "hamplanet",
}

COMPONENT_LABELS = {
    "Q_fit":          "ODE Fit\n(Q_fit)",
    "Q_backfire":     "Backfire\nAccuracy\n(Q_back)",
    "Q_user":         "User\nConsistency\n(Q_user)",
    "Q_regime":       "Regime\nMargin\n(Q_reg)",
}

COMPONENT_COLS = ["Q_fit_scaled", "Q_backfire",
                  "Q_user", "Q_regime"]


# ---------------------------------------------------------------------------
# Load all score files
# ---------------------------------------------------------------------------

def load_scores(conf_dir: Path) -> dict:
    """
    Loads all CSVs / parquets / JSON produced by confidence_score.py.
    Returns a dict of dataframes + scalar dicts.
    """
    required = {
        "mci":     conf_dir / "subreddit_mci.csv",
        "scalars": conf_dir / "tables" / "scalar_results.json",
    }
    optional = {
        "rho":      conf_dir / "user_behavioral.parquet",
        "regime":   conf_dir / "tables" / "regime_scores.csv",
        "backfire": conf_dir / "tables" / "backfire_scores.csv",
        "fit":      conf_dir / "tables" / "fit_scores.csv",
    }

    data = {}
    for key, path in required.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing: {path}\n"
                "Run confidence_score.py first."
            )
        if path.suffix == ".csv":
            data[key] = pd.read_csv(path)
        else:
            with open(path) as f:
                data[key] = json.load(f)

    for key, path in optional.items():
        if path.exists():
            if path.suffix == ".parquet":
                data[key] = pd.read_parquet(path)
            else:
                data[key] = pd.read_csv(path)
        else:
            log.warning(f"Optional file not found (skipping plots that need it): {path.name}")
            data[key] = None

    return data


# ---------------------------------------------------------------------------
# Figure 1: MCI composite bar chart
# ---------------------------------------------------------------------------

def plot_mci_bar(mci: pd.DataFrame, plot_dir: Path):
    """
    Horizontal bar chart — one bar per subreddit — coloured by stability regime.
    Component breakdown shown as stacked segments.
    """
    df = mci.copy().sort_values("MCI", ascending=True)

    fig, (ax_main, ax_comp) = plt.subplots(
        1, 2, figsize=(16, max(6, len(df) * 0.55)),
        gridspec_kw={"width_ratios": [1, 2]}
    )

    y = np.arange(len(df))

    # Left panel: composite MCI bar
    for i, (_, row) in enumerate(df.iterrows()):
        regime = str(row.get("regime", ""))
        color  = REGIME_COLORS.get(regime, PALETTE["control"])
        if row["is_banned"]:
            color = PALETTE["banned"]
        ax_main.barh(i, row["MCI"], color=color, alpha=0.85, height=0.65)
        ax_main.text(row["MCI"] + 0.005, i,
                     f"{row['MCI']:.3f}", va="center", fontsize=8)

    labels = [
        f"r/{r['subreddit']}  {'[BANNED]' if r['is_banned'] else ''}"
        for _, r in df.iterrows()
    ]
    ax_main.set_yticks(y)
    ax_main.set_yticklabels(labels, fontsize=9)
    ax_main.set_xlabel("Model Credibility Index (MCI)", fontsize=11)
    ax_main.set_xlim(0, 1.08)
    ax_main.axvline(0.65, color="black", linestyle="--", linewidth=1,
                    label="Strong credibility (0.65)")
    ax_main.axvline(0.45, color="gray",  linestyle=":",  linewidth=1,
                    label="Partial credibility (0.45)")
    ax_main.legend(fontsize=8, loc="lower right")
    ax_main.set_title("Composite MCI\n(coloured by regime)", fontsize=11, fontweight="bold")

    # Right panel: stacked component breakdown
    comp_map = {
        "Q_fit_scaled":    ("ODE Fit",       "#1D9E75"),
        "Q_backfire":      ("Backfire",       "#378ADD"),
        "Q_substitution":  ("T–C Sub.",       "#EF9F27"),
        "Q_user":          ("User Consist.",  "#9B59B6"),
        "Q_regime":        ("Regime Margin",  "#E24B4A"),
        "Q_spillover":     ("Spillover",      "#E67E22"),
    }

    left = np.zeros(len(df))
    for col, (label, color) in comp_map.items():
        if col not in df.columns:
            continue
        vals = df[col].fillna(0).clip(0, 1).values / 6.0   # each worth 1/6 of MCI
        ax_comp.barh(y, vals, left=left, color=color, alpha=0.80,
                     height=0.65, label=label)
        left += vals

    ax_comp.set_yticks(y)
    ax_comp.set_yticklabels([""] * len(df))
    ax_comp.set_xlabel("MCI contribution by component (each capped at 1/6)", fontsize=11)
    ax_comp.set_xlim(0, 1.05)
    ax_comp.legend(fontsize=8, loc="lower right", ncol=2)
    ax_comp.set_title("Component breakdown\n(stacked)", fontsize=11, fontweight="bold")

    patches = [
        mpatches.Patch(color=v, label=k)
        for k, v in REGIME_COLORS.items()
        if k in mci.get("regime", pd.Series()).values
    ]
    patches.append(mpatches.Patch(color=PALETTE["banned"], label="Banned subreddit"))
    fig.legend(handles=patches, fontsize=8, loc="lower center",
               ncol=len(patches), bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        "Model Credibility Index (MCI): Coordination-Legitimacy Game validation\n"
        "Higher MCI = stronger empirical support for the structural model",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(plot_dir / "mci_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved: mci_bar.png")


# ---------------------------------------------------------------------------
# Figure 2: Radar / spider chart
# ---------------------------------------------------------------------------

def plot_mci_radar(mci: pd.DataFrame, plot_dir: Path):
    """
    One spider chart per subreddit (up to 12, arranged in a grid).
    Each spoke = one MCI component.  Shows which dimensions are strong/weak.
    """
    component_cols = ["Q_fit_scaled", "Q_backfire", "Q_substitution",
                      "Q_user", "Q_regime", "Q_spillover"]
    labels = ["ODE Fit", "Backfire\nAccuracy", "T–C\nSubstitution",
              "User\nConsistency", "Regime\nMargin", "Spillover"]

    df = mci.dropna(subset=["MCI"]).copy()
    n  = min(len(df), 15)
    df = df.sort_values("MCI", ascending=False).head(n)

    cols = min(5, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(4.5 * cols, 4 * rows),
        subplot_kw={"projection": "polar"}
    )
    axes = np.array(axes).flatten()

    num_vars = len(component_cols)
    angles   = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles  += angles[:1]   # close the polygon

    for i, (_, row) in enumerate(df.iterrows()):
        ax = axes[i]
        values = [
            float(row[c]) if (c in row.index and not np.isnan(row[c])) else 0.0
            for c in component_cols
        ]
        values = [np.clip(v, 0, 1) for v in values]
        values_plot = values + values[:1]

        # Regime / banned colour
        regime = str(row.get("regime", ""))
        if row["is_banned"]:
            color = PALETTE["banned"]
        else:
            color = REGIME_COLORS.get(regime, PALETTE["control"])

        ax.plot(angles, values_plot, color=color, linewidth=2)
        ax.fill(angles, values_plot, color=color, alpha=0.20)

        # Reference circles
        for level in [0.25, 0.50, 0.75, 1.0]:
            ax.plot(angles, [level] * (num_vars + 1), color="gray",
                    linewidth=0.5, linestyle=":", alpha=0.5)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.50, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=5, color="gray")

        banned_tag = "  [BANNED]" if row["is_banned"] else ""
        ax.set_title(
            f"r/{row['subreddit']}{banned_tag}\n"
            f"MCI = {row['MCI']:.3f}",
            fontsize=9, fontweight="bold",
            color=color, pad=12
        )

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "MCI component profiles by subreddit\n"
        "Each spoke = one validity dimension (0 = weakest, 1 = strongest)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(plot_dir / "mci_radar.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved: mci_radar.png")


# ---------------------------------------------------------------------------
# Figure 3: MCI vs g*
# ---------------------------------------------------------------------------

def plot_mci_vs_gstar(mci: pd.DataFrame, plot_dir: Path):
    """
    Scatter: g* (x-axis) vs MCI (y-axis), coloured by banned/control status
    and shaped by regime.

    Prediction: structurally fragile communities (low g*) may have lower
    Q_backfire but similar Q_fit — this appears as a diagonal slope in the scatter.
    """
    df = mci.dropna(subset=["MCI", "g_star"]).copy()
    if df.empty:
        log.warning("  Skipping mci_vs_gstar.png — no valid data.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    for _, row in df.iterrows():
        color   = PALETTE["banned"] if row["is_banned"] else PALETTE["control"]
        marker  = "D" if row["is_banned"] else "o"
        regime  = str(row.get("regime", ""))
        msize   = 120 if row["is_banned"] else 80

        ax.scatter(row["g_star"], row["MCI"],
                   color=color, marker=marker, s=msize,
                   alpha=0.85, zorder=3, edgecolors="white", linewidth=0.5)
        ax.annotate(
            f"r/{row['subreddit']}",
            (row["g_star"], row["MCI"]),
            fontsize=7.5, alpha=0.8,
            xytext=(5, 3), textcoords="offset points"
        )

    # Trend line (all subreddits)
    if len(df) > 3:
        z = np.polyfit(df["g_star"], df["MCI"], 1)
        g_line = np.linspace(df["g_star"].min(), df["g_star"].max(), 100)
        ax.plot(g_line, np.poly1d(z)(g_line),
                "k--", linewidth=1.2, alpha=0.5,
                label=f"Trend (slope={z[0]:.3f})")

    # Reference lines
    ax.axhline(0.65, color="gray", linestyle=":", linewidth=1,
               label="Strong credibility threshold (0.65)")
    ax.axhline(0.45, color="lightgray", linestyle=":", linewidth=1,
               label="Partial credibility threshold (0.45)")

    legend_patches = [
        mpatches.Patch(color=PALETTE["banned"],  label="Banned subreddit"),
        mpatches.Patch(color=PALETTE["control"], label="Control subreddit"),
    ]
    ax.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color="k", linestyle="--", label="Trend")
    ], fontsize=9)

    ax.set_xlabel("g* — Backfire Threshold (structural fragility; lower = more fragile)",
                  fontsize=12)
    ax.set_ylabel("Model Credibility Index (MCI)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "MCI vs Backfire Threshold g*\n"
        "Tests whether structurally fragile communities are also empirically harder to fit",
        fontsize=11, fontweight="bold"
    )
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(plot_dir / "mci_vs_gstar.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved: mci_vs_gstar.png")


# ---------------------------------------------------------------------------
# Figure 4: T–C substitution scatter
# ---------------------------------------------------------------------------

#def plot_tc_substitution(daily_metrics_path: Path, plot_dir: #ath):
#   """
#   Scatter of mean T_norm vs mean C_smooth per subreddit-month.
#   Partial regression line from Q_substitution OLS (β₁ on T controlling compliance).
#
#   Annotates each subreddit with its name.  Coloured by banned/control.
#   """
#   if not daily_metrics_path.exists():
#       log.warning("  Skipping tc_substitution.png — daily_metrics.parquet not found.")
#       return
#
#   dm = pd.read_parquet(daily_metrics_path)
#   dm["day"] = pd.to_datetime(dm["day"], utc=True)
#
#   sub_agg = dm.groupby("subreddit").agg(
#       T_mean     = ("T_norm",   "mean"),
#       C_mean     = ("C_smooth", "mean"),
#       compliance = ("C_raw",    lambda s: 1 - s.mean()),
#   ).reset_index().dropna()
#
#   if sub_agg.empty:
#       log.warning("  Skipping tc_substitution.png — empty after aggregation.")
#       return
#
#   sub_agg["is_banned"] = sub_agg["subreddit"].isin(BANNED_SUBREDDITS)
#   compliance = sub_agg["compliance"].values
#
#   fig, ax = plt.subplots(figsize=(10, 7))
#
#   for _, row in sub_agg.iterrows():
#       color  = PALETTE["banned"] if row["is_banned"] else PALETTE["control"]
#       marker = "D" if row["is_banned"] else "o"
#       ax.scatter(row["T_mean"], row["C_mean"],
#                  c=color, marker=marker, s=100,
#                  alpha=0.85, zorder=3,
#                  edgecolors="white", linewidth=0.5)
#       ax.annotate(f"r/{row['subreddit']}",
#                   (row["T_mean"], row["C_mean"]),
#                   fontsize=7.5, alpha=0.75,
#                   xytext=(4, 3), textcoords="offset points")
#
#   # Partial regression line: residuals of C on compliance vs residuals of T on compliance
#   if len(sub_agg) > 5:
#       X_c = compliance.reshape(-1, 1) - compliance.mean()
#       def ols_resid(y):
#           b = np.linalg.lstsq(X_c, y - y.mean(), rcond=None)[0]
#           return (y - y.mean()) - X_c @ b
#
#       resid_T = ols_resid(sub_agg["T_mean"].values)
#       resid_C = ols_resid(sub_agg["C_mean"].values)
#       if resid_T.std() > 1e-8:
#           z = np.polyfit(resid_T, resid_C, 1)
#           t_line = np.linspace(resid_T.min(), resid_T.max(), 100)
#           t_plot = t_line + sub_agg["T_mean"].mean()
#           c_plot = np.poly1d(z)(t_line) + sub_agg["C_mean"].mean()
#           ax.plot(t_plot, c_plot, color="black", linewidth=1.5, linestyle="--",
#                   label=f"Partial regression (slope={z[0]:.3f})\n[controlling for compliance]")
#
#   # Labels and style
#   ax.set_xlabel("Mean T_norm — Community Legitimacy / Trust", fontsize=12)
#   ax.set_ylabel("Mean C_smooth — Enforcement Intensity (removal rate)", fontsize=12)
#   ax.set_title(
#       "T–C Substitution Space\n"
#       "Negative partial slope confirms: high-trust communities achieve equal\n"
#       "compliance with lower enforcement (Tyler 1990; Levi 1988)",
#       fontsize=11, fontweight="bold"
#   )
#
#   legend_patches = [
#       mpatches.Patch(color=PALETTE["banned"],  label="Banned subreddit"),
#       mpatches.Patch(color=PALETTE["control"], label="Control subreddit"),
#   ]
#   ax.legend(handles=legend_patches + [
#       plt.Line2D([0], [0], color="black", linestyle="--", label="Partial regression")
#   ], fontsize=9)
#   ax.grid(alpha=0.3)
#   plt.tight_layout()
#   fig.savefig(plot_dir / "tc_substitution.png", dpi=150, bbox_inches="tight")
#   plt.close()
#   log.info("  Saved: tc_substitution.png")


# ---------------------------------------------------------------------------
# Figure 5: User ρ_u violin plot
# ---------------------------------------------------------------------------

def plot_user_consistency_violin(rho_df: pd.DataFrame, plot_dir: Path):
    """
    Violin + strip plot of ρ_u (enforcement-removal correlation) by user_type.

    Prediction (Granovetter 1978):
        JOINER ρ_u < 0  (exploit low-enforcement windows)
        INITIATOR ρ_u ≈ 0  (enforcement-insensitive)
        COMPLIER ρ_u ≈ 0   (never removed)
    """
    if rho_df is None or rho_df.empty:
        log.warning("  Skipping user_consistency_violin.png — no rho data.")
        return

    order   = ["INITIATOR", "JOINER", "COMPLIER"]
    colors  = [PALETTE["initiator"], PALETTE["joiner"], PALETTE["complier"]]
    labels  = {
        "INITIATOR": "INITIATOR\n(θ < 0: enforcement-\ninsensitive rule-breakers)",
        "JOINER":    "JOINER\n(0 < θ < 1: exploit\nlow-enforcement windows)",
        "COMPLIER":  "COMPLIER\n(θ > 1: never\nbreak rules)",
    }

    fig, ax = plt.subplots(figsize=(11, 7))

    positions = list(range(len(order)))
    for pos, (utype, color) in enumerate(zip(order, colors)):
        subset = rho_df[rho_df["user_type"] == utype]["rho_u"].dropna()
        if subset.empty:
            continue

        # Violin
        parts = ax.violinplot(subset, positions=[pos], showmedians=True,
                              showextrema=True, widths=0.65)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.55)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2)

        # Jittered strip (subsample if large)
        sample = subset.sample(min(500, len(subset)), random_state=42)
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(sample))
        ax.scatter(pos + jitter, sample, color=color,
                   alpha=0.25, s=4, zorder=2)

        # Annotate mean + n
        ax.text(pos, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.5,
                f"μ={subset.mean():.3f}\nn={len(subset):,}",
                ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")

    ax.axhline(0, color="black", linestyle="--", linewidth=1.2,
               label="ρ = 0 (no correlation)")

    ax.set_xticks(positions)
    ax.set_xticklabels([labels[t] for t in order], fontsize=10)
    ax.set_ylabel("ρ_u — Pearson r(C_smooth, is_removed) per user", fontsize=12)
    ax.set_title(
        "User Behavioral Consistency by Type\n"
        "Prediction: JOINER ρ_u < 0 (opportunistic); "
        "INITIATOR & COMPLIER ρ_u ≈ 0 (enforcement-insensitive)\n"
        "[Validates user heterogeneity assumption — Granovetter 1978]",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(plot_dir / "user_consistency_violin.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved: user_consistency_violin.png")


# ---------------------------------------------------------------------------
# Figure 6: Spillover — migrant vs native removal rates
# ---------------------------------------------------------------------------
#
#def plot_spillover_rates(scalars: dict, plot_dir: Path):
#    """
#    Bar chart comparing mean post-ban removal rates:
#    Migrant INITIATORs vs Native users.
#
#    Shows Cohen's d and Mann-Whitney p as annotations.
#    """
#    sp = scalars.get("Q_spillover", {})
#    mr = sp.get("mean_rate_migrants", np.nan)
#    nr = sp.get("mean_rate_natives",  np.nan)
#    d  = sp.get("cohens_d",           np.nan)
#    p  = sp.get("mw_p",               np.nan)
#    nm = sp.get("n_migrants",          0)
#    nn = sp.get("n_natives",           0)
#
#    if np.isnan(mr) or np.isnan(nr):
#        log.warning("  Skipping spillover_rates.png — no #spillover data.")
#        return
#
#    fig, ax = plt.subplots(figsize=(8, 6))
#
#    bars = ax.bar(
#        ["Migrant INITIATORs\n(from banned subs)",
#         "Native users\n(control subs)"],
#        [mr, nr],
#        color=[PALETTE["initiator"], PALETTE["control"]],
#        alpha=0.80, width=0.45, edgecolor="white"
#    )
#
#    # Count labels
#    for bar, n_val in zip(bars, [nm, nn]):
#        ax.text(bar.get_x() + bar.get_width() / 2,
#                bar.get_height() + 0.002,
#                f"n={n_val:,}", ha="center", va="bottom", #fontsize=10)
#
#    # Effect size annotation
#    sig_stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" #if p < 0.05 else "ns"
#    y_bracket = max(mr, nr) + 0.015
#    ax.plot([0, 1], [y_bracket, y_bracket], color="black", #linewidth=1.2)
#    ax.text(
#        0.5, y_bracket + 0.003,
#        f"d={d:.3f}  ({sig_stars})\nMann-Whitney p={p:.4f}",
#        ha="center", va="bottom", fontsize=10, fontweight="bold"
#    )
#
#    interp = ("LARGE" if abs(d) > 0.5 else "MEDIUM" if abs(d) > 0.#2 else "SMALL")
#    ax.set_ylabel("Post-ban removal rate", fontsize=12)
#    ax.set_title(
#        "Spillover: Migrant INITIATORs vs Native Users\n"
#        f"Effect size: Cohen's d = {d:.3f} ({interp})\n"
#        "[Validates behavioral type persistence — P4 of the model]#",
#        fontsize=11, fontweight="bold"
#    )
#    ax.set_ylim(0, max(mr, nr) * 1.35)
#    ax.grid(alpha=0.3, axis="y")
#    plt.tight_layout()
#    fig.savefig(plot_dir / "spillover_rates.png", dpi=150, #bbox_inches="tight")
#    plt.close()
#    log.info("  Saved: spillover_rates.png")


# ---------------------------------------------------------------------------
# Figure 7: Sensitivity heatmap
# ---------------------------------------------------------------------------

def plot_sensitivity_heatmap(mci: pd.DataFrame, plot_dir: Path,
                              n_draws: int = 500, seed: int = 42):
    """
    Heatmap: x-axis = weight draw index, y-axis = subreddit.
    Cell colour = rank of that subreddit under that weight draw.
    Shows whether the MCI ranking is stable or sensitive to weighting.
    """
    cols = ["Q_fit_scaled", "Q_backfire", "Q_substitution",
            "Q_user", "Q_regime", "Q_spillover"]

    df = mci[["subreddit"] + cols].copy()
    for c in cols:
        df[c] = df[c].fillna(df[c].mean())
    df = df.dropna(thresh=3)

    if len(df) < 4:
        log.warning("  Skipping sensitivity heatmap — too few subreddits.")
        return

    rng   = np.random.default_rng(seed)
    ranks = np.zeros((len(df), n_draws), dtype=float)

    for j in range(n_draws):
        w    = rng.uniform(0.05, 0.35, size=len(cols))
        w    = w / w.sum()
        mci_j = (df[cols].values * w).sum(axis=1)
        ranks[:, j] = pd.Series(mci_j).rank(ascending=False).values

    fig, ax = plt.subplots(figsize=(min(18, n_draws // 15 + 4), max(5, len(df) * 0.5)))
    im = ax.imshow(ranks, aspect="auto", cmap="RdYlGn_r",
                   vmin=1, vmax=len(df))
    plt.colorbar(im, ax=ax, label="Rank (1 = highest MCI)")

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(
        [f"r/{r['subreddit']}  {'[B]' if r['subreddit'] in BANNED_SUBREDDITS else '   '}"
         for _, r in df.iterrows()],
        fontsize=8
    )
    ax.set_xlabel(f"Weight draw (n={n_draws})", fontsize=11)
    ax.set_title(
        "MCI Ranking Stability under Random Weight Perturbations\n"
        "Stable colours = robust ranking regardless of weight choice\n"
        "High variance rows = subreddits whose rank is sensitive to weighting",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(plot_dir / "mci_sensitivity_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved: mci_sensitivity_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 8: Regime margin vs C volatility
# ---------------------------------------------------------------------------

def plot_regime_margin_volatility(regime_df: pd.DataFrame, plot_dir: Path):
    """
    Scatter: stability margin (Q5 input) vs C_volatility.
    Prediction: larger margin → lower volatility (β < 0).
    """
    if regime_df is None or regime_df.empty:
        log.warning("  Skipping regime_margin_volatility.png — no regime data.")
        return

    df = regime_df.dropna(subset=["stability_margin", "C_volatility"]).copy()
    if len(df) < 4:
        log.warning("  Skipping regime_margin_volatility.png — insufficient data.")
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    for _, row in df.iterrows():
        regime = str(row.get("regime", ""))
        color  = REGIME_COLORS.get(regime, PALETTE["neutral"])
        if row["subreddit"] in BANNED_SUBREDDITS:
            color = PALETTE["banned"]
        ax.scatter(row["stability_margin"], row["C_volatility"],
                   color=color, s=90, alpha=0.85, zorder=3,
                   edgecolors="white", linewidth=0.5)
        ax.annotate(f"r/{row['subreddit']}",
                    (row["stability_margin"], row["C_volatility"]),
                    fontsize=7.5, alpha=0.75,
                    xytext=(4, 3), textcoords="offset points")

    # Trend line
    if len(df) > 3:
        z = np.polyfit(df["stability_margin"], df["C_volatility"], 1)
        m_line = np.linspace(df["stability_margin"].min(),
                             df["stability_margin"].max(), 100)
        ax.plot(m_line, np.poly1d(z)(m_line), "k--", linewidth=1.2, alpha=0.6,
                label=f"Trend (slope={z[0]:.4f})")

    # Zero margin line
    ax.axvline(0, color="gray", linestyle=":", linewidth=1.2,
               label="Stability boundary (margin=0)")

    patches = [mpatches.Patch(color=v, label=k)
               for k, v in REGIME_COLORS.items() if k in df["regime"].values]
    patches.append(mpatches.Patch(color=PALETTE["banned"], label="Banned"))
    ax.legend(handles=patches + [
        plt.Line2D([0], [0], color="k", linestyle="--", label="Trend")
    ], fontsize=8)

    ax.set_xlabel("Stability margin  (α_gov − α_gov,min) / |α_gov,min|", fontsize=12)
    ax.set_ylabel("C volatility (std of weekly enforcement)", fontsize=12)
    ax.set_title(
        "Stage 2 Regime: Stability Margin vs Enforcement Volatility\n"
        "Prediction: larger margin → smoother enforcement (Routh-Hurwitz criterion)\n"
        "[Validates Stage 2 governance dynamics — P1 extension]",
        fontsize=11, fontweight="bold"
    )
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(plot_dir / "regime_margin_volatility.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved: regime_margin_volatility.png")


# ---------------------------------------------------------------------------
# Table 1: MCI decomposition (main paper table)
# ---------------------------------------------------------------------------

def make_table1(mci: pd.DataFrame, tables_dir: Path):
    """
    Produces the main MCI table for the paper.
    Columns: Subreddit | Type | Regime | Q_fit | Q_back | Q_user | Q_reg | MCI | N_components
    """
    df = mci.copy().sort_values("MCI", ascending=False)
    df["Type"] = df["is_banned"].map({True: "Banned", False: "Control"})

    keep = ["subreddit", "Type", "regime",
            "Q_fit", "Q_backfire",
            "Q_user", "Q_regime",
            "MCI", "n_components", "g_star"]
    df = df[[c for c in keep if c in df.columns]]

    # Round floats
    float_cols = ["Q_fit", "Q_backfire",
                  "Q_user", "Q_regime", "MCI", "g_star"]
    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].round(4)

    df.to_csv(tables_dir / "table1_mci_decomposition.csv", index=False)
    log.info(f"  Saved: table1_mci_decomposition.csv  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Table 2: Regression summary
# ---------------------------------------------------------------------------

def make_table2(scalars: dict, tables_dir: Path):
    """
    Two regressions in one table:
      Reg 1: C_smooth = β₀ + β₁·T_norm + β₂·compliance  (T–C substitution)
      Reg 2: post_ban_rate = β₀ + β₁·theta_proxy         (dose-response)
    """
    q_sub   = scalars.get("Q_substitution", {})
    q_spill = scalars.get("Q_spillover",    {})

    rows = [
        {
            "Regression":     "T–C Substitution",
            "DV":             "C_smooth",
            "IV":             "T_norm (partial, controlling compliance)",
            "beta":           q_sub.get("beta1_TC",   np.nan),
            "SE":             q_sub.get("beta1_SE",   np.nan),
            "t_stat":         q_sub.get("beta1_t",    np.nan),
            "p_value":        q_sub.get("beta1_p",    np.nan),
            "R2":             q_sub.get("r2_pooled",  np.nan),
            "partial_r":      q_sub.get("partial_r_TC", np.nan),
            "N":              q_sub.get("n_obs",       np.nan),
            "Prediction":     "β < 0 (T–C substitution, Tyler 1990)",
            "Confirmed":      "YES" if q_sub.get("beta1_TC", 0) < 0
                              and q_sub.get("beta1_p", 1) < 0.05 else "NO",
        },
        {
            "Regression":     "Dose-Response (Spillover)",
            "DV":             "post_ban_removal_rate",
            "IV":             "theta_proxy (migrant users only)",
            "beta":           q_spill.get("dose_beta_theta", np.nan),
            "SE":             np.nan,
            "t_stat":         np.nan,
            "p_value":        q_spill.get("dose_beta_p", np.nan),
            "R2":             q_spill.get("dose_r2",      np.nan),
            "partial_r":      np.nan,
            "N":              q_spill.get("n_migrants",   np.nan),
            "Prediction":     "β > 0 (higher theta → higher post-ban removal)",
            "Confirmed":      "YES" if q_spill.get("dose_beta_theta", 0) > 0
                              and q_spill.get("dose_beta_p", 1) < 0.05 else "NO",
        },
    ]

    df = pd.DataFrame(rows)
    for c in ["beta", "SE", "t_stat", "p_value", "R2", "partial_r"]:
        df[c] = df[c].round(6)
    df.to_csv(tables_dir / "table2_regressions.csv", index=False)
    log.info("  Saved: table2_regressions.csv")
    return df


# ---------------------------------------------------------------------------
# Table 3: Statistical tests summary
# ---------------------------------------------------------------------------

def make_table3(scalars: dict, mci: pd.DataFrame, tables_dir: Path):
    """
    One-stop summary of every hypothesis test in the paper's validation section.
    """
    q_user  = scalars.get("Q_user",         {})
    q_sens  = scalars.get("sensitivity",    {})

    # Q_backfire hit rate
    backfire_hit = np.nan
    if "Q_backfire" in mci.columns:
        backfire_hit = mci["Q_backfire"].mean()

    rows = [
        # Q2 — backfire hit rate
        {
            "Test":        "Backfire prediction accuracy (Q_backfire)",
            "Statistic":   f"Hit rate = {backfire_hit:.4f}" if not np.isnan(backfire_hit) else "n/a",
            "p_value":     np.nan,
            "Effect_size": f"Point-biserial r = {mci['r_pointbiser'].iloc[0]:.4f}"
                           if "r_pointbiser" in mci.columns else "n/a",
            "Null_H0":     "Hit rate = 0.50 (chance)",
            "Direction":   "Hit rate > 0.50",
            "Confirmed":   "YES" if (not np.isnan(backfire_hit) and backfire_hit > 0.50) else "NO",
            "Reference":   "Lakatos (1978); Angrist & Pischke (2009)",
        },
        # Q4 — ANOVA
        {
            "Test":        "User behavioral consistency ANOVA (Q_user)",
            "Statistic":   f"F = {q_user.get('F_stat', np.nan):.4f}",
            "p_value":     q_user.get("p_anova", np.nan),
            "Effect_size": f"η² = {q_user.get('eta2', np.nan):.4f}",
            "Null_H0":     "Mean ρ_u equal across user types",
            "Direction":   "JOINER ρ_u < INITIATOR ρ_u",
            "Confirmed":   "YES" if (q_user.get("p_anova", 1) < 0.05
                           and q_user.get("mean_rho_JOIN", 0) < q_user.get("mean_rho_INIT", 0))
                           else "NO",
            "Reference":   "Granovetter (1978); Oliver & Marwell (1988)",
        },
        # Q4 — post-hoc INITIATOR vs JOINER
        {
            "Test":        "Post-hoc: INITIATOR > JOINER ρ_u (Mann-Whitney)",
            "Statistic":   f"U = {q_user.get('mw_IJ_U', np.nan):.0f}",
            "p_value":     q_user.get("mw_IJ_p", np.nan),
            "Effect_size": "n/a",
            "Null_H0":     "INITIATOR ρ_u ≤ JOINER ρ_u",
            "Direction":   "INITIATOR ρ_u > JOINER ρ_u",
            "Confirmed":   "YES" if q_user.get("mw_IJ_p", 1) < 0.05 else "NO",
            "Reference":   "Granovetter (1978)",
        },
        
        # Sensitivity
        {
            "Test":        "MCI ranking stability (Spearman ρ, 500 weight draws)",
            "Statistic":   f"Mean ρ = {q_sens.get('rho_mean', np.nan):.4f}",
            "p_value":     np.nan,
            "Effect_size": f"Min ρ = {q_sens.get('rho_min', np.nan):.4f}",
            "Null_H0":     "Ranking changes arbitrarily with weights",
            "Direction":   "Mean ρ > 0.85",
            "Confirmed":   "YES" if q_sens.get("rho_mean", 0) > 0.85 else "NO",
            "Reference":   "Mäki (2009) — robustness of model credibility claims",
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(tables_dir / "table3_statistical_tests.csv", index=False)
    log.info("  Saved: table3_statistical_tests.csv")
    return df

def make_negative_results(scalars: dict, tables_dir: Path):
    """
    Document any important negative or null results that don't fit in the main table.
    """
    # Placeholder for future negative results documentation
    pass

# ---------------------------------------------------------------------------
# Print confirmation summary
# ---------------------------------------------------------------------------

def print_summary(t1: pd.DataFrame, t2: pd.DataFrame, t3: pd.DataFrame):
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE — PREDICTIONS CONFIRMED/REJECTED")
    print("=" * 70)

    pred_map = {
        "P1 (Backfire Spearman ρ)": "Backfire prediction",
        "P3 (Post-shock divergence)": "divergence",
        "P5 (Control decay)": "decay",
    }

    for pred_label, test_keyword in pred_map.items():
        match = t3[t3["Test"].str.contains(test_keyword, case=False)]
        if match.empty:
            status = "?? (no test found)"
        else:
            confirmed = match.iloc[0]["Confirmed"]
            p_val     = match.iloc[0]["p_value"]
            p_str     = f"  p={p_val:.4f}" if not (isinstance(p_val, float) and np.isnan(p_val)) else ""
            status    = f"{'✓ CONFIRMED' if confirmed == 'YES' else '✗ NOT CONFIRMED'}{p_str}"
        print(f"  {pred_label:<30}  {status}")

    print("\n--- MCI Summary ---")
    print(f"  {'Subreddit':<26} {'MCI':>6}  {'Regime':<25}")
    print("  " + "-" * 60)
    for _, r in t1.sort_values("MCI", ascending=False).iterrows():
        ban = "[B]" if r.get("Type") == "Banned" else "   "
        mci_v = f"{r['MCI']:.3f}" if not np.isnan(r["MCI"]) else " n/a"
        print(f"  {ban} r/{r['subreddit']:<23} {mci_v}  {str(r.get('regime', 'n/a')):<25}")
    print("=" * 70)
    print("\n--- Negative Results (Not used in MCI) ---")
    print("  - T–C substitution: no significant negative relationship")
    print("  - Spillover: no migrant signal detected")



def plot_attention_flags(attention_df: pd.DataFrame, plot_dir: Path):
    """
    Two-panel figure:
    Left: Tier 1 scores (ranked)
    Right: Tier 2 scores (ranked)

    Color = attention_flag (Tier1 / Tier2 / Both / Neither)
    Only includes High/Moderate confidence subreddits
    """
# ─────────────────────────────────────────────
    # 1. Filter High / Moderate confidence only
    # ─────────────────────────────────────────────
    df = attention_df[
        attention_df["confidence_level"].isin(["High", "Moderate"])
    ].copy()

    if df.empty:
        print("No High/Moderate confidence data available for plotting.")
        return

    # Sort by Tier 1 for consistent ordering
    df = df.sort_values("tier1_rank")

    # ─────────────────────────────────────────────
    # 2. Color mapping
    # ─────────────────────────────────────────────
    color_map = {
        "BOTH": "red",
        "TIER1_ONLY": "orange",
        "TIER2_ONLY": "blue",
        "NEITHER": "gray",
    }

    colors = df["attention_flag"].map(color_map).fillna("black")

    subreddits = df["subreddit"].values
    x = np.arange(len(df))

    # ─────────────────────────────────────────────
    # 3. Create figure
    # ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # ─────────────────────────────────────────────
    # LEFT: Tier 1
    # ─────────────────────────────────────────────
    axes[0].bar(x, df["tier1_rank"], color=colors)
    axes[0].set_title("Tier 1: Observable Deterioration", fontsize=12)
    axes[0].set_ylabel("Rank (lower = worse)", fontsize=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(subreddits, rotation=90, fontsize=8)

    # Highlight top quartile cutoff
    cutoff = int(np.ceil(len(df) * 0.25))
    axes[0].axhline(cutoff, linestyle="--")

    # ─────────────────────────────────────────────
    # RIGHT: Tier 2
    # ─────────────────────────────────────────────
    axes[1].bar(x, df["tier2_rank"], color=colors)
    axes[1].set_title("Tier 2: Structural Fragility (C / g*)", fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(subreddits, rotation=90, fontsize=8)

    axes[1].axhline(cutoff, linestyle="--")

    # ─────────────────────────────────────────────
    # 4. Legend
    # ─────────────────────────────────────────────
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="red", label="Both"),
        Patch(facecolor="orange", label="Tier1 Only"),
        Patch(facecolor="blue", label="Tier2 Only"),
        Patch(facecolor="gray", label="Neither"),
    ]

    fig.legend(handles=legend_elements, loc="upper center", ncol=4)

    # ─────────────────────────────────────────────
    # 5. Layout & save
    # ─────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    output_path = plot_dir / "attention_flags.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved attention flags plot → {output_path}")

def plot_attention_scatter(attention_df: pd.DataFrame, plot_dir: Path):
    import matplotlib.pyplot as plt
    import numpy as np

    # ─────────────────────────────────────────────
    # 1. Filter High / Moderate confidence
    # ─────────────────────────────────────────────
    df = attention_df[
        attention_df["confidence_level"].isin(["High", "Moderate"])
    ].copy()

    if df.empty:
        print("No High/Moderate confidence data available.")
        return

    # Use ranks (your current system)
    x = df["tier2_rank"]   # Structural fragility
    y = df["tier1_rank"]   # Observable deterioration

    # ─────────────────────────────────────────────
    # 2. Color mapping
    # ─────────────────────────────────────────────
    color_map = {
        "BOTH": "red",
        "TIER1_ONLY": "orange",
        "TIER2_ONLY": "blue",
        "NEITHER": "gray",
    }

    colors = df["attention_flag"].map(color_map).fillna("black")

    # ─────────────────────────────────────────────
    # 3. Plot
    # ─────────────────────────────────────────────
    plt.figure(figsize=(8, 6))

    plt.scatter(x, y, c=colors, s=80)

    # Annotate points
    for i, row in df.iterrows():
        plt.text(
            row["tier2_rank"] + 0.1,
            row["tier1_rank"] + 0.1,
            row["subreddit"],
            fontsize=8
        )

    # ─────────────────────────────────────────────
    # 4. Threshold lines (top quartile)
    # ─────────────────────────────────────────────
    n = len(df)
    cutoff = int(np.ceil(n * 0.25))

    plt.axvline(cutoff, linestyle="--")  # Tier 2 cutoff
    plt.axhline(cutoff, linestyle="--")  # Tier 1 cutoff

    # ─────────────────────────────────────────────
    # 5. Labels
    # ─────────────────────────────────────────────
    plt.xlabel("Tier 2: Structural Fragility Rank (C / g*)")
    plt.ylabel("Tier 1: Observable Deterioration Rank")

    plt.title("Attention Map: Deterioration vs Structural Fragility")

    # ─────────────────────────────────────────────
    # 6. Legend
    # ─────────────────────────────────────────────
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="red", label="Both"),
        Patch(facecolor="orange", label="Tier1 Only"),
        Patch(facecolor="blue", label="Tier2 Only"),
        Patch(facecolor="gray", label="Neither"),
    ]

    plt.legend(handles=legend_elements, loc="upper right")

    # ─────────────────────────────────────────────
    # 7. Layout & save
    # ─────────────────────────────────────────────
    plt.tight_layout()

    output_path = plot_dir / "attention_scatter.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved scatter plot → {output_path}")


def plot_preban_divergence(daily_metrics_path: Path, plot_dir: Path):
    """
    Event study:
    Enforcement trajectory before and after June 10

    Compare:
    - banned subreddits
    - control subreddits

    Shows divergence (core empirical result)
    """

        # ─────────────────────────────────────────────
    # 1. Load data
    # ─────────────────────────────────────────────
    df = pd.read_parquet(daily_metrics_path).copy()
    df["day"] = pd.to_datetime(df["day"])

    # Mark treatment group
    df["is_banned"] = df["subreddit"].isin(BANNED_SUBREDDITS)
    BAN_DATE = "2015-06-10"
    # ─────────────────────────────────────────────
    # 2. Create event time (days from June 10)
    # ─────────────────────────────────────────────
    event_date = pd.Timestamp(BAN_DATE)
    df["event_day"] = (df["day"] - event_date).dt.days

    # Restrict window (clean visualization)
    df = df[(df["event_day"] >= -30) & (df["event_day"] <= 30)]

    # ─────────────────────────────────────────────
    # 3. Aggregate trajectories
    # ─────────────────────────────────────────────
    agg = (
        df.groupby(["event_day", "is_banned"])["C_raw"]
        .mean()
        .reset_index()
    )

    banned = agg[agg["is_banned"] == True]
    control = agg[agg["is_banned"] == False]

    # ─────────────────────────────────────────────
    # 4. Smooth (optional but recommended)
    # ─────────────────────────────────────────────
    def smooth(series, window=5):
        return series.rolling(window=window, center=True, min_periods=1).mean()

    banned["C_smooth"] = smooth(banned["C_raw"])
    control["C_smooth"] = smooth(control["C_raw"])

    # ─────────────────────────────────────────────
    # 5. Plot
    # ─────────────────────────────────────────────
    plt.figure(figsize=(10, 6))

    plt.plot(
        banned["event_day"], banned["C_smooth"],
        label="Banned Subreddits",
        linewidth=2
    )

    plt.plot(
        control["event_day"], control["C_smooth"],
        linestyle="--",
        label="Control Subreddits",
        linewidth=2
    )

    # Event line (June 10)
    plt.axvline(0, linestyle=":", linewidth=2)

    # ─────────────────────────────────────────────
    # 6. Labels & title
    # ─────────────────────────────────────────────
    plt.xlabel("Days Relative to Ban (June 10)")
    plt.ylabel("Average Removal Rate (C_raw)")
    plt.title("Pre- and Post-Ban Enforcement Trajectories")

    plt.legend()

    # ─────────────────────────────────────────────
    # 7. Layout & save
    # ─────────────────────────────────────────────
    plt.tight_layout()

    output_path = plot_dir / "preban_divergence.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved divergence plot → {output_path}")
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate paper tables and plots from MCI scores."
    )
    parser.add_argument("--out_dir", default="../Reddit-2015-v2/output",
                        help="Path to the pipeline output directory.")
    args = parser.parse_args()

    out_dir    = Path(args.out_dir)
    conf_dir   = out_dir / "confidence"
    plot_dir   = conf_dir / "plots"
    tables_dir = conf_dir / "tables"

    for d in [plot_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("CONFIDENCE VERIFICATION  —  confidence_verification.py")
    log.info("=" * 60)

    log.info("\nLoading score files...")
    data = load_scores(conf_dir)
    attention_df = pd.read_csv(conf_dir / "attention_flags.csv") if (conf_dir / "attention_flags.csv").exists() else None

    mci        = data["mci"]
    scalars    = data["scalars"]
    rho_df     = data.get("rho")
    regime_df  = data.get("regime")

    # Enrich MCI with banned flag if not present
    if "is_banned" not in mci.columns:
        mci["is_banned"] = mci["subreddit"].isin(BANNED_SUBREDDITS)

    # ---- Figures ----
    log.info("\nGenerating plots...")
    plot_mci_bar(mci, plot_dir)
    plot_mci_radar(mci, plot_dir)
    plot_mci_vs_gstar(mci, plot_dir)
    #plot_tc_substitution(out_dir / "daily_metrics.parquet", plot_dir)
    plot_user_consistency_violin(rho_df, plot_dir)
    #plot_spillover_rates(scalars, plot_dir)
    plot_sensitivity_heatmap(mci, plot_dir)
    plot_regime_margin_volatility(regime_df, plot_dir)
    plot_attention_flags(attention_df, plot_dir)
    plot_attention_scatter(attention_df, plot_dir)
    plot_preban_divergence(out_dir / "daily_metrics.parquet", plot_dir)

    # ---- Tables ----
    log.info("\nGenerating tables...")
    t1 = make_table1(mci, tables_dir)
    t2 = make_table2(scalars, tables_dir)
    t3 = make_table3(scalars, mci, tables_dir)

    # ---- Summary ----
    print_summary(t1, t2, t3)

    log.info(f"\nAll outputs saved to: {conf_dir}/")
    log.info("Plots:")
    for p in sorted(plot_dir.glob("*.png")):
        log.info(f"  {p.name}")
    log.info("Tables:")
    for p in sorted(tables_dir.glob("*.csv")):
        log.info(f"  {p.name}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
