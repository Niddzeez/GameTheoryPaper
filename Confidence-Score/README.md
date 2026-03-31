# Model Credibility Index (MCI)
## Empirical Validation Framework for the Coordination-Legitimacy Game

---

## Overview

This folder contains the complete empirical validation pipeline for the paper. The core research question is not just *"does the model fit the data?"* but *"how much of the model's theoretical structure is actually supported by the evidence?"*

To answer this, we construct a **Model Credibility Index (MCI)** — a composite score built from six theoretically-grounded components, each targeting a distinct type of validity claim. The MCI is computed per subreddit (community-level) and per user (agent-level), then used to produce the paper's key validation tables and figures.

The pipeline runs entirely on top of the existing output files from `reddit_pipeline_safe.py`, `model_calibration.py`, and `stage2_analysis.py`. No re-processing of raw data is required.

---

## Files in This Folder

```
Confidence-Score/
├── README.md                   ← This file
├── confidence_score.py         ← Computes all six MCI components, writes CSVs
└── confidence_verification.py  ← Generates paper tables and publication figures
```

**Run order:**
```bash
# Step 1: Compute scores (reads from ../Reddit 2015/output/ or ../Reddit-2015-v2/output/)
python confidence_score.py --out_dir "../Reddit-2015-v2/output"

# Step 2: Generate paper materials
python confidence_verification.py --out_dir "../Reddit-2015-v2/output"
```

All outputs are written into `<out_dir>/confidence/`.

---

## What We Are Trying to Prove

The Coordination-Legitimacy Game makes four falsifiable predictions:

| Prediction | Claim | Where Tested |
|---|---|---|
| P1 | There is a backfire threshold g*: over-enforcement increases violations | Q_backfire component |
| P2 | Legitimacy (T) and enforcement (C) are substitutes | Q_substitution component |
| P3 | Fringe communities harbor more initiators (θ < 0) | Q_user component |
| P4 | Banned users migrate and maintain toxic type in new communities | Q_spillover component |

The MCI measures how strongly each of these is supported by the Reddit 2015 data.

---

## The Six Components: Conceptual Backing

### Q1 — ODE Fit Quality (`Q_fit`)

**What it is:** A normalized R²-equivalent measuring how well the Stage 1 ODE reproduces the observed daily removal-rate time series for each subreddit.

**Formula:**
```
SS_obs  = Var(C_raw_observed) × n_days       # total variance in observed series
Q_fit   = 1 − (fit_nll / SS_obs)             # 1 = perfect fit, 0 = no better than mean
```
`fit_nll` is the residual sum of squares from the Nelder-Mead calibration in `model_calibration.py`.

**Why we normalize:** Raw `fit_nll` is scale-dependent — a subreddit with 90 days always has higher RSS than one with 20. Dividing by SS_obs puts all subreddits on the same [−∞, 1] scale (identical to the OLS R²).

**Interpretation:**
- `Q_fit > 0.70` — Strong structural fit. The ODE's feedback mechanism, not just curve-fitting, captures the dominant dynamics.
- `0.40 < Q_fit < 0.70` — Moderate fit. Model captures general trend but misses some volatility.
- `Q_fit < 0.40` — Weak fit. ODE structure not adequate for this community (possibly too few pre-ban days, or non-stationarity).

**Literature backing:** Box & Jenkins (1976) model identification; Ramsay & Silverman (2005) *Functional Data Analysis* — normalized goodness-of-fit for differential equation models. Used in the same way that R² is reported for structural VAR models in macroeconomics (Sims 1980).

**Data sources:** `calibrated_params.csv` (fit_nll, n_days), `daily_metrics.parquet` (C_raw)

---

### Q2 — Backfire Threshold Accuracy (`Q_backfire`)

**What it is:** A hit-rate measuring whether g* (derived from pre-ban calibration) correctly predicts which subreddits entered enforcement collapse post-ban. This is the core out-of-sample falsification test.

**Formula:**
```
For each subreddit i in the post-ban window (Jun 10 – Aug 31):
    slope_i        = OLS slope of C_raw on time (positive = violations rising)
    C_preban_mean  = mean(C_smooth) in Jun 1–9
    prediction_i   = "backfire" if C_preban_mean > g_star_i, else "stable"
    outcome_i      = "backfire" if slope_i > 0 and p < 0.10, else "stable"
    hit_i          = 1 if prediction_i == outcome_i

Q_backfire = mean(hit_i)   # proportion of correct predictions
```

Also compute the **Point-Biserial correlation** between (C_preban > g*) and observed slope sign — this is the continuous effect size.

**Why this is not circular:** g* is estimated using Jun 1–9 only. Post-ban dynamics (Jun 10–Aug 31) are entirely out-of-sample. The model has zero information about the post-ban period when g* is derived. This satisfies the Popperian standard of novel prediction.

**Interpretation:**
- `Q_backfire > 0.75` — Strong predictive validity. g* discriminates well between communities that collapsed and those that stabilized.
- `Q_backfire ≈ 0.50` — No better than chance. Threshold is not predictive.
- `Q_backfire < 0.50` — Model has systematic directional error (predicts backfire when community actually stabilizes, and vice versa).

**Literature backing:** Lakatos (1978) — scientific programmes are progressive to the extent they make novel, confirmed predictions. Smets & Wouters (2007) — evaluation of DSGE models via out-of-sample forecasting. The Jun 10 ban wave is a natural experiment (Angrist & Pischke 2009): exogenous enforcement shock lets us test the model's response function without selection bias.

**Data sources:** `calibrated_params.csv` (g_star), `daily_metrics.parquet` (C_raw, C_smooth, day, post_ban, subreddit)

---

### Q3 — Trust–Enforcement Substitution (`Q_substitution`)

**What it is:** The partial correlation between community legitimacy (T_norm) and enforcement intensity (C_smooth), controlling for the achieved compliance rate. A strong negative partial correlation confirms Prediction 2: high-trust communities achieve equal compliance with less coercion.

**Formula:**
```
For each subreddit-day observation (i):
    compliance_i  = 1 − C_raw_i           # daily compliance rate

OLS regression:
    C_smooth_i = β₀ + β₁·T_norm_i + β₂·compliance_i + ε_i

Q_substitution = β₁ (should be negative and significant)
```

Also report the partial correlation coefficient `r(T, C | compliance)` as the standardized effect size.

**Why partial, not raw:** Raw T–C correlation may be positive if both rise together in high-activity periods (a spurious confound). Holding compliance constant isolates the *substitution* channel: for the same compliance outcome, does more trust allow less enforcement?

**Interpretation:**
- `β₁ < 0` and `p < 0.05` — T–C substitution is operating. Communities with more legitimacy achieve equal compliance with lower enforcement intensity. Confirms P2.
- `β₁ ≈ 0` — No substitution detected. Trust and enforcement are orthogonal in this dataset (may indicate T_norm does not capture the right legitimacy concept, or the panel is too short).
- `β₁ > 0` — T and C move together (escalation dynamic). Suggests the data reflects governance crisis rather than equilibrium — consistent with Stage 2 Governance Trap classification.

**Literature backing:** Tyler (1990) *Why People Obey the Law* — legitimacy as a normative motivator distinct from deterrence. Levi (1988) *Of Rule and Revenue* — "contingent consent": citizens' willingness to comply depends on their trust in the institution's procedural fairness. Sunshine & Tyler (2003) — empirical confirmation that legitimacy reduces the policing burden in real communities.

**Data sources:** `daily_metrics.parquet` (T_norm, C_smooth, C_raw, subreddit, day)

---

### Q4 — User Behavioral Consistency (`Q_user`)

**What it is:** Validation that the three user types (INITIATOR, JOINER, COMPLIER) differ in the *way* their removal probability responds to enforcement intensity, not just in their average removal rate. This validates the fundamental heterogeneity assumption of the model.

**Formula:**
```
For each user u who posted on at least 10 days in subreddit s:
    ρ_u = Pearson correlation(C_smooth_on_posting_days, is_removed_indicator)

Theoretical predictions:
    INITIATOR:  ρ_u ≈ 0   (posts violating content regardless of enforcement level)
    JOINER:     ρ_u < 0   (posts violations more when enforcement is LOW — opportunistic)
    COMPLIER:   ρ_u ≈ 0   (never removed, insensitive to C)

Test: One-way ANOVA with user_type as grouping variable, ρ_u as outcome
      F-statistic + η² (effect size) = Q_user
      Post-hoc: Tukey HSD pairwise tests (INITIATOR vs JOINER, JOINER vs COMPLIER)
```

The critical test is whether JOINER ρ_u is significantly *more negative* than INITIATOR ρ_u. INITIATORs are enforcement-insensitive; JOINERs exploit low-enforcement windows.

**Interpretation:**
- ANOVA `F` significant at p < 0.05, `η² > 0.06` (medium effect) — User types differ behaviorally, not just definitionally. Classification is capturing real heterogeneity.
- Tukey HSD shows JOINER < INITIATOR in mean ρ_u — JOINERs are more strategically sensitive to enforcement, consistent with threshold model.
- COMPLIER ρ_u distribution centered near 0 — Compliers are enforcement-insensitive (they never violate regardless of C).

**Literature backing:** Granovetter (1978) *Threshold Models of Collective Action* — the core theoretical claim that populations are heterogeneous in their sensitivity to the social environment. Oliver & Marwell (1988) *The Paradox of Group Size* — "critical mass" initiators have intrinsic motivation, making them insensitive to incentive levels. Ryan & Deci (2000) *Self-Determination Theory* — intrinsic motivation (B in the model) is by definition unresponsive to external enforcement.

**Data sources:** `user_thresholds.parquet` (user_type, author, subreddit, n_posts), `daily_metrics.parquet` (C_smooth, subreddit, day), `comments_filtered.csv` (author, subreddit, created_utc, is_removed)

---

### Q5 — Stage 2 Regime Stability Margin (`Q_regime`)

**What it is:** A continuous measure of how far each subreddit is from the Stage 2 stability boundary (the Routh-Hurwitz condition). The binary regime classification (Stable / Governance Trap / Structural Instability) is useful for categorization, but the *margin* from the boundary is the correct measure of how robust the stability claim is.

**Formula:**
```
margin_i = (α_gov_i − α_gov_min_i) / |α_gov_min_i|
    # positive → safely stable (α_gov exceeds minimum needed)
    # negative → in danger zone (moderators too slow)
    # −1.0 → α_gov is only half of what stability requires

Q_regime_i = tanh(margin_i)
    # maps (−∞, +∞) → (−1, +1) with natural saturation at extremes
    # tanh(0) = 0 (exactly at boundary), tanh(1) ≈ 0.76, tanh(−1) ≈ −0.76
```

**Cross-validation:** Regress Q_regime against observed C_volatility (std of weekly enforcement):
```
C_volatility_i = β₀ + β₁·Q_regime_i + ε_i
Expected: β₁ < 0  (stable margin → smooth enforcement → low volatility)
```

**Interpretation of Ξ (Xi, the Routh-Hurwitz discriminant):**
- `Ξ > 0` — The equilibrium Jacobian satisfies the necessary stability condition. Stabilization is *possible* given fast enough governance response.
- `Ξ ≤ 0` — Structural instability. No governance response speed, however fast, can stabilize the system. The community is in a regime where enforcement and trust co-erode irreversibly.
- `Ξ >> 0` — Deep stability. Large buffer before the system transitions to instability.

**Interpretation of α_gov_min:**
- The minimum moderator response speed (dC/dt per unit deviation from target) required to prevent enforcement oscillations. Communities where observed α_gov < α_gov_min are in the "governance trap" — they react, but too slowly to prevent the x–T–C feedback loop from amplifying.

**Literature backing:** Routh (1877) / Hurwitz (1895) stability criterion — standard tool for linear stability analysis of equilibria in ODEs. Åström & Murray (2008) *Feedback Systems* — "integral windup" as an analog for governance trap: a controller that responds too slowly relative to the system's natural frequency. Samuelson (1941) *The Stability of Equilibrium* — comparative statics are only valid at stable equilibria; the regime classification determines which subreddits' equilibria can be interpreted structurally.

**Data sources:** `stage2_results.csv` (alpha_gov, alpha_gov_min, Xi, C_volatility, regime, g_star)

---

### Q6 — Spillover Prediction Score (`Q_spillover`)

**What it is:** Tests whether users classified as INITIATOR in banned subreddits maintain elevated removal rates in control subreddits post-ban. This is the strongest test in the entire framework: it is a pre-registered-style causal prediction on a group the model was not calibrated on, using an exogenous treatment (the ban wave).

**Formula:**
```
Group A: users where from_banned_sub = True AND user_type = INITIATOR
Group B: native users in the same post-ban control subreddits (from_banned_sub = False)

For each user u: post_ban_removal_rate_u = n_removed_post_ban / n_posts_post_ban

Primary test:
    Mann-Whitney U test (Group A vs Group B)
    Effect size: Cohen's d = (mean_A − mean_B) / pooled_std

Dose-response test (continuous):
    OLS: post_ban_removal_rate_u = β₀ + β₁·theta_proxy_u + β₂·[user_type dummies] + ε_u
    Expected: β₁ > 0 (higher theta_proxy → higher post-ban removal rate)
```

Also compute spillover *per destination subreddit* — the migration wave should affect communities proportionally to the fraction of users arriving from banned communities.

**Why this is the strongest test:**
1. The ban wave is exogenous: Reddit's decision was not based on predicted user behavior in control subreddits.
2. The population (migrant INITIATORs) was not used to calibrate any model parameter.
3. The prediction is directional and specific: INITIATORs should persist; JOINERs should assimilate; COMPLIERs should be indistinguishable from natives.

**Interpretation:**
- Mann-Whitney U p < 0.01, Cohen's d > 0.2 — Large effect. Migrant INITIATORs are detectably more disruptive than native users. Behavioral type is a stable trait, not a community artifact. Confirms P4.
- θ_proxy regression β₁ significant — Continuous dose-response: users with higher resistance thresholds in banned communities are *more* disruptive post-migration. The theta_proxy score is a valid predictor of future behavior.
- No significant difference — Either the classification is noisy, or the control communities absorbed migrants so effectively that behavioral differences are undetectable. Report as a null finding.

**Literature backing:** Guerette & Bowers (2009) *Assessing the Extent of Crime Displacement* — the displacement hypothesis from criminology. Sunstein (2017) *#Republic* — behavioral persistence of online polarization. Chandrasekharan et al. (2017) *"You Can't Stay Here"* — empirical study of the same Reddit ban wave (use as a comparison benchmark: do our INITIATOR migrants match their finding that ~80% of banned-sub users decreased overall activity, while ~20% maintained it?).

**Data sources:** `panel.parquet` (author, subreddit, week, resisted, from_banned_sub, post_ban, is_banned_sub), `user_thresholds.parquet` (theta_proxy, user_type, author, subreddit)

---

## Composite MCI Score

```
MCI_i = (1/6) × [Q_fit_i + Q_backfire_i + Q_substitution_i + Q_user_i + Q_regime_i + Q_spillover_i]
```

Before aggregating, each component is rescaled to [0, 1]:

| Component | Rescaling |
|---|---|
| Q_fit | clip(Q_fit, 0, 1) — already ≈ R² |
| Q_backfire | already [0, 1] by construction |
| Q_substitution | (−β₁_normalized + 1) / 2 — flips sign so negative β₁ → higher score |
| Q_user | normalized η² from ANOVA, scaled to [0, 1] |
| Q_regime | (tanh(margin) + 1) / 2 — maps [−1, +1] to [0, 1] |
| Q_spillover | 1 − p_value of Mann-Whitney U (crude); Cohen's d normalized |

**Sensitivity analysis:** Weights are varied uniformly over a grid (w ∈ [0.05, 0.35] per component, Σw = 1) across 500 random draws. Spearman rank correlation of MCI ordering across draws is reported. If ρ > 0.85 across all draws, the composite score is robust to weighting choice.

**Global MCI interpretation:**
- `MCI > 0.65` — Model is credible for this subreddit. All major structural claims are supported.
- `0.45 < MCI < 0.65` — Partial credibility. Model fits but some predictions fail.
- `MCI < 0.45` — Low credibility. Either insufficient data or the model's structure is wrong for this community type.

---

## Validity Framework Summary

| Component | Validity Type | Test Logic | Key Reference |
|---|---|---|---|
| Q_fit | Internal | ODE R²-equivalent on calibration window | Box & Jenkins (1976) |
| Q_backfire | Predictive (out-of-sample) | g* vs. post-ban slope direction | Lakatos (1978), Angrist & Pischke (2009) |
| Q_substitution | Structural (mechanism) | Partial correlation T ↔ C controlling compliance | Tyler (1990), Levi (1988) |
| Q_user | Structural (agent heterogeneity) | ANOVA on ρ_u across user types | Granovetter (1978) |
| Q_regime | Comparative statics | Regime margin vs. observed enforcement volatility | Routh-Hurwitz, Samuelson (1941) |
| Q_spillover | Predictive (causal, OOS) | Migrant INITIATOR removal rate vs. native baseline | Guerette & Bowers (2009) |

---

## Output Files

All outputs go to `<out_dir>/confidence/`:

```
confidence/
├── subreddit_mci.csv              # Per-subreddit: all 6 components + composite MCI
├── user_behavioral.parquet        # Per-user: rho_u, matched type prediction T/F
├── confidence_report.txt          # All statistical tests, p-values, effect sizes
├── tables/
│   ├── table1_mci_decomposition.csv   # Main results table (LaTeX-ready)
│   ├── table2_regressions.csv         # T→C substitution + theta_proxy regression
│   └── table3_statistical_tests.csv   # All hypothesis tests in one place
└── plots/
    ├── mci_radar.png                  # Spider plot: 6 components per subreddit
    ├── mci_bar.png                    # MCI composite bar chart, coloured by regime
    ├── mci_vs_gstar.png               # MCI vs g* scatter coloured by banned/control
    ├── user_consistency_violin.png    # ρ_u distributions by user type
    ├── spillover_rates.png            # Migrant vs. native post-ban removal rates
    ├── tc_substitution.png            # T vs C scatter with partial regression line
    └── mci_sensitivity_heatmap.png   # Rank stability under 500 weight draws
```

---

## Quick Interpretation Reference for the Paper

### Confirming each Prediction

**Prediction 1 (Backfire):** Report `Q_backfire` as the hit-rate + point-biserial r. State: *"g* correctly predicted post-ban enforcement trajectory direction in X of N subreddits (Q_backfire = X/N, r_pb = Y, p = Z)."*

**Prediction 2 (T–C Substitution):** Report β₁ from the partial regression. State: *"Holding compliance constant, a one-standard-deviation increase in T_norm is associated with a β₁-unit decrease in enforcement intensity (β₁ = X, SE = Y, p = Z), confirming the substitutability of legitimacy and coercion."*

**Prediction 3 (Initiator Concentration):** Report the ANOVA F-statistic and η². State: *"User types differ significantly in their enforcement-sensitivity (F(2, N) = X, p = Y, η² = Z). Post-hoc tests confirm that JOINERs show significantly more negative ρ_u than INITIATORs (Tukey HSD, p = W), consistent with threshold theory."*

**Prediction 4 (Spillover):** Report Mann-Whitney U and Cohen's d. State: *"Migrant INITIATORs showed significantly higher post-ban removal rates than native users in destination communities (Mann-Whitney U = X, p = Y, Cohen's d = Z), confirming that user type is a stable trait, not a community artifact."*

---

## Dependencies

```
pandas >= 1.5
numpy >= 1.23
scipy >= 1.9
matplotlib >= 3.6
seaborn >= 0.12
statsmodels >= 0.13   (for OLS regression tables)
pyarrow >= 10.0       (for parquet I/O)
```

Install: `pip install pandas numpy scipy matplotlib seaborn statsmodels pyarrow`
