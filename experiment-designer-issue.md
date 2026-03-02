# ExperimentDesigner: Posterior-aware experiment design for lift tests

## Summary

Add an `ExperimentDesigner` to PyMC-Marketing that recommends which marketing experiment to run — which channel, at what spend level, for how long — based on a fitted MMM's posterior uncertainty about channel response functions. The v1 scope is **national-level experiments** on multiple spend channels, analysed via Interrupted Time Series (ITS).

**Project home:** PyMC-Marketing (not CausalPy). The `ExperimentDesigner` consumes a fitted multidimensional `MMM` object (from `pymc_marketing.mmm.multidimensional`) directly — it reads posterior samples from the model's `InferenceData`, accesses the saturation and adstock function objects, and derives current spend and residual noise from the training data. No cross-repo protocol or intermediate representation is needed. Post-experiment analysis (ITS, Synthetic Control) remains in CausalPy.

The core innovation: GeoLift asks the user to guess the expected effect size (`effect_size = seq(0, 0.25, 0.05)`). The `ExperimentDesigner` replaces that guess with the posterior distribution of the predicted lift from a fitted model. Instead of "what's my power if the lift is 10%?", it answers "what's my power given what the model believes the lift will be?" — for every channel, at every candidate spend level.

This same principle extends to power: instead of computing power at one guessed effect size, the tool computes **posterior-predictive power** (Bayesian assurance) — the expected power averaged over the posterior distribution of the true effect.

**Relationship to CausalPy [#759](https://github.com/pymc-labs/CausalPy/issues/759):** Issue #759 tracks GeoLift feature parity (power analysis, market selection, augmented SC) in CausalPy. This feature is complementary — it adds the upstream decision layer that GeoLift doesn't have (which channel? what spend level?) while partially overlapping with #759's G3 (power analysis). The `ExperimentDesigner` can later consume #759's geo-level power analysis and market selection as they're built.

---

## The problem

A practitioner has a fitted MMM with multiple channels whose parameters are uncertain — often severely so, due to correlated spend patterns. They know a lift test would help, but face design questions:

- **Which channel** should they test?
- **At what spend level** — where on the saturation curve should the test probe?
- **What magnitude** of spend change is needed for a detectable effect?
- **For how long** — what experiment duration balances power against cost?
- **What is the cost** of each candidate experiment?

Currently these decisions are made by intuition. GeoLift helps with the downstream question (which geos, how long, what MDE) but only after the channel and design type have already been chosen. No existing open source tool makes the upstream decisions based on model uncertainty.

---

## Relationship to GeoLift

GeoLift is the closest open-source tool. It provides an end-to-end workflow for geo-based incrementality experiments: `GeoLiftMarketSelection()` optimises which geos to use, test duration, number of treatment markets, and computes MDE, power curves, and budget estimates. It does this well.

But by the time you use GeoLift, two decisions have already been made:
1. **Which channel** to test (user decides)
2. **That you're running a geo experiment** (GeoLift only supports geo holdout with Augmented SCM)

The `ExperimentDesigner` makes those upstream decisions, informed by the model's posterior:

| Capability | GeoLift | ExperimentDesigner (v1) |
|---|---|---|
| Which channel to test | User decides | Ranks channels by posterior uncertainty |
| Expected effect size | User guesses (`effect_size = seq(...)`) | Posterior-predicted from the response function |
| Power / MDE | Yes — simulation-based on historical data | Yes — posterior-predictive (Bayesian assurance) |
| Adstock-aware lift prediction | No | Yes — accounts for ramp-up/decay over experiment duration |
| Cost model | Budget from CPIC | Direct additional spend over experiment duration |
| MMM posterior awareness | None | Core feature |
| Geo selection | Yes — core strength | Deferred to v2 |
| Design type comparison | Geo holdout only | Sustained ITS and go-dark in v1; geo designs in v2 |

The two tools are complementary. In a complete workflow:
1. `ExperimentDesigner` (v1) says: "Test TV at +20% spend per week for 6 weeks via a sustained ITS"
2. In v2, the `ExperimentDesigner` could additionally say: "Run this as a geo holdout using {Chicago, Portland} as treatment geos" (integrating with #759 G4)
3. After the experiment runs, CausalPy's `InterruptedTimeSeries` (or `SyntheticControl`) analyses the results and the lift estimate feeds back into the MMM via `add_lift_test_measurements()`

---

## What the ExperimentDesigner actually computes

The tool is a **posterior calculator**, not a model. It does not run, refit, or forecast with an MMM. It reads posterior samples from the fitted MMM's `InferenceData` and evaluates the model's saturation functions at candidate spend levels, accounting for adstock dynamics. The entire computation is vectorised numpy arithmetic on pre-existing arrays — no MCMC, no PyTensor graph compilation, no sampling.

### The core computation

For every candidate experiment design $d = (\text{channel } c, \Delta x_{\text{weekly}}, T_{\text{active}})$, the computation has two steps:

**Step 1: Adstock-aware lift prediction.** A spend change doesn't produce its full effect instantly — the adstocked spend ramps toward a new steady state over time. For each posterior draw, we compute the lift at each week of the experiment and sum:

```python
# phi_samples: shape (n_draws, n_params) — saturation parameters
# alpha_samples: shape (n_draws,) — adstock decay parameter
# l_max: int — maximum adstock lag from mmm.adstock.l_max
# normalize: bool — whether adstock weights are normalised (mmm.adstock.normalize)
#
# x_ss: current steady-state adstocked spend.
#        Assumes current spend has been approximately constant long enough
#        for adstock to reach its geometric series steady state. This is a
#        simplifying assumption — the designer uses the recent average of
#        weekly spend from the training data, not the full spend history,
#        so the steady state is the best available approximation.
#
#        The steady-state value depends on the adstock normalisation convention
#        used during fitting (must match):
#          - normalize=True  (default): x_ss = x_current
#            (normalised weights sum to 1, so steady-state adstocked spend
#             equals the raw spend)
#          - normalize=False: x_ss = x_current * (1 - alpha^l_max) / (1 - alpha)
#            (unnormalised weights sum to the truncated geometric series)
#
# delta_x: weekly spend change (per week)

alpha_safe = np.clip(alpha_samples, 1e-6, 1 - 1e-6)

# Normalisation constant for the truncated geometric series
S = (1 - alpha_safe ** l_max) / (1 - alpha_safe)

for t in range(T_active):
    # Cumulative weight through week t (truncated geometric partial sum)
    partial_sum = (1 - alpha_safe ** (t + 1)) / (1 - alpha_safe)

    if normalize:
        # Normalised weights: ramp is the fraction of total weight accumulated
        ramp = partial_sum / S
        # x_ss = x_current (steady state equals raw spend under normalisation)
    else:
        # Unnormalised weights: ramp is the raw partial sum
        ramp = partial_sum
        # x_ss = x_current * S (steady state is inflated by total weight)

    effective_spend_t = x_ss + delta_x * ramp
    weekly_lift_t = g(effective_spend_t, phi_samples) - g(x_ss, phi_samples)

# Total predicted lift over the experiment (sum of weekly lifts)
predicted_lift = sum(weekly_lift_t for t in range(T_active))  # shape: (n_draws,)
```

In practice this is vectorised over draws and weeks with shape `(n_draws, T_active)`, then summed over the time axis. No loops needed.

**Step 2: Summary statistics.**

```python
expected_lift = np.mean(predicted_lift)
lift_hdi = az.hdi(predicted_lift, hdi_prob=0.94)
snr = expected_lift / sigma_d
```

### Posterior-predictive power (Bayesian assurance)

Traditional power analysis picks a single effect size and computes P(detect | that effect). The `ExperimentDesigner` computes **posterior-predictive power** — the expected power averaged over the posterior distribution of the true effect. This mirrors the posterior-derived effect size: instead of guessing one number, we integrate over what the model believes.

For each posterior draw, the predicted lift represents a possible true state of the world. If that draw is the truth, the experiment would observe:

$$\text{observed lift} \sim \mathcal{N}(\text{predicted\_lift}_i, \sigma_d)$$

The probability of detecting this effect (two-sided test at significance level $\alpha$) is:

$$\text{power}_i = 1 - \Phi\!\left(z_{\alpha/2} - \frac{|\text{predicted\_lift}_i|}{\sigma_d}\right) + \Phi\!\left(-z_{\alpha/2} - \frac{|\text{predicted\_lift}_i|}{\sigma_d}\right)$$

The posterior-predictive power (assurance) averages over all posterior draws:

```python
from scipy.stats import norm

z_alpha = norm.ppf(1 - 0.05 / 2)  # ≈ 1.96 for α = 0.05 two-sided
ncp = np.abs(predicted_lift) / sigma_d  # non-centrality parameter, shape (n_draws,)

# Per-draw power: P(detect | this draw is truth)
per_draw_power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)

# Posterior-predictive power (Bayesian assurance)
assurance = np.mean(per_draw_power)
```

This produces intuitive behaviour:
- **Well-identified channel, large effect:** Most draws predict a large lift → per-draw power is high for nearly all draws → assurance ≈ 1.
- **Uncertain channel, moderate effect:** Some draws predict a large lift (high per-draw power), others predict a small lift (low per-draw power) → assurance reflects the average.
- **Posterior includes zero:** Draws near zero contribute ≈ $\alpha$ to the assurance → pulls it down, correctly reflecting the risk that the true effect is undetectable.

### How adstock affects the lift prediction

Without adstock, the predicted lift would be the **steady-state** difference: $g(x + \Delta x) - g(x)$. This is what you'd observe if the experiment ran forever. In reality, adstock means the effect ramps up gradually:

- **Spend increase:** The adstocked spend doesn't jump to the new steady state instantly. For fast-decaying channels (low α), the ramp-up takes a few weeks. For slow-decaying channels (high α), it can take many weeks. A short experiment on a high-adstock channel sees only a fraction of the steady-state effect, reducing power.
- **Go dark:** When spend drops to zero, the adstocked effect decays rather than vanishing. Week 1 still has most of the residual effect; week 6 has much less. The total observed lift (negative) over T weeks depends on how fast the adstock drains — which is uncertain, since α is a posterior quantity.

The adstock ramp fraction — the ratio of the time-averaged lift to the steady-state lift — is a useful diagnostic:

$$\text{ramp\_fraction}(T, \alpha) = \frac{\text{time-averaged lift over } T \text{ weeks}}{\text{steady-state lift}}$$

The ramp fraction's formula depends on the adstock normalisation convention. With normalised adstock (default), the cumulative weight at week $t$ is $(1 - \alpha^{t+1}) / (1 - \alpha^{l_{\max}})$, so the ramp fraction after $T$ weeks is the average of these cumulative weights relative to their steady-state value of 1. With unnormalised adstock, the cumulative weight is $(1 - \alpha^{t+1}) / (1 - \alpha)$ and the steady state is $(1 - \alpha^{l_{\max}}) / (1 - \alpha)$. In both cases the ratio is the same:

$$\text{ramp\_fraction}(T, \alpha) = \frac{1}{T} \sum_{t=0}^{T-1} \frac{1 - \alpha^{t+1}}{1 - \alpha^{l_{\max}}}$$

For fast-decaying channels (α ≈ 0.3), the ramp fraction approaches 1 within a few weeks. For slow-decaying channels (α ≈ 0.85), a 4-week experiment may only capture 60–70% of the steady-state effect. The `ExperimentDesigner` computes this per posterior draw, so the uncertainty in α propagates through to the power calculation.

### How uncertainty flows through

If a channel's saturation and adstock parameters are well-identified (narrow posteriors), `predicted_lift` will be a tight distribution and assurance will be high. If the parameters are highly uncertain (wide posteriors), `predicted_lift` will be wide — some posterior draws might predict a total lift of 8000, others might predict 800.

This creates a natural tension the tool surfaces transparently:
- **Uncertain channels** are where you'd **learn the most** (high information gain), but the experiment is **riskier** (lower assurance, because the true lift might be too small to detect)
- **Well-identified channels** produce **reliable experiments** (high assurance), but you **learn less** (the model already knows the answer)

The recommendation table shows both dimensions — the user sees information gain alongside assurance and decides based on their risk appetite.

### Candidate grid and performance

The `recommend()` method iterates over a grid of candidate designs:

| Dimension | Typical values | Count |
|---|---|---|
| Channels | tv, search, social, display, ... | 3–8 |
| Weekly spend changes | ±10%, ±20%, ±30%, ±50%, −100% (go dark) | 5–9 |
| Durations | 4, 6, 8, 12 weeks | 3–4 |

A typical grid: 5 channels × 7 spend changes × 4 durations = **140 candidates**. Each candidate requires one vectorised numpy call on ~2000 posterior draws across T weeks — microseconds per candidate. The entire `recommend()` call completes in **well under a second**. The plotting is slower than the computation.

This means there's no need for computational shortcuts, approximations, or sampling strategies. We evaluate every candidate exhaustively.

---

## Scope: v1 (this issue)

### Scope boundaries

v1 is explicitly **national-level**: one outcome time series, multiple spend channels, no geo-level data. The experiment designs are national sustained spend changes analysed via Interrupted Time Series (ITS), plus the special case of going dark on a channel.

This is a deliberate scoping decision. Geo-level designs (geo holdout, staggered geo, synthetic control) require geo-level panel data, a geo-specific noise model (σ from SC fit quality rather than ITS residuals), and market selection algorithms. These are substantial capabilities tracked by CausalPy [#759](https://github.com/pymc-labs/CausalPy/issues/759) and deferred to v2.

The v1 tool is useful without geos because the upstream questions — which channel to test, at what spend level, for how long — can be answered from the MMM's posterior alone. The user takes the recommendation and decides separately whether to implement it nationally (ITS) or as a geo experiment using existing tools like GeoLift.

### Causal identification: one intervention at a time

National-level ITS has a single outcome time series. Causal identification requires that **only one channel's spend changes at the intervention point**. If TV and Search both change in the same week, the observed outcome change is the sum of both effects — it cannot be decomposed. This is the same structural problem that motivates lift tests in the first place: two treatments, one outcome series, no way to separate them.

The recommendation table is therefore a **priority ranking for sequential execution**, not a portfolio to run in parallel. The user should:

1. Run the top-ranked experiment.
2. Wait for the intervention period to end.
3. Wait an additional **buffer period** of at least $2 \times l_{\max}$ weeks (where $l_{\max}$ is the maximum adstock lag across channels) to allow carryover from the completed experiment to decay. Without this gap, residual adstock from the first experiment contaminates the baseline for the second.
4. Analyse the results (via ITS).
5. Optionally update the MMM posterior with the new lift test measurement and re-run the `ExperimentDesigner` for the next experiment.

The buffer period matters because adstock means a spend change doesn't stop affecting outcomes when the experiment ends. For fast-decaying channels (α ≈ 0.3), 4–6 weeks of buffer suffices. For slow-decaying channels (α ≈ 0.85), the buffer may need to be 12+ weeks. The `ExperimentDesigner` does not model sequential dependencies between experiments in v1 — each recommendation is computed independently. Sequential planning (where the recommendation for experiment 2 depends on the outcome of experiment 1) is deferred to v2.

If the user wants to test multiple channels in parallel, they should use **geo-level designs** with non-overlapping geo assignments — different channels tested in different regions. This requires geo-level data and is part of the v2 scope.

### Inputs

The `ExperimentDesigner` accepts a fitted PyMC-Marketing `MMM` object directly (the multidimensional `MMM` class from `pymc_marketing.mmm.multidimensional`). Even though the multidimensional MMM supports geo-level `dims`, v1 targets national-level models (`dims=None`) — a single outcome time series with no geo dimension. It extracts everything it needs from the model:

```python
from pymc_marketing.mmm import MMM
from pymc_marketing.mmm.experiment_design import ExperimentDesigner

# Option A: From a fitted MMM (primary workflow)
mmm = MMM(
    date_column="date",
    channel_columns=["tv", "search", "social"],
    target_column="y",
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
)
idata = mmm.fit(X, y)
designer = ExperimentDesigner(mmm)

# Option B: From a saved InferenceData fixture (for demos and testing)
designer = ExperimentDesigner.from_idata(
    az.from_netcdf("simulated_3channel.nc"),
    saturation="logistic",
    adstock="geometric",
)
```

**What the designer reads from the fitted MMM:**

| Quantity | Source | How it's used |
|---|---|---|
| Posterior samples (saturation params, beta, alpha) | `mmm.idata.posterior` — variables `saturation_lam`, `saturation_beta`, `adstock_alpha` (per-channel, dims `("channel",)`) | Lift prediction, assurance calculation |
| Saturation function | `mmm.saturation` (e.g., `LogisticSaturation`) | Evaluating `beta × sat(x)` at candidate spend levels |
| Adstock function spec | `mmm.adstock` (e.g., `GeometricAdstock`) | Determines ramp computation (geometric series in v1) |
| Adstock normalisation | `mmm.adstock.normalize` (default `True`) | Controls steady-state and ramp formulas (see Step 1 pseudocode) |
| Adstock max lag | `mmm.adstock.l_max` | Truncated geometric series length for ramp computation |
| Channel names | `mmm.channel_columns` | Labelling and iteration |
| Current weekly spend | `mmm.X[mmm.channel_columns]` (last rows, recent average) | Current operating point for each channel |
| Residual std | `np.std(y - mmm.predict(X))` — std of per-week residuals (`predict()` returns the posterior mean) | Measurement noise σ(d) |
| Spend correlation | `mmm.X[mmm.channel_columns].corr()` | Scoring dimension (optional) |
| Channel / target scales | `mmm.scalers["_channel"]`, `mmm.scalers["_target"]` | Back-transforming costs and lifts to original currency units for display |

Since the designer lives in PyMC-Marketing, it has direct access to the model's internal representations — saturation and adstock function objects, the scalers, the training data. No intermediate protocol or context object is needed.

**Scaling convention:** The designer operates on the **model's internal scale** throughout — the same scale used during fitting. The multidimensional MMM scales channels by their maximum absolute value (via `mmm.scalers["_channel"]`, an xarray DataArray with one value per channel) and the target by `mmm.scalers["_target"]`. All posterior samples, spend values, and response function evaluations are in this scaled space. The `spend_change_abs` field in the output can be back-transformed to original currency units by multiplying by the appropriate channel scale. This back-transformation is a display convenience; the computation and ranking are scale-invariant.

**Ordering assumption:** v1 assumes `adstock_first=True` — the standard pipeline where adstock is applied before saturation: `contribution = beta × sat(adstock(x))`. This is the default in PyMC-Marketing and the most common convention. The `adstock_first=False` ordering (saturation before adstock) would require a different lift computation and is deferred to v2.

**Adstock normalisation assumption:** The designer must match the normalisation convention used during fitting (see Step 1 pseudocode for the branching logic). PyMC-Marketing's `GeometricAdstock` defaults to `normalize=True`, meaning adstock weights are divided by their sum so they total 1. This changes the steady-state adstocked spend (equal to the raw spend under normalisation, vs. inflated by the geometric series sum without normalisation) and the ramp-up dynamics. The designer reads `mmm.adstock.normalize` and `mmm.adstock.l_max` to select the correct formulas. See the Step 1 pseudocode for the exact computation.

**How response functions are called:** The designer evaluates `beta × sat(adstocked_spend)` for each posterior draw. It accesses the saturation function directly from the fitted MMM's saturation object and calls it with numpy arrays:

```python
def logistic_saturation(x, lam, beta):
    """Logistic saturation: beta * (1 - exp(-lam*x)) / (1 + exp(-lam*x))"""
    return beta * (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))
```

In practice, all parameter arrays have shape `(n_draws,)` and spend arrays have shape `(n_draws,)` or `(n_draws, T_active)`, so the function must support numpy broadcasting. The designer does **not** call the saturation function's PyTensor graph — it uses a lightweight numpy equivalent for speed.

**Adstock in v1:** The geometric adstock ramp is computed analytically by the designer using the `alpha` posterior samples, `mmm.adstock.l_max`, and `mmm.adstock.normalize` (see Step 1 pseudocode). The `adstock` object from the MMM is inspected to confirm the functional form, extract parameter names, and read `normalize` and `l_max`. v2 will support non-geometric adstock forms (delayed, Weibull) that require calling the adstock function rather than using a closed-form ramp.

### Core functionality

```python
designer = ExperimentDesigner(mmm)

# Recommend experiments across all channels
recommendations = designer.recommend(
    spend_changes=[0.1, 0.2, 0.3, 0.5, -0.2, -0.5, -1.0],  # fractional per-week change
    durations=[4, 6, 8, 12],         # weeks (optimised over)
    min_snr=2.0,                     # filter out undetectable designs
    significance_level=0.05,         # for power calculation (two-sided)
)

# Returns a ranked table:
# | rank | channel | delta_x  | duration | expected_lift | lift_hdi       | snr | assurance | ramp_frac | net_cost | score |
# |------|---------|----------|----------|---------------|----------------|-----|-----------|-----------|----------|-------|
# | 1    | tv      | +20%/wk  | 8 wks    | 8400 ± 5200   | [1800, 15600]  | 2.8 | 0.79      | 0.88      | $80k     | 0.73  |
# | 2    | search  | +30%/wk  | 6 wks    | 4200 ± 1800   | [1900, 7100]   | 3.5 | 0.92      | 0.95      | $54k     | 0.71  |
# | ...  | ...     | ...      | ...      | ...           | ...            | ... | ...       | ...       | ...      | ...   |
```

The `spend_changes` parameter specifies fractional changes to weekly spend. A value of `0.2` means "increase weekly spend on this channel by 20%." A value of `-1.0` means "go dark" (reduce weekly spend to zero). Duration is a parameter in the candidate grid, optimised over alongside channel and spend change.

### Plotting

```python
designer.plot_channel_diagnostics()
designer.plot_power_cost(recommendations)
designer.plot_lift_distributions(channel="tv")
designer.plot_saturation_curve(channel="tv")
designer.plot_adstock_ramp(recommendations)
```

**Plot specifications:**

| Method | Layout | Content |
|---|---|---|
| `plot_channel_diagnostics()` | Multi-panel figure, one row per channel (or grouped bar chart). | For each channel: (1) posterior HDI width (product of parameter HDIs), (2) mean absolute spend correlation with other channels, (3) saturation gradient at the current operating point, (4) posterior mean of adstock α. Gives a visual summary of "which channels are most uncertain, most correlated, and fastest/slowest to respond." |
| `plot_power_cost(recs)` | Scatter plot, one point per candidate design. | x-axis: absolute net cost ($\|\text{net\_cost}\|$). y-axis: assurance. Points coloured by channel, shaped by spend direction (increase vs decrease vs go-dark). Optionally annotated with top-ranked designs. |
| `plot_lift_distributions(channel)` | Grid of density plots: rows = spend changes, columns = durations. | Each panel shows the posterior distribution of total predicted lift for that (channel, Δx, T) combination. Shaded HDI region. Vertical line at zero. |
| `plot_saturation_curve(channel)` | Single panel with uncertainty band. | x-axis: adstocked spend. y-axis: `response_fn(x)`. Posterior draws shown as semi-transparent lines or a credible band. Vertical line at the current steady-state operating point. Optional vertical lines at candidate spend levels to show where experiments would probe. |
| `plot_adstock_ramp(recs)` | Line plot, one line per channel. | x-axis: experiment duration (weeks). y-axis: mean adstock ramp fraction (posterior mean). Shows how quickly each channel approaches steady-state effect. Uncertainty band from posterior draws of α. |

### Design type catalogue (v1)

v1 supports two design types, both analysed at the national level via ITS:

| Design type | Parameters | What it identifies |
|---|---|---|
| **Sustained spend change** (ITS) | Δx (per week), T_active | One point on the saturation curve. Adstock ramp-up means the observed effect grows over time, providing partial adstock information. |
| **Go dark** (Δx = −x, special case) | T_dark | Absolute scale of channel contribution. Adstock decay means the effect is gradual — early weeks retain residual effect, later weeks reveal the full impact. |

Go dark is implemented as a sustained spend change with `spend_change_frac = -1.0`. It's called out separately because it has distinct properties: it probes the full range from current spend to zero, provides the strongest constraint on the saturation scale, and the adstock decay during go-dark is a clean signal for the decay rate.

Pulse, switchback, and geo-based designs are deferred to v2.

### Measurement noise estimation

For v1, $\sigma(d)$ uses a simple parametric model appropriate for national-level ITS. Because `predicted_lift` is the **cumulative** lift (sum of weekly lifts over $T_{\text{active}}$ weeks), the measurement noise must also be on the cumulative scale:

$$\sigma(d) \approx \hat{\sigma}_{\text{residual}} \cdot \sqrt{T_{\text{active}}}$$

where $\hat{\sigma}_{\text{residual}}$ is the standard deviation of the MMM's **per-week** residuals on the **outcome scale** (same units as the predicted lift). For a fitted PyMC-Marketing MMM, this is `np.std(y - mmm.predict(X))` on the training data (`predict()` returns the posterior mean). The $\sqrt{T}$ scaling follows from summing approximately independent weekly residuals: $\operatorname{Var}\!\left(\sum \varepsilon_t\right) = T \cdot \sigma^2$. Despite the noise growing with duration, the **SNR improves** because the signal (total lift) grows approximately linearly with $T$ while the noise grows only as $\sqrt{T}$:

$$\text{SNR} \approx \frac{\bar{\mu} \cdot T}{\hat{\sigma}_{\text{residual}} \cdot \sqrt{T}} = \frac{\bar{\mu} \cdot \sqrt{T}}{\hat{\sigma}_{\text{residual}}}$$

This is the standard result: power increases with $\sqrt{T_{\text{active}}}$, giving diminishing returns to longer experiments. v2 can upgrade to simulation-based σ estimation (placebo-in-time, as in GeoLift's power analysis) and geo-specific σ from synthetic control fit quality.

### Cost model (v1)

For v1, the cost model is the **direct additional spend** over the experiment duration:

$$\text{net\_cost}(d) = \Delta x_{\text{abs}} \cdot T_{\text{active}}$$

For spend increases, this is positive (additional marketing outlay). For spend decreases, this is negative (cost savings). For go-dark ($\Delta x_{\text{abs}} = -x_{\text{current}}$), the cost is the total saved spend (a large negative value).

**Units:** The cost is computed in the same units as `current_weekly_spend` — which is in the **model's internal scale** (see Scaling Convention above). When the designer is constructed from a fitted `MMM`, it has access to the model's scaler and can back-transform costs to original currency units for display. When constructed from a saved `InferenceData` fixture, costs are displayed in model-scale units unless a scaler is provided — still valid for **ranking** experiments (the ordering is scale-invariant).

v1 does **not** model revenue impact (no margin parameter). This is a deliberate simplification: the revenue impact depends on the same uncertain parameters we're trying to learn, creating a circularity. The v1 cost model is sufficient for ranking experiments by "how much additional spend does this require?" v2 can add a margin parameter and expected revenue offset for a net-cost model.

### Channel ranking score

Channels and designs are ranked by a **weighted sum** of normalised scoring dimensions:

$$\text{score}(d) = \sum_k w_k \cdot \tilde{s}_k(d)$$

where each $\tilde{s}_k$ is min-max normalised to $[0, 1]$ across all candidates, and $w_k$ are user-configurable weights (with sensible defaults). Note that $s_1$ (uncertainty) and $s_2$ (correlation) are per-channel quantities — all designs for the same channel share the same value. If a dimension has zero range across all candidates (e.g., only one channel), the normalised value is set to 0.5 (neutral) and its weight is effectively inert. The scoring dimensions are:

1. **Posterior uncertainty** ($s_1$) — product of HDI widths for the channel's response function parameters. Wider posteriors → higher priority (more to learn).
2. **Spend correlation** ($s_2$) — mean absolute correlation of this channel's spend with all other channels. Highly correlated channels contribute most to the identification problem. Computed automatically from the MMM's training data when available. If unavailable (e.g., when using `from_idata()` without spend data), this dimension is dropped and its weight is redistributed proportionally across the remaining dimensions. The `rationale` template omits the correlation sentence in this case.
3. **Saturation gradient** ($s_3$) — $\partial g / \partial x$ at the current operating point, evaluated at the posterior mean. Larger gradient → bigger Δy for a given Δx → better SNR.
4. **Assurance** ($s_4$) — posterior-predictive power as defined above.
5. **Cost efficiency** ($s_5$) — assurance per dollar of absolute spend disruption ($|\text{net\_cost}|$). Uses absolute value so that both spend increases and decreases are ranked by the magnitude of the budget commitment.

The specific weights are a design choice to be determined during implementation and user testing. Reasonable defaults might emphasise assurance and cost efficiency (the "will this work and is it affordable?" dimensions) while using uncertainty and correlation as tiebreakers. The weights should be exposed as a parameter so users can adjust them to their priorities:

```python
recommendations = designer.recommend(
    ...,
    score_weights={"uncertainty": 0.2, "correlation": 0.1, "gradient": 0.1,
                   "assurance": 0.3, "cost_efficiency": 0.3},
)
```

### Output object

```python
@dataclass
class ExperimentRecommendation:
    channel: str
    spend_change_frac: float             # fractional per-week change, e.g. 0.2 for +20%/wk
    spend_change_abs: float              # absolute weekly Δx in spend units
    duration_weeks: int                  # active intervention period
    expected_lift: float                 # posterior mean of total lift over T weeks
    expected_lift_hdi: tuple[float, float]
    snr: float
    assurance: float                     # posterior-predictive power (Bayesian assurance)
    adstock_ramp_fraction: float         # mean fraction of steady-state effect captured
    net_cost: float                       # Δx_abs × T_active (direct additional spend)
    score: float                         # weighted composite score
    rationale: str                       # auto-generated explanation (template-based)
```

The `rationale` field is template-based, e.g.: *"TV has the widest posterior uncertainty (HDI product 4.2) and high spend correlation with Digital (r = 0.91). A +20%/wk change for 8 weeks produces an expected total lift of 8400 (94% HDI: [1800, 15600]) with assurance 0.79. Adstock ramp fraction 0.88 — the channel's moderate adstock (α ≈ 0.5) means 8 weeks captures most of the steady-state effect. Additional spend: $80k."*

The template string is defined in the codebase (not generated by an LLM). It interpolates the recommendation's numeric fields into a fixed structure with three sentences: (1) why this channel (uncertainty + correlation), (2) what the experiment produces (lift, HDI, assurance), (3) adstock and cost context.

### Simulation helper

A `generate_experiment_fixture()` utility generates realistic posteriors by fitting an actual MMM to simulated data. This produces posterior structure (parameter correlations, degeneracies, identification challenges) that hand-crafted distributions cannot replicate.

**Process:**
1. **Simulate spend data:** Generate 104 weeks of spend for N channels using correlated log-normal random walks to produce realistic positive, correlated, trending spend series. Correlation structure is controlled via a specified correlation matrix (e.g., TV–Search correlation of 0.7).
2. **Simulate outcomes:** Apply the DGP: $y_t = \text{intercept} + \sum_c \beta_c \cdot \text{sat}_c(\text{adstock}_c(x_{c,t})) + \varepsilon_t$ where $\varepsilon_t \sim \mathcal{N}(0, \sigma^2)$. No trend or seasonality — these are nuisance parameters irrelevant to the experiment designer. The ground-truth parameters are stored for later validation.
3. **Fit a PyMC-Marketing MMM:** Fit an `MMM` with logistic saturation (`LogisticSaturation`) and geometric adstock (`GeometricAdstock`) on the simulated data. MCMC settings: 2 chains × 2000 draws × 1000 tune, `target_accept=0.9`. This takes 2–5 minutes and runs once.
4. **Save as InferenceData:** Save the fitted model's `InferenceData` via `az.to_netcdf()`. The `.nc` file contains the posterior samples with PyMC-Marketing's standard parameter names. Additional metadata — ground-truth parameters, `current_weekly_spend`, `residual_std`, `spend_correlation`, model specification (saturation/adstock type), and adstock settings (`normalize`, `l_max`) — is stored in the InferenceData's `constant_data` group and attributes. The fixture lives in `pymc_marketing/mmm/data/`.

```python
from pymc_marketing.mmm.experiment_design import generate_experiment_fixture

# One-time generation (run offline, result committed to repo):
idata = generate_experiment_fixture(
    channels=["tv", "search", "social"],
    saturation="logistic",
    adstock="geometric",
    true_params={
        "tv": {"lam": 0.5, "beta": 3000, "alpha": 0.7},
        "search": {"lam": 2.0, "beta": 1500, "alpha": 0.3},
        "social": {"lam": 1.0, "beta": 800, "alpha": 0.5},
    },
    n_weeks=104,
    seed=42,
)
az.to_netcdf(idata, "simulated_3channel.nc")

# For demos and tests (fixture ships with PyMC-Marketing):
designer = ExperimentDesigner.from_idata(
    az.from_netcdf("simulated_3channel.nc"),
    saturation="logistic",
    adstock="geometric",
)
```

Using InferenceData (arviz's standard format) rather than a custom `.npz` schema has several advantages: the posterior samples use PyMC-Marketing's standard parameter names with no mapping needed, the format is familiar to any PyMC user, and `az.to_netcdf()` / `az.from_netcdf()` are well-tested round-trip serialisation. The `constant_data` group stores derived quantities (current spend, residual std, spend correlation, ground truth) alongside the posterior in a single file.

Pre-built fixtures ship with PyMC-Marketing as `.nc` files so that demos, tests, and documentation work out of the box. The `generate_experiment_fixture()` function is used only for generating new fixtures or regenerating existing ones with different ground-truth parameters.

Because the ground truth is known, unit tests can validate that the `ExperimentDesigner` correctly identifies the channel with the most uncertain parameters, that power calculations are calibrated (recommendations with assurance 0.8 should succeed ~80% of the time when run against the true DGP), and that adstock ramp fractions match the known α values.

---

## Scope: v2 (future)

- **Geo-level designs:** Accept historical geo-level panel data and integrate with CausalPy #759's power analysis (G3) and market selection (G4). Add geo holdout and staggered geo as design types with geo-specific σ models (SC fit quality, donor pool assessment). Replace the simple σ model with simulation-based σ from placebo-in-time analysis (potentially consuming CausalPy's `SyntheticControl` for σ estimation). Recommend specific geos, not just the channel and spend level.
- **Pulse and switchback designs:** Add design types that produce sharper adstock identification. Pulse tests (brief spike + dark period) give a clean impulse response; switchback tests (alternating on/off) provide replicated decay observations with precision scaling as ~1/√K.
- **Cross-channel information gain via hierarchical priors:** For MMMs with hierarchical priors across channels (e.g., partial pooling of saturation parameters), testing one channel updates beliefs about others. v2's scoring should account for this cross-channel information transfer, which is especially relevant when channels share a common saturation function family.
- **Fisher Information engine:** Replace the heuristic scoring with Jacobian-based Fisher Information for principled information gain computation.
- **Pareto frontier:** Visualise the full set of candidate designs as an information-vs-cost scatter with the Pareto frontier highlighted.
- **Sequential design:** `recommend_next()` for iterative experiment planning (recommend, test, update posterior, repeat).
- **Revenue-aware cost model:** Add a margin parameter and compute net cost as direct spend change minus expected revenue impact. This requires the predicted lift (which creates a circularity with the uncertain parameters, but is useful for more realistic cost ranking).

---

## Module structure

```
pymc_marketing/
  mmm/
    experiment_design/
      __init__.py              # public API: ExperimentDesigner,
                               #   ExperimentRecommendation
      designer.py              # ExperimentDesigner class: __init__(mmm),
                               #   from_idata(), recommend(), scoring, plotting
      recommendation.py        # ExperimentRecommendation dataclass, rationale template
      functions.py             # Lightweight numpy response functions:
                               #   logistic_saturation(x, lam, beta)
                               #   (geometric adstock ramp is computed analytically
                               #    in designer.py, not as a standalone function)
      fixture.py               # generate_experiment_fixture() utility
    data/
      simulated_3channel.nc    # Pre-built InferenceData fixture

tests/
  mmm/
    test_experiment_design/
      test_designer.py         # recommend() output, scoring, filtering
      test_functions.py        # Response function correctness, broadcasting
      test_fixture.py          # InferenceData round-trip, from_idata() loading
      test_assurance.py        # Calibration checks (slow, simulation-based)
```

## Deliverables

- [ ] `ExperimentDesigner` class with `__init__(mmm)` and `from_idata()` constructors
- [ ] `ExperimentDesigner.recommend()` method
- [ ] `ExperimentRecommendation` dataclass with template-based rationale
- [ ] Lightweight numpy response functions in `functions.py` (`logistic_saturation`)
- [ ] Adstock-aware lift prediction (time-averaged over experiment duration)
- [ ] Posterior-predictive power (Bayesian assurance) calculation
- [ ] Channel ranking score (weighted sum of normalised dimensions, configurable weights)
- [ ] Simple cost model (direct additional spend: Δx_abs × T_active)
- [ ] Measurement noise model (σ = σ_residual · √T for cumulative lift in national ITS)
- [ ] `generate_experiment_fixture()` utility and pre-built InferenceData fixture (`.nc`)
- [ ] `plot_channel_diagnostics()`, `plot_power_cost()`, `plot_lift_distributions()`, `plot_saturation_curve()`, `plot_adstock_ramp()`
- [ ] Documentation page in PyMC-Marketing's docs: a concise, end-to-end walkthrough of the full workflow on simulated data (load fixture → inspect posteriors → recommend → interpret results → plot diagnostics)
- [ ] Unit tests with known ground-truth validation (see testing strategy below)

### Testing strategy

Tests fall into two categories:

**Fast deterministic tests** (run on every CI push):
- **Response functions:** `logistic_saturation` matches known analytical values; broadcasting works for scalar, 1D, and 2D inputs.
- **Adstock ramp:** For known α values, the ramp fraction at duration T matches the analytical geometric series `(1 - α^(T+1)) / (1 - α)` (normalised by the steady-state series `1 / (1 - α)`).
- **Scoring:** Min-max normalisation handles edge cases (zero range → 0.5, single candidate, missing `spend_correlation`). Weights sum to 1 after redistribution.
- **Fixture round-trip:** `az.to_netcdf()` then `az.from_netcdf()` recovers all posterior samples and `constant_data` metadata. `ExperimentDesigner.from_idata()` correctly reconstructs the designer from a loaded fixture.
- **Recommend output:** Returns a list of `ExperimentRecommendation` sorted by score. Filtering by `min_snr` removes low-SNR designs. Go-dark designs have `spend_change_frac == -1.0`.

**Slow simulation-based tests** (run nightly or on-demand, marked `@pytest.mark.slow`):
- **Assurance calibration:** For the shipped fixture with known ground-truth DGP, simulate N=500 synthetic experiments for designs at various assurance levels. The fraction of experiments where the true lift exceeds the detection threshold should be within ±0.10 of the reported assurance (binomial 95% CI at N=500 is ~±0.04, so 0.10 is conservative). This validates that assurance is calibrated, not just internally consistent.
- **Channel ranking:** The `ExperimentDesigner` should rank the channel with the widest posterior (by design, in the simulated fixture) as the highest-uncertainty channel.

---

## Related issues (CausalPy)

These CausalPy issues are complementary — they cover post-experiment analysis and geo-level capabilities that the `ExperimentDesigner` can consume in v2:

- [CausalPy #759](https://github.com/pymc-labs/CausalPy/issues/759) — GeoLift feature parity (G3 power analysis, G4 market selection — v2 integration point for geo designs)
- [CausalPy #569](https://github.com/pymc-labs/CausalPy/issues/569) — Bayesian Assurance Framework (power analysis)
- [CausalPy #721](https://github.com/pymc-labs/CausalPy/issues/721) — Experiment duration tool
- [CausalPy #758](https://github.com/pymc-labs/CausalPy/issues/758) — CausalImpact parity (complementary — ITS/BSTS analysis)

## References

- Meta GeoLift: https://github.com/facebookincubator/GeoLift
- GeoLift walkthrough: https://facebookincubator.github.io/GeoLift/docs/GettingStarted/Walkthrough/
- O'Hagan, Stevens & Campbell (2005), "Assurance in clinical trial design"
- Chaloner & Verdinelli (1995), "Bayesian Experimental Design: A Review"
- Foster et al. (2021), "Deep Adaptive Design"
- Johnson, Lewis, & Nubbemeyer (2017), geo experiments for advertising
