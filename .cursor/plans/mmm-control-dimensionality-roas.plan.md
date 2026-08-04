<!-- 75677e4a-af28-44de-a487-8e21e8a68bf4 -->
---
todos:
  - id: "env"
    content: "Run `uv sync --extra docs` to install nutpie, numpyro and jax, which are optional-dependency-only and currently missing from the environment"
    status: pending
  - id: "prototype-dgp"
    content: "Prototype the paper-faithful DGP in sandbox/: media spend from the LKJ Cholesky generative model in mmm_data_generator.ipynb used verbatim, K_max=100 iid standardized controls independent of spend with constant true coefficient, media beta solved for a target delta_media share of explained variance, sigma solved for a target true R2, single dataset generated once"
    status: pending
  - id: "r2d2-prior"
    content: "Implement R2D2Prior as a SpecialPrior subclass usable directly in model_config['gamma_control'] with dims ('control',), using a Beta R2, an unnormalised-Gamma simplex over controls and a plug-in reference scale; first check whether pymc_extras R2D2M2CP can be wrapped instead. Verify it builds and samples via sample_prior"
    status: pending
  - id: "prior-r2"
    content: "Build the implied-prior-R2 figure (Figure 4 left analogue) from prior predictive draws only, comparing Normal (concentrating toward 1 as K grows) against split R2D2 (stable), decomposed into media and control shares"
    status: pending
  - id: "experiment-loop"
    content: "Prototype fit_for_k over nested subsets K in {0, 5, 10, 25, 50, 100} crossed with the two control priors, with control_columns=None for K=0, identical media priors and seeds, extracting total ROAS via mmm.incrementality.compute_incremental_contribution divided by total spend"
    status: pending
  - id: "repetitions"
    content: "Wrap the DGP and fit loop into simulate(seed) and run_experiment(n_reps) so the closing section can compute interval length, coverage and RMSE of total ROAS across K and prior, defaulting to n_reps=1"
    status: pending
  - id: "notebook"
    content: "Author docs/source/notebooks/mmm/mmm_control_dimensionality.ipynb with the full narrative: motivation, paper mapping, DGP, the two control priors, prior R2 mechanism, single-dataset ROAS comparison, diagnostics, repetition study, conclusion"
    status: pending
  - id: "gallery"
    content: "Add a card entry to docs/source/gallery/gallery.yaml (the source of truth) and run `uv run python scripts/generate_gallery.py` to regenerate gallery.md and extract the thumbnail; required or the gallery-in-sync pre-commit hook fails"
    status: pending
  - id: "validate"
    content: "Run the mock notebook runner, do a real execution, and run pre-commit on the changed files"
    status: pending
isProject: false
---
# Case study: control dimensionality and ROAS in an MMM

Translate Experiment 4 of [To select or not to select](https://arxiv.org/abs/2606.22850) (Section 5.4, Figure 9) into a media mix modelling setting: hold the data-generating process fixed, fit an MMM on expanding nested subsets of control variables, and compare how the total media ROAS estimate evolves under independent Normal priors versus a split R2D2 prior on the control coefficients.

## Review finding: synthetic control generation

There is **no reusable utility** for generating synthetic controls anywhere in the repo, so this is new code (living in the notebook, per your choice):

- No importable MMM data generator exists in `pymc_marketing`. [docs/source/notebooks/mmm/mmm_data_generator.ipynb](docs/source/notebooks/mmm/mmm_data_generator.ipynb) is a hand-rolled parameter-recovery walkthrough that sets a `gamma_control` prior in `model_config` but never passes `control_columns`, so that entry is dead code.
- The only multidimensional generator is the un-packaged [scripts/data_generators/mmm_data_generation.py](scripts/data_generators/mmm_data_generation.py), whose controls are two hardcoded binary spikes identical across geos.
- Elsewhere: binary event dummies with hand-picked coefficients ([mmm_example.ipynb](docs/source/notebooks/mmm/mmm_example.ipynb)) or the Gaussian-bump holiday loop in [mmm_causal_identification.ipynb](docs/source/notebooks/mmm/mmm_causal_identification.ipynb). Nothing generates many controls programmatically or budgets their variance.

## Mapping the paper onto the MMM

- Treatment `z` becomes media spend, passed through adstock and saturation. Fixed across all `K`.
- Treatment coefficient `alpha` becomes **total ROAS**, the quantity whose posterior we track.
- Covariates `X`, drawn iid and independent of `z`, become control variables drawn iid and independent of media spend. In the language of `cinelli_crash_2024` these are neutral controls: they cannot bias ROAS, only affect its variance. Any drift we observe is therefore pure prior geometry, not confounding.
- `M_base` (treatment only) becomes an MMM with `control_columns=None`. This is important: Table 4 of the paper shows the Normal full model is *worse* than the base model (RMSE 0.30 vs 0.27, coverage 0.87 vs 0.92), so the baseline is the punchline, not a footnote.
- `M_full` at `p in {10, 50, 100}` becomes the MMM fit on the first `K` controls. With a single geo and weekly data over three years we get `n = 156`, close enough to the paper's `n_obs: [150]` that the same `K` grid can be used and the `K/n` ratio at `K = 100` lands at 0.64 against the paper's 0.67.
- Of the paper's three prior specifications, this notebook implements **Normal** and **split (R2D2 on controls, unchanged priors on media)**. The joint R2D2, which would also shrink the media coefficients and which the paper shows to be badly biased, is out of scope.

## Why the Normal prior is expected to fail

The dims-based `MMM` (now in [pymc_marketing/mmm/mmm.py](pymc_marketing/mmm/mmm.py); `multidimensional.py` is a deprecation shim) builds controls as a plain linear term with an independent Normal prior per control:

```2337:2349:pymc_marketing/mmm/mmm.py
            if self.control_columns is not None and len(self.control_columns) > 0:
                gamma_control = self.model_config["gamma_control"].create_variable(
                    name="gamma_control", xdist=True
                )

                control_data_ = pmd.Data("control_data", self.xarray_dataset._control)

                control_contribution = pmd.Deterministic(
                    "control_contribution",
                    control_data_ * gamma_control,
                )

                mu_var += control_contribution.sum(dim="control")
```

The default is `Prior("Normal", mu=0, sigma=2, dims=(*self.dims, "control"))`. Because `sigma` does not depend on `K`, the implied prior on `Var(mu)` grows linearly in `K` while the residual scale prior stays put, so the implied prior `R^2` concentrates at 1. Note also that controls are **not scaled** by the model (only target and channels are), so `gamma_control` is in scaled-target-per-raw-control units. Standardising the generated controls keeps this interpretable.

## Data-generating process

Generated **once**, following `scripts/dgp_treatment.R` from the reference repo. `dims=()`, a single geo, weekly, `n = 156`.

### Media spend: LKJ generative model

Reuse the LKJ Cholesky approach from [mmm_data_generator.ipynb](docs/source/notebooks/mmm/mmm_data_generator.ipynb) (cell 7) rather than inventing an AR(1) scheme. With a single geo it is used **verbatim**, no extension needed. It gives realistically correlated channel spend with per-channel trends and a hard non-negativity constraint:

```python
with pm.Model(coords=coords) as covariates_model:
    t_data = pm.Data("t", t, dims=("date",))
    L, _, _ = pm.LKJCholeskyCov(
        "L", n=len(coords["channel"]), eta=chol_eta,
        sd_dist=pm.Exponential.dist(lam=1 / 3),
    )
    a = pm.Normal("a", mu=0, sigma=1, dims="channel")
    b = pm.Normal("b", mu=0, sigma=1, dims="channel")
    mu = pm.Deterministic("mu", a + b * t_data[..., None], dims=("date", "channel"))
    x_raw = pm.MvNormal("x_raw", mu=mu, chol=L, dims=("date", "channel"))
    x = pm.Deterministic("x", pt.softplus(x_raw), dims=("date", "channel"))
```

Spend is drawn once with `pm.draw(covariates_model.x, draws=1, random_seed=rng)` and held fixed for every `K`.

### Controls and target

- `K_max = 100` controls, iid and standardized, independent of spend, with a **constant** true coefficient across all of them (the paper uses `beta <- array(0.1, c(p, 1))`).
- True media contribution comes from `geometric_adstock` then `logistic_saturation` in `pymc_marketing.mmm.transformers`, called directly with fixed true `alpha`, `lam`, `beta`, so the truth matches the model's functional form exactly.
- Media beta scaled so media accounts for a share `delta_media` of the explained variance, mirroring equation (34). Default `delta_media = 0.2`.
- `sigma` solved to hit a target true `R^2` (paper footnote 15). Default `R^2 = 0.2`, the low-signal regime where the pathology is sharpest; exposed as a knob since real MMMs often sit higher.
- No seasonality or trend, so controls are the only thing that varies across models.

Calling the transformers directly departs from the `pm.do` plus `sample_prior_predictive` route used for the target in `mmm_data_generator.ipynb`. The reason: `beta` enters `LogisticSaturation` as a pure multiplicative scale, so a single `beta = 1` evaluation lets us solve the variance budget above in closed form, which the `do` route would need a two-pass loop to achieve.

Ground-truth total ROAS is `total_media_contribution / total_spend`, a single number, identical for every `K` because the data never changes.

## The two control priors

Both fit in `model_config["gamma_control"]`, so the media priors are untouched by construction. That is exactly what makes the second one a *split* prior rather than a joint one. **No `MuEffect` subclass is needed.**

### 1. Normal (vanilla)

```python
"gamma_control": Prior("Normal", mu=0, sigma=0.5, dims=("control",))
```

`sigma` is deliberately **not** a function of `K`.

### 2. Split R2D2

`gamma_control` is consumed via `create_variable(name="gamma_control", xdist=True)`, and `pymc_marketing/special_priors.py` already defines a `SpecialPrior` ABC for priors that behave like `Prior` but need custom graph construction (`LaplacePrior` and `LogNormalPrior` are the existing examples). A new `R2D2Prior(SpecialPrior)` only needs `_checks` and `create_variable`, built from `pymc.dims` primitives:

```python
r2 = pmd.Beta(f"{name}_R2", alpha=r2_mean * r2_prec, beta=(1 - r2_mean) * r2_prec)
g = pmd.Gamma(f"{name}_gamma", alpha=concentration, beta=1, dims=("control",))
psi = g / g.sum("control")
tau2 = sigma_ref**2 * r2 / (1 - r2)
z = pmd.Normal(f"{name}_z", mu=0, sigma=1, dims=("control",))
return pmd.Deterministic(name, z * pmd.math.sqrt(tau2 * psi))
```

Defaults follow the paper's `config.yaml`: `r2_mean = 1/3`, `r2_prec = 3` (so `Beta(1, 2)`), symmetric concentration `a = 1`. The unnormalised-simplex trick (`Gamma` then divide by its sum) is lifted straight from `stan/r2_normal.stan`, which avoids needing a Dirichlet with a named dim.

This is the mechanism that matters: the Dirichlet splits a **fixed** `tau2` among however many controls there are, so the total control variance budget does not grow with `K`.

Two honest caveats to state in the notebook:

- `stan/r2_normal.stan` sets `beta = beta_z .* sqrt(sigma^2 * tau2 * psi)`, conditioning the scale on the residual `sigma`. In the MMM, `gamma_control` is created at `mmm.py:2337` while the likelihood `sigma` is created later, so it cannot be referenced. We use a plug-in `sigma_ref` instead. This is defensible because the MMM max-scales its target, so the scaled target has `O(1)` variance, and it leaves the property the experiment turns on fully intact. Wiring in the sampled `sigma` would need a lazy hook or an effect that owns the likelihood, which is worth a note as genuine future work.
- R2D2 assumes standardized covariates (paper footnote 7). Our controls are standardized by construction, which matters here because the MMM does not scale controls itself.

Rather than hand-rolling this, it is worth first checking whether `pymc_extras.distributions.R2D2M2CP` (available in the pinned `pymc-extras`) can be wrapped directly; it is built for plain PyMC rather than `pymc.dims`, so the wrapper may be more trouble than the ten lines above.

## Experiment

Nested subsets `K in {0, 5, 10, 25, 50, 100}` of the same `K_max = 100` controls, so `y` is byte-identical across every fit. `K = 0` must pass `control_columns=None`, not `[]`, because of the Pydantic `min_length=1` constraint.

Each `K > 0` is fit twice, once per control prior; `K = 0` is fit once since the two priors coincide when there are no controls. That is 11 fits. Everything else is held identical: same media priors, same seeds, same sampler.

Total ROAS per fit:

```python
incr = mmm.incrementality.compute_incremental_contribution(frequency="all_time")
total_spend = mmm.data.get_channel_spend().sum()
total_roas = incr.sum("channel") / total_spend
```

`nutpie`, `numpyro` and `jax` live in the `docs` and `test` optional-dependency groups and are **not currently installed**, so step one is `uv sync --extra docs`, then fit with `nuts_sampler="nutpie"`.

Code is factored as `simulate(seed) -> (data, truth)` and `fit_for_k(data, k, prior, seed) -> total_roas_posterior`, so the closing repetition study is a loop over seeds rather than a rewrite.

## Notebook outline

1. Motivation, and the mapping from the paper's RCT setting to MMM.
2. The DGP, with plots of spend and of the control block.
3. The two control priors, including the `R2D2Prior(SpecialPrior)` implementation.
4. **Implied prior `R^2` vs `K` for both priors** (Figure 4 left analogue), from prior predictive draws only, no MCMC. The Normal prior concentrates at 1 as `K` grows while R2D2 stays put. This is the mechanism figure and it is cheap.
5. Single-dataset experiment across `K` and both priors.
6. **Total ROAS posterior densities across `K`, one series per prior**, against the true value and the `K = 0` baseline (Figure 9 analogue). The headline figure.
7. Supporting views: posterior media vs control contribution shares against truth, and posterior `R^2` (`Var(mu) / (Var(mu) + sigma^2)`, as in the reference `generated quantities` block) against the true `R^2`.
8. Diagnostics table: divergences and max r-hat per fit, showing the Normal fit does not visibly degrade even as its ROAS does.
9. Repetition study over `n_reps` datasets producing 90% interval length, coverage and RMSE of total ROAS per `K` and prior (Table 4 analogue). Default `n_reps = 1`, written so raising it is the only change needed.
10. Conclusion.

## Out of scope

- The joint R2D2 prior (R2D2 spanning media and control coefficients together), which the paper shows is badly biased.
- Multiple geos. The `MMM` used here is the dims-based class either way, so adding `dims=("geo",)` later is mostly a matter of widening the prior dims and the spend generator.
- Promoting `R2D2Prior` into `pymc_marketing/special_priors.py` proper, with tests and serialization registration. The notebook defines it locally; if it proves useful, upstreaming it alongside `LaplacePrior` and `LogNormalPrior` is a natural follow-up.
- Conditioning the R2D2 scale on the sampled likelihood `sigma`, which the current `gamma_control` hook cannot reach.

## Files

- New: `docs/source/notebooks/mmm/mmm_control_dimensionality.ipynb` (name open to change).
- Edit: [docs/source/gallery/gallery.yaml](docs/source/gallery/gallery.yaml), which is the **single source of truth** for the gallery. `gallery.md` is generated from it and must not be hand-edited. Add a card under the relevant MMM subsection:

```yaml
- title: Control Dimensionality and ROAS
  notebook: mmm/mmm_control_dimensionality
```

Then run `uv run python scripts/generate_gallery.py` to regenerate `gallery.md` and extract `docs/source/gallery/images/mmm_control_dimensionality.png` from the notebook's first image cell. Commit all three. This is not optional: the `gallery-in-sync` pre-commit hook fails when a notebook exists on disk but is missing from the yaml, so skipping it will block the commit. See [docs/source/gallery/README.md](docs/source/gallery/README.md).

- Scratch prototype in `sandbox/` (gitignored).

## Validation

Prototype the DGP and loop as a `sandbox/` script with a tiny grid first, since iterating on a long notebook cell-by-cell with real MCMC is slow. Then port to the notebook, run `uv run python scripts/run_notebooks/runner.py --notebooks docs/source/notebooks/mmm/mmm_control_dimensionality.ipynb`, do a real execution, and run `uv run pre-commit run --files ...`.
