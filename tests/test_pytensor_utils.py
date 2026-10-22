#   Copyright 2022 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Tests for pytensor_utils module."""

import builtins as _builtins
import sys

import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytensor.xtensor as ptx
import pytest
import xarray as xr
from pymc.model.fgraph import clone_model, fgraph_from_model
from pytensor import function, shared
from pytensor.graph.traversal import ancestors

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer
from pymc_marketing.mmm.mmm import (
    MMM,
    BudgetOptimizerWrapper,
)
from pymc_marketing.pytensor_utils import (
    ConditionedModel,
    ModelSamplerEstimator,
    SharedPosterior,
    VariableLayout,
    _prefix_model,
    align_posterior_dim,
    aligned_posterior,
    check_aligned_dims,
    check_aligned_shape,
    extract_response_distribution,
    merge_models,
    needed_posterior_names,
    non_sample_dims,
    reorder_to_model_coords,
    transpose_to_model_dims,
    validate_unique_value_vars,
    variable_layout,
)


@pytest.fixture
def sample_multidim_data():
    """Create sample data with country dimension for testing."""
    np.random.seed(42)
    n_obs_per_country = 12  # days of data per country
    countries = ["A", "B"]
    dates = pd.date_range(start="2023-01-01", periods=n_obs_per_country, freq="W-MON")

    # Create multi-indexed data with country and date
    data_list = []
    for country in countries:
        country_data = pd.DataFrame(
            {
                "date": dates,
                "country": country,
                "C1": np.random.uniform(10, 50, n_obs_per_country),
                "C2": np.random.uniform(5, 40, n_obs_per_country),
                "control": np.random.normal(10, 2, n_obs_per_country),
            }
        )

        # Create target with some relationship to channels
        country_data["y"] = (
            0.5 * country_data["C1"]
            + 0.3 * country_data["C2"]
            + 2 * country_data["control"]
            + (20 if country == "A" else 30)  # Different baseline per country
            + np.random.normal(0, 3, n_obs_per_country)
        )

        data_list.append(country_data)

    # Combine all country data
    data = pd.concat(data_list, ignore_index=True)
    return data


@pytest.fixture
def fitted_multidim_mmm(sample_multidim_data, mock_pymc_sample):
    """Create and fit a multidimensional MMM model."""
    mmm = MMM(
        date_column="date",
        channel_columns=["C1", "C2"],
        control_columns=["control"],
        dims=("country",),
        target_column="y",
        adstock=GeometricAdstock(l_max=5),
        saturation=LogisticSaturation(),
    )

    X = sample_multidim_data.drop(columns=["y"])
    y = sample_multidim_data["y"]

    # Fit with mocked sampling for speed and determinism
    mmm.fit(X, y, draws=100, chains=2, random_seed=42, tune=100)

    return mmm


@pytest.fixture
def simple_pymc_model_no_channel_data():
    with pm.Model() as m:
        pm.Normal("x", 0.0, 1.0)
    return m


@pytest.fixture
def simple_pymc_model_with_channel_data():
    with pm.Model() as m:
        x = pm.Normal("x", 0.0, 1.0)
        pm.Deterministic("channel_data", x + 1.0)
    return m


def test_extract_response_distribution_vs_sample_response(
    fitted_multidim_mmm, sample_multidim_data
):
    """Test that extract_response_distribution gives similar results to sample_response_distribution."""
    # Get date range from the data
    dates = sample_multidim_data["date"].unique()
    start_date = dates[-1] + pd.Timedelta(
        days=7
    )  # Start one week after last training date
    end_date = start_date + pd.Timedelta(weeks=8)  # 8 weeks of future data

    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")
    print(f"Number of periods: {(end_date - start_date) // 7}")

    fitted_multidim_mmm.add_original_scale_contribution_variable(
        ["channel_contribution"]
    )

    # Wrap the model in BudgetOptimizerWrapper
    optimizable_model = BudgetOptimizerWrapper(
        model=fitted_multidim_mmm,
        start_date=start_date,
        end_date=end_date,
    )

    # Define allocation strategy: 2 units per channel per country
    allocation_values = np.array(
        [
            [2.0, 2.0],  # Country A: C1=2, C2=2
            [2.0, 2.0],  # Country B: C1=2, C2=2
        ]
    )

    allocation_strategy = xr.DataArray(
        allocation_values,
        dims=["country", "channel"],
        coords={
            "country": ["A", "B"],
            "channel": ["C1", "C2"],
        },
    )

    print(
        "\n=== Testing sample_response_distribution vs extract_response_distribution ==="
    )
    print(f"Allocation strategy:\n{allocation_strategy}")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")

    # Create a BudgetOptimizer instance to mimic what happens internally

    budget_optimizer = BudgetOptimizer(
        num_periods=optimizable_model.num_periods,
        model=optimizable_model,
        response_variable="total_media_contribution_original_scale",
    )
    # Compile channel data
    response_fun_inputs = budget_optimizer.extract_response_distribution("channel_data")

    input_fun = function([budget_optimizer._budgets_flat], response_fun_inputs)
    response_fun_inputs_values = input_fun(allocation_strategy.values.flatten())
    print(f"Shape of channel data: {response_fun_inputs_values.shape}")
    print("Dimension must match: (N_periods, N_country, N_channels)")
    assert response_fun_inputs_values.shape == (
        optimizable_model.num_periods + optimizable_model.adstock.l_max,
        len(optimizable_model.idata.posterior.coords["country"]),
        len(optimizable_model.idata.posterior.coords["channel"]),
    ), "Dimension mismatch"

    # Compile and evaluate to get the response values
    response_fun = budget_optimizer.extract_response_distribution(
        "total_media_contribution_original_scale"
    )

    eval_response_values = function([budget_optimizer._budgets_flat], response_fun)
    response_fun_values = eval_response_values(allocation_strategy.values.flatten())

    print("\n--- extract_response_distribution results ---")

    mean_from_extract = np.mean(response_fun_values)
    std_from_extract = np.std(response_fun_values)

    print(f"Mean total contribution: {mean_from_extract:.2f}")
    print(f"Std total contribution: {std_from_extract:.2f}")

    # Method 1: Use sample_response_distribution
    response_data = optimizable_model.sample_response_distribution(
        allocation_strategy=allocation_strategy,
        include_carryover=True,
        include_last_observations=False,
        noise_level=1e-17,
    )

    # Get the channels information
    data_values_for_model = xr.concat(
        [response_data[channel] for channel in optimizable_model.channel_columns],
        dim=pd.Index(optimizable_model.channel_columns, name="channel"),
    ).transpose(..., "channel")
    print(f"Shape of channel data: {data_values_for_model.shape}")
    print("Dimension must match: (N_periods, N_country, N_channels)")
    assert data_values_for_model.shape == (
        optimizable_model.num_periods + optimizable_model.adstock.l_max,
        len(optimizable_model.idata.posterior.coords["country"]),
        len(optimizable_model.idata.posterior.coords["channel"]),
    ), "Dimension mismatch"

    # Extract the total media contribution
    total_contribution_samples = response_data[
        "total_media_contribution_original_scale"
    ]

    print("\n--- sample_response_distribution results ---")
    print(f"Shape: {total_contribution_samples.shape}")
    print(f"Dims: {list(total_contribution_samples.dims)}")

    # Calculate mean and std from samples
    mean_from_samples = total_contribution_samples.mean().values
    std_from_samples = total_contribution_samples.std().values

    print(f"Mean total contribution: {mean_from_samples:.2f}")
    print(f"Std total contribution: {std_from_samples:.2f}")

    # Data inputs checks
    ## Calculate the diff between them per day.
    diff = response_fun_inputs_values - data_values_for_model.values
    print(f"Diff shape: {diff.shape}")
    ##assume dimension is (N_periods, 2, 2) pick the country 0 and channel 0
    diff_country_0_channel_0 = diff[:, 0, 0]
    print(f"Diff country 0 channel 0: {diff_country_0_channel_0}")

    # printing each day for channel 0 and country 0 on response_fun_inputs_values, and data_values_for_model
    print(f"Response fun inputs values: {response_fun_inputs_values[:, 0, 0]}")
    print(f"Data values for model: {data_values_for_model.values[:, 0, 0]}")

    print(f"Number of periods: {optimizable_model.num_periods}")
    print(f"Max adstock lag: {optimizable_model.adstock.l_max}")
    print(
        f"Number of periods + max adstock lag: {optimizable_model.num_periods + optimizable_model.adstock.l_max}"
    )

    ## Assert that response_fun_inputs_values[:, 0, 0] have length equal
    ## to optimizable_model.num_periods + optimizable_model.adstock.l_max

    ## Assert that data_values_for_model.values[:, 0, 0] have length equal
    ## to optimizable_model.num_periods + optimizable_model.adstock.l_max
    assert (
        len(response_fun_inputs_values[:, 0, 0])
        == optimizable_model.num_periods + optimizable_model.adstock.l_max
    ), "Response fun inputs values length mismatch"
    assert (
        len(data_values_for_model.values[:, 0, 0])
        == optimizable_model.num_periods + optimizable_model.adstock.l_max
    ), "Data values for model length mismatch"

    ## Assert that the count of values with zero its equal to optimizable_model.adstock.l_max
    assert (
        np.sum(response_fun_inputs_values[:, 0, 0] == 0)
        == optimizable_model.adstock.l_max
    ), "Posterior predictive dataset: Number of values with zero mismatch"
    assert (
        np.sum(data_values_for_model.values[:, 0, 0] == 0)
        == optimizable_model.adstock.l_max
    ), "Pytensor extraction function:Number of values with zero mismatch"

    ## Assert that both are the same
    assert np.allclose(response_fun_inputs_values, data_values_for_model.values), (
        "Data inputs are not the same across the two methods"
    )

    # Calculate expected number of periods correctly using pd.date_range
    expected_periods = len(pd.date_range(start=start_date, end=end_date, freq="W-MON"))
    assert optimizable_model.num_periods == expected_periods, (
        f"Number of periods mismatch: {optimizable_model.num_periods} != {expected_periods}"
    )

    # Compare the results
    print("\n--- Comparison ---")
    mean_diff = abs(mean_from_samples - mean_from_extract)
    std_diff = abs(std_from_samples - std_from_extract)

    print(f"Mean difference: {mean_diff:.4f}")
    print(f"Std difference: {std_diff:.4f}")

    # Calculate relative differences
    mean_rel_diff = (
        mean_diff / abs(mean_from_samples) * 100 if mean_from_samples != 0 else 0
    )
    std_rel_diff = std_diff / std_from_samples * 100 if std_from_samples != 0 else 0

    print(f"Mean relative difference: {mean_rel_diff:.2f}%")
    print(f"Std relative difference: {std_rel_diff:.2f}%")

    # Verify that both methods give similar results
    # Allow for larger tolerance since we're comparing with noisy data
    assert mean_rel_diff < 1.0, (
        f"Mean relative difference too large: {mean_rel_diff:.2f}%"
    )
    assert std_rel_diff < 1.0, f"Std relative difference too large: {std_rel_diff:.2f}%"

    # Additional checks
    assert np.all(np.isfinite(response_fun_values)), (
        "extract_response_distribution produced non-finite values"
    )
    assert np.all(np.isfinite(total_contribution_samples.values)), (
        "sample_response_distribution produced non-finite values"
    )

    # Both should have the same number of posterior samples
    expected_samples = 200  # 100 draws * 2 chains
    assert response_fun_values.shape[0] == expected_samples, (
        f"Expected {expected_samples} samples from extract_response_distribution"
    )
    assert total_contribution_samples.shape[0] == expected_samples, (
        "Sample count mismatch"
    )

    print("\n✓ Both methods produce consistent results!")


@pytest.mark.parametrize("draws, tune", [(50, 50)])
def test_model_sampler_estimator_with_simple_model(monkeypatch, draws, tune):
    """Smoke test for ModelSamplerEstimator on a tiny PyMC model."""

    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 1.0)
        sigma = pm.HalfNormal("sigma", 1.0)
        pm.Normal("y", mu=mu, sigma=sigma, observed=np.random.normal(0, 1, size=10))

    est = ModelSamplerEstimator(
        tune=tune, draws=draws, chains=2, sequential_chains=1, seed=123
    )
    df = est.run(model)

    # Check schema and basic values
    expected_columns = {
        "model_name",
        "num_steps",
        "eval_time_seconds",
        "sequential_chains",
        "estimated_sampling_time_seconds",
        "estimated_sampling_time_minutes",
        "estimated_sampling_time_hours",
        "tune",
        "draws",
        "chains",
        "seed",
        "timestamp",
    }
    assert set(df.columns) >= expected_columns


def test_model_sampler_estimator_eval_time_multidim_model(fitted_multidim_mmm):
    """Measure eval time for a fitted multidimensional MMM's PyMC model."""
    pm_model = fitted_multidim_mmm.model

    est = ModelSamplerEstimator(
        tune=50, draws=50, chains=1, sequential_chains=1, seed=123
    )

    steps = est.estimate_num_steps_sampling(pm_model)
    assert steps > 0, f"Expected positive number of steps, got {steps}"
    assert isinstance(steps, int), f"Expected steps to be an integer, got {type(steps)}"

    est_df = est.run(pm_model)

    # Check schema and basic values
    expected_columns = {
        "model_name",
        "num_steps",
        "eval_time_seconds",
        "sequential_chains",
        "estimated_sampling_time_seconds",
        "estimated_sampling_time_minutes",
        "estimated_sampling_time_hours",
        "tune",
        "draws",
        "chains",
        "seed",
        "timestamp",
    }
    assert set(est_df.columns) >= expected_columns, (
        f"Expected columns {expected_columns} not found in {set(est_df.columns)}"
    )

    assert isinstance(est.default_mcmc_kwargs, dict), (
        f"Expected default_mcmc_kwargs to be a dict, got {type(est.default_mcmc_kwargs)}"
    )
    assert isinstance(est.default_nuts_kwargs, dict), (
        f"Expected default_nuts_kwargs to be a dict, got {type(est.default_nuts_kwargs)}"
    )


def test_jax_numpyro_not_available(monkeypatch, fitted_multidim_mmm):
    """Ensure estimate_model_eval_time raises when JAX is unavailable."""
    # Simulate that JAX (and pymc's jax helper) are not importable
    monkeypatch.delitem(sys.modules, "jax", raising=False)
    monkeypatch.delitem(sys.modules, "pymc.sampling.jax", raising=False)

    real_import = _builtins.__import__

    def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "jax" or name.startswith("pymc.sampling.jax"):
            raise ImportError("blocked for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(_builtins, "__import__", _blocked_import)

    with pytest.raises(
        RuntimeError, match="JAX backend is required for ModelSamplerEstimator"
    ):
        est = ModelSamplerEstimator(
            tune=50, draws=50, chains=1, sequential_chains=1, seed=123
        )
        est.estimate_model_eval_time(fitted_multidim_mmm.model)


def test_merge_models_prefix_and_merge_on_channel_data(
    fitted_multidim_mmm, sample_multidim_data
):
    # Derive a short future window
    dates = sample_multidim_data["date"].unique()
    start_date = dates[-1] + pd.Timedelta(days=7)
    end_date = start_date + pd.Timedelta(weeks=4)

    # Create two optimizer-compatible wrappers from the same fitted model
    wrapper1 = BudgetOptimizerWrapper(
        model=fitted_multidim_mmm, start_date=start_date, end_date=end_date
    )
    wrapper2 = BudgetOptimizerWrapper(
        model=fitted_multidim_mmm, start_date=start_date, end_date=end_date
    )

    # Build per-wrapper optimization models
    m1 = wrapper1._set_predictors_for_optimization(num_periods=wrapper1.num_periods)
    m2 = wrapper2._set_predictors_for_optimization(num_periods=wrapper2.num_periods)

    # Merge with explicit prefixes, sharing on 'channel_data'
    merged = merge_models(
        models=[m1, m2], prefixes=["model1", "model2"], merge_on="channel_data"
    )

    # 'channel_data' should be present and unprefixed exactly once
    var_names = set(merged.named_vars)
    assert "channel_data" in var_names
    assert "model1_channel_data" not in var_names
    assert "model2_channel_data" not in var_names

    # Prefixed response variables should exist for each model
    assert "model1_total_media_contribution_original_scale" in var_names
    assert "model2_total_media_contribution_original_scale" in var_names

    # Prefixed channel contribution should also exist
    assert "model1_channel_contribution" in var_names
    assert "model2_channel_contribution" in var_names

    # The shared dims on channel_data should retain their original names
    channel_data_dims = merged.named_vars_to_dims["channel_data"]
    for d in ("date", "channel", "country"):
        assert d in channel_data_dims


def test_merge_models_value_vars_unique_and_logp_compiles(
    fitted_multidim_mmm, sample_multidim_data
):
    # Build two wrapped models with identical structure to force potential name collisions
    dates = sample_multidim_data["date"].unique()
    start_date = dates[-1] + pd.Timedelta(days=7)
    end_date = start_date + pd.Timedelta(weeks=4)

    wrapper1 = BudgetOptimizerWrapper(
        model=fitted_multidim_mmm, start_date=start_date, end_date=end_date
    )
    wrapper2 = BudgetOptimizerWrapper(
        model=fitted_multidim_mmm, start_date=start_date, end_date=end_date
    )

    m1 = wrapper1._set_predictors_for_optimization(num_periods=wrapper1.num_periods)
    m2 = wrapper2._set_predictors_for_optimization(num_periods=wrapper2.num_periods)

    merged = merge_models(
        models=[m1, m2], prefixes=["model1", "model2"], merge_on="channel_data"
    )

    # Validate uniqueness of value var names and mapping consistency
    validate_unique_value_vars(merged)

    # Additionally ensure PyMC can create the logp+dlogp function without raising
    _ = merged.logp_dlogp_function(ravel_inputs=True)


def test_merge_models_raises_with_too_few_models(fitted_multidim_mmm):
    m1 = fitted_multidim_mmm.model
    with pytest.raises(ValueError, match="Need at least 2 models to merge"):
        merge_models([m1])


def test_merge_models_raises_when_prefixes_length_mismatch(fitted_multidim_mmm):
    m1 = fitted_multidim_mmm.model
    m2 = fitted_multidim_mmm.model
    with pytest.raises(
        ValueError, match=r"Number of prefixes \(1\) must match number of models \(2\)"
    ):
        merge_models([m1, m2], prefixes=["a"])


def test_merge_models_raises_when_merge_on_missing_in_first_model(
    simple_pymc_model_no_channel_data, simple_pymc_model_with_channel_data
):
    m1 = simple_pymc_model_no_channel_data
    m2 = simple_pymc_model_with_channel_data
    with pytest.raises(
        ValueError, match="Variable 'channel_data' not found in first model"
    ):
        merge_models([m1, m2], prefixes=["a", "b"], merge_on="channel_data")


def test_merge_models_raises_when_merge_on_missing_in_other_models(
    simple_pymc_model_no_channel_data, simple_pymc_model_with_channel_data
):
    m1 = simple_pymc_model_with_channel_data
    m2 = simple_pymc_model_no_channel_data
    with pytest.raises(
        ValueError, match="Variable 'channel_data' not found in model 2"
    ):
        merge_models([m1, m2], prefixes=["a", "b"], merge_on="channel_data")


def test_prefix_model_exclude_none_renames_vars_dims_and_coords():
    with pm.Model(coords={"d": [0, 1, 2]}) as m:
        pm.Normal("x", 0.0, 1.0, dims="d")

    fg, _ = fgraph_from_model(m, inlined_views=True)
    fg2 = _prefix_model(fg, prefix="pfx", exclude_vars=None)

    # Variable names should be prefixed
    out_names = {v.name for v in fg2.outputs}
    assert any(name.startswith("pfx_") for name in out_names)

    # Dims/coords should be prefixed too
    coords_keys = set(fg2._coords.keys())  # type: ignore[attr-defined]
    assert "pfx_d" in coords_keys
    assert "d" not in coords_keys


def _model_with_deterministic_alias():
    with pm.Model(coords={"obs": range(5)}) as model:
        x = pm.Normal("x")
        pm.Deterministic("x_alias", x)
        d = pm.Deterministic("d", 2 * x)
        pm.Deterministic("dd", d + 1)
        y_obs = pm.Data("y_obs", np.zeros(5), dims="obs")
        pm.Normal("y", x, observed=y_obs, dims="obs")
    return model


def test_merge_models_deterministic_alias_of_rv():
    """A Deterministic that aliases an RV must not duplicate the RV after merging.

    Rebuilding the ``ModelVar`` ops has to replace consumers before producers,
    otherwise the stale ``ModelFreeRV`` node is re-imported through the alias.
    """
    merged = merge_models(
        [_model_with_deterministic_alias(), _model_with_deterministic_alias()],
        prefixes=["a", "b"],
    )

    assert {rv.name for rv in merged.free_RVs} == {"a_x", "b_x"}
    expected_names = {
        f"{prefix}_{name}"
        for prefix in ("a", "b")
        for name in ("x", "x_alias", "d", "dd", "y_obs", "y")
    }
    assert set(merged.named_vars) == expected_names
    assert tuple(merged.named_vars_to_dims["a_y"]) == ("a_obs",)
    assert tuple(merged.named_vars_to_dims["b_y_obs"]) == ("b_obs",)
    assert set(merged.coords) == {"a_obs", "b_obs"}
    assert set(merged.dim_lengths) == {"a_obs", "b_obs"}
    validate_unique_value_vars(merged)
    assert np.isfinite(merged.compile_logp()(merged.initial_point()))


def test_merge_models_variables_already_starting_with_prefix():
    """Variables whose name already starts with ``<prefix>_`` still get unique value vars."""

    def make():
        with pm.Model() as model:
            pm.Normal("a_x")
            pm.HalfNormal("a_s")
        return model

    merged = merge_models([make(), make()], prefixes=["a", "b"])

    assert set(merged.named_vars) == {"a_a_x", "a_a_s", "b_a_x", "b_a_s"}
    assert {v.name for v in merged.value_vars} == {
        "a_a_x",
        "a_a_s_log__",
        "b_a_x",
        "b_a_s_log__",
    }
    validate_unique_value_vars(merged)
    assert np.isfinite(merged.compile_logp()(merged.initial_point()))


def test_merge_models_merge_on_shares_dim_lengths():
    """``set_data`` on a shared dim after ``merge_on`` resizes both merged models."""

    def make(seed):
        coords = {"channel": ["c1", "c2"], "date": np.arange(4)}
        with pm.Model(coords=coords) as model:
            X = pm.Data(
                "channel_data", np.ones((4, 2)) * seed, dims=("date", "channel")
            )
            y_obs = pm.Data("y_obs", np.ones(4), dims="date")
            beta = pm.HalfNormal("beta", dims="channel")
            mu = pm.Deterministic("mu", (X * beta).sum(-1), dims="date")
            eps = pm.Normal("eps", dims="date")
            pm.Normal("y", mu + eps, 1.0, observed=y_obs, dims="date")
        return model

    merged = merge_models(
        [make(1), make(2)], prefixes=["a", "b"], merge_on="channel_data"
    )
    assert set(merged.dim_lengths) == {"channel", "date"}

    with merged:
        pm.set_data(
            {
                "channel_data": np.ones((5, 2)),
                "a_y_obs": np.ones(5),
                "b_y_obs": np.ones(5),
            },
            coords={"date": np.arange(5)},
        )

    assert tuple(merged["a_eps"].shape.eval()) == (5,)
    assert tuple(merged["b_eps"].shape.eval()) == (5,)
    assert tuple(merged["a_mu"].shape.eval()) == (5,)
    assert tuple(merged["b_mu"].shape.eval()) == (5,)
    assert np.isfinite(merged.compile_logp()(merged.initial_point()))


def test_extract_response_distribution_accepts_several_variables():
    """Several variables extract in one pass, sharing the subgraph they have in common.

    Callers that need more than one node -- incrementality evaluating
    ``channel_contribution`` alongside a mediated effect, both reading the same
    spend through the same adstock -- would otherwise pay for that subgraph once
    per node.  Sharing it is the point of the sequence form, so the test checks
    the shared node is literally the same object, not just that the values agree.
    """
    # xtensor variables throughout: the conditioning step substitutes posterior
    # draws as xtensor constants and vectorizes over them.
    data_values = np.array([1.0, 2.0, 3.0])
    with pm.Model(coords={"d": [0, 1, 2]}) as model:
        data = pmd.Data("data", data_values, dims="d")
        slope = pmd.Normal("slope", mu=0.0, sigma=1.0)
        doubled = pmd.Deterministic("doubled", 2.0 * data * slope)
        pmd.Deterministic("tripled", 3.0 * doubled)

    with model:
        prior = pm.sample_prior_predictive(draws=4, random_seed=42)
    idata = xr.DataTree.from_dict({"/posterior": prior["/prior"].to_dataset()})

    single = extract_response_distribution(model, idata, "doubled")
    both = extract_response_distribution(model, idata, ["doubled", "tripled"])

    # A single name keeps the historical scalar return type.
    assert not isinstance(single, list)
    assert isinstance(both, list) and len(both) == 2

    doubled_graph, tripled_graph = both

    # The shared subgraph is shared, not duplicated: one conditioning pass, so
    # the posterior draws of ``slope`` are the same node in both graphs.  Not the
    # returned outputs themselves -- the final rewrite can wrap one of them.
    def conditioned_slope(graph):
        return {
            node
            for node in ancestors([graph])
            if getattr(node, "name", None) == "slope"
        }

    assert conditioned_slope(doubled_graph) & conditioned_slope(tripled_graph)
    # And the assertion has teeth: a separate call conditions its own copy.
    assert not conditioned_slope(doubled_graph) & conditioned_slope(single)

    doubled_values, tripled_values = function([], both)()
    np.testing.assert_allclose(tripled_values, 3.0 * doubled_values)
    drawn = prior["/prior"].to_dataset()["slope"].values.reshape(-1)
    np.testing.assert_allclose(doubled_values, np.outer(drawn, 2.0 * data_values))
    # Same graph, same numbers, whichever form was asked for.
    np.testing.assert_allclose(function([], [single])()[0], doubled_values)

    with pytest.raises(ValueError, match="at least one variable"):
        extract_response_distribution(model, idata, [])


def _channel_layout(dims=("channel",), *, shape=(2,), coords=None):
    return VariableLayout(
        dims=dims,
        coords={} if coords is None else coords,
        shape=shape,
    )


def test_non_sample_dims_keeps_storage_order_and_drops_samples():
    posterior_da = xr.DataArray(np.ones((2, 3, 4)), dims=("chain", "channel", "draw"))
    assert non_sample_dims(posterior_da, sample_dims=("chain", "draw")) == ("channel",)


def test_check_aligned_dims_compares_the_dim_set_not_its_order():
    layout = _channel_layout(dims=("channel", "date"), shape=(2, 3))

    check_aligned_dims("beta", ("date", "channel"), layout)  # order is free

    with pytest.raises(ValueError, match="has dims"):
        check_aligned_dims("beta", ("channel",), layout)


@pytest.mark.parametrize(
    "given, bound, match",
    [
        pytest.param(
            None,
            pd.Index(["a", "b"]),
            "labeled on one side only",
            id="given-unlabeled",
        ),
        pytest.param(
            pd.Index(["a", "b"]),
            None,
            "labeled on one side only",
            id="bound-unlabeled",
        ),
        pytest.param(
            pd.Index(["a", "b"]),
            pd.Index(["a", "a"]),
            "has duplicated channel labels",
            id="duplicated-bound-labels",
        ),
        pytest.param(
            pd.Index(["a", "c"]),
            pd.Index(["a", "b"]),
            "has different channel labels",
            id="different-labels",
        ),
        pytest.param(
            pd.Index(["a", "b", "c"]),
            pd.Index(["a", "b"]),
            "has different channel labels",
            id="different-length",
        ),
        pytest.param(
            pd.Index(["a", "a"]),
            pd.Index(["a", "b"]),
            "has different channel labels",
            id="duplicated-given-labels",
        ),
    ],
)
def test_align_posterior_dim_refuses_unusable_labels(given, bound, match):
    posterior_da = xr.DataArray(
        np.ones((2 if given is None else len(given), 3)),
        dims=("channel", "sample"),
        coords={} if given is None else {"channel": given},
    )

    with pytest.raises(ValueError, match=match):
        align_posterior_dim("beta", posterior_da, "channel", bound)


def test_align_posterior_dim_reorders_the_axis_to_the_bound_labels():
    posterior_da = xr.DataArray(
        [1.0, 2.0], dims="channel", coords={"channel": ["b", "a"]}
    )

    aligned = align_posterior_dim("beta", posterior_da, "channel", pd.Index(["a", "b"]))

    assert list(aligned.indexes["channel"]) == ["a", "b"]
    np.testing.assert_allclose(aligned.values, [2.0, 1.0])
    # The same array comes back when neither side is labeled.
    unlabeled = xr.DataArray(np.ones((2, 3)), dims=("channel", "sample"))
    assert align_posterior_dim("beta", unlabeled, "channel", None) is unlabeled


def test_transpose_to_model_dims_moves_only_a_reordering_of_the_declared_dims():
    """A variable over the declared dims in another order is transposed.

    The sample dims stay in front, a variable already in order or over another
    set of dims is left as given, and so is one the model does not declare.
    """
    model_dims = {
        "lam": ("geo", "channel"),
        "alpha": ("geo", "channel"),
        "sigma": ("geo",),
    }
    rng = np.random.default_rng(0)
    posterior = xr.Dataset(
        {
            "lam": (("chain", "draw", "channel", "geo"), rng.normal(size=(1, 2, 3, 3))),
            "alpha": (
                ("chain", "draw", "geo", "channel"),
                rng.normal(size=(1, 2, 3, 3)),
            ),
            "sigma": (("chain", "draw", "geo", "extra"), rng.normal(size=(1, 2, 3, 2))),
            "free": (
                ("chain", "draw", "channel", "geo"),
                rng.normal(size=(1, 2, 3, 3)),
            ),
        }
    )

    laid_out = transpose_to_model_dims(posterior, model_dims)

    assert laid_out["lam"].dims == ("chain", "draw", "geo", "channel")
    xr.testing.assert_identical(
        laid_out["lam"], posterior["lam"].transpose(*laid_out["lam"].dims)
    )
    for name in ("alpha", "sigma", "free"):
        assert laid_out[name].dims == posterior[name].dims
    # Nothing to move returns the dataset itself.
    assert transpose_to_model_dims(laid_out, model_dims) is laid_out


def test_check_aligned_shape_allows_only_the_sample_length_to_change():
    layout = _channel_layout(shape=(2,))

    check_aligned_shape("beta", (2,), layout)

    with pytest.raises(ValueError, match="Only the sample dimension may change"):
        check_aligned_shape("beta", (3,), layout)


def test_align_posterior_dim_compares_datetime_labels_by_value():
    """Datetime labels stored in another unit are the same labels."""
    dates = pd.DatetimeIndex(["2024-01-01", "2024-01-08"])
    posterior_da = xr.DataArray(
        [1.0, 2.0], dims="date", coords={"date": dates[::-1].as_unit("ns")}
    )

    aligned = align_posterior_dim("beta", posterior_da, "date", dates.as_unit("s"))

    np.testing.assert_allclose(aligned.values, [2.0, 1.0])


def test_variable_layout_records_the_non_sample_dims_labels_and_shape():
    """Only a labeled dim carries labels, and the sample dims may have any names."""
    posterior_da = xr.DataArray(
        np.zeros((2, 4, 3, 5)),
        dims=("chain", "draw", "channel", "lag"),
        coords={"channel": ["a", "b", "c"]},
    )

    layout = variable_layout(posterior_da, sample_dims=("chain", "draw"))

    assert layout.dims == ("channel", "lag")
    assert list(layout.coords) == ["channel"]
    assert list(layout.coords["channel"]) == ["a", "b", "c"]
    assert layout.shape == (3, 5)
    stacked = posterior_da.stack(sample=("chain", "draw")).transpose("sample", ...)
    stacked_layout = variable_layout(stacked, sample_dims=("sample",))
    assert stacked_layout.dims == layout.dims
    assert stacked_layout.shape == layout.shape


def _geo_lag_layout() -> VariableLayout:
    """The layout of a ``(chain, draw, geo, lag)`` variable with labeled ``geo``."""
    constructed = xr.DataArray(
        np.zeros((1, 2, 2, 3)),
        dims=("chain", "draw", "geo", "lag"),
        coords={"geo": ["n", "s"]},
    )
    return variable_layout(constructed, sample_dims=("chain", "draw"))


def test_aligned_posterior_lays_a_variable_out_like_its_layout():
    """Labels are reordered and dims transposed to the layout, sample dims first.

    The number of draws is free: only the non-sample geometry is checked.
    """
    values = np.arange(1 * 5 * 3 * 2, dtype=float).reshape(1, 5, 3, 2)
    incoming = xr.DataArray(
        values,
        dims=("chain", "draw", "lag", "geo"),
        coords={"geo": ["s", "n"]},
    ).transpose("lag", "chain", "geo", "draw")

    aligned = aligned_posterior(
        "beta", incoming, _geo_lag_layout(), sample_dims=("chain", "draw")
    )

    assert aligned.dims == ("chain", "draw", "geo", "lag")
    assert list(aligned.indexes["geo"]) == ["n", "s"]
    # Position (0, 0) of the laid-out geo and lag is geo "n", stored second.
    np.testing.assert_array_equal(aligned.values[0, :, 0, 0], values[0, :, 0, 1])


@pytest.mark.parametrize(
    "dims, geo, lag, match",
    [
        pytest.param(("chain", "draw", "geo"), ["n", "s"], None, "has dims", id="dims"),
        pytest.param(
            ("chain", "draw", "geo", "lag"),
            ["n", "x"],
            3,
            "has different geo labels",
            id="labels",
        ),
        pytest.param(
            ("chain", "draw", "geo", "lag"),
            None,
            3,
            "labeled on one side only",
            id="one-sided-labels",
        ),
        pytest.param(
            ("chain", "draw", "geo", "lag"),
            ["n", "s"],
            4,
            "has shape",
            id="non-sample-length",
        ),
    ],
)
def test_aligned_posterior_refuses_another_layout(dims, geo, lag, match):
    """Another dim set, label set, one-sided labeling or non-sample length is refused."""
    sizes = {"chain": 1, "draw": 2, "geo": 2, "lag": lag}
    incoming = xr.DataArray(
        np.zeros(tuple(sizes[dim] for dim in dims)),
        dims=dims,
        coords={} if geo is None else {"geo": geo},
    )

    with pytest.raises(ValueError, match=match):
        aligned_posterior(
            "beta", incoming, _geo_lag_layout(), sample_dims=("chain", "draw")
        )


def _scaled_response_model() -> pm.Model:
    """``resp = scaled * gamma`` with ``scaled = beta * spend``, and an unused RV."""
    with pm.Model(coords={"channel": ["a", "b"]}) as model:
        spend = pmd.Data("spend", np.array([1.0, 2.0]), dims="channel")
        beta = pmd.Normal("beta", dims="channel")
        gamma = pmd.Normal("gamma")
        pmd.Normal("unused")
        scaled = pmd.Deterministic("scaled", beta * spend)
        pmd.Deterministic("resp", scaled * gamma + beta)
        pmd.Deterministic("resp_scaled", scaled * gamma)
    return model


def test_needed_posterior_names_lists_each_free_variable_a_response_reads_once():
    """Data and the deterministics on the way are traversed, not listed."""
    model = _scaled_response_model()

    names = needed_posterior_names(model, [model["resp"]])

    assert sorted(names) == ["beta", "gamma"]


def test_needed_posterior_names_stops_at_a_frozen_deterministic():
    """A frozen deterministic is listed instead of the variables behind it.

    A frozen name the model does not have is ignored.
    """
    model = _scaled_response_model()

    names = needed_posterior_names(
        model, [model["resp_scaled"]], frozen_deterministics=["scaled", "missing"]
    )

    assert sorted(names) == ["gamma", "scaled"]


def test_reorder_to_model_coords_puts_a_matching_label_set_in_the_model_order():
    """Every labeled dim whose labels are the model's goes to the model's order.

    ``chain`` and ``draw`` are never reordered, even when the model has coords
    of that name, and a dataset already in order comes back as is.
    """
    values = np.arange(1 * 2 * 2 * 3, dtype=float).reshape(1, 2, 2, 3)
    posterior = xr.Dataset(
        {"beta": (("chain", "draw", "geo", "channel"), values)},
        coords={
            "chain": [0],
            "draw": [1, 0],
            "geo": ["s", "n"],
            "channel": ["c", "a", "b"],
        },
    )
    model_coords = {
        "geo": ["n", "s"],
        "channel": ["a", "b", "c"],
        "draw": [0, 1],
    }

    reordered = reorder_to_model_coords(posterior, model_coords)

    assert list(reordered.indexes["geo"]) == ["n", "s"]
    assert list(reordered.indexes["channel"]) == ["a", "b", "c"]
    assert list(reordered.indexes["draw"]) == [1, 0]
    # Position (0, 0) is geo "n" and channel "a", both stored second.
    np.testing.assert_array_equal(
        reordered["beta"].values[0, :, 0, 0], values[0, :, 1, 1]
    )
    assert reorder_to_model_coords(reordered, model_coords) is reordered


def test_reorder_to_model_coords_compares_datetime_labels_by_value():
    """Model dates in another storage unit still reorder the posterior's."""
    dates = pd.DatetimeIndex(["2024-01-01", "2024-01-08"])
    posterior = xr.Dataset(
        {"beta": (("chain", "draw", "date"), np.array([[[1.0, 2.0]]]))},
        coords={"date": dates[::-1].as_unit("ns")},
    )

    reordered = reorder_to_model_coords(
        posterior, {"date": dates.as_unit("s").to_numpy()}
    )

    np.testing.assert_allclose(reordered["beta"].values[0, 0], [2.0, 1.0])


@pytest.mark.parametrize(
    "labels, model_coords",
    [
        pytest.param(["s", "n"], {"geo": ["n", "x"]}, id="other-labels"),
        pytest.param(["s", "n"], {"geo": ["n", "s", "e"]}, id="other-length"),
        pytest.param(["s", "n"], {}, id="no-model-coords"),
        pytest.param(None, {"geo": ["n", "s"]}, id="unlabeled"),
        pytest.param(
            ["n", "s", "n"], {"geo": ["s", "n", "n"]}, id="duplicated-model-labels"
        ),
    ],
)
def test_reorder_to_model_coords_leaves_a_dim_it_cannot_match_as_given(
    labels, model_coords
):
    """A dim whose labels are not the model's (unique) labels is stored as given."""
    size = 2 if labels is None else len(labels)
    posterior = xr.Dataset(
        {"beta": (("chain", "draw", "geo"), np.zeros((1, 1, size)))},
        coords={} if labels is None else {"geo": labels},
    )

    assert reorder_to_model_coords(posterior, model_coords) is posterior


def test_transpose_to_model_dims_keeps_a_stacked_sample_dim_in_front():
    """Any sample dims can be named, and they stay in front."""
    posterior = xr.Dataset(
        {"lam": (("channel", "sample", "geo"), np.zeros((3, 4, 2)))},
    )

    laid_out = transpose_to_model_dims(
        posterior, {"lam": ("geo", "channel")}, sample_dims=("sample",)
    )

    assert laid_out["lam"].dims == ("sample", "geo", "channel")


def _slope_model_and_draws(draws: int, seed: int):
    """A tiny dims model plus a prior-predictive ``posterior`` to condition on."""
    data_values = np.array([1.0, 2.0, 3.0])
    with pm.Model(coords={"d": [0, 1, 2]}) as model:
        data = pmd.Data("data", data_values, dims="d")
        slope = pmd.Normal("slope", mu=0.0, sigma=1.0)
        pmd.Deterministic("doubled", 2.0 * data * slope)
        pmd.Deterministic("tripled", 3.0 * data * slope)
    with model:
        prior = pm.sample_prior_predictive(draws=draws, random_seed=seed)
    idata = xr.DataTree.from_dict({"/posterior": prior["/prior"].to_dataset()})
    return model, idata, data_values


def _two_channel_model():
    """``resp = beta * gamma`` over a labelled channel axis."""
    with pm.Model(coords={"channel": ["a", "b"]}) as model:
        beta = pmd.Normal("beta", dims="channel")
        gamma = pmd.Normal("gamma", dims="channel")
        pmd.Deterministic("resp", beta * gamma)
    return model


def _two_channel_posterior(beta, gamma, channel=("a", "b")):
    return xr.Dataset(
        {
            "beta": (("chain", "draw", "channel"), np.asarray(beta, float)[None, None]),
            "gamma": (
                ("chain", "draw", "channel"),
                np.asarray(gamma, float)[None, None],
            ),
        },
        coords={"channel": list(channel)},
    )


def _gamma_model():
    """``base`` reads ``beta`` alone; ``resp`` reads both gammas, so either can be the missing one."""
    spend = np.array([10.0, 20.0])
    with pm.Model(coords={"channel": ["a", "b"]}) as model:
        spend_var = pmd.Data("spend", spend, dims="channel")
        beta = pmd.Normal("beta", dims="channel")
        gamma1 = pmd.Normal("gamma1", dims="channel")
        gamma2 = pmd.Normal("gamma2", dims="channel")
        pmd.Deterministic("base", beta * spend_var)
        pmd.Deterministic("resp", gamma2 + gamma1)
    return model, spend


def _gamma_posterior(beta=(1.0, 1.0), gamma1=None, gamma2=None, channel=("a", "b")):
    """A channel posterior holding ``beta`` and whichever gammas are asked for."""
    data = {"beta": (("chain", "draw", "channel"), np.asarray(beta, float)[None, None])}
    for name, values in (("gamma1", gamma1), ("gamma2", gamma2)):
        if values is not None:
            data[name] = (
                ("chain", "draw", "channel"),
                np.asarray(values, float)[None, None],
            )
    return xr.Dataset(data, coords={"channel": list(channel)})


def test_shared_posterior_rebinds_compiled_graph_without_recompile():
    """Draws bound through ``SharedPosterior`` follow ``set_posterior``.

    The compiled function is the same object before and after the rebind, the
    values match what constants would have given for each posterior, and the
    number of draws is free to change between rebinds.
    """
    model, idata_a, data_values = _slope_model_and_draws(draws=4, seed=1)
    _, idata_b, _ = _slope_model_and_draws(draws=6, seed=2)

    shared_posterior = SharedPosterior()
    graph = extract_response_distribution(
        model, idata_a, "doubled", shared_posterior=shared_posterior
    )
    assert set(shared_posterior.variables) == {"slope"}
    assert shared_posterior.dims["slope"][0] == "sample"

    fn = function([], graph)

    def expected(idata):
        drawn = idata["posterior"]["slope"].values.reshape(-1)
        return np.outer(drawn, 2.0 * data_values)

    np.testing.assert_allclose(fn(), expected(idata_a))

    shared_posterior.set_posterior(idata_b)
    np.testing.assert_allclose(fn(), expected(idata_b))
    assert fn().shape[0] == 6

    # The constant path is untouched by the rebind: a graph extracted without
    # ``shared_posterior`` still evaluates its own draws.
    constant_graph = extract_response_distribution(model, idata_a, "doubled")
    np.testing.assert_allclose(function([], constant_graph)(), expected(idata_a))


def test_shared_posterior_rebinds_every_graph_at_once():
    """Two graphs extracted with one instance move together on one rebind.

    A later extraction has to be conditioned on the draws already bound; a
    different posterior would otherwise be silently ignored, so it raises.
    """
    model = _two_channel_model()
    posterior_a = _two_channel_posterior([1.0, 1.0], [2.0, 2.0])
    posterior_b = _two_channel_posterior([10.0, 10.0], [3.0, 3.0])

    shared_posterior = SharedPosterior()
    resp = extract_response_distribution(
        model, posterior_a, "resp", shared_posterior=shared_posterior
    )
    beta = extract_response_distribution(
        model, posterior_a, "beta", shared_posterior=shared_posterior
    )
    resp_fn, beta_fn = function([], resp), function([], beta)
    np.testing.assert_allclose(resp_fn(), [[2.0, 2.0]])
    np.testing.assert_allclose(beta_fn(), [[1.0, 1.0]])

    shared_posterior.set_posterior(posterior_b)
    np.testing.assert_allclose(resp_fn(), [[30.0, 30.0]])
    np.testing.assert_allclose(beta_fn(), [[10.0, 10.0]])

    with pytest.raises(
        ValueError, match=r"already binds '(beta|gamma)' to different draws"
    ):
        extract_response_distribution(
            model, posterior_a, "resp", shared_posterior=shared_posterior
        )
    # Same draws as bound: fine, and the new graph reads the shared variable.
    again = extract_response_distribution(
        model, posterior_b, "resp", shared_posterior=shared_posterior
    )
    np.testing.assert_allclose(function([], again)(), [[30.0, 30.0]])


def test_shared_posterior_rebind_is_atomic():
    """A posterior that fails on one variable leaves every variable untouched.

    The corrupt variable is the *second* one bound, so an implementation that
    validates and writes in a single loop would have already swapped the first
    one when it raises.  Two failure modes are covered: a layout mismatch,
    caught before any cast, and a value that passes the layout checks but
    cannot be cast to the variable's dtype.
    """
    model = _two_channel_model()
    shared_posterior = SharedPosterior()
    graph = extract_response_distribution(
        model,
        _two_channel_posterior([1.0, 1.0], [2.0, 2.0]),
        "resp",
        shared_posterior=shared_posterior,
    )
    fn = function([], graph)
    first, second = list(shared_posterior.variables)
    bound = {
        name: shared_posterior.variables[name].get_value().copy()
        for name in (first, second)
    }

    def assert_untouched():
        for name, values in bound.items():
            np.testing.assert_array_equal(
                shared_posterior.variables[name].get_value(), values
            )
        np.testing.assert_allclose(fn(), [[2.0, 2.0]])

    # Layout failure on the second variable.
    wrong_dims = _two_channel_posterior([9.0, 9.0], [9.0, 9.0]).assign(
        {second: lambda ds: ds[second].expand_dims(extra=[0])}
    )
    with pytest.raises(ValueError, match=rf"'{second}' has dims"):
        shared_posterior.set_posterior(wrong_dims)
    assert_untouched()

    # Cast failure on the second variable: right dims and shape, wrong values.
    uncastable = _two_channel_posterior([9.0, 9.0], [9.0, 9.0])
    uncastable[second] = uncastable[second].astype(object)
    uncastable[second][0, 0, 0] = "x"
    with pytest.raises(ValueError, match="could not convert"):
        shared_posterior.set_posterior(uncastable)
    assert_untouched()


def test_shared_posterior_aligns_coordinates_on_rebind():
    """Labelled coordinates are matched by label, not position; new labels are refused."""
    model = _two_channel_model()
    shared_posterior = SharedPosterior()
    graph = extract_response_distribution(
        model,
        _two_channel_posterior([1.0, 2.0], [10.0, 10.0]),
        "resp",
        shared_posterior=shared_posterior,
    )
    fn = function([], graph)
    np.testing.assert_allclose(fn(), [[10.0, 20.0]])

    reordered = _two_channel_posterior([2.0, 1.0], [10.0, 10.0], channel=("b", "a"))
    shared_posterior.set_posterior(reordered)
    np.testing.assert_allclose(fn(), [[10.0, 20.0]])

    relabelled = _two_channel_posterior([1.0, 2.0], [10.0, 10.0], channel=("a", "c"))
    with pytest.raises(ValueError, match=r"channel labels") as info:
        shared_posterior.set_posterior(relabelled)
    # The message summarises the difference rather than dumping both axes.
    assert "1 not bound, 1 missing" in str(info.value)


def test_shared_posterior_realigns_permuted_multi_dim_variables():
    """A variable over two non-sample dims rebinds from a permuted, reordered layout.

    The posterior may store ``beta`` as ``(channel, country)`` with the
    countries in another order while it was bound as ``(country, channel)``;
    the values have to land in the bound layout.
    """
    countries, channels = ["x", "y", "z"], ["a", "b"]
    with pm.Model(coords={"country": countries, "channel": channels}) as model:
        beta = pmd.Normal("beta", dims=("country", "channel"))
        pmd.Deterministic("doubled", 2.0 * beta)

    values = np.arange(6.0).reshape(3, 2)  # country x channel
    bound = xr.Dataset(
        {"beta": (("chain", "draw", "country", "channel"), values[None, None])},
        coords={"country": countries, "channel": channels},
    )
    shared_posterior = SharedPosterior()
    fn = function(
        [],
        extract_response_distribution(
            model, bound, "doubled", shared_posterior=shared_posterior
        ),
    )
    np.testing.assert_allclose(fn(), 2.0 * values[None])

    # Same draws, dims swapped and countries reversed.
    permuted = xr.Dataset(
        {
            "beta": (
                ("chain", "draw", "channel", "country"),
                (10.0 + values).T[:, ::-1][None, None],
            )
        },
        coords={"country": countries[::-1], "channel": channels},
    )
    shared_posterior.set_posterior(permuted)
    np.testing.assert_allclose(fn(), 2.0 * (10.0 + values)[None])


def test_shared_posterior_mixed_type_labels_raise_value_error():
    """Labels that cannot be sorted still get the documented ValueError."""
    with pm.Model(coords={"k": ["a", 1]}) as model:
        beta = pmd.Normal("beta", dims="k")
        pmd.Deterministic("doubled", 2.0 * beta)
    posterior = xr.Dataset(
        {"beta": (("chain", "draw", "k"), np.ones((1, 1, 2)))}, coords={"k": ["a", 1]}
    )
    shared_posterior = SharedPosterior()
    fn = function(
        [],
        extract_response_distribution(
            model, posterior, "doubled", shared_posterior=shared_posterior
        ),
    )
    # Same labels, other order: accepted.
    shared_posterior.set_posterior(
        xr.Dataset(
            {"beta": (("chain", "draw", "k"), np.array([[[3.0, 5.0]]]))},
            coords={"k": [1, "a"]},
        )
    )
    np.testing.assert_allclose(fn(), [[10.0, 6.0]])
    # A different label: ValueError, not TypeError from sorting.
    with pytest.raises(ValueError, match=r"different k labels"):
        shared_posterior.set_posterior(posterior.assign_coords(k=["a", 2]))


def test_shared_posterior_refuses_duplicated_labels_at_bind_time():
    """A duplicated label cannot be realigned by ``sel`` later, so it is refused up front.

    Without this a bound ``["a", "a"]`` axis would accept a rebind from
    ``["a", "b"]`` (same length, no missing label) and silently select the
    ``a`` column twice.
    """
    with pm.Model(coords={"k": ["a", "a"]}) as model:
        beta = pmd.Normal("beta", dims="k")
        pmd.Deterministic("doubled", 2.0 * beta)
    duplicated = xr.Dataset(
        {"beta": (("chain", "draw", "k"), np.ones((1, 1, 2)))}, coords={"k": ["a", "a"]}
    )
    with pytest.raises(ValueError, match=r"duplicated k labels: \['a'\]"):
        extract_response_distribution(
            model, duplicated, "doubled", shared_posterior=SharedPosterior()
        )

    # The constant path is unaffected: it never realigns by label.
    np.testing.assert_allclose(
        function([], extract_response_distribution(model, duplicated, "doubled"))(),
        [[2.0, 2.0]],
    )

    # And a duplicate on the rebind side is caught by the set check.
    unique = duplicated.assign_coords(k=["a", "b"])
    shared_posterior = SharedPosterior()
    extract_response_distribution(
        model, unique, "doubled", shared_posterior=shared_posterior
    )
    with pytest.raises(ValueError, match=r"different k labels"):
        shared_posterior.set_posterior(duplicated)


def test_shared_posterior_failed_extract_keeps_registries_consistent():
    """A duplicated-label variable is refused before it is inserted.

    The instance already binds a valid ``beta``; the extraction that adds a
    ``gamma`` with duplicated labels must raise and leave ``_variables`` and
    ``_layouts`` keyed alike, so the instance stays usable.
    """
    model = _two_channel_model()
    shared_posterior = SharedPosterior()
    extract_response_distribution(
        model,
        _two_channel_posterior([1.0, 2.0], [2.0, 2.0]),
        "beta",
        shared_posterior=shared_posterior,
    )
    assert set(shared_posterior.variables) == {"beta"}

    duplicated_gamma = xr.Dataset(
        {"gamma": (("chain", "draw", "channel"), np.ones((1, 1, 2)))},
        coords={"channel": ["a", "a"]},
    )
    with pytest.raises(ValueError, match=r"duplicated channel labels: \['a'\]"):
        extract_response_distribution(
            model, duplicated_gamma, "gamma", shared_posterior=shared_posterior
        )

    registries = (
        set(shared_posterior._variables),
        set(shared_posterior._layouts),
    )
    assert registries == ({"beta"}, {"beta"})
    # The instance still rebinds every graph it handed out.
    shared_posterior.set_posterior(_two_channel_posterior([3.0, 4.0], [5.0, 6.0]))


def _raise_from_vectorize(*_args, **_kwargs):
    """Stand-in for ``vectorize_graph`` that fails the extraction."""
    raise RuntimeError("vectorize failed")


def _missing_gamma2_posterior(_monkeypatch):
    """A posterior that fails in the loop, after ``gamma1`` has been registered."""
    return _gamma_posterior(gamma1=(2.0, 2.0))


def _vectorize_failing_posterior(monkeypatch):
    """A valid posterior with ``vectorize_graph`` patched to fail after both gammas bind."""
    monkeypatch.setattr(
        "pymc_marketing.pytensor_utils.vectorize_graph", _raise_from_vectorize
    )
    return _gamma_posterior(gamma1=(2.0, 2.0), gamma2=(3.0, 3.0))


@pytest.mark.parametrize(
    "posterior_for_call, expected_error",
    [
        pytest.param(
            _missing_gamma2_posterior, (KeyError, "gamma2"), id="missing-variable"
        ),
        pytest.param(
            _vectorize_failing_posterior,
            (RuntimeError, "vectorize failed"),
            id="vectorize-fails",
        ),
    ],
)
def test_extract_response_distribution_failure_registers_nothing(
    posterior_for_call, expected_error, monkeypatch
):
    """A shared extraction that fails at any point registers no variable.

    ``resp`` reads ``gamma1`` then ``gamma2``, so ``gamma1`` is registered before
    either failure: a missing variable in the placeholder loop, or an error while
    vectorizing the graph afterwards.  The instance must be left as it began.
    """
    model, _ = _gamma_model()
    shared_posterior = SharedPosterior()
    posterior = posterior_for_call(monkeypatch)
    error, match = expected_error

    with pytest.raises(error, match=match):
        extract_response_distribution(
            model, posterior, "resp", shared_posterior=shared_posterior
        )

    assert set(shared_posterior._variables) == set()
    assert set(shared_posterior._layouts) == set()


def test_shared_posterior_coordinate_mismatch_message_stays_short():
    """A long labelled axis is reported by counts and a few examples, not in full."""
    dates = pd.date_range("2020-02-02", periods=208, freq="W").to_pydatetime()
    with pm.Model(coords={"date": list(dates)}) as model:
        latent = pmd.Normal("latent", dims="date")
        pmd.Deterministic("out", latent.sum())
    posterior = xr.Dataset(
        {"latent": (("chain", "draw", "date"), np.ones((1, 1, dates.size)))},
        coords={"date": list(dates)},
    )
    shared_posterior = SharedPosterior()
    extract_response_distribution(
        model, posterior, "out", shared_posterior=shared_posterior
    )

    # A day's shift makes every label differ, which is the worst case: the
    # message has to truncate both sides rather than list 208 labels each.
    shifted = posterior.assign_coords(date=list(dates + pd.Timedelta(days=1)))
    with pytest.raises(ValueError, match=r"208 not bound, 208 missing") as info:
        shared_posterior.set_posterior(shifted)
    assert len(str(info.value)) < 400


def test_shared_posterior_accepts_datetime_labels_across_units():
    """A datetime axis compares by value, not by storage unit.

    A netCDF round trip (``ModelBuilder.save`` then ``load``) turns a
    ``datetime64[us]`` coordinate into ``datetime64[ns]``; the labels are the
    same, so the reloaded posterior must rebind.
    """
    dates = pd.date_range("2020-01-05", periods=3, freq="W")
    with pm.Model(coords={"date": dates}) as model:
        latent = pmd.Normal("latent", dims="date")
        pmd.Deterministic("out", 2.0 * latent)
    values = np.arange(3.0)[None, None]
    as_us = xr.Dataset(
        {"latent": (("chain", "draw", "date"), values)},
        coords={"date": dates.values.astype("datetime64[us]")},
    )
    as_ns = as_us.assign_coords(date=dates.values.astype("datetime64[ns]"))
    assert as_us["date"].dtype != as_ns["date"].dtype

    shared_posterior = SharedPosterior()
    fn = function(
        [],
        extract_response_distribution(
            model, as_us, "out", shared_posterior=shared_posterior
        ),
    )
    shared_posterior.set_posterior(as_ns.assign(latent=as_ns["latent"] + 1.0))
    np.testing.assert_allclose(fn(), [[2.0, 4.0, 6.0]])

    # The mismatch message renders timestamps on both sides, not integers.
    off = as_ns.assign_coords(date=as_ns["date"].values + np.timedelta64(1, "D"))
    with pytest.raises(ValueError, match=r"2020-01-06") as info:
        shared_posterior.set_posterior(off)
    assert "2020-01-05" in str(info.value)


def test_shared_posterior_refuses_labelled_against_unlabelled_dims():
    """Labels on one side and none on the other cannot be checked, so it raises."""
    model = _two_channel_model()

    def unlabelled(beta, gamma):
        return _two_channel_posterior(beta, gamma).drop_vars("channel")

    # Bound with labels, rebound without.
    with_labels = SharedPosterior()
    fn = function(
        [],
        extract_response_distribution(
            model,
            _two_channel_posterior([1.0, 2.0], [2.0, 2.0]),
            "resp",
            shared_posterior=with_labels,
        ),
    )
    with pytest.raises(ValueError, match=r"'channel' is labeled on one side only"):
        with_labels.set_posterior(unlabelled([2.0, 1.0], [2.0, 2.0]))
    np.testing.assert_allclose(fn(), [[2.0, 4.0]])

    # Bound without labels, rebound with.
    without_labels = SharedPosterior()
    fn = function(
        [],
        extract_response_distribution(
            model,
            unlabelled([1.0, 2.0], [2.0, 2.0]),
            "resp",
            shared_posterior=without_labels,
        ),
    )
    with pytest.raises(ValueError, match=r"'channel' is labeled on one side only"):
        without_labels.set_posterior(
            _two_channel_posterior([2.0, 1.0], [2.0, 2.0], channel=("b", "a"))
        )
    np.testing.assert_allclose(fn(), [[2.0, 4.0]])
    # Unlabelled on both sides is positional, as documented.
    without_labels.set_posterior(unlabelled([2.0, 1.0], [2.0, 2.0]))
    np.testing.assert_allclose(fn(), [[4.0, 2.0]])


def test_shared_posterior_reuse_tolerates_nan_draws():
    """A NaN in the bound draws must not make an identical posterior look different."""
    model = _two_channel_model()
    posterior = _two_channel_posterior([1.0, np.nan], [2.0, 2.0])
    shared_posterior = SharedPosterior()
    extract_response_distribution(
        model, posterior, "resp", shared_posterior=shared_posterior
    )
    # The documented reuse path: a second graph extracted from the same draws.
    extract_response_distribution(
        model, posterior, "beta", shared_posterior=shared_posterior
    )

    different = _two_channel_posterior([1.0, np.nan], [3.0, 2.0])
    with pytest.raises(ValueError, match=r"already binds 'gamma' to different draws"):
        extract_response_distribution(
            model, different, "resp", shared_posterior=shared_posterior
        )


def test_shared_posterior_set_posterior_validates_input():
    """A posterior missing a bound variable, or with other dims, is refused."""
    model, idata, _ = _slope_model_and_draws(draws=3, seed=4)
    shared_posterior = SharedPosterior()
    extract_response_distribution(
        model, idata, "doubled", shared_posterior=shared_posterior
    )

    without_slope = idata["posterior"].to_dataset().drop_vars("slope")
    with pytest.raises(KeyError, match=r"missing variables.*slope"):
        shared_posterior.set_posterior(without_slope)

    wrong_dims = (
        idata["posterior"]
        .to_dataset()
        .assign(slope=lambda ds: ds["slope"].expand_dims(extra=[0, 1]))
    )
    with pytest.raises(ValueError, match=r"slope.*dims"):
        shared_posterior.set_posterior(wrong_dims)


def test_shared_posterior_rejects_changed_non_sample_lengths():
    """Only the sample axis may change length; a resized channel axis is refused."""
    with pm.Model(coords={"channel": ["a", "b"]}) as model:
        beta = pmd.Normal("beta", mu=0.0, sigma=1.0, dims="channel")
        pmd.Deterministic("doubled", 2.0 * beta)
    with model:
        prior = pm.sample_prior_predictive(draws=3, random_seed=5)
    # Bound without channel labels so the length check, not the label check,
    # is what refuses the resized axis below.
    posterior = prior["/prior"].to_dataset().drop_vars("channel")

    shared_posterior = SharedPosterior()
    extract_response_distribution(
        model,
        xr.DataTree.from_dict({"/posterior": posterior}),
        "doubled",
        shared_posterior=shared_posterior,
    )
    assert "beta" in repr(shared_posterior)

    # A plain posterior Dataset (chain, draw) is accepted as well as a tree,
    # and a different draw count is fine.
    shared_posterior.set_posterior(posterior.isel(draw=[0]))
    assert shared_posterior.variables["beta"].get_value().shape == (1, 2)

    three_channels = xr.Dataset(
        {"beta": (("chain", "draw", "channel"), np.zeros((1, 3, 3)))}
    )
    with pytest.raises(ValueError, match=r"beta.*Only the sample dimension"):
        shared_posterior.set_posterior(three_channels)


def test_extract_response_distribution_one_variable_posterior():
    """Regression: a posterior holding a single variable extracts on the constant path.

    ``arviz.extract`` squeezes a one-variable group to a ``DataArray``, which
    used to make the placeholder lookup fail with ``KeyError``.
    """
    with pm.Model(coords={"channel": ["a", "b"]}) as model:
        beta = pmd.Normal("beta", dims="channel")
        pmd.Deterministic("resp", 2.0 * beta)
    posterior = xr.Dataset(
        {"beta": (("chain", "draw", "channel"), np.array([[[1.0, 3.0]]]))}
    )
    graph = extract_response_distribution(
        model, xr.DataTree.from_dict({"/posterior": posterior}), "resp"
    )
    np.testing.assert_allclose(function([], graph)(), [[2.0, 6.0]])


def test_shared_posterior_interrupted_write_keeps_the_previous_draws(monkeypatch):
    """An interrupt partway through the write puts every variable back.

    ``resp`` reads ``beta`` and ``gamma``, so a rebind writes two shared
    variables.  An interrupt on the second write lands after the first was
    swapped; every variable, and a function compiled over them, must still hold
    the previous draws rather than a mix of two posteriors.
    """
    model = _two_channel_model()
    shared_posterior = SharedPosterior()
    fn = function(
        [],
        extract_response_distribution(
            model,
            _two_channel_posterior([1.0, 2.0], [3.0, 4.0]),
            "resp",
            shared_posterior=shared_posterior,
        ),
    )
    variables = shared_posterior.variables
    assert len(variables) == 2
    before = {name: var.get_value().copy() for name, var in variables.items()}

    second = list(variables.values())[1]
    original_set_value = second.set_value
    calls = []

    def interrupted_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise KeyboardInterrupt
        return original_set_value(*args, **kwargs)

    monkeypatch.setattr(second, "set_value", interrupted_once)

    with pytest.raises(KeyboardInterrupt):
        shared_posterior.set_posterior(
            _two_channel_posterior([10.0, 20.0], [30.0, 40.0])
        )

    for name, var in shared_posterior.variables.items():
        np.testing.assert_array_equal(var.get_value(), before[name])
    np.testing.assert_allclose(fn(), [[3.0, 8.0]])

    # The binding is intact: the next rebind writes both variables.
    shared_posterior.set_posterior(_two_channel_posterior([10.0, 20.0], [30.0, 40.0]))
    np.testing.assert_allclose(fn(), [[300.0, 800.0]])


def _expected_doubled(idata, data_values) -> np.ndarray:
    drawn = idata["posterior"]["slope"].values.reshape(-1)
    return np.outer(drawn, 2.0 * data_values)


@pytest.mark.parametrize(
    "rebinder",
    ["shared_posterior", "conditioned_model", "conditioned_model_before_extract"],
)
def test_a_rebind_ignores_an_unread_variable_it_could_not_stack(rebinder):
    """A posterior variable no graph reads may change freely on a rebind.

    Regression: a rebind stacked the whole posterior into one ``sample`` axis
    before it picked the bound variables, so every rebind copied every
    deterministic of a lazily loaded posterior.  An unread variable that could
    not be stacked, here one that already has a ``sample`` dim of another
    length, made the rebind fail, also before any graph was extracted.
    """
    model, first, data_values = _slope_model_and_draws(draws=4, seed=1)
    _, second, _ = _slope_model_and_draws(draws=6, seed=2)
    posterior = second["posterior"].to_dataset().assign(junk=("sample", np.zeros(5)))
    if rebinder == "shared_posterior":
        bound = SharedPosterior()
        graph = extract_response_distribution(
            model, first, "doubled", shared_posterior=bound
        )
    else:
        bound = ConditionedModel(model, first, intervention={"data": data_values})
        if rebinder == "conditioned_model_before_extract":
            bound.set_posterior(posterior)
        graph = bound.extract("doubled")
    fn = function([], graph)

    bound.set_posterior(posterior)

    np.testing.assert_allclose(fn(), _expected_doubled(second, data_values))


def test_a_rebind_with_nothing_bound_still_needs_chain_and_draw():
    """With nothing bound a rebind writes nothing, but still refuses unindexed draws."""
    shared_posterior = SharedPosterior()
    unindexed = xr.Dataset({"slope": (("draw",), np.zeros(3))})

    with pytest.raises(ValueError, match="indexed by 'chain' and 'draw'"):
        shared_posterior.set_posterior(unindexed)


def _channel_conditioned_model():
    """A model with ``resp = beta * spend`` and ``resp_g = gamma * spend``."""
    spend = np.array([10.0, 20.0])
    with pm.Model(coords={"channel": ["a", "b"]}) as model:
        spend_var = pmd.Data("spend", spend, dims="channel")
        beta = pmd.Normal("beta", dims="channel")
        gamma = pmd.Normal("gamma", dims="channel")
        pmd.Deterministic("resp", beta * spend_var)
        pmd.Deterministic("resp_g", gamma * spend_var)
    return model, spend


def _channel_posterior(beta, channel=("a", "b"), *, gamma=None):
    data_vars = {
        "beta": (("chain", "draw", "channel"), np.asarray(beta, float)[None, None])
    }
    if gamma is not None:
        data_vars["gamma"] = (
            ("chain", "draw", "channel"),
            np.asarray(gamma, float)[None, None],
        )
    return xr.Dataset(data_vars, coords={"channel": list(channel)})


@pytest.fixture
def slope_posteriors():
    """A ``doubled = 2 * data * slope`` model, its data, and three posteriors."""
    model, first, data_values = _slope_model_and_draws(draws=4, seed=1)
    _, second, _ = _slope_model_and_draws(draws=6, seed=2)
    _, third, _ = _slope_model_and_draws(draws=2, seed=3)
    return model, data_values, first, second, third


@pytest.fixture
def conditioned_slope(slope_posteriors):
    """The conditioned model, intervening on ``data``, not yet rebound."""
    model, data_values, first, _, _ = slope_posteriors
    return ConditionedModel(model, first, intervention={"data": data_values})


def test_conditioned_model_extract_is_cached_by_key(conditioned_slope):
    graph = conditioned_slope.extract("doubled")
    assert conditioned_slope.extract("doubled") is graph
    # A one-element sequence shares the extraction but keeps the list form.
    assert conditioned_slope.extract(["doubled"])[0] is graph
    # A second response name keys a separate extraction.
    other_graph = conditioned_slope.extract("tripled")
    assert other_graph is not graph
    assert conditioned_slope.extract("tripled") is other_graph


def _derived_responses(conditioned, *factors):
    """Distinct derived nodes over the clone, one per factor."""
    return [conditioned.model["doubled"] * factor for factor in factors]


def test_conditioned_model_cache_keeps_a_hit_and_evicts_the_least_recent(
    conditioned_slope, monkeypatch
):
    """A hit keeps its graph, and an insert evicts the least recent key.

    An evicted key re-extracts to a new graph.  A hit makes its key the most
    recent, so the next insert evicts the other one instead.
    """
    monkeypatch.setattr("pymc_marketing.pytensor_utils._GRAPH_CACHE_SIZE", 2)
    first, second, third = _derived_responses(conditioned_slope, 2.0, 3.0, 4.0)

    first_graph = conditioned_slope.extract(first)
    conditioned_slope.extract(second)
    conditioned_slope.extract(third)  # the third key evicts the least recent, ``first``
    assert conditioned_slope.extract(first) is not first_graph

    first_graph = conditioned_slope.extract(
        first
    )  # cached again, so the next call hits
    conditioned_slope.extract(second)
    # A hit moves ``first`` to the most recent position.
    assert conditioned_slope.extract(first) is first_graph
    conditioned_slope.extract(third)  # evicts ``second``, not the just-used ``first``
    assert conditioned_slope.extract(first) is first_graph


def test_conditioned_model_evicted_key_reextracts_and_follows_a_later_rebind(
    slope_posteriors, monkeypatch
):
    """An evicted key is a miss, so it re-extracts onto the shared posterior.

    Eviction drops the cache entry only: the graph handed out before it still
    reads the shared variables, so a function compiled from it before a later
    rebind follows that rebind, and so does the re-extracted graph.
    """
    model, data_values, _, second, third = slope_posteriors
    conditioned = ConditionedModel(model, second, intervention={"data": data_values})
    monkeypatch.setattr("pymc_marketing.pytensor_utils._GRAPH_CACHE_SIZE", 1)
    kept, evicting = _derived_responses(conditioned, 2.0, 3.0)

    kept_graph = conditioned.extract(kept)
    evicted_fn = function([], kept_graph)
    conditioned.extract(evicting)  # the cap of one evicts ``kept``

    re_extracted = conditioned.extract(kept)
    assert re_extracted is not kept_graph
    np.testing.assert_allclose(
        evicted_fn(), 2.0 * _expected_doubled(second, data_values)
    )

    conditioned.set_posterior(third)
    expected = 2.0 * _expected_doubled(third, data_values)
    np.testing.assert_allclose(evicted_fn(), expected)
    np.testing.assert_allclose(function([], re_extracted)(), expected)


def test_conditioned_model_evicted_graph_still_requires_its_variables(monkeypatch):
    """An evicted graph keeps reading its variables, so a rebind must supply them.

    ``resp_g`` is evicted by ``resp``, but a function compiled from it still
    reads ``gamma``: a posterior without ``gamma`` is refused, and that
    function keeps the previous draws.
    """
    model, spend = _channel_conditioned_model()
    conditioned = ConditionedModel(
        model,
        _channel_posterior([1.0, 2.0], gamma=[3.0, 4.0]),
        intervention={"spend": spend},
    )
    monkeypatch.setattr("pymc_marketing.pytensor_utils._GRAPH_CACHE_SIZE", 1)
    evicted_fn = function([], conditioned.extract("resp_g"))
    conditioned.extract("resp")  # the cap of one evicts ``resp_g``

    with pytest.raises(KeyError, match="gamma"):
        conditioned.set_posterior(_channel_posterior([5.0, 6.0]))

    np.testing.assert_allclose(evicted_fn(), [[30.0, 80.0]])
    conditioned.set_posterior(_channel_posterior([5.0, 6.0], gamma=[7.0, 8.0]))
    np.testing.assert_allclose(evicted_fn(), [[70.0, 160.0]])


def test_conditioned_model_names_distinguish_shared_name(
    conditioned_slope, slope_posteriors
):
    """Two nodes that share a name key apart in the cache.

    An anonymous intermediate of the clone is renamed to the registered node's name.
    The two return different graphs that each evaluate to their own expected values and stay cached separately.
    """
    _first_model, data_values, first, _, _ = slope_posteriors
    registered = conditioned_slope.model["doubled"]
    anonymous = next(
        node
        for node in ancestors([registered])
        if getattr(node, "name", None) is None
        and getattr(node, "owner", None) is not None
    )
    anonymous.name = "doubled"

    registered_graph = conditioned_slope.extract(registered)
    anonymous_graph = conditioned_slope.extract(anonymous)

    assert registered_graph is not anonymous_graph
    assert conditioned_slope.extract(registered) is registered_graph
    assert conditioned_slope.extract(anonymous) is anonymous_graph

    np.testing.assert_allclose(
        function([], registered_graph)(), _expected_doubled(first, data_values)
    )
    np.testing.assert_allclose(function([], anonymous_graph)(), 2.0 * data_values)


def test_conditioned_model_rebind_with_an_empty_cache_writes_and_requires_bound_variables(
    monkeypatch,
):
    """An empty cache means no graph is cached, not that no draws are bound.

    With a cache size of zero nothing stays cached, yet the graph handed out
    reads the shared variables: a rebind still writes ``beta``, so the compiled
    function follows it, and a rebind that lacks ``beta`` raises ``KeyError``
    and leaves the draws and ``posterior`` as they were.
    """
    model, spend = _channel_conditioned_model()
    conditioned = ConditionedModel(
        model, _channel_posterior([1.0, 2.0]), intervention={"spend": spend}
    )
    monkeypatch.setattr("pymc_marketing.pytensor_utils._GRAPH_CACHE_SIZE", 0)
    graph = conditioned.extract("resp")
    assert conditioned.extract("resp") is not graph  # nothing was cached
    fn = function([], graph)

    conditioned.set_posterior(_channel_posterior([7.0, 8.0]))
    np.testing.assert_allclose(fn(), [[70.0, 160.0]])

    posterior = conditioned.posterior
    without_beta = xr.Dataset({"gamma": (("chain", "draw"), np.zeros((1, 1)))})
    with pytest.raises(KeyError, match="beta"):
        conditioned.set_posterior(without_beta)
    assert conditioned.posterior is posterior
    np.testing.assert_allclose(fn(), [[70.0, 160.0]])


def test_conditioned_model_later_rebind_reuses_graph_and_follows_new_draws(
    conditioned_slope, slope_posteriors
):
    """Every rebind swaps draws in the one shared posterior, whatever the draw count."""
    _, data_values, _, second, third = slope_posteriors
    shared = conditioned_slope.shared_posterior
    assert isinstance(shared, SharedPosterior)
    graph = conditioned_slope.extract("doubled")
    fn = function([], graph)

    conditioned_slope.set_posterior(second)  # four draws to six
    np.testing.assert_allclose(fn(), _expected_doubled(second, data_values))
    conditioned_slope.set_posterior(third)  # six draws to two

    assert conditioned_slope.shared_posterior is shared
    # No re-extraction: the same compiled function follows the new draw count.
    assert conditioned_slope.extract("doubled") is graph
    np.testing.assert_allclose(fn(), _expected_doubled(third, data_values))


def test_conditioned_model_graph_extracted_at_construction_follows_a_rebind(
    conditioned_slope, slope_posteriors
):
    """A graph handed out before the first rebind follows it.

    Regression: a graph extracted before the first ``set_posterior`` was
    conditioned on constant draws, so a function compiled from it silently kept
    the construction draws after the rebind.
    """
    _, data_values, first, second, _ = slope_posteriors
    graph = conditioned_slope.extract("doubled")
    fn = function([], graph)
    np.testing.assert_allclose(fn(), _expected_doubled(first, data_values))

    conditioned_slope.set_posterior(second)  # four draws to six

    assert conditioned_slope.extract("doubled") is graph
    np.testing.assert_allclose(fn(), _expected_doubled(second, data_values))


def test_conditioned_model_failed_rebind_keeps_the_previous_binding(
    conditioned_slope, slope_posteriors
):
    _, data_values, first, second, _ = slope_posteriors
    fn = function([], conditioned_slope.extract("doubled"))
    posterior = conditioned_slope.posterior

    # A posterior the shared variables cannot be rebound to is refused atomically,
    # so the previous draws and ``posterior`` stay in place.
    without_slope = second["posterior"].to_dataset().drop_vars("slope")
    with pytest.raises(KeyError, match="missing variables"):
        conditioned_slope.set_posterior(without_slope)

    assert conditioned_slope.posterior is posterior
    np.testing.assert_allclose(fn(), _expected_doubled(first, data_values))


def test_conditioned_model_interrupted_rebind_keeps_the_previous_draws(monkeypatch):
    """An interrupt partway through a rebind leaves every graph on the previous draws.

    ``resp`` and ``resp_g`` read ``beta`` and ``gamma``, so a rebind writes two
    shared variables.  An interrupt on the second write lands after the first
    was swapped; the compiled function and ``posterior`` must still be on the
    previous posterior rather than on a mix of two.
    """
    model, spend = _channel_conditioned_model()
    conditioned = ConditionedModel(
        model,
        _channel_posterior([1.0, 2.0], gamma=[3.0, 4.0]),
        intervention={"spend": spend},
    )
    fn = function([], conditioned.extract(["resp", "resp_g"]))
    posterior = conditioned.posterior
    variables = conditioned.shared_posterior.variables
    assert list(variables) == ["beta", "gamma"]

    second = variables["gamma"]
    original_set_value = second.set_value
    calls = []

    def interrupted_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise KeyboardInterrupt
        return original_set_value(*args, **kwargs)

    monkeypatch.setattr(second, "set_value", interrupted_once)

    with pytest.raises(KeyboardInterrupt):
        conditioned.set_posterior(_channel_posterior([5.0, 6.0], gamma=[7.0, 8.0]))

    assert conditioned.posterior is posterior
    resp, resp_g = fn()
    np.testing.assert_allclose(resp, [[10.0, 40.0]])
    np.testing.assert_allclose(resp_g, [[30.0, 80.0]])

    # The binding is intact: the next rebind writes both variables.
    conditioned.set_posterior(_channel_posterior([5.0, 6.0], gamma=[7.0, 8.0]))
    resp, resp_g = fn()
    np.testing.assert_allclose(resp, [[50.0, 120.0]])
    np.testing.assert_allclose(resp_g, [[70.0, 160.0]])


def test_conditioned_model_failed_extract_leaves_the_binding_unchanged():
    """An extract that fails on a missing variable leaves no trace.

    The shared posterior binds ``beta`` only.  Extracting ``resp``, which reads
    ``gamma2 + gamma1``, fails on the missing ``gamma2``; nothing new may be
    bound, so a rebind without either gamma still succeeds and is followed.
    """
    model, spend = _gamma_model()
    conditioned = ConditionedModel(
        model,
        _gamma_posterior(gamma1=(2.0, 2.0), gamma2=(3.0, 3.0)),
        intervention={"spend": spend},
    )
    fn = function([], conditioned.extract("base"))
    conditioned.set_posterior(_gamma_posterior(gamma1=(2.0, 2.0)))
    assert set(conditioned.shared_posterior.variables) == {"beta"}

    with pytest.raises(KeyError, match="gamma2"):
        conditioned.extract("resp")

    assert set(conditioned.shared_posterior.variables) == {"beta"}
    # A rebind that satisfies every handed-out graph is accepted.
    conditioned.set_posterior(_gamma_posterior(beta=(2.0, 3.0)))
    np.testing.assert_allclose(fn(), [[20.0, 60.0]])


def test_conditioned_model_refuses_pre_intervention_variable(
    conditioned_slope, slope_posteriors
):
    model, *_ = slope_posteriors
    with pytest.raises(ValueError, match="reads inputs of the source model"):
        conditioned_slope.extract(model["doubled"])


@pytest.mark.parametrize(
    "intervention, expected",
    [
        (None, 50.0),
        ({"spend": np.array([30.0, 40.0])}, 110.0),
    ],
    ids=["no-intervention", "with-intervention"],
)
def test_conditioned_model_accepts_a_derived_expression_over_the_clone(
    intervention, expected
):
    """A graph built on the clone resolves, with or without an intervention."""
    model, _ = _channel_conditioned_model()
    conditioned = ConditionedModel(
        model, _channel_posterior([1.0, 2.0]), intervention=intervention
    )

    graph = conditioned.extract(conditioned.model["resp"].sum("channel"))

    np.testing.assert_allclose(function([], graph)(), expected)


def test_conditioned_model_accepts_a_free_symbolic_input():
    """A free input is a function argument, not a pre-intervention value."""
    model, spend = _channel_conditioned_model()
    conditioned = ConditionedModel(
        model, _channel_posterior([1.0, 2.0]), intervention={"spend": spend}
    )
    price = ptx.xtensor("price", dims=())

    graph = conditioned.extract(conditioned.model["resp"] * price)

    np.testing.assert_allclose(function([price], graph)(2.0), [[20.0, 80.0]])


def test_conditioned_model_without_an_intervention_keeps_the_construction_data():
    """Without an intervention the model is still cloned, so its data stay fixed.

    Regression: ``intervention=None`` used the source model itself, so a
    ``pm.set_data`` on it after construction reached every compiled graph,
    while the same object built with an intervention kept the construction
    data.
    """
    model, _ = _channel_conditioned_model()
    conditioned = ConditionedModel(model, _channel_posterior([1.0, 2.0]))
    fn = function([], conditioned.extract("resp"))

    pm.set_data({"spend": np.array([100.0, 200.0])}, model=model)

    np.testing.assert_allclose(fn(), [[10.0, 40.0]])
    # A graph first extracted after the change reads the clone's data too.
    later = conditioned.extract(conditioned.model["resp"] * 2.0)
    np.testing.assert_allclose(function([], later)(), [[20.0, 80.0]])


@pytest.mark.parametrize(
    "with_intervention", [False, True], ids=["no-intervention", "with-intervention"]
)
@pytest.mark.parametrize(
    "build_response",
    [
        pytest.param(lambda model: model["resp"], id="source-node"),
        pytest.param(lambda model: model["resp"].sum("channel"), id="source-derived"),
    ],
)
def test_conditioned_model_refuses_a_source_model_graph(
    build_response, with_intervention
):
    """A source-model node and a graph derived over it are refused.

    The clone holds its own copy of every input, so a graph over the source
    model would read the source's live data, and no intervention.
    """
    model, spend = _channel_conditioned_model()
    conditioned = ConditionedModel(
        model,
        _channel_posterior([1.0, 2.0]),
        intervention={"spend": spend} if with_intervention else None,
    )

    with pytest.raises(ValueError, match="reads inputs of the source model"):
        conditioned.extract(build_response(model))


def test_conditioned_model_refusal_names_the_node_itself():
    """The refusal prints the offending node, even when it is anonymous."""
    model, spend = _channel_conditioned_model()
    conditioned = ConditionedModel(
        model, _channel_posterior([1.0, 2.0]), intervention={"spend": spend}
    )
    response = model["resp"].sum("channel")
    assert response.name is None

    with pytest.raises(ValueError) as excinfo:
        conditioned.extract(response)

    message = str(excinfo.value)
    assert repr(response) in message
    assert "None" not in message
    # The named inputs are listed; the source model's anonymous RNG is not.
    assert "'spend'" in message
    assert "<RandomGeneratorSharedVariable>" not in message


def test_conditioned_model_refusal_names_the_outside_input():
    """A named input from the source model is named, not just the response."""
    model, spend = _channel_conditioned_model()
    conditioned = ConditionedModel(
        model, _channel_posterior([1.0, 2.0]), intervention={"spend": spend}
    )

    with pytest.raises(ValueError) as excinfo:
        conditioned.extract(conditioned.model["beta"] * model["spend"])

    assert "'spend'" in str(excinfo.value)


def test_conditioned_model_refusal_names_a_source_random_variable_not_its_size():
    """A source random variable is named itself, not the dim length that sizes it."""
    model, spend = _channel_conditioned_model()
    conditioned = ConditionedModel(
        model, _channel_posterior([1.0, 2.0]), intervention={"spend": spend}
    )

    with pytest.raises(ValueError, match="reads inputs of the source model") as excinfo:
        conditioned.extract(conditioned.model["resp"] + model["beta"].sum())

    message = str(excinfo.value)
    assert "'beta'" in message
    assert "'channel'" not in message


def test_conditioned_model_refusal_names_a_source_dim_length_as_a_dim():
    """A dim length read directly is named as a dim, not as a data variable."""
    model, spend = _channel_conditioned_model()
    conditioned = ConditionedModel(
        model, _channel_posterior([1.0, 2.0]), intervention={"spend": spend}
    )
    channels = ptx.as_xtensor(model.dim_lengths["channel"], dims=())

    with pytest.raises(ValueError, match="reads inputs of the source model") as excinfo:
        conditioned.extract(conditioned.model["resp"] * channels)

    assert "the length of dim 'channel'" in str(excinfo.value)


@pytest.mark.parametrize(
    "build_response, named",
    [
        pytest.param(
            lambda other, resp: resp + other["spend"].sum(),
            "'spend'",
            id="other-model-data",
        ),
        pytest.param(
            lambda other, resp: (
                resp * ptx.as_xtensor(shared(np.array(2.0), name="price"), dims=())
            ),
            "'price'",
            id="model-less-shared",
        ),
        pytest.param(
            lambda other, resp: resp + other["beta"].sum(),
            "'beta'",
            id="other-model-random-variable",
        ),
        pytest.param(
            lambda other, resp: resp + pmd.Normal.dist(),
            "<XTensorVariable>",
            id="model-less-random-variable",
        ),
    ],
)
def test_conditioned_model_refuses_an_outside_shared_or_random_variable(
    build_response, named
):
    """A shared or random variable from outside the clone is refused, by name.

    Regression: another model's data was embedded in the graph and read live,
    so a later ``pm.set_data`` on that model changed every compiled function,
    against the promise that the data are fixed at construction.  A random
    variable from outside reached the generic "RVs found ... likely a bug"
    ``RuntimeError``.
    """
    model, spend = _channel_conditioned_model()
    conditioned = ConditionedModel(
        model, _channel_posterior([1.0, 2.0]), intervention={"spend": spend}
    )
    other = clone_model(model)

    with pytest.raises(
        ValueError, match="from outside the model this object holds"
    ) as excinfo:
        conditioned.extract(build_response(other, conditioned.model["resp"]))

    message = str(excinfo.value)
    assert named in message
    # A random variable is named itself, not by the inputs that size it.
    assert "'channel'" not in message


def test_conditioned_model_accepts_the_clones_own_unused_dim_length():
    """A dim no variable is sized by is still an input of the clone.

    Its length is an ancestor of no named variable, so a graph that reads it
    must not be taken for one that reads from outside the clone.
    """
    with pm.Model(coords={"channel": ["a", "b"], "week": [0, 1, 2]}) as model:
        spend = pmd.Data("spend", np.array([10.0, 20.0]), dims="channel")
        beta = pmd.Normal("beta", dims="channel")
        pmd.Deterministic("resp", beta * spend)
    conditioned = ConditionedModel(model, _channel_posterior([1.0, 2.0]))
    weeks = ptx.as_xtensor(conditioned.model.dim_lengths["week"], dims=())

    graph = conditioned.extract(conditioned.model["resp"] * weeks)

    np.testing.assert_allclose(function([], graph)(), [[30.0, 120.0]])


def test_conditioned_model_accepts_a_node_added_to_the_clone_after_construction():
    """A variable added to the clone later, and a graph over it, are the clone's own."""
    model, spend = _channel_conditioned_model()
    conditioned = ConditionedModel(
        model, _channel_posterior([1.0, 2.0]), intervention={"spend": spend}
    )
    with conditioned.model:
        weight = pmd.Data("weight", np.array([2.0, 3.0]), dims="channel")
        late = pmd.Deterministic("late", conditioned.model["resp"] * weight)

    graph = conditioned.extract(late)

    np.testing.assert_allclose(function([], graph)(), [[20.0, 120.0]])


@pytest.mark.parametrize(
    "build_response",
    [
        pytest.param(
            lambda model, resp: (
                resp
                + model["beta"].sum()
                + ptx.as_xtensor(shared(np.array(1.0), name="price"), dims=())
            ),
            id="source-and-outside-shared",
        ),
        pytest.param(
            lambda model, resp: resp + pmd.Normal.dist(mu=model["beta"].sum()),
            id="outside-random-variable-over-the-source",
        ),
    ],
)
def test_conditioned_model_source_refusal_comes_before_the_outside_one(build_response):
    """A graph that reads the source model is refused as such, whatever else it reads.

    The source refusal says what to pass instead, a node of the clone, so it
    comes first, also when the source variable sits behind a random variable
    from outside both models.
    """
    model, spend = _channel_conditioned_model()
    conditioned = ConditionedModel(
        model, _channel_posterior([1.0, 2.0]), intervention={"spend": spend}
    )

    with pytest.raises(ValueError, match="reads inputs of the source model") as excinfo:
        conditioned.extract(build_response(model, conditioned.model["resp"]))

    assert "'beta'" in str(excinfo.value)


def test_describe_outside_inputs_names_an_anonymous_input_by_type():
    """An anonymous input is named by its type, never by a repr with an address."""
    message = ConditionedModel._describe_outside_inputs([shared(np.array(1.0))])

    assert "<TensorSharedVariable>" in message
    assert "0x" not in message


def test_describe_outside_inputs_deduplicates_named_inputs():
    """The same name reached through two roots is listed once."""
    first = shared(np.array(1.0), name="spend")
    second = shared(np.array(2.0), name="spend")

    message = ConditionedModel._describe_outside_inputs([first, second])

    assert message.count("'spend'") == 1


def test_describe_outside_inputs_caps_the_named_inputs():
    """Only five names are shown, followed by the number left out."""
    outside = [shared(np.array(1.0), name=f"outside{index}") for index in range(6)]

    message = ConditionedModel._describe_outside_inputs(outside)

    assert sum(f"'outside{index}'" in message for index in range(6)) == 5
    assert "and 1 more" in message


def test_conditioned_model_rebind_realigns_labels_and_refuses_a_resized_axis():
    model, spend = _channel_conditioned_model()

    # Reordered labels are realigned against the construction layout, not bound
    # by position, so a function compiled before the rebind reads ``a=1, b=2``.
    reordered = ConditionedModel(
        model, _channel_posterior([1.0, 2.0]), intervention={"spend": spend}
    )
    fn = function([], reordered.extract("resp"))
    reordered.set_posterior(_channel_posterior([2.0, 1.0], channel=("b", "a")))
    np.testing.assert_allclose(fn(), [[10.0, 40.0]])

    # A resized axis on a bound variable is refused before anything is written,
    # so the graph, its draws and ``posterior`` are kept.
    conditioned = ConditionedModel(
        model, _channel_posterior([1.0, 2.0]), intervention={"spend": spend}
    )
    graph = conditioned.extract("resp")
    fn = function([], graph)
    resized = xr.Dataset(
        {"beta": (("chain", "draw", "channel"), np.ones((1, 1, 3)))},
        coords={"channel": ["a", "b", "c"]},
    )
    with pytest.raises(ValueError, match=r"'beta' has different channel labels"):
        conditioned.set_posterior(resized)
    assert conditioned.posterior.indexes["channel"].tolist() == ["a", "b"]
    assert conditioned.extract("resp") is graph
    np.testing.assert_allclose(fn(), [[10.0, 40.0]])


def test_conditioned_model_reordered_rebind_realigns_handed_out_and_new_keys():
    """A rebind realigns at the door, so every later read sees model order.

    A posterior stored as ``(b, a)`` used to give ``[[20, 20]]`` from a graph
    extracted after the rebind, and a new ``resp_g`` key gave ``[[40, 60]]``
    instead of ``[[30, 80]]``.
    """
    model, spend = _channel_conditioned_model()
    conditioned = ConditionedModel(
        model,
        _channel_posterior([1.0, 2.0], gamma=[3.0, 4.0]),
        intervention={"spend": spend},
    )
    fn = function([], conditioned.extract("resp"))  # handed out before the rebind

    conditioned.set_posterior(
        _channel_posterior([20.0, 10.0], channel=("b", "a"), gamma=[4.0, 3.0])
    )

    # The stored posterior has the construction layout, not the incoming one.
    assert conditioned.posterior.indexes["channel"].tolist() == ["a", "b"]
    np.testing.assert_allclose(fn(), [[100.0, 400.0]])
    np.testing.assert_allclose(
        function([], conditioned.extract("resp_g"))(), [[30.0, 80.0]]
    )


def test_conditioned_model_rebind_before_any_extract_realigns_and_defers_validation():
    """A rebind before any extraction realigns the labels and validates nothing.

    Nothing is bound yet, so the next ``extract`` builds its graph on the
    model's label order and a later rebind is followed.  A resized axis is
    stored as given at the door and refused when a graph reads it.
    """
    model, spend = _channel_conditioned_model()
    conditioned = ConditionedModel(
        model, _channel_posterior([1.0, 2.0]), intervention={"spend": spend}
    )

    conditioned.set_posterior(_channel_posterior([2.0, 1.0], channel=("b", "a")))

    assert conditioned.posterior.indexes["channel"].tolist() == ["a", "b"]
    fn = function([], conditioned.extract("resp"))
    np.testing.assert_allclose(fn(), [[10.0, 40.0]])
    conditioned.set_posterior(_channel_posterior([3.0, 4.0]))
    np.testing.assert_allclose(fn(), [[30.0, 80.0]])

    # The door stores a resized axis as given; the read against the
    # construction layout is what refuses it.
    resized = ConditionedModel(
        model, _channel_posterior([1.0, 2.0]), intervention={"spend": spend}
    )
    resized.set_posterior(_channel_posterior([1.0, 2.0, 3.0], channel=("a", "b", "c")))
    with pytest.raises(ValueError, match="different channel labels"):
        resized.extract("resp")
    resized.set_posterior(_channel_posterior([1.0, 2.0]))
    np.testing.assert_allclose(function([], resized.extract("resp"))(), [[10.0, 40.0]])


def test_conditioned_model_empty_cache_refuses_a_one_sided_labeling():
    """An unlabeled dim cannot be read by position once the construction draws are labeled.

    The door stores it as given; the read against the construction layout is
    what refuses it.  The error mirrors ``SharedPosterior``'s wording.
    """
    model, spend = _channel_conditioned_model()
    conditioned = ConditionedModel(
        model, _channel_posterior([1.0, 2.0]), intervention={"spend": spend}
    )

    conditioned.set_posterior(_channel_posterior([2.0, 1.0]).drop_vars("channel"))
    with pytest.raises(ValueError, match="labeled on one side only"):
        conditioned.extract("resp")

    # The refusal wrote nothing: a labeled rebind still reads the construction order.
    conditioned.set_posterior(_channel_posterior([2.0, 1.0], channel=("b", "a")))
    fn = function([], conditioned.extract("resp"))
    np.testing.assert_allclose(fn(), [[10.0, 40.0]])

    # Once bound, the same one-sided rebind is refused at the door.
    with pytest.raises(ValueError, match="labeled on one side only"):
        conditioned.set_posterior(_channel_posterior([2.0, 1.0]).drop_vars("channel"))
    np.testing.assert_allclose(fn(), [[10.0, 40.0]])


def test_conditioned_model_empty_cache_accepts_a_subsample_shaped_posterior():
    """``subsample_draws`` drops the ``chain``/``draw`` labels but keeps ``channel``.

    The skipped sample dims must not be mistaken for the one-sided case.
    """
    model, spend = _channel_conditioned_model()
    conditioned = ConditionedModel(
        model,
        _channel_posterior([1.0, 2.0]).assign_coords(chain=[0], draw=[0]),
        intervention={"spend": spend},
    )
    subsampled = (
        _channel_posterior([2.0, 1.0], channel=("b", "a"))
        .assign_coords(chain=[0], draw=[0])
        .drop_vars(["chain", "draw"])
    )
    assert "chain" not in subsampled.indexes and "channel" in subsampled.indexes

    conditioned.set_posterior(subsampled)

    assert conditioned.posterior.indexes["channel"].tolist() == ["a", "b"]
    np.testing.assert_allclose(
        function([], conditioned.extract("resp"))(), [[10.0, 40.0]]
    )


def test_conditioned_model_empty_cache_skips_a_missing_dimension():
    """A thinned posterior that drops a variable and its dimension is accepted.

    A dimension is realigned only when both posteriors label it.
    """
    model, spend = _channel_conditioned_model()
    conditioned = ConditionedModel(
        model, _channel_posterior([1.0, 2.0]), intervention={"spend": spend}
    )
    thinned = xr.Dataset({"gamma": (("chain", "draw"), np.zeros((1, 1)))})

    conditioned.set_posterior(thinned)
    assert "channel" not in conditioned.posterior.sizes

    # The construction layout survives the thinned rebind, so a reordered channel is still realigned.
    conditioned.set_posterior(_channel_posterior([2.0, 1.0], channel=("b", "a")))
    np.testing.assert_allclose(
        function([], conditioned.extract("resp"))(), [[10.0, 40.0]]
    )


@pytest.mark.parametrize(
    ("replaced", "incoming", "message"),
    [
        (["a", "a"], ["a", "b"], "duplicated channel labels"),
        (["a", "b"], ["a", "a"], "different channel labels"),
    ],
)
def test_conditioned_model_refuses_duplicated_labels(replaced, incoming, message):
    """A duplicated label would make ``sel`` drop a column instead of realigning.

    Incoming duplicates surface as a different label set.
    Duplicates on the reference side pass that check, so they are refused separately.
    """
    model, spend = _channel_conditioned_model()
    conditioned = ConditionedModel(
        model,
        _channel_posterior([1.0, 2.0], channel=replaced),
        intervention={"spend": spend},
    )

    conditioned.set_posterior(_channel_posterior([1.0, 2.0], channel=incoming))
    with pytest.raises(ValueError, match=message):
        conditioned.extract("resp")


def _dated_conditioned_model() -> pm.Model:
    """``resp`` reads ``beta``; ``channel_contribution`` is date-indexed but unread."""
    dates = pd.date_range("2025-01-06", periods=3, freq="W-MON")
    spend = np.tile([1.0, 2.0], (3, 1))
    with pm.Model(coords={"date": dates, "channel": ["a", "b"]}) as model:
        spend_var = pmd.Data("spend", spend, dims=("date", "channel"))
        pmd.Normal("beta", dims="channel")
        pmd.Normal("gamma", dims="channel")
        pmd.Deterministic(
            "channel_contribution",
            spend_var * model["gamma"],
            dims=("date", "channel"),
        )
        pmd.Deterministic("resp", (spend_var * model["beta"]).sum())
    return model


def _dated_posterior(dates, beta, channel=("a", "b")):
    size = len(dates)
    return xr.Dataset(
        {
            "beta": (
                ("chain", "draw", "channel"),
                np.asarray(beta, float)[None, None],
            ),
            "gamma": (("chain", "draw", "channel"), np.ones((1, 1, 2))),
            "channel_contribution": (
                ("chain", "draw", "date", "channel"),
                np.ones((1, 1, size, 2)),
            ),
        },
        coords={"channel": list(channel), "date": list(dates)},
    )


def test_conditioned_model_ignores_unread_date_labels_on_rebind():
    """A refit date window on an unread deterministic is accepted.

    ``resp`` reads only ``beta``, so the newer ``date`` labels of the unread
    ``channel_contribution`` must not refuse the rebind.  A reordered
    ``channel`` is still realigned, so the new draws are not permuted.
    """
    model = _dated_conditioned_model()
    construction = pd.date_range("2025-01-06", periods=3, freq="W-MON")
    refit = pd.date_range("2025-02-03", periods=4, freq="W-MON")
    conditioned = ConditionedModel(model, _dated_posterior(construction, [1.0, 2.0]))
    fn = function([], conditioned.extract("resp"))

    conditioned.set_posterior(_dated_posterior(refit, [20.0, 10.0], channel=("b", "a")))

    assert conditioned.posterior.indexes["date"].tolist() == list(refit)
    np.testing.assert_allclose(fn(), [150.0])


def test_conditioned_model_first_read_after_reordered_rebind():
    """A variable first read after a reordered rebind is read in the model's order.

    The construction posterior carries no ``channel`` labels (a thinned
    posterior), so only the model's coords can order the rebind.  ``gamma`` is
    absent from it and arrives stored as ``(b, a)``; it must be read as the
    model's ``(a, b)``: ``a=40`` and ``b=30``.
    """
    model, spend = _channel_conditioned_model()
    conditioned = ConditionedModel(
        model,
        _channel_posterior([1.0, 2.0]).drop_vars("channel"),
        intervention={"spend": spend},
    )

    conditioned.set_posterior(
        _channel_posterior([10.0, 20.0], channel=("b", "a"), gamma=[30.0, 40.0])
    )

    np.testing.assert_allclose(
        function([], conditioned.extract("resp_g"))(), [[400.0, 600.0]]
    )
    assert conditioned.posterior.indexes["channel"].tolist() == ["a", "b"]


def test_conditioned_model_construction_posterior_in_non_model_order():
    """Draws stored in another label order are read in the model's order.

    The model's coords are ``(a, b)``; the construction posterior stores
    ``(b, a)`` with ``beta=[2, 1]`` and ``gamma=[4, 3]``.  The stored dataset is
    reordered on the way in, so ``posterior`` is in the model's order and the
    graphs read ``a=1`` and ``b=2``.
    """
    model, spend = _channel_conditioned_model()
    conditioned = ConditionedModel(
        model,
        _channel_posterior([2.0, 1.0], channel=("b", "a"), gamma=[4.0, 3.0]),
        intervention={"spend": spend},
    )

    assert conditioned.posterior.indexes["channel"].tolist() == ["a", "b"]
    np.testing.assert_allclose(
        function([], conditioned.extract("resp"))(), [[10.0, 40.0]]
    )
    np.testing.assert_allclose(
        function([], conditioned.extract("resp_g"))(), [[30.0, 80.0]]
    )


def test_conditioned_model_refuses_a_per_variable_dim_change_at_the_read():
    """``beta`` gaining a dimension is refused when a graph reads it, not at the door."""
    model, spend = _channel_conditioned_model()
    conditioned = ConditionedModel(
        model, _channel_posterior([1.0, 2.0]), intervention={"spend": spend}
    )
    gained = xr.Dataset(
        {"beta": (("chain", "draw", "channel", "geo"), np.ones((1, 1, 2, 3)))},
        coords={"channel": ["a", "b"], "geo": ["x", "y", "z"]},
    )

    conditioned.set_posterior(gained)

    with pytest.raises(ValueError, match=r"beta.*has dims"):
        conditioned.extract("resp")


def test_conditioned_model_takes_a_variable_missing_at_construction_as_given():
    """A variable thinned out of the construction posterior has no known layout."""
    model, spend = _channel_conditioned_model()
    conditioned = ConditionedModel(
        model, _channel_posterior([1.0, 2.0]), intervention={"spend": spend}
    )

    conditioned.set_posterior(_channel_posterior([1.0, 2.0], gamma=[3.0, 4.0]))

    fn = function([], conditioned.extract("resp_g"))
    np.testing.assert_allclose(fn(), [[30.0, 80.0]])
    # Once read, it is bound like any other variable and follows a rebind.
    conditioned.set_posterior(_channel_posterior([1.0, 2.0], gamma=[5.0, 6.0]))
    np.testing.assert_allclose(fn(), [[50.0, 120.0]])
