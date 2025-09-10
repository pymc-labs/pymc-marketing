#   Copyright 2022 - 2025 The PyMC Labs Developers
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

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from pymc_extras.prior import Prior
from pytensor import function

from pymc_marketing.mmm import (
    DelayedAdstock,
    GeometricAdstock,
    LogisticSaturation,
    MichaelisMentenSaturation,
)
from pymc_marketing.mmm.multidimensional import (
    MMM,
    MultiDimensionalBudgetOptimizerWrapper,
)
from pymc_marketing.pytensor_utils import (
    MaskedDist,
    ModelSamplerEstimator,
    merge_models,
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
                "C1": np.random.randint(10, 50, n_obs_per_country),
                "C2": np.random.randint(5, 40, n_obs_per_country),
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

    # Fit with minimal sampling for speed
    mmm.fit(X, y, draws=100, chains=2, random_seed=42, tune=100)

    return mmm


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

    # Wrap the model in MultiDimensionalBudgetOptimizerWrapper
    optimizable_model = MultiDimensionalBudgetOptimizerWrapper(
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
    from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer

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


def test_MaskedDist_masked_prior_basic():
    # dims over 2x3 grid
    coords = {
        "country": ["A", "B"],
        "channel": ["C1", "C2", "C3"],
    }
    prior = Prior("Normal", mu=0, sigma=1, dims=("country", "channel"))
    # mask: activate A:C1 and B:C3 only
    mask = np.array([[1, 0, 0], [0, 0, 1]], dtype=bool)

    masked = MaskedDist(prior, mask=mask)

    with pm.Model(coords=coords):
        out = masked.create_variable("vmax_full")

        # Active RV exists and has length 2
        assert "vmax_full_dist" in pm.modelcontext(None).named_vars, (
            "Active RV 'vmax_full_dist' not found in model"
        )
        active_rv = pm.modelcontext(None).named_vars["vmax_full_dist"]
        assert active_rv.ndim == 1, f"Expected 1D active RV, got {active_rv.ndim}D"
        assert active_rv.shape[0].eval() == 2, (
            f"Expected 2 active elements, got {active_rv.shape[0].eval()}"
        )

        # The filled tensor should have zeros where mask is False
        f = function([], out)
        val = f()
        assert val.shape == (2, 3), f"Expected shape (2, 3), got {val.shape}"
        # Zero positions
        assert np.all(val[0, 1:] == 0), (
            "Expected zeros at masked positions [0, 1:] but found non-zero values"
        )
        assert np.all(val[1, :2] == 0), (
            "Expected zeros at masked positions [1, :2] but found non-zero values"
        )
        # Non-zero positions correspond to active entries
        # We can't know exact numeric values (random), but they should not be all zeros
        assert not np.all(val[0, 0] == 0), (
            "Expected non-zero value at active position [0, 0] but got zero"
        )
        assert not np.all(val[1, 2] == 0), (
            "Expected non-zero value at active position [1, 2] but got zero"
        )


@pytest.mark.parametrize(
    "component_factory,expected_vars",
    [
        (
            # Current implementation: LogisticSaturation with lam and beta masked
            lambda mask: LogisticSaturation(
                priors={
                    "lam": MaskedDist(
                        Prior("HalfNormal", sigma=1.0, dims=("country", "region")),
                        mask=mask,
                    ),
                    "beta": MaskedDist(
                        Prior("HalfNormal", sigma=1.0, dims=("country", "region")),
                        mask=mask,
                    ),
                }
            ),
            ["saturation_lam", "saturation_beta"],
        ),
        (
            # Additional saturation: Michaelis-Menten with alpha and lam masked
            lambda mask: MichaelisMentenSaturation(
                priors={
                    "alpha": MaskedDist(
                        Prior("HalfNormal", sigma=1.0, dims=("country", "region")),
                        mask=mask,
                    ),
                    "lam": MaskedDist(
                        Prior("HalfNormal", sigma=1.0, dims=("country", "region")),
                        mask=mask,
                    ),
                }
            ),
            ["saturation_alpha", "saturation_lam"],
        ),
        (
            # Adstock: Geometric with alpha masked
            lambda mask: GeometricAdstock(
                l_max=3,
                priors={
                    "alpha": MaskedDist(
                        Prior("Beta", alpha=1, beta=3, dims=("country", "region")),
                        mask=mask,
                    )
                },
            ),
            ["adstock_alpha"],
        ),
        (
            # Adstock: Delayed with alpha and theta masked
            lambda mask: DelayedAdstock(
                l_max=3,
                priors={
                    "alpha": MaskedDist(
                        Prior("Beta", alpha=1, beta=3, dims=("country", "region")),
                        mask=mask,
                    ),
                    "theta": MaskedDist(
                        Prior("HalfNormal", sigma=1.0, dims=("country", "region")),
                        mask=mask,
                    ),
                },
            ),
            ["adstock_alpha", "adstock_theta"],
        ),
    ],
)
def test_MaskedDist_inside_component_without_explicit_coords(
    component_factory, expected_vars
):
    # dims over 2x2 grid
    coords = {
        "country": ["A", "B"],
        "region": ["R1", "R2"],
    }

    # 2x2 mask activates positions (0,0) and (1,1)
    mask = np.array([[True, False], [False, True]])
    comp = component_factory(mask)

    prior_ds = comp.sample_prior(coords=coords, random_seed=1)
    for var in expected_vars:
        assert var in prior_ds, (
            f"Expected variable '{var}' not found in prior dataset. Available variables: {list(prior_ds.keys())}"
        )


def test_ModelSamplerEstimator_dataframe():
    # Build a minimal PyMC model (structure won't be used due to monkeypatching)
    with pm.Model() as model:
        pm.Normal("x", mu=0.0, sigma=1.0, observed=np.array([0.0]))

    estimator = ModelSamplerEstimator(
        tune=100, draws=200, chains=1, sequential_chains=1, seed=123
    )

    df = estimator.run(model)

    assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df)}"
    assert len(df) == 1, f"Expected DataFrame with 1 row, got {len(df)} rows"

    required_cols = {
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
    assert required_cols.issubset(set(df.columns)), (
        f"Missing required columns: {required_cols - set(df.columns)}"
    )

    # Check meta values
    assert df.loc[0, "sequential_chains"] == 1, (
        f"Expected sequential_chains=1, got {df.loc[0, 'sequential_chains']}"
    )
    assert df.loc[0, "tune"] == 100, f"Expected tune=100, got {df.loc[0, 'tune']}"
    assert df.loc[0, "draws"] == 200, f"Expected draws=200, got {df.loc[0, 'draws']}"
    assert df.loc[0, "chains"] == 1, f"Expected chains=1, got {df.loc[0, 'chains']}"
    assert df.loc[0, "seed"] == 123, f"Expected seed=123, got {df.loc[0, 'seed']}"

    # Model name and timestamp sanity
    assert isinstance(df.loc[0, "model_name"], str), (
        f"Expected model_name to be str, got {type(df.loc[0, 'model_name'])}"
    )
    assert isinstance(df.loc[0, "timestamp"], pd.Timestamp), (
        f"Expected timestamp to be pd.Timestamp, got {type(df.loc[0, 'timestamp'])}"
    )


def test_MaskedDist_with_likelihood_masks_geo_dates():
    # Coords: 4 dates, 2 geos
    coords = {
        "date": np.arange(4),
        "geo": ["A", "B"],
    }

    # Mask: only sample contributions for dates [0, 2] in geo A; all others not sampled
    mask = np.array(
        [
            [True, False],  # date 0: A True, B False
            [False, False],  # date 1: A False, B False
            [True, False],  # date 2: A True, B False
            [False, False],  # date 3: A False, B False
        ]
    )

    # Prior over (date, geo) grid for a contribution to the mean of y
    mu_prior = Prior("Normal", mu=0.0, sigma=1.0, dims=("date", "geo"))
    masked_mu = MaskedDist(mu_prior, mask=mask)

    observed = np.zeros((len(coords["date"]), len(coords["geo"])))

    likelihood_prior = Prior(
        "Normal",
        sigma=Prior("HalfNormal", sigma=1.0),
        dims=("date", "geo"),
    )
    masked_lik = MaskedDist(likelihood_prior, mask=mask)

    with pm.Model(coords=coords):
        # Masked deterministic mean over full grid (zeros where mask is False)
        mu_full = masked_mu.create_variable("mu_full")

        # Build likelihood as a Prior and mask it, so both mu and observed are masked
        masked_lik.create_likelihood_variable(
            name="y",
            mu=mu_full,
            observed=observed,
        )

        # Active RV for mu exists and has expected size (number of True in mask)
        active = pm.modelcontext(None).named_vars["mu_full_dist"]
        assert active.ndim == 1, (
            f"Expected active RV to be 1-dimensional, got {active.ndim} dimensions"
        )
        assert active.shape[0].eval() == int(mask.sum()), (
            f"Expected active RV shape to be {int(mask.sum())}, got {active.shape[0].eval()}"
        )

        # Deterministic has zeros where mask is False
        f = function([], mu_full)
        mu_val = f()
        assert mu_val.shape == (len(coords["date"]), len(coords["geo"])), (
            f"Expected mu_val shape to be {(len(coords['date']), len(coords['geo']))}, "
            f"got {mu_val.shape}"
        )
        assert np.all(mu_val[~mask] == 0), (
            "Expected mu_val to be zero where mask is False, but found non-zero values"
        )

        # Observed likelihood should exist and be defined over active dim only
        y_rv = pm.modelcontext(None).named_vars["y"]
        y_dims = pm.modelcontext(None).named_vars_to_dims[y_rv.name]
        # single active dimension from MaskedDist
        assert len(y_dims) == 1, (
            f"Expected y_rv to have 1 dimension (active dim only), got {len(y_dims)} dimensions"
        )


def _build_toy_model(X: np.ndarray, y: np.ndarray):
    coords = {
        "date": [f"date_{i}" for i in range(X.shape[0])],
        "feature": [f"feature_{i}" for i in range(X.shape[1])],
    }
    with pm.Model(coords=coords) as m:
        pm.Data("shared_input", X, dims=("date", "feature"))
        beta = pm.Normal("beta", 0.0, 1.0, dims=("feature",))
        # Sum over the feature axis (axis=1) to get a per-date mean
        mu = (m["shared_input"] * beta).sum(axis=-1)
        pm.Deterministic("mu", mu, dims=("date",))
        sigma = pm.HalfNormal("sigma", 1.0)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y, dims=("date",))
    return m


def test_merge_models_with_shared_input_container():
    # Random small dataset
    rng = np.random.default_rng(0)
    n, p = 15, 3
    X = rng.normal(size=(n, p))
    y1 = rng.normal(size=n)
    y2 = rng.normal(size=n)
    y3 = rng.normal(size=n)

    # Three independent models sharing the same input container name 'shared_input'
    m1 = _build_toy_model(X, y1)
    m2 = _build_toy_model(X, y2)
    m3 = _build_toy_model(X, y3)

    # Merge on the shared input container so it is not prefixed
    merged = merge_models(
        [m1, m2, m3], prefixes=["m1", "m2", "m3"], merge_on="shared_input"
    )

    # The shared variable should exist unprefixed
    assert "shared_input" in merged.named_vars, (
        "Expected unprefixed shared_input in merged model"
    )

    # Each model's deterministics and RVs should be present with prefixes
    for prefix in ("m1", "m2", "m3"):
        assert f"{prefix}_beta" in merged.named_vars, (
            f"Missing {prefix}_beta in merged model"
        )
        assert f"{prefix}_mu" in merged.named_vars, (
            f"Missing {prefix}_mu in merged model"
        )
        assert f"{prefix}_sigma" in merged.named_vars, (
            f"Missing {prefix}_sigma in merged model"
        )
        assert f"{prefix}_y" in merged.named_vars, f"Missing {prefix}_y in merged model"

    # Dimensions for the shared input must remain unprefixed and present in coords
    assert "date" in merged.coords and "feature" in merged.coords, (
        "Shared dims 'date' and 'feature' should be present in merged coords"
    )
    # And the prefixed models should have their own dim names for any of their internal dims (none here),
    # so we simply confirm the merged graph is compilable
    with merged:
        f = function([], merged["m1_mu"])  # smoke test compile
        out = f()
        assert out.shape == (n,), "Merged model produced unexpected shape"


def test_simple_masked_linear_model_with_oos_extension(mock_pymc_sample):
    rng = np.random.default_rng(0)

    # Dimensions
    T = 12
    geos = ["A", "B"]
    channels = ["C1", "C2"]

    # Data: X over (date, geo, channel)
    X = rng.normal(loc=10.0, scale=3.0, size=(T, len(geos), len(channels)))

    # make all dates for B geo and channel C2 zero
    X[:, 0, 1] = 0.0

    # make the first 3 days for geo A and channel C1 zero
    X[:3, 0, 0] = 0.0

    # Static channel betas
    beta_vals = np.array([0.6, -0.25], dtype=float)

    # True mean and observations
    mu_true = (X * beta_vals).sum(axis=-1)
    y = rng.normal(loc=mu_true, scale=0.2)

    coords = {
        "date": np.arange(T),
        "geo": geos,
        "channel": channels,
    }

    beta = Prior("Normal", mu=0.0, sigma=1.0, dims=("geo", "channel"))
    likelihood_prior = Prior(
        "Normal", sigma=Prior("HalfNormal", sigma=1.0), dims=("date", "geo")
    )

    # create a mask with size len(geos) x len(channels) where we exclude all dates for B geo and channel C2
    mask = np.ones((len(geos), len(channels)), dtype=bool)
    mask[1, 1] = False  # exclude all dates for B geo and channel C2

    masked_beta = MaskedDist(beta, mask=mask)

    # create a mask with size T x len(geos) where we exclude the first 3 days for geo A
    mask_y = np.ones((T, len(geos)), dtype=bool)
    mask_y[:3, 0] = False

    masked_likelihood = MaskedDist(likelihood_prior, mask=mask_y)

    with pm.Model(coords=coords) as m:
        pm.Data("X", X, dims=("date", "geo", "channel"))
        beta = masked_beta.create_variable("beta")
        mu = (m["X"] * beta).sum(axis=-1)
        masked_likelihood.create_likelihood_variable("y", mu=mu, observed=y)

        idata = pm.sample(
            draws=150, tune=150, chains=2, cores=1, random_seed=22, progressbar=False
        )
        # sample posterior pred
        idata.extend(
            pm.sample_posterior_predictive(idata, var_names=["y"], progressbar=False)
        )

    # New set of T values (take the last value of np.arange(T) and add 5)
    # Follow PyMC out-of-sample logic: update pm.Data and coords, then draw PPC
    T_new = T + 5
    new_dates = np.arange(T_new)
    X_future = rng.normal(loc=3.0, scale=0.5, size=(5, len(geos), len(channels)))
    X_extended = np.concatenate([X, X_future], axis=0)

    with m:
        pm.set_data({"X": X_extended}, coords={"date": new_dates})
        # Draw new posterior predictive with updated inputs
        pm.sample_posterior_predictive(idata, var_names=["y"], progressbar=False)


def test_test_only_oos_with_masked_likelihood_raises(mock_pymc_sample):
    rng = np.random.default_rng(1)

    # Train dimensions
    T = 12
    geos = ["A", "B"]
    channels = ["C1", "C2"]

    # Training data
    X = rng.normal(loc=10.0, scale=3.0, size=(T, len(geos), len(channels)))
    X[:, 0, 1] = 0.0
    X[:3, 0, 0] = 0.0

    beta_vals = np.array([0.6, -0.25], dtype=float)
    mu_true = (X * beta_vals).sum(axis=-1)
    y = rng.normal(loc=mu_true, scale=0.2)

    coords = {
        "date": np.arange(T),
        "geo": geos,
        "channel": channels,
    }

    # Priors
    beta_prior = Prior("Normal", mu=0.0, sigma=1.0, dims=("geo", "channel"))
    # Unmasked likelihood prior to allow flexible OOS dims
    likelihood_prior = Prior(
        "Normal",
        sigma=Prior("HalfNormal", sigma=1.0),
        dims=("date", "geo"),
    )

    # Mask for beta only (e.g., disable B:C2)
    mask = np.ones((len(geos), len(channels)), dtype=bool)
    mask[1, 1] = False
    masked_beta = MaskedDist(beta_prior, mask=mask)

    with pm.Model(coords=coords) as m:
        pm.Data("X", X, dims=("date", "geo", "channel"))
        beta = masked_beta.create_variable("beta")
        mu = (m["X"] * beta).sum(axis=-1)
        likelihood_prior.create_likelihood_variable("y", mu=mu, observed=y)

        idata = pm.sample(
            draws=150, tune=150, chains=2, cores=1, random_seed=22, progressbar=False
        )

    # Predict only over test set (last 5 periods) by updating coords and X only for test
    T_test = 5
    X_test = rng.normal(loc=3.0, scale=0.5, size=(T_test, len(geos), len(channels)))
    new_dates = np.arange(T_test)

    with m:
        pm.set_data({"X": X_test}, coords={"date": new_dates})
        # Predicting only over the test set is incompatible with masked likelihood over training dims.
        # Expect a coordinate/shape error when attempting PPC with shorter coords.
        with pytest.raises(Exception, match=r"conflicting sizes.*dimension 'date'"):
            pm.sample_posterior_predictive(idata, var_names=["y"], progressbar=False)


def test_test_only_oos_without_masked_likelihood_succeeds():
    rng = np.random.default_rng(2)

    # Train dims
    T = 12
    geos = ["A", "B"]
    channels = ["C1", "C2"]

    # Training data
    X = rng.normal(loc=10.0, scale=3.0, size=(T, len(geos), len(channels)))
    X[:, 0, 1] = 0.0
    X[:3, 0, 0] = 0.0

    beta_vals = np.array([0.6, -0.25], dtype=float)
    mu_true = (X * beta_vals).sum(axis=-1)
    y = rng.normal(loc=mu_true, scale=0.2)

    coords = {
        "date": np.arange(T),
        "geo": geos,
        "channel": channels,
    }

    # Only mask priors for betas; likelihood is NOT masked
    beta_prior = Prior("Normal", mu=0.0, sigma=1.0, dims=("geo", "channel"))
    mask = np.ones((len(geos), len(channels)), dtype=bool)
    mask[1, 1] = False
    masked_beta = MaskedDist(beta_prior, mask=mask)

    likelihood_prior = Prior(
        "Normal",
        sigma=Prior("HalfNormal", sigma=1.0),
        dims=("date", "geo"),
    )

    with pm.Model(coords=coords) as m:
        pm.Data("X", X, dims=("date", "geo", "channel"))
        beta = masked_beta.create_variable("beta")
        mu = (m["X"] * beta).sum(axis=-1)
        likelihood_prior.create_likelihood_variable("y", mu=mu, observed=y)
        idata = pm.sample(
            draws=150, tune=150, chains=2, cores=1, random_seed=22, progressbar=False
        )

    # Test-only OOS
    T_test = 5
    X_test = rng.normal(loc=3.0, scale=0.5, size=(T_test, len(geos), len(channels)))
    new_dates = np.arange(T_test)

    with m:
        pm.set_data({"X": X_test}, coords={"date": new_dates})
        # Use return_inferencedata=False to avoid packaging observed_data (which has training dims)
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["y"],
            progressbar=False,
            return_inferencedata=False,
        )

    assert "y" in ppc, "y not present in PPC dict"
    # Shape: (chains, draws, T_test, N_geo) in PyMC 5
    y_pp = ppc["y"]
    assert y_pp.shape[-2:] == (T_test, len(geos)), (
        "PPC shape mismatch for test-only OOS"
    )
