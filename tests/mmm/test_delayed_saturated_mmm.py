import os
from typing import Dict, List, Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from matplotlib import pyplot as plt

from pymc_marketing.mmm.delayed_saturated_mmm import (
    BaseDelayedSaturatedMMM,
    DelayedSaturatedMMM,
)

seed: int = sum(map(ord, "pymc_marketing"))
rng: np.random.Generator = np.random.default_rng(seed=seed)


@pytest.fixture(scope="module")
def generate_data():
    def _generate_data(date_data: pd.DatetimeIndex) -> pd.DataFrame:
        n: int = date_data.size

        return pd.DataFrame(
            data={
                "date": date_data,
                "channel_1": rng.integers(low=0, high=400, size=n),
                "channel_2": rng.integers(low=0, high=50, size=n),
                "control_1": rng.gamma(shape=1000, scale=500, size=n),
                "control_2": rng.gamma(shape=100, scale=5, size=n),
                "other_column_1": rng.integers(low=0, high=100, size=n),
                "other_column_2": rng.normal(loc=0, scale=1, size=n),
            }
        )

    return _generate_data


@pytest.fixture(scope="module")
def toy_X(generate_data) -> pd.DataFrame:
    date_data: pd.DatetimeIndex = pd.date_range(
        start="2019-06-01", end="2021-12-31", freq="W-MON"
    )

    return generate_data(date_data)


@pytest.fixture(scope="class")
def model_config_requiring_serialization() -> Dict:
    model_config = {
        "intercept": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 2}},
        "beta_channel": {
            "dist": "HalfNormal",
            "kwargs": {"sigma": np.array([0.4533017, 0.25488063])},
        },
        "alpha": {
            "dist": "Beta",
            "kwargs": {
                "alpha": np.array([3, 3]),
                "beta": np.array([3.55001301, 2.87092431]),
            },
        },
        "lam": {
            "dist": "Gamma",
            "kwargs": {
                "alpha": np.array([3, 3]),
                "beta": np.array([4.12231653, 5.02896872]),
            },
        },
        "likelihood": {
            "dist": "Normal",
            "kwargs": {
                "sigma": {"dist": "HalfNormal", "kwargs": {"sigma": 2}},
            },
        },
        "gamma_control": {"dist": "HalfNormal", "kwargs": {"mu": 0, "sigma": 2}},
        "gamma_fourier": {"dist": "HalfNormal", "kwargs": {"mu": 0, "b": 1}},
    }
    return model_config


@pytest.fixture(scope="module")
def toy_y(toy_X: pd.DataFrame) -> pd.Series:
    return pd.Series(data=rng.integers(low=0, high=100, size=toy_X.shape[0]))


@pytest.fixture(scope="module")
def mmm() -> DelayedSaturatedMMM:
    return DelayedSaturatedMMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        adstock_max_lag=4,
        control_columns=["control_1", "control_2"],
    )


@pytest.fixture(scope="module")
def mmm_with_fourier_features() -> DelayedSaturatedMMM:
    return DelayedSaturatedMMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        adstock_max_lag=4,
        control_columns=["control_1", "control_2"],
        yearly_seasonality=2,
    )


@pytest.fixture(scope="module")
def mmm_fitted(
    mmm: DelayedSaturatedMMM, toy_X: pd.DataFrame, toy_y: pd.Series
) -> DelayedSaturatedMMM:
    mmm.fit(X=toy_X, y=toy_y, target_accept=0.8, draws=3, chains=2, random_seed=rng)
    return mmm


@pytest.fixture(scope="module")
def mmm_fitted_with_fourier_features(
    mmm_with_fourier_features: DelayedSaturatedMMM,
    toy_X: pd.DataFrame,
    toy_y: pd.Series,
) -> DelayedSaturatedMMM:
    mmm_with_fourier_features.fit(
        X=toy_X, y=toy_y, target_accept=0.8, draws=3, chains=2, random_seed=rng
    )
    return mmm_with_fourier_features


class TestDelayedSaturatedMMM:
    def test_save_load_with_not_serializable_model_config(
        self, model_config_requiring_serialization, toy_X, toy_y
    ):
        def deep_equal(dict1, dict2):
            for key, value in dict1.items():
                if key not in dict2:
                    return False
                if isinstance(value, dict):
                    if not deep_equal(value, dict2[key]):
                        return False
                elif isinstance(value, np.ndarray):
                    if not np.array_equal(value, dict2[key]):
                        return False
                else:
                    if value != dict2[key]:
                        return False
            return True

        model = DelayedSaturatedMMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock_max_lag=4,
            model_config=model_config_requiring_serialization,
        )
        model.fit(
            toy_X, toy_y, target_accept=0.81, draws=100, chains=2, random_seed=rng
        )
        model.save("test_save_load")
        model2 = DelayedSaturatedMMM.load("test_save_load")
        assert model.date_column == model2.date_column
        assert model.control_columns == model2.control_columns
        assert model.channel_columns == model2.channel_columns
        assert model.adstock_max_lag == model2.adstock_max_lag
        assert model.validate_data == model2.validate_data
        assert model.yearly_seasonality == model2.yearly_seasonality
        assert deep_equal(model.model_config, model2.model_config)

        assert model.sampler_config == model2.sampler_config
        os.remove("test_save_load")

    @pytest.mark.parametrize(
        argnames="adstock_max_lag",
        argvalues=[1, 4],
        ids=["adstock_max_lag=1", "adstock_max_lag=4"],
    )
    @pytest.mark.parametrize(
        argnames="control_columns",
        argvalues=[None, ["control_1"], ["control_1", "control_2"]],
        ids=["no_control", "one_control", "two_controls"],
    )
    @pytest.mark.parametrize(
        argnames="channel_columns",
        argvalues=[
            (["channel_1"]),
            (["channel_1", "channel_2"]),
        ],
        ids=[
            "single_channel",
            "multiple_channel",
        ],
    )
    @pytest.mark.parametrize(
        argnames="yearly_seasonality",
        argvalues=[None, 2],
        ids=["no_yearly_seasonality", "yearly_seasonality"],
    )
    def test_init(
        self,
        toy_X: pd.DataFrame,
        toy_y: pd.Series,
        yearly_seasonality: Optional[int],
        channel_columns: List[str],
        control_columns: List[str],
        adstock_max_lag: int,
    ) -> None:
        mmm = BaseDelayedSaturatedMMM(
            date_column="date",
            channel_columns=channel_columns,
            control_columns=control_columns,
            adstock_max_lag=adstock_max_lag,
            yearly_seasonality=yearly_seasonality,
        )
        mmm.build_model(X=toy_X, y=toy_y)
        n_channel: int = len(mmm.channel_columns)
        samples: int = 3
        with mmm.model:
            prior_predictive: az.InferenceData = pm.sample_prior_predictive(
                samples=samples, random_seed=rng
            )

        assert (
            az.extract(
                prior_predictive, group="prior", var_names=["intercept"], combined=True
            )
            .to_numpy()
            .size
            == samples
        )
        assert az.extract(
            data=prior_predictive,
            group="prior",
            var_names=["beta_channel"],
            combined=True,
        ).to_numpy().shape == (
            n_channel,
            samples,
        )
        assert az.extract(
            data=prior_predictive, group="prior", var_names=["alpha"], combined=True
        ).to_numpy().shape == (
            n_channel,
            samples,
        )
        assert az.extract(
            data=prior_predictive, group="prior", var_names=["lam"], combined=True
        ).to_numpy().shape == (
            n_channel,
            samples,
        )

        if control_columns is not None:
            n_control = len(control_columns)
            assert az.extract(
                data=prior_predictive,
                group="prior",
                var_names=["gamma_control"],
                combined=True,
            ).to_numpy().shape == (
                n_control,
                samples,
            )
        if yearly_seasonality is not None:
            assert az.extract(
                data=prior_predictive,
                group="prior",
                var_names=["gamma_fourier"],
                combined=True,
            ).to_numpy().shape == (
                2 * yearly_seasonality,
                samples,
            )

    def test_fit(self, toy_X: pd.DataFrame, toy_y: pd.Series) -> None:
        draws: int = 100
        chains: int = 2

        mmm = BaseDelayedSaturatedMMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            control_columns=["control_1", "control_2"],
            adstock_max_lag=2,
            yearly_seasonality=2,
        )
        assert mmm.version == "0.0.2"
        assert mmm._model_type == "DelayedSaturatedMMM"
        assert mmm.model_config is not None
        n_channel: int = len(mmm.channel_columns)
        n_control: int = len(mmm.control_columns)
        fourier_terms: int = 2 * mmm.yearly_seasonality
        mmm.fit(
            X=toy_X,
            y=toy_y,
            target_accept=0.81,
            draws=draws,
            chains=chains,
            random_seed=rng,
        )
        idata: az.InferenceData = mmm.fit_result
        assert (
            az.extract(data=idata, var_names=["intercept"], combined=True)
            .to_numpy()
            .size
            == draws * chains
        )
        assert az.extract(
            data=idata, var_names=["beta_channel"], combined=True
        ).to_numpy().shape == (n_channel, draws * chains)
        assert az.extract(
            data=idata, var_names=["alpha"], combined=True
        ).to_numpy().shape == (n_channel, draws * chains)
        assert az.extract(
            data=idata, var_names=["lam"], combined=True
        ).to_numpy().shape == (n_channel, draws * chains)
        assert az.extract(
            data=idata, var_names=["gamma_control"], combined=True
        ).to_numpy().shape == (
            n_channel,
            draws * chains,
        )

        mean_model_contributions_ts = mmm.compute_mean_contributions_over_time(
            original_scale=True
        )
        assert mean_model_contributions_ts.shape == (
            toy_X.shape[0],
            n_channel + n_control + fourier_terms + 1,
        )
        assert mean_model_contributions_ts.columns.tolist() == [
            "channel_1",
            "channel_2",
            "control_1",
            "control_2",
            "sin_order_1",
            "cos_order_1",
            "sin_order_2",
            "cos_order_2",
            "intercept",
        ]

    @pytest.mark.parametrize(
        argnames="yearly_seasonality",
        argvalues=[None, 1, 2],
        ids=["no_yearly_seasonality", "yearly_seasonality=1", "yearly_seasonality=2"],
    )
    def test_get_fourier_models_data(
        self, toy_X: pd.DataFrame, toy_y: pd.Series, yearly_seasonality: Optional[int]
    ) -> None:
        mmm = BaseDelayedSaturatedMMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            control_columns=["control_1", "control_2"],
            adstock_max_lag=2,
            yearly_seasonality=yearly_seasonality,
        )
        if yearly_seasonality is None:
            with pytest.raises(ValueError):
                mmm._get_fourier_models_data(toy_X)

        else:
            fourier_modes_data: Optional[pd.DataFrame] = mmm._get_fourier_models_data(
                toy_X
            )
            assert fourier_modes_data.shape == (
                toy_X.shape[0],
                2 * yearly_seasonality,
            )
            assert fourier_modes_data.max().max() <= 1
            assert fourier_modes_data.min().min() >= -1

    def test_channel_contributions_forward_pass_recovers_contribution(
        self, mmm_fitted: DelayedSaturatedMMM
    ) -> None:
        channel_data = mmm_fitted.preprocessed_data["X"][
            mmm_fitted.channel_columns
        ].to_numpy()
        channel_contributions_forward_pass = (
            mmm_fitted.channel_contributions_forward_pass(channel_data=channel_data)
        )
        channel_contributions_forward_pass_mean = (
            channel_contributions_forward_pass.mean(axis=(0, 1))
        )
        channel_contributions_mean = mmm_fitted.fit_result[
            "channel_contributions"
        ].mean(dim=["draw", "chain"])
        assert (
            channel_contributions_forward_pass_mean.shape
            == channel_contributions_mean.shape
        )
        # The forward pass results should be in the original scale of the target variable.
        # The trace fits the model with scaled data, so when scaling back, they should match.
        # Since we are using a `MaxAbsScaler`, the scaling factor is the maximum absolute, i.e y.max()
        np.testing.assert_array_almost_equal(
            x=channel_contributions_forward_pass_mean / channel_contributions_mean,
            y=mmm_fitted.y.max(),
        )

    def test_channel_contributions_forward_pass_is_consistent(
        self, mmm_fitted: DelayedSaturatedMMM
    ) -> None:
        channel_data = mmm_fitted.preprocessed_data["X"][
            mmm_fitted.channel_columns
        ].to_numpy()
        channel_contributions_forward_pass = (
            mmm_fitted.channel_contributions_forward_pass(channel_data=channel_data)
        )
        # use a grid [0, 1, 2] which corresponds to
        # - no-spend -> forward pass should be zero
        # - spend input for the model -> should match the forward pass
        # - doubling the spend -> should be higher than the forward pass with the original spend
        channel_contributions_forward_pass_grid = (
            mmm_fitted.get_channel_contributions_forward_pass_grid(
                start=0, stop=2, num=3
            )
        )
        assert channel_contributions_forward_pass_grid[0].sum().item() == 0
        np.testing.assert_equal(
            actual=channel_contributions_forward_pass,
            desired=channel_contributions_forward_pass_grid[1].to_numpy(),
        )
        assert (
            channel_contributions_forward_pass_grid[2].to_numpy()
            >= channel_contributions_forward_pass
        ).all()

    def test_get_channel_contributions_forward_pass_grid_shapes(
        self, mmm_fitted: DelayedSaturatedMMM
    ) -> None:
        n_channels = len(mmm_fitted.channel_columns)
        data_range = mmm_fitted.X.shape[0]
        draws = 3
        chains = 2
        grid_size = 2
        contributions = mmm_fitted.get_channel_contributions_forward_pass_grid(
            start=0, stop=1.5, num=grid_size
        )
        assert contributions.shape == (
            grid_size,
            chains,
            draws,
            data_range,
            n_channels,
        )

    def test_bad_start_get_channel_contributions_forward_pass_grid(
        self, mmm_fitted: DelayedSaturatedMMM
    ) -> None:
        with pytest.raises(
            expected_exception=ValueError,
            match="start must be greater than or equal to 0.",
        ):
            mmm_fitted.get_channel_contributions_forward_pass_grid(
                start=-0.5, stop=1.5, num=2
            )

    @pytest.mark.parametrize(
        argnames="absolute_xrange",
        argvalues=[False, True],
        ids=["relative_xrange", "absolute_xrange"],
    )
    def test_plot_channel_contributions_grid(
        self, mmm_fitted: DelayedSaturatedMMM, absolute_xrange: bool
    ) -> None:
        fig = mmm_fitted.plot_channel_contributions_grid(
            start=0, stop=1.5, num=2, absolute_xrange=absolute_xrange
        )
        assert isinstance(fig, plt.Figure)

    def test_data_setter(self, toy_X, toy_y):
        base_delayed_saturated_mmm = BaseDelayedSaturatedMMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock_max_lag=4,
        )
        base_delayed_saturated_mmm.fit(
            X=toy_X, y=toy_y, target_accept=0.81, draws=100, chains=2, random_seed=rng
        )

        X_correct_ndarray = np.random.randint(low=0, high=100, size=(135, 2))
        y_correct_ndarray = np.random.randint(low=0, high=100, size=135)

        X_incorrect = "Incorrect data"
        y_incorrect = "Incorrect data"

        with pytest.raises(TypeError):
            base_delayed_saturated_mmm._data_setter(X_incorrect, toy_y)

        with pytest.raises(TypeError):
            base_delayed_saturated_mmm._data_setter(toy_X, y_incorrect)

        with pytest.raises(KeyError):
            X_wrong_df = pd.DataFrame(
                {"column1": np.random.rand(135), "column2": np.random.rand(135)}
            )
            base_delayed_saturated_mmm._data_setter(X_wrong_df, toy_y)

        try:
            base_delayed_saturated_mmm._data_setter(toy_X, toy_y)
        except Exception as e:
            pytest.fail(f"_data_setter failed with error {e}")

        with pytest.raises(TypeError, match="X must be a pandas DataFrame"):
            base_delayed_saturated_mmm._data_setter(
                X_correct_ndarray, y_correct_ndarray
            )

    def test_save_load(self, mmm_fitted):
        model = mmm_fitted

        model.save("test_save_load")
        model2 = BaseDelayedSaturatedMMM.load("test_save_load")
        assert model.date_column == model2.date_column
        assert model.control_columns == model2.control_columns
        assert model.channel_columns == model2.channel_columns
        assert model.adstock_max_lag == model2.adstock_max_lag
        assert model.validate_data == model2.validate_data
        assert model.yearly_seasonality == model2.yearly_seasonality
        assert model.model_config == model2.model_config
        assert model.sampler_config == model2.sampler_config
        os.remove("test_save_load")

    def test_fail_id_after_load(self, monkeypatch, toy_X, toy_y):
        # This is the new behavior for the property
        def mock_property(self):
            return "for sure not correct id"

        # Now create an instance of MyClass
        DSMMM = DelayedSaturatedMMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock_max_lag=4,
        )

        # Check that the property returns the new value
        DSMMM.fit(
            toy_X, toy_y, target_accept=0.81, draws=100, chains=2, random_seed=rng
        )
        DSMMM.save("test_model")
        # Apply the monkeypatch for the property
        monkeypatch.setattr(DelayedSaturatedMMM, "id", property(mock_property))
        with pytest.raises(
            ValueError,
            match="The file 'test_model' does not contain an inference data of the same model or configuration as 'DelayedSaturatedMMM'",
        ):
            DelayedSaturatedMMM.load("test_model")
        os.remove("test_model")

    @pytest.mark.parametrize(
        argnames="model_config",
        argvalues=[
            None,
            {
                "intercept": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 2}},
                "beta_channel": {
                    "dist": "HalfNormal",
                    "kwargs": {"sigma": np.array([0.4533017, 0.25488063])},
                },
                "alpha": {
                    "dist": "Beta",
                    "kwargs": {
                        "alpha": np.array([3, 3]),
                        "beta": np.array([3.55001301, 2.87092431]),
                    },
                },
                "lam": {
                    "dist": "Gamma",
                    "kwargs": {
                        "alpha": np.array([3, 3]),
                        "beta": np.array([4.12231653, 5.02896872]),
                    },
                },
                "likelihood": {
                    "dist": "StudentT",
                    "kwargs": {"nu": 3, "sigma": 2},
                },
                "gamma_control": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 2}},
                "gamma_fourier": {"dist": "Laplace", "kwargs": {"mu": 0, "b": 1}},
            },
        ],
        ids=["default_config", "custom_config"],
    )
    def test_model_config(
        self, model_config: Dict, toy_X: pd.DataFrame, toy_y: pd.Series
    ):
        # Create model instance with specified config
        model = DelayedSaturatedMMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock_max_lag=2,
            yearly_seasonality=2,
            model_config=model_config,
        )

        model.build_model(X=toy_X, y=toy_y.to_numpy())
        # Check for default configuration
        if model_config is None:
            # assert observed RV type, and priors of some/all free_RVs.
            assert isinstance(
                model.model.observed_RVs[0].owner.op, pm.Normal
            )  # likelihood
            # Add more asserts as needed for default configuration

        # Check for custom configuration
        else:
            # assert custom configuration is applied correctly
            assert isinstance(
                model.model.observed_RVs[0].owner.op, pm.StudentT
            )  # likelihood
            assert isinstance(
                model.model["beta_channel"].owner.op, pm.HalfNormal
            )  # beta_channel


def new_date_ranges_to_test():
    yield from [
        # 2021-12-31 is the last date in the toy data
        # Old and New dates
        pd.date_range("2021-11-01", "2022-03-01", freq="W-MON"),
        # Only old dates
        pd.date_range("2019-06-01", "2021-12-31", freq="W-MON"),
        # Only new dates
        pd.date_range("2022-01-01", "2022-03-01", freq="W-MON"),
        # Less than the adstock_max_lag (4) of the model
        pd.date_range("2022-01-01", freq="W-MON", periods=1),
    ]


@pytest.mark.parametrize(
    "model_name", ["mmm_fitted", "mmm_fitted_with_fourier_features"]
)
@pytest.mark.parametrize(
    "new_dates",
    new_date_ranges_to_test(),
)
@pytest.mark.parametrize("combined", [True, False])
@pytest.mark.parametrize("original_scale", [True, False])
def test_new_data_sample_posterior_predictive_method(
    generate_data,
    toy_X,
    model_name: str,
    new_dates: pd.DatetimeIndex,
    combined: bool,
    original_scale: bool,
    request,
) -> None:
    """This is the method that is used in all the other methods that generate predictions."""
    mmm = request.getfixturevalue(model_name)
    X_pred = generate_data(new_dates)

    posterior_predictive = mmm.sample_posterior_predictive(
        X_pred=X_pred,
        extend_idata=False,
        combined=combined,
        original_scale=original_scale,
    )
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(posterior_predictive.coords["date"]),
        new_dates,
    )


@pytest.mark.parametrize(
    "model_name", ["mmm_fitted", "mmm_fitted_with_fourier_features"]
)
@pytest.mark.parametrize(
    "new_dates",
    [pd.date_range("2022-01-01", "2022-03-01", freq="W-MON")],
)
def test_new_data_include_last_observation_same_dims(
    generate_data,
    model_name: str,
    new_dates: pd.DatetimeIndex,
    request,
) -> None:
    mmm = request.getfixturevalue(model_name)
    X_pred = generate_data(new_dates)

    pp_without = mmm.predict_posterior(
        X_pred,
        include_last_observations=False,
    )
    pp_with = mmm.predict_posterior(
        X_pred,
        include_last_observations=True,
    )
    assert pp_without.coords.equals(pp_with.coords)
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(pp_without.coords["date"]),
        new_dates,
    )


@pytest.mark.parametrize(
    "model_name", ["mmm_fitted", "mmm_fitted_with_fourier_features"]
)
@pytest.mark.parametrize(
    "new_dates",
    [pd.date_range("2022-01-01", "2022-03-01", freq="W-MON")],
)
def test_new_data_predict_method(
    generate_data,
    toy_y,
    model_name: str,
    new_dates: pd.DatetimeIndex,
    request,
) -> None:
    mmm = request.getfixturevalue(model_name)
    X_pred = generate_data(new_dates)

    posterior_predictive_mean = mmm.predict(X_pred=X_pred)

    assert isinstance(posterior_predictive_mean, np.ndarray)
    assert posterior_predictive_mean.shape[0] == new_dates.size
    # Original scale constraint
    assert np.all(posterior_predictive_mean >= 0)

    # Domain kept close
    lower, upper = np.quantile(a=posterior_predictive_mean, q=[0.025, 0.975], axis=0)
    assert lower < toy_y.mean() < upper


def test_get_valid_distribution(mmm):
    normal_dist = mmm._get_distribution({"dist": "Normal"})
    assert normal_dist is pm.Normal


def test_get_invalid_distribution(mmm):
    with pytest.raises(ValueError, match="does not exist in PyMC"):
        mmm._get_distribution({"dist": "NonExistentDist"})


def test_invalid_likelihood_type(mmm):
    with pytest.raises(
        ValueError,
        match="The distribution used for the likelihood is not allowed",
    ):
        mmm._create_likelihood_distribution(
            dist={"dist": "Cauchy", "kwargs": {"alpha": 2, "beta": 4}},
            mu=np.array([0]),
            observed=np.random.randn(100),
            dims="obs_dim",
        )


def test_create_likelihood_invalid_kwargs_structure(mmm):
    with pytest.raises(
        ValueError, match="either a dictionary with a 'dist' key or a numeric value"
    ):
        mmm._create_likelihood_distribution(
            dist={"dist": "Normal", "kwargs": {"sigma": "not a dictionary or numeric"}},
            mu=np.array([0]),
            observed=np.random.randn(100),
            dims="obs_dim",
        )


def test_create_likelihood_mu_in_top_level_kwargs(mmm):
    with pytest.raises(
        ValueError, match="'mu' key is not allowed directly within 'kwargs'"
    ):
        mmm._create_likelihood_distribution(
            dist={"dist": "Normal", "kwargs": {"mu": 0, "sigma": 2}},
            mu=np.array([0]),
            observed=np.random.randn(100),
            dims="obs_dim",
        )


def new_contributions_property_checks(new_contributions, X, model):
    assert isinstance(new_contributions, xr.DataArray)

    coords = new_contributions.coords
    assert coords["channel"].values.tolist() == model.channel_columns
    np.testing.assert_allclose(
        coords["time_since_spend"].values,
        np.arange(-model.adstock_max_lag, model.adstock_max_lag + 1),
    )

    # Channel contributions are non-negative
    assert (new_contributions >= 0).all()


def test_new_spend_contributions(mmm_fitted) -> None:
    new_spend = np.ones(len(mmm_fitted.channel_columns))
    new_contributions = mmm_fitted.new_spend_contributions(new_spend)

    new_contributions_property_checks(new_contributions, mmm_fitted.X, mmm_fitted)


def test_new_spend_contributions_prior_error(mmm) -> None:
    new_spend = np.ones(len(mmm.channel_columns))
    match = "sample_prior_predictive"
    with pytest.raises(RuntimeError, match=match):
        mmm.new_spend_contributions(new_spend, prior=True)


@pytest.mark.parametrize("original_scale", [True, False])
def test_new_spend_contributions_prior(original_scale, mmm, toy_X) -> None:
    mmm.sample_prior_predictive(
        X_pred=toy_X,
        extend_idata=True,
    )

    new_spend = np.ones(len(mmm.channel_columns))
    new_contributions = mmm.new_spend_contributions(
        new_spend, prior=True, original_scale=original_scale, random_seed=0
    )

    new_contributions_property_checks(new_contributions, toy_X, mmm)


def test_plot_new_spend_contributions_original_scale(mmm_fitted) -> None:
    ax = mmm_fitted.plot_new_spend_contributions(
        spend_amount=1, original_scale=True, random_seed=0
    )

    assert isinstance(ax, plt.Axes)


@pytest.fixture(scope="module")
def mmm_with_prior(mmm) -> DelayedSaturatedMMM:
    n_chains = 1
    n_samples = 100

    channels = mmm.channel_columns
    n_channels = len(channels)

    idata = az.from_dict(
        prior={
            # Arbitrary but close to the default parameterization
            "alpha": rng.uniform(size=(n_chains, n_samples, n_channels)),
            "lam": rng.exponential(size=(n_chains, n_samples, n_channels)),
            "beta_channel": np.abs(rng.normal(size=(n_chains, n_samples, n_channels))),
        },
        coords={"channel": channels},
        dims={
            "alpha": ["chain", "draw", "channel"],
            "lam": ["chain", "draw", "channel"],
            "beta_channel": ["chain", "draw", "channel"],
        },
    )
    mmm.idata = idata

    return mmm


def test_plot_new_spend_contributions_prior(mmm_with_prior) -> None:
    ax = mmm_with_prior.plot_new_spend_contributions(
        spend_amount=1, prior=True, random_seed=0
    )
    assert isinstance(ax, plt.Axes)


def test_plot_new_spend_contributions_prior_select_channels(
    mmm_with_prior,
) -> None:
    ax = mmm_with_prior.plot_new_spend_contributions(
        spend_amount=1, prior=True, channels=["channel_2"], random_seed=0
    )

    assert isinstance(ax, plt.Axes)


@pytest.fixture
def df_lift_test() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "channel": ["channel_1", "channel_1"],
            "x": [1, 2],
            "delta_x": [1, 1],
            "delta_y": [1, 1],
            "sigma": [1, 1],
        }
    )


def test_add_lift_test_measurements(mmm, toy_X, toy_y, df_lift_test) -> None:
    mmm.build_model(X=toy_X, y=toy_y)

    name = "lift_measurements"
    assert name not in mmm.model

    mmm.add_lift_test_measurements(
        df_lift_test,
        name=name,
    )

    assert name in mmm.model


def test_add_lift_test_measurements_no_model() -> None:
    mmm = DelayedSaturatedMMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        adstock_max_lag=4,
        control_columns=["control_1", "control_2"],
    )
    with pytest.raises(RuntimeError, match="The model has not been built yet."):
        mmm.add_lift_test_measurements(
            pd.DataFrame(),
        )
