import os
from typing import List, Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
from matplotlib import pyplot as plt

from pymc_marketing.mmm.delayed_saturated_mmm import (
    BaseDelayedSaturatedMMM,
    DelayedSaturatedMMM,
)

seed: int = sum(map(ord, "pymc_marketing"))
rng: np.random.Generator = np.random.default_rng(seed=seed)


@pytest.fixture(scope="class")
def toy_X() -> pd.DataFrame:
    date_data: pd.DatetimeIndex = pd.date_range(
        start="2019-06-01", end="2021-12-31", freq="W-MON"
    )

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


@pytest.fixture(scope="class")
def toy_y(toy_X: pd.DataFrame) -> pd.Series:
    return pd.Series(data=rng.integers(low=0, high=100, size=toy_X.shape[0]))


@pytest.fixture(scope="class")
def mmm() -> DelayedSaturatedMMM:
    return DelayedSaturatedMMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        adstock_max_lag=4,
        control_columns=["control_1", "control_2"],
    )


@pytest.fixture(scope="class")
def mmm_fitted(
    mmm: DelayedSaturatedMMM, toy_X: pd.DataFrame, toy_y: pd.Series
) -> DelayedSaturatedMMM:
    mmm.fit(X=toy_X, y=toy_y, target_accept=0.8, draws=3, chains=2)
    return mmm


class TestDelayedSaturatedMMM:
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
        channel_data = mmm_fitted.X[mmm_fitted.channel_columns].to_numpy()
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
        channel_data = mmm_fitted.X[mmm_fitted.channel_columns].to_numpy()
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

        with pytest.raises(RuntimeError):
            X_wrong_df = pd.DataFrame(
                {"column1": np.random.rand(135), "column2": np.random.rand(135)}
            )
            base_delayed_saturated_mmm._data_setter(X_wrong_df, toy_y)

        try:
            base_delayed_saturated_mmm._data_setter(toy_X, toy_y)
        except Exception as e:
            pytest.fail(f"_data_setter failed with error {e}")

        try:
            base_delayed_saturated_mmm._data_setter(
                X_correct_ndarray, y_correct_ndarray
            )
        except Exception as e:
            pytest.fail(f"_data_setter failed with error {e}")

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
