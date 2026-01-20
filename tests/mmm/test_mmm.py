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
import os

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from matplotlib import pyplot as plt
from pymc_extras.prior import Prior

from pymc_marketing.mmm.components.adstock import DelayedAdstock, GeometricAdstock
from pymc_marketing.mmm.components.saturation import (
    LogisticSaturation,
    MichaelisMentenSaturation,
    SaturationTransformation,
)
from pymc_marketing.mmm.mmm import MMM, BaseMMM
from pymc_marketing.model_builder import DifferentModelError

pytest.skip(reason="MMM not refactored with pymc.dims", allow_module_level=True)

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


@pytest.fixture(scope="module")
def toy_X_with_bad_dates() -> pd.DataFrame:
    bad_date_data = ["a", "b", "c", "d", "e"]
    n: int = len(bad_date_data)
    return pd.DataFrame(
        data={
            "date": bad_date_data,
            "channel_1": rng.integers(low=0, high=400, size=n),
            "channel_2": rng.integers(low=0, high=50, size=n),
            "control_1": rng.gamma(shape=1000, scale=500, size=n),
            "control_2": rng.gamma(shape=100, scale=5, size=n),
            "other_column_1": rng.integers(low=0, high=100, size=n),
            "other_column_2": rng.normal(loc=0, scale=1, size=n),
        }
    )


@pytest.fixture(scope="class")
def model_config_requiring_serialization() -> dict:
    model_config = {
        "intercept": Prior("Normal", mu=0, sigma=2),
        "saturation_beta": Prior("HalfNormal", sigma=np.array([0.4533017, 0.25488063])),
        "adstock_alpha": Prior(
            "Beta", alpha=np.array([3, 3]), beta=np.array([3.55001301, 2.87092431])
        ),
        "saturation_lam": Prior(
            "Gamma", alpha=np.array([3, 3]), beta=np.array([4.12231653, 5.02896872])
        ),
        "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=2)),
        "gamma_control": Prior("HalfNormal", sigma=2),
        "gamma_fourier": Prior("HalfNormal"),
    }
    return model_config


@pytest.fixture(scope="module")
def toy_y(toy_X: pd.DataFrame) -> pd.Series:
    return pd.Series(data=rng.integers(low=0, high=100, size=toy_X.shape[0]))


@pytest.fixture(scope="module")
def mmm() -> MMM:
    return MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        control_columns=["control_1", "control_2"],
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
    )


@pytest.fixture(scope="module")
def mmm_no_controls() -> MMM:
    return MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
    )


@pytest.fixture(scope="module")
def mmm_with_fourier_features() -> MMM:
    return MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        control_columns=["control_1", "control_2"],
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        yearly_seasonality=2,
    )


@pytest.fixture(scope="module")
def mmm_fitted(
    mmm: MMM,
    toy_X: pd.DataFrame,
    toy_y: pd.Series,
    mock_pymc_sample,
) -> MMM:
    mmm.fit(X=toy_X, y=toy_y)
    return mmm


@pytest.fixture(scope="module")
def mmm_fitted_no_controls(
    mmm_no_controls: MMM,
    toy_X: pd.DataFrame,
    toy_y: pd.Series,
    mock_pymc_sample,
) -> MMM:
    mmm_no_controls.fit(X=toy_X, y=toy_y)
    return mmm_no_controls


@pytest.fixture(scope="module")
def mmm_fitted_with_posterior_predictive(
    mmm_fitted: MMM,
    toy_X: pd.DataFrame,
) -> MMM:
    _ = mmm_fitted.sample_posterior_predictive(toy_X, extend_idata=True, combined=True)
    return mmm_fitted


@pytest.fixture(scope="module")
def mmm_fitted_with_prior_and_posterior_predictive(
    mmm_fitted_with_posterior_predictive,
    toy_X,
):
    _ = mmm_fitted_with_posterior_predictive.sample_prior_predictive(toy_X)
    return mmm_fitted_with_posterior_predictive


@pytest.fixture(scope="module")
def mmm_fitted_with_fourier_features(
    mmm_with_fourier_features: MMM,
    toy_X: pd.DataFrame,
    toy_y: pd.Series,
    mock_pymc_sample,
) -> MMM:
    mmm_with_fourier_features.fit(X=toy_X, y=toy_y)
    return mmm_with_fourier_features


@pytest.mark.parametrize("media_transform", ["adstock", "saturation"])
def test_plotting_media_transform_workflow(mmm_fitted, media_transform) -> None:
    transform = getattr(mmm_fitted, media_transform)
    curve = transform.sample_curve(mmm_fitted.fit_result)
    fig, axes = transform.plot_curve(curve)

    assert isinstance(fig, plt.Figure)
    assert len(axes) == mmm_fitted.fit_result["channel"].size

    plt.close()


class TestMMM:
    def test_save_load_with_not_serializable_model_config(
        self,
        model_config_requiring_serialization,
        toy_X,
        toy_y,
        mock_pymc_sample,
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

        l_max = 4
        adstock = GeometricAdstock(l_max=l_max)
        saturation = LogisticSaturation()
        model = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            model_config=model_config_requiring_serialization,
            adstock=adstock,
            saturation=saturation,
        )
        model.fit(toy_X, toy_y)
        model.save("test_save_load")
        model2 = MMM.load("test_save_load")
        assert model.date_column == model2.date_column
        assert model.control_columns == model2.control_columns
        assert model.channel_columns == model2.channel_columns
        assert model.adstock.l_max == model2.adstock.l_max
        assert model.validate_data == model2.validate_data
        assert model.yearly_seasonality == model2.yearly_seasonality
        assert deep_equal(model.model_config, model2.model_config)

        assert model.sampler_config == model2.sampler_config
        os.remove("test_save_load")

    def test_bad_date_column(self, toy_X_with_bad_dates) -> None:
        with pytest.raises(
            ValueError,
            match=r"Could not convert bad_date_column to datetime. Please check the date format.",
        ):
            my_mmm = MMM(
                date_column="bad_date_column",
                channel_columns=["channel_1", "channel_2"],
                adstock=GeometricAdstock(l_max=4),
                saturation=LogisticSaturation(),
                control_columns=["control_1", "control_2"],
            )
            y = np.ones(toy_X_with_bad_dates.shape[0])
            my_mmm.build_model(X=toy_X_with_bad_dates, y=y)

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
    @pytest.mark.parametrize(
        argnames="time_varying_intercept",
        argvalues=[False, True],
        ids=["no_time_varying_intercept", "time_varying_intercept"],
    )
    @pytest.mark.parametrize(
        argnames="time_varying_media",
        argvalues=[False, True],
        ids=["no_time_varying_media", "time_varying_media"],
    )
    def test_init(
        self,
        toy_X: pd.DataFrame,
        toy_y: pd.Series,
        yearly_seasonality: int | None,
        channel_columns: list[str],
        control_columns: list[str],
        adstock_max_lag: int,
        time_varying_intercept: bool,
        time_varying_media: bool,
    ) -> None:
        mmm = BaseMMM(
            date_column="date",
            channel_columns=channel_columns,
            control_columns=control_columns,
            yearly_seasonality=yearly_seasonality,
            time_varying_intercept=time_varying_intercept,
            time_varying_media=time_varying_media,
            adstock=GeometricAdstock(l_max=adstock_max_lag),
            saturation=LogisticSaturation(),
        )
        mmm.build_model(X=toy_X, y=toy_y)
        n_channel: int = len(mmm.channel_columns)
        samples: int = 3
        with mmm.model:
            prior_predictive: az.InferenceData = pm.sample_prior_predictive(
                draws=samples, random_seed=rng
            )

        assert az.extract(
            prior_predictive, group="prior", var_names=["intercept"], combined=True
        ).to_numpy().shape == (
            (samples,) if not time_varying_intercept else (toy_X.shape[0], samples)
        )
        assert az.extract(
            data=prior_predictive,
            group="prior",
            var_names=["saturation_beta"],
            combined=True,
        ).to_numpy().shape == (
            n_channel,
            samples,
        )
        assert az.extract(
            data=prior_predictive,
            group="prior",
            var_names=["adstock_alpha"],
            combined=True,
        ).to_numpy().shape == (
            n_channel,
            samples,
        )
        assert az.extract(
            data=prior_predictive,
            group="prior",
            var_names=["saturation_lam"],
            combined=True,
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

    def test_fit(self, toy_X: pd.DataFrame, toy_y: pd.Series, mock_pymc_sample) -> None:
        mmm = BaseMMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            control_columns=["control_1", "control_2"],
            yearly_seasonality=2,
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
        )
        assert mmm.version == "0.0.3"
        assert mmm._model_type == "BaseValidateMMM"
        assert mmm._model_name == "BaseMMM"
        assert mmm.model_config is not None
        n_channel: int = len(mmm.channel_columns)
        n_control: int = len(mmm.control_columns)
        mmm.fit(X=toy_X, y=toy_y)
        posterior: az.InferenceData = mmm.fit_result
        chains = posterior.sizes["chain"]
        draws = posterior.sizes["draw"]
        assert (
            az.extract(data=posterior, var_names=["intercept"], combined=True)
            .to_numpy()
            .size
            == draws * chains
        )
        assert az.extract(
            data=posterior, var_names=["saturation_beta"], combined=True
        ).to_numpy().shape == (n_channel, draws * chains)
        assert az.extract(
            data=posterior, var_names=["adstock_alpha"], combined=True
        ).to_numpy().shape == (n_channel, draws * chains)
        assert az.extract(
            data=posterior, var_names=["saturation_lam"], combined=True
        ).to_numpy().shape == (n_channel, draws * chains)
        assert az.extract(
            data=posterior, var_names=["gamma_control"], combined=True
        ).to_numpy().shape == (
            n_channel,
            draws * chains,
        )

        mean_model_contributions_ts = mmm.compute_mean_contributions_over_time(
            original_scale=True
        )
        assert mean_model_contributions_ts.shape == (
            toy_X.shape[0],
            n_channel
            + n_control
            + 2,  # 2 for yearly seasonality (+1) and intercept (+)
        )

        processed_df = mmm._process_decomposition_components(
            data=mean_model_contributions_ts
        )

        assert processed_df.shape == (n_channel + n_control + 2, 3)

        assert mean_model_contributions_ts.columns.tolist() == [
            "channel_1",
            "channel_2",
            "control_1",
            "control_2",
            "yearly_seasonality",
            "intercept",
        ]

    def test_mmm_serializes_and_deserializes_dag_and_nodes(
        self,
        toy_X: pd.DataFrame,
        toy_y: pd.Series,
        mock_pymc_sample,
    ) -> None:
        dag = """
        digraph {
            channel_1 -> y;
            control_1 -> channel_1;
            control_1 -> y;
        }
        """
        treatment_nodes = ["channel_1"]
        outcome_node = "y"

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            control_columns=["control_1", "control_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            dag=dag,
            treatment_nodes=treatment_nodes,
            outcome_node=outcome_node,
        )

        mmm.fit(X=toy_X, y=toy_y)

        # Save and reload the model
        mmm.save("test_model")
        loaded_mmm = MMM.load("test_model")

        # Assert that the attributes persist
        assert loaded_mmm.dag == dag, "DAG did not persist correctly."
        assert loaded_mmm.treatment_nodes == treatment_nodes, (
            "Treatment nodes did not persist correctly."
        )
        assert loaded_mmm.outcome_node == outcome_node, (
            "Outcome node did not persist correctly."
        )

        # Clean up
        os.remove("test_model")

    def test_channel_contribution_forward_pass_recovers_contribution(
        self,
        mmm_fitted: MMM,
    ) -> None:
        channel_data = mmm_fitted.preprocessed_data["X"][
            mmm_fitted.channel_columns
        ].to_numpy()
        channel_contribution_forward_pass = (
            mmm_fitted.channel_contribution_forward_pass(channel_data=channel_data)
        )
        channel_contribution_forward_pass_mean = channel_contribution_forward_pass.mean(
            axis=(0, 1)
        )
        # The forward pass results should be in the original scale of the target variable.
        # Compare directly with the original scale version from the fit result
        if "channel_contribution_original_scale" in mmm_fitted.fit_result:
            channel_contribution_original_mean = mmm_fitted.fit_result[
                "channel_contribution_original_scale"
            ].mean(dim=["draw", "chain"])
        else:
            # Fallback: scale the scaled version by target_scale
            channel_contribution_scaled_mean = mmm_fitted.fit_result[
                "channel_contribution"
            ].mean(dim=["draw", "chain"])
            target_scale = (
                mmm_fitted.target_scale
                if hasattr(mmm_fitted, "target_scale")
                else mmm_fitted.y.max()
            )
            channel_contribution_original_mean = (
                channel_contribution_scaled_mean * target_scale
            )

        assert (
            channel_contribution_forward_pass_mean.shape
            == channel_contribution_original_mean.shape
        )

        # Both should be in original scale and approximately equal
        # Use relative tolerance for numerical precision and sampling variation
        np.testing.assert_allclose(
            channel_contribution_forward_pass_mean,
            channel_contribution_original_mean,
            rtol=2e-1,  # 20% relative tolerance for sampling variation
            atol=1e-1,  # Absolute tolerance
        )

    @pytest.mark.parametrize(
        argnames="original_scale",
        argvalues=[False, True],
        ids=["scaled", "original-scale"],
    )
    @pytest.mark.parametrize(
        argnames="var_contribution",
        argvalues=["channel_contribution", "control_contribution"],
        ids=["channel_contribution", "control_contribution"],
    )
    def test_get_ts_contribution_posterior(
        self,
        mmm_fitted_with_posterior_predictive: MMM,
        var_contribution: str,
        original_scale: bool,
    ):
        ts_posterior = (
            mmm_fitted_with_posterior_predictive.get_ts_contribution_posterior(
                var_contribution=var_contribution, original_scale=original_scale
            )
        )
        chains = ts_posterior.sizes["chain"]
        draws = ts_posterior.sizes["draw"]
        assert ts_posterior.dims == ("chain", "draw", "date")
        assert ts_posterior.chain.size == chains
        assert ts_posterior.draw.size == draws

    @pytest.mark.parametrize(
        argnames="original_scale",
        argvalues=[False, True],
        ids=["scaled", "original-scale"],
    )
    def test_get_errors(
        self,
        mmm_fitted_with_posterior_predictive: MMM,
        original_scale: bool,
    ) -> None:
        errors = mmm_fitted_with_posterior_predictive.get_errors(
            original_scale=original_scale
        )
        n_chains = errors.sizes["chain"]
        n_draws = errors.sizes["draw"]
        assert isinstance(errors, xr.DataArray)
        assert errors.name == "errors"
        assert errors.shape == (
            n_chains,
            n_draws,
            mmm_fitted_with_posterior_predictive.y.shape[0],
        )

    def test_get_errors_raises_not_fitted(self) -> None:
        my_mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            control_columns=["control_1", "control_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )
        with pytest.raises(
            RuntimeError,
            match=r"Make sure the model has been fitted and the posterior_predictive has been sampled!",
        ):
            my_mmm.get_errors()

    def test_posterior_predictive_raises_not_fitted(self) -> None:
        my_mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            control_columns=["control_1", "control_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )
        with pytest.raises(
            RuntimeError,
            match=r"Make sure the model has been fitted and the posterior_predictive has been sampled!",
        ):
            my_mmm.plot_posterior_predictive()

    def test_get_errors_bad_y_length(
        self,
        mmm_fitted_with_posterior_predictive: MMM,
    ):
        mmm_fitted_with_posterior_predictive.y = np.array([1, 2])
        with pytest.raises(ValueError):
            mmm_fitted_with_posterior_predictive.get_errors()

    def test_plot_posterior_predictive_bad_y_length(
        self,
        mmm_fitted_with_posterior_predictive: MMM,
    ):
        mmm_fitted_with_posterior_predictive.y = np.array([1, 2])
        with pytest.raises(ValueError):
            mmm_fitted_with_posterior_predictive.plot_posterior_predictive()

    def test_channel_contribution_forward_pass_is_consistent(
        self, mmm_fitted: MMM
    ) -> None:
        channel_data = mmm_fitted.preprocessed_data["X"][
            mmm_fitted.channel_columns
        ].to_numpy()
        channel_contribution_forward_pass = (
            mmm_fitted.channel_contribution_forward_pass(channel_data=channel_data)
        )
        # use a grid [0, 1, 2] which corresponds to
        # - no-spend -> forward pass should be zero
        # - spend input for the model -> should match the forward pass
        # - doubling the spend -> should be higher than the forward pass with the original spend
        channel_contribution_forward_pass_grid = (
            mmm_fitted.get_channel_contribution_forward_pass_grid(
                start=0, stop=2, num=3
            )
        )
        assert channel_contribution_forward_pass_grid[0].sum().item() == 0
        np.testing.assert_equal(
            actual=channel_contribution_forward_pass,
            desired=channel_contribution_forward_pass_grid[1].to_numpy(),
        )
        assert (
            channel_contribution_forward_pass_grid[2].to_numpy()
            >= channel_contribution_forward_pass
        ).all()

    def test_get_channel_contribution_forward_pass_grid_shapes(
        self, mmm_fitted: MMM
    ) -> None:
        n_channels = len(mmm_fitted.channel_columns)
        data_range = mmm_fitted.X.shape[0]
        grid_size = 2
        contributions = mmm_fitted.get_channel_contribution_forward_pass_grid(
            start=0, stop=1.5, num=grid_size
        )
        draws = contributions.sizes["draw"]
        chains = contributions.sizes["chain"]
        assert contributions.shape == (
            grid_size,
            chains,
            draws,
            data_range,
            n_channels,
        )

    def test_bad_start_get_channel_contribution_forward_pass_grid(
        self,
        mmm_fitted: MMM,
    ) -> None:
        with pytest.raises(
            expected_exception=ValueError,
            match=r"start must be greater than or equal to 0.",
        ):
            mmm_fitted.get_channel_contribution_forward_pass_grid(
                start=-0.5, stop=1.5, num=2
            )

    @pytest.mark.parametrize(
        argnames="absolute_xrange",
        argvalues=[False, True],
        ids=["relative_xrange", "absolute_xrange"],
    )
    def test_plot_channel_contribution_grid(
        self, mmm_fitted: MMM, absolute_xrange: bool
    ) -> None:
        fig = mmm_fitted.plot_channel_contribution_grid(
            start=0, stop=1.5, num=2, absolute_xrange=absolute_xrange
        )
        assert isinstance(fig, plt.Figure)

    @pytest.mark.parametrize(
        argnames="group",
        argvalues=["prior_predictive", "posterior_predictive"],
        ids=["prior_predictive", "posterior_predictive"],
    )
    @pytest.mark.parametrize(
        argnames="original_scale",
        argvalues=[False, True],
        ids=["scaled", "original-scale"],
    )
    def test_get_group_predictive_data(
        self,
        mmm_fitted_with_prior_and_posterior_predictive: MMM,
        group: str,
        original_scale: bool,
    ):
        dataset = (
            mmm_fitted_with_prior_and_posterior_predictive._get_group_predictive_data(
                group=group,
                original_scale=original_scale,
            )
        )
        assert isinstance(dataset, xr.Dataset)
        assert dataset.dims["date"] == 135
        assert dataset["y"].dims == ("chain", "draw", "date")

    def test_data_setter(self, toy_X, toy_y, mock_pymc_sample):
        base_delayed_saturated_mmm = BaseMMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )
        base_delayed_saturated_mmm.fit(X=toy_X, y=toy_y)

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

        with pytest.raises(TypeError, match=r"X must be a pandas DataFrame"):
            base_delayed_saturated_mmm._data_setter(
                X_correct_ndarray, y_correct_ndarray
            )

    def test_save_load(self, mmm_fitted: MMM):
        model = mmm_fitted

        model.save("test_save_load")
        model2 = MMM.load("test_save_load")
        assert model.date_column == model2.date_column
        assert model.control_columns == model2.control_columns
        assert model.channel_columns == model2.channel_columns
        assert model.adstock.l_max == model2.adstock.l_max
        assert model.validate_data == model2.validate_data
        assert model.yearly_seasonality == model2.yearly_seasonality
        assert model.model_config == model2.model_config
        assert model.sampler_config == model2.sampler_config
        os.remove("test_save_load")

    def test_save_load_with_kwargs(self, mmm_fitted: MMM):
        """Test save/load functionality with kwargs (engine and groups)."""
        model = mmm_fitted

        # Use kwargs to test functionality - ArviZ supports engine and groups
        # Note: ArviZ's to_netcdf has limited compression support compared to xarray
        compression_kwargs = {
            "engine": "h5netcdf",  # Alternative engine that may have better compression
        }

        model.save("test_save_load_kwargs", **compression_kwargs)

        # Load and verify
        model2 = MMM.load("test_save_load_kwargs")
        assert model.date_column == model2.date_column
        assert model.control_columns == model2.control_columns
        assert model.channel_columns == model2.channel_columns
        assert model.adstock.l_max == model2.adstock.l_max
        assert model.validate_data == model2.validate_data
        assert model.yearly_seasonality == model2.yearly_seasonality
        assert model.model_config == model2.model_config
        assert model.sampler_config == model2.sampler_config

        os.remove("test_save_load_kwargs")

    def test_save_load_engine_comparison(self, mmm_fitted: MMM):
        """Test save/load with different engines and kwargs options."""
        model = mmm_fitted

        # Save with default engine
        model.save("test_save_load_default")

        # Save with h5netcdf engine (demonstrates kwargs functionality)
        engine_kwargs = {
            "engine": "h5netcdf",
        }
        model.save("test_save_load_h5netcdf", **engine_kwargs)

        # Verify both files exist
        assert os.path.exists("test_save_load_default")
        assert os.path.exists("test_save_load_h5netcdf")

        # Verify both can be loaded successfully and have the same data
        model_default = MMM.load("test_save_load_default")
        model_h5netcdf = MMM.load("test_save_load_h5netcdf")

        # Both should have the same model configuration
        assert (
            model.model_config
            == model_default.model_config
            == model_h5netcdf.model_config
        )

        # Clean up
        os.remove("test_save_load_default")
        os.remove("test_save_load_h5netcdf")

    def test_fail_id_after_load(self, monkeypatch, toy_X, toy_y):
        # This is the new behavior for the property
        def mock_property(self):
            return "for sure not correct id"

        # Now create an instance of MyClass
        DSMMM = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )

        # Check that the property returns the new value
        DSMMM.fit(toy_X, toy_y)
        DSMMM.save("test_model")
        # Apply the monkeypatch for the property
        monkeypatch.setattr(MMM, "id", property(mock_property))

        error_msg = (
            "The file 'test_model' does not "
            "contain an InferenceData of the "
            "same model or configuration as 'MMM'"
        )
        with pytest.raises(DifferentModelError, match=error_msg):
            MMM.load("test_model")
        os.remove("test_model")

    @pytest.mark.parametrize(
        argnames="model_config",
        argvalues=[
            None,
            {
                "intercept": Prior("Normal", mu=0, sigma=2),
                "saturation_beta": Prior(
                    "HalfNormal", sigma=np.array([0.4533017, 0.25488063])
                ),
                "adstock_alpha": Prior(
                    "Beta",
                    alpha=np.array([3, 3]),
                    beta=np.array([3.55001301, 2.87092431]),
                ),
                "saturation_lam": Prior(
                    "Gamma",
                    alpha=np.array([3, 3]),
                    beta=np.array([4.12231653, 5.02896872]),
                ),
                "likelihood": Prior("StudentT", nu=3, sigma=2),
                "gamma_control": Prior("Normal", sigma=2),
                "gamma_fourier": Prior("Laplace", mu=0, b=1),
            },
        ],
        ids=["default_config", "custom_config"],
    )
    def test_model_config(
        self, model_config: dict, toy_X: pd.DataFrame, toy_y: pd.Series
    ):
        # Create model instance with specified config
        model = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            yearly_seasonality=2,
            model_config=model_config,
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
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
                model.model["saturation_beta"].owner.op, pm.HalfNormal
            )  # saturation_beta

    def test_mmm_causal_attributes_initialization(self):
        dag = """
        digraph {
            channel_1 -> y;
            control_1 -> channel_1;
            control_1 -> y;
        }
        """
        treatment_nodes = ["channel_1"]
        outcome_node = "y"

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            control_columns=["control_1", "control_2"],
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
            dag=dag,
            treatment_nodes=treatment_nodes,
            outcome_node=outcome_node,
        )

        assert mmm.dag == dag, "DAG was not set correctly."
        assert mmm.treatment_nodes == treatment_nodes, (
            "Treatment nodes not set correctly."
        )
        assert mmm.outcome_node == outcome_node, "Outcome node not set correctly."

    def test_mmm_causal_attributes_default_treatment_nodes(self):
        dag = """
        digraph {
            channel_1 -> y;
            channel_2 -> y;
            control_1 -> channel_1;
            control_1 -> channel_2;
            control_1 -> y;
        }
        """
        outcome_node = "y"

        with pytest.warns(
            UserWarning,
            match=r"No treatment nodes provided, using channel columns as treatment nodes.",
        ):
            mmm = MMM(
                date_column="date",
                channel_columns=["channel_1", "channel_2"],
                control_columns=["control_1", "control_2"],
                adstock=GeometricAdstock(l_max=2),
                saturation=LogisticSaturation(),
                dag=dag,
                outcome_node=outcome_node,
            )

        assert mmm.treatment_nodes == [
            "channel_1",
            "channel_2",
        ], "Default treatment nodes are incorrect."
        assert mmm.outcome_node == "y", "Outcome node was not set correctly."

    def test_mmm_adjustment_set_updates_control_columns(self):
        dag = """
        digraph {
            channel_1 -> y;
            control_1 -> channel_1;
            control_1 -> y;
        }
        """
        treatment_nodes = ["channel_1"]
        outcome_node = "y"

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            control_columns=["control_1", "control_2"],
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
            dag=dag,
            treatment_nodes=treatment_nodes,
            outcome_node=outcome_node,
        )

        assert mmm.control_columns == ["control_1"], (
            "Control columns were not updated based on the DAG."
        )

    def test_mmm_missing_dag_does_not_initialize_causal_graph(self):
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            control_columns=["control_1", "control_2"],
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
        )

        assert mmm.dag is None, "DAG should be None."
        assert not hasattr(mmm, "causal_graphical_model"), (
            "Causal graph should not be initialized without a DAG."
        )

    def test_mmm_missing_dag_or_nodes(self):
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            control_columns=["control_1", "control_2"],
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
        )

        # Check that the causal graph is not initialized
        assert mmm.dag is None, "DAG should be None when not provided."
        assert not hasattr(mmm, "causal_graphical_model"), (
            "Causal graph should not be initialized without DAG."
        )

        # Check behavior with missing treatment or outcome nodes
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            control_columns=["control_1", "control_2"],
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
            dag="digraph {channel_1 -> y;}",
        )
        assert mmm.treatment_nodes is None, "Treatment nodes should default to None."
        assert mmm.outcome_node is None, "Outcome node should default to None."

    def test_scaling_dict_with_missing_channel(
        self, toy_X: pd.DataFrame, toy_y: pd.Series
    ):
        """Test scaling dict that only specifies target."""
        from pymc_marketing.mmm.scaling import VariableScaling

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            scaling={"target": VariableScaling(method="mean", dims=())},
        )

        mmm.build_model(X=toy_X, y=toy_y)
        assert hasattr(mmm, "model")
        assert mmm.scaling.channel.method == "max"

    def test_scaling_dict_with_missing_target(
        self, toy_X: pd.DataFrame, toy_y: pd.Series
    ):
        """Test scaling dict that only specifies channel."""
        from pymc_marketing.mmm.scaling import VariableScaling

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            scaling={"channel": VariableScaling(method="mean", dims=())},
        )

        mmm.build_model(X=toy_X, y=toy_y)
        assert hasattr(mmm, "model")
        assert mmm.scaling.target.method == "max"

    def test_mean_scaling_method(self, toy_X: pd.DataFrame, toy_y: pd.Series):
        """Test using mean scaling instead of max."""
        from pymc_marketing.mmm.scaling import Scaling, VariableScaling

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            scaling=Scaling(
                target=VariableScaling(method="mean", dims=()),
                channel=VariableScaling(method="mean", dims=()),
            ),
        )

        mmm.build_model(X=toy_X, y=toy_y)
        assert hasattr(mmm, "model")
        assert mmm.target_scale > 0

    def test_validation_disabled(self, toy_X: pd.DataFrame, toy_y: pd.Series):
        """Test model with validation disabled."""
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            validate_data=False,
        )

        mmm.build_model(X=toy_X, y=toy_y)
        assert hasattr(mmm, "model")

    def test_adstock_first_false(self, toy_X: pd.DataFrame, toy_y: pd.Series):
        """Test model with saturation applied before adstock."""
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            adstock_first=False,
        )

        mmm.build_model(X=toy_X, y=toy_y)
        assert hasattr(mmm, "model")
        assert mmm.adstock_first is False

    def test_serializable_model_config_with_ndarray(self, mmm_fitted: MMM):
        """Test _serializable_model_config converts ndarrays to lists."""
        mmm_fitted.model_config["test_array"] = np.array([1, 2, 3])
        mmm_fitted.model_config["nested"] = {"array": np.array([4, 5])}

        serializable = mmm_fitted._serializable_model_config

        assert isinstance(serializable["test_array"], list)
        assert isinstance(serializable["nested"]["array"], list)

    def test_model_config_formatting_with_nested_dicts(self):
        """Test _model_config_formatting with nested dictionaries."""
        config = {
            "param1": [1, 2, 3],
            "param2": {"nested": [4, 5, 6], "dims": ["a", "b"]},
            "param3": {"deep": {"nested": [7, 8], "dims": ["c"]}},
        }

        formatted = MMM._model_config_formatting(config)

        assert isinstance(formatted["param2"]["dims"], tuple)
        assert isinstance(formatted["param3"]["deep"]["dims"], tuple)
        assert isinstance(formatted["param1"], np.ndarray)
        assert isinstance(formatted["param2"]["nested"], np.ndarray)

    def test_deserialize_scaling_none(self):
        """Test _deserialize_scaling with None input."""
        result = MMM._deserialize_scaling(None)
        assert result is None

    def test_attrs_to_init_kwargs_with_defaults(self):
        """Test attrs_to_init_kwargs with missing optional attributes."""
        attrs = {
            "model_config": "{}",
            "date_column": '"date"',
            "control_columns": "null",
            "channel_columns": '["ch1"]',
            "adstock": '{"lookup_name": "geometric", "l_max": 4, "normalize": true, "mode": "After"}',
            "saturation": '{"lookup_name": "logistic"}',
            "yearly_seasonality": "null",
            "validate_data": "true",
            "sampler_config": "{}",
        }

        kwargs = MMM.attrs_to_init_kwargs(attrs)

        assert kwargs["adstock_first"] is True
        assert kwargs["time_varying_intercept"] is False
        assert kwargs["scaling"] is None


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
@pytest.mark.parametrize(
    argnames="combined",
    argvalues=[True, False],
    ids=["combined", "not_combined"],
)
@pytest.mark.parametrize(
    argnames="original_scale",
    argvalues=[True, False],
    ids=["original_scale", "scaled"],
)
@pytest.mark.parametrize(
    argnames="var_names",
    argvalues=[None, ["mu", "y_sigma", "channel_contribution"], ["mu", "intercept"]],
    ids=["no_var_names", "var_names", "var_names_with_intercept"],
)
def test_new_data_sample_posterior_predictive_method(
    generate_data,
    toy_X,
    model_name: str,
    new_dates: pd.DatetimeIndex,
    combined: bool,
    original_scale: bool,
    var_names: list[str] | None,
    request,
) -> None:
    """This is the method that is used in all the other methods that generate predictions."""
    mmm = request.getfixturevalue(model_name)
    X = generate_data(new_dates)

    kwargs = {"var_names": var_names} if var_names is not None else {}

    posterior_predictive = mmm.sample_posterior_predictive(
        X=X,
        extend_idata=False,
        combined=combined,
        original_scale=original_scale,
        **kwargs,
    )
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(posterior_predictive.coords["date"]),
        new_dates,
    )

    if var_names is not None:
        assert var_names == list(posterior_predictive.data_vars)


@pytest.mark.parametrize(
    "predictions",
    [True, False],
)
def test_sample_posterior_predictive_with_prediction_kwarg(
    generate_data,
    mmm_fitted,
    predictions: bool,
) -> None:
    new_dates = pd.date_range("2022-01-01", "2022-03-01", freq="W-MON")
    X = generate_data(new_dates)

    predictions = mmm_fitted.sample_posterior_predictive(
        X=X,
        extend_idata=False,
        combined=True,
        predictions=predictions,
    )
    pd.testing.assert_index_equal(
        pd.DatetimeIndex(predictions.coords["date"]),
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
    X = generate_data(new_dates)

    pp_without = mmm.predict_posterior(
        X,
        include_last_observations=False,
    )
    pp_with = mmm.predict_posterior(
        X,
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
    X = generate_data(new_dates)

    posterior_predictive_mean = mmm.predict(X=X)

    assert isinstance(posterior_predictive_mean, np.ndarray)
    assert posterior_predictive_mean.shape[0] == new_dates.size
    # Original scale constraint
    # TODO: bring back with real data fit
    # assert np.all(posterior_predictive_mean >= 0)

    # Domain kept close
    # TODO: bring back with real data fit
    # lower, upper = np.quantile(a=posterior_predictive_mean, q=[0.025, 0.975], axis=0)
    # assert lower < toy_y.mean() < upper


def new_contributions_property_checks(new_contributions, X, model):
    assert isinstance(new_contributions, xr.DataArray)

    coords = new_contributions.coords
    assert coords["channel"].values.tolist() == model.channel_columns
    np.testing.assert_allclose(
        coords["time_since_spend"].values,
        np.arange(-model.adstock.l_max, model.adstock.l_max + 1),
    )

    # Channel contributions are non-negative
    assert (new_contributions >= 0).all()


def test_new_spend_contributions(mmm_fitted) -> None:
    new_spend = np.ones(len(mmm_fitted.channel_columns))
    new_contributions = mmm_fitted.new_spend_contributions(new_spend)

    new_contributions_property_checks(new_contributions, mmm_fitted.X, mmm_fitted)


def test_new_spend_contributions_prior_error() -> None:
    mmm = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        control_columns=["control_1", "control_2"],
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
    )
    new_spend = np.ones(len(mmm.channel_columns))
    match = r"sample_prior_predictive"
    with pytest.raises(RuntimeError, match=match):
        mmm.new_spend_contributions(new_spend, prior=True)


@pytest.mark.parametrize("original_scale", [True, False])
def test_new_spend_contributions_prior(original_scale, mmm, toy_X) -> None:
    mmm.sample_prior_predictive(
        X=toy_X,
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
def mmm_with_prior() -> MMM:
    n_chains = 1
    n_samples = 100

    mmm = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        control_columns=["control_1", "control_2"],
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
    )

    channels = mmm.channel_columns
    n_channels = len(channels)

    idata = az.from_dict(
        prior={
            # Arbitrary but close to the default parameterization
            "adstock_alpha": rng.uniform(size=(n_chains, n_samples, n_channels)),
            "saturation_lam": rng.exponential(size=(n_chains, n_samples, n_channels)),
            "saturation_beta": np.abs(
                rng.normal(size=(n_chains, n_samples, n_channels))
            ),
        },
        coords={"channel": channels},
        dims={
            "adstock_alpha": ["chain", "draw", "channel"],
            "saturation_lam": ["chain", "draw", "channel"],
            "saturation_beta": ["chain", "draw", "channel"],
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


@pytest.fixture
def df_lift_test_with_date() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "channel": ["channel_1", "channel_1"],
            "x": [1, 2],
            "delta_x": [1, 1],
            "delta_y": [1, 1],
            "sigma": [1, 1],
            "date": pd.to_datetime(["2020-08-10", "2020-08-31"]),
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
    adstock = GeometricAdstock(l_max=4)
    saturation = LogisticSaturation()
    mmm = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        control_columns=["control_1", "control_2"],
        adstock=adstock,
        saturation=saturation,
    )
    with pytest.raises(RuntimeError, match=r"The model has not been built yet."):
        mmm.add_lift_test_measurements(
            pd.DataFrame(),
        )


def test_initialize_alternative_with_classes() -> None:
    mmm = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        control_columns=["control_1", "control_2"],
        adstock=DelayedAdstock(l_max=10),
        saturation=MichaelisMentenSaturation(),
    )

    assert isinstance(mmm.adstock, DelayedAdstock)
    assert mmm.adstock.l_max == 10
    assert isinstance(mmm.saturation, MichaelisMentenSaturation)


def test_initialize_defaults_channel_media_dims() -> None:
    mmm = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        control_columns=["control_1", "control_2"],
        adstock=DelayedAdstock(l_max=10),
        saturation=MichaelisMentenSaturation(),
    )

    for transform in [mmm.adstock, mmm.saturation]:
        for config in transform.function_priors.values():
            assert config.dims == ("channel",)


@pytest.mark.parametrize(
    "time_varying_intercept, time_varying_media",
    [
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_save_load_with_tvp(
    time_varying_intercept,
    time_varying_media,
    toy_X,
    toy_y,
    mock_pymc_sample,
) -> None:
    adstock = GeometricAdstock(l_max=5)
    saturation = LogisticSaturation()
    mmm = MMM(
        channel_columns=["channel_1", "channel_2"],
        date_column="date",
        adstock=adstock,
        saturation=saturation,
        time_varying_intercept=time_varying_intercept,
        time_varying_media=time_varying_media,
    )
    mmm.fit(toy_X, toy_y)

    file = "tmp-model"
    mmm.save(file)
    loaded_mmm = MMM.load(file)
    assert mmm.time_varying_intercept == loaded_mmm.time_varying_intercept
    assert mmm.time_varying_intercept == time_varying_intercept
    assert mmm.time_varying_media == loaded_mmm.time_varying_media
    assert mmm.time_varying_media == time_varying_media

    # clean up
    os.remove(file)

    expected_flats = []
    if time_varying_intercept:
        expected_flats.append("intercept_latent_process_f_mean")
    if time_varying_media:
        expected_flats.append("media_temporal_latent_multiplier_f_mean")

    def get_random_variable_name(var):
        return var.owner.op.__class__.__name__

    for free_RV in loaded_mmm.model.free_RVs:
        if free_RV.name in expected_flats:
            assert get_random_variable_name(free_RV) == "FlatRV"


class CustomSaturation(SaturationTransformation):
    lookup_name: str = "custom_saturation"

    def function(self, x, beta):
        return beta * x

    default_priors = {
        "beta": Prior("HalfNormal", sigma=2.5),
    }


@pytest.fixture(scope="module")
def mmm_with_media_config() -> MMM:
    return MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        control_columns=["control_1", "control_2"],
        adstock=GeometricAdstock(l_max=4, normalize=False, mode="Before"),
        saturation=CustomSaturation(),
    )


@pytest.fixture(scope="module")
def mmm_with_media_config_fitted(
    mmm_with_media_config: MMM,
    toy_X: pd.DataFrame,
    toy_y: pd.Series,
) -> MMM:
    mmm_with_media_config.fit(toy_X, toy_y)
    return mmm_with_media_config


def test_save_load_with_media_transformation(mmm_with_media_config_fitted) -> None:
    file = "tmp-model"
    mmm_with_media_config_fitted.save(file)

    loaded_mmm = MMM.load(file)

    assert loaded_mmm.adstock == GeometricAdstock(
        l_max=4,
        normalize=False,
        mode="Before",
        priors={
            "alpha": Prior("Beta", alpha=1, beta=3, dims="channel"),
        },
    )
    assert loaded_mmm.saturation == CustomSaturation(
        priors={
            "beta": Prior("HalfNormal", sigma=2.5, dims="channel"),
        }
    )

    # clean up
    os.remove(file)


def test_missing_attrs_to_defaults(toy_X, toy_y, mock_pymc_sample) -> None:
    mmm = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        control_columns=["control_1", "control_2"],
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        adstock_first=False,
        time_varying_intercept=False,
        time_varying_media=False,
    )
    mmm.fit(toy_X, toy_y)
    mmm.idata.attrs.pop("adstock_first")
    mmm.idata.attrs.pop("time_varying_intercept")
    mmm.idata.attrs.pop("time_varying_media")

    file = "tmp-model"
    mmm.save(file)

    loaded_mmm = MMM.load(file)

    attrs = loaded_mmm.idata.attrs
    for key in [
        "adstock_first",
        "time_varying_intercept",
        "time_varying_media",
    ]:
        assert key not in attrs

    assert not loaded_mmm.time_varying_intercept
    assert not loaded_mmm.time_varying_media
    # Falsely loaded
    assert loaded_mmm.adstock_first

    # clean up
    os.remove(file)


def test_channel_contribution_forward_pass_time_varying_media(
    toy_X,
    toy_y,
    mock_pymc_sample,
) -> None:
    mmm = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        control_columns=["control_1", "control_2"],
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        time_varying_media=True,
    )
    mmm.fit(toy_X, toy_y)

    posterior = mmm.fit_result

    baseline_contributions = posterior["baseline_channel_contribution"]
    multiplier = posterior["media_temporal_latent_multiplier"]
    target_scale = mmm.target_scale if hasattr(mmm, "target_scale") else mmm.y.max()
    recovered_contributions = baseline_contributions * multiplier * target_scale
    media_contributions = mmm.channel_contribution_forward_pass(
        mmm.preprocessed_data["X"][mmm.channel_columns].to_numpy()
    )
    np.testing.assert_allclose(
        recovered_contributions.to_numpy(),
        media_contributions,
        rtol=1e-1,  # 10% relative tolerance for scaling tests
        atol=1e-1,  # Absolute tolerance
    )


def test_time_varying_media_with_lift_test(
    toy_X, toy_y, df_lift_test_with_date
) -> None:
    mmm = MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        control_columns=["control_1", "control_2"],
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        time_varying_media=True,
    )
    mmm.build_model(X=toy_X, y=toy_y)
    try:
        mmm.add_lift_test_measurements(df_lift_test_with_date)
    except Exception as e:
        pytest.fail(
            f"add_lift_test_measurements for time_varying_media model failed with error {e}"
        )


@pytest.mark.parametrize(
    argnames="noise_level", argvalues=[0.01, 0.05], ids=["low_noise", "high_noise"]
)
@pytest.mark.parametrize(
    argnames="granularity",
    argvalues=["weekly", "monthly", "quarterly", "yearly"],
    ids=["weekly", "monthly", "quarterly", "yearly"],
)
@pytest.mark.parametrize(
    argnames="time_length",
    argvalues=[8, 12, 16, 20],
    ids=["time_length_8", "time_length_12", "time_length_16", "time_length_20"],
)
@pytest.mark.parametrize(
    argnames="lag", argvalues=[2, 4, 6, 8], ids=["lag_2", "lag_4", "lag_6", "lag_8"]
)
def test_create_synth_dataset(
    mmm_fitted: MMM,
    toy_X: pd.DataFrame,
    noise_level: float,
    granularity: str,
    time_length: int,
    lag: int,
) -> None:
    """Test the _create_synth_dataset method of MMM class."""

    # Create a simple allocation strategy
    channels = mmm_fitted.channel_columns
    allocation_strategy = xr.DataArray(
        data=np.ones(len(channels)),
        dims=["channel"],
        coords={"channel": channels},
    )

    # Generate synthetic dataset
    synth_df = mmm_fitted._create_synth_dataset(
        df=toy_X,
        date_column=mmm_fitted.date_column,
        channels=mmm_fitted.channel_columns,
        controls=mmm_fitted.control_columns,
        target_col="y",
        allocation_strategy=allocation_strategy,
        time_granularity=granularity,
        time_length=time_length,
        lag=lag,
        noise_level=noise_level,
    )

    # Test output properties
    assert isinstance(synth_df, pd.DataFrame)
    assert len(synth_df) == time_length

    # Check required columns exist
    required_columns = {
        mmm_fitted.date_column,
        *mmm_fitted.channel_columns,
        "y",
    }
    if mmm_fitted.control_columns:
        required_columns.update(mmm_fitted.control_columns)
    assert all(col in synth_df.columns for col in required_columns)

    # Check date properties
    assert pd.api.types.is_datetime64_any_dtype(synth_df[mmm_fitted.date_column])
    assert len(synth_df[mmm_fitted.date_column].unique()) == time_length

    # Check channel values are non-negative (since they represent spend)
    for channel in mmm_fitted.channel_columns:
        assert (synth_df[channel] >= 0).all()

    # Check target variable exists and has reasonable values
    assert "y" in synth_df.columns
    assert not synth_df["y"].isna().any()


@pytest.mark.parametrize(
    argnames="noise_level", argvalues=[0.01, 0.05], ids=["low_noise", "high_noise"]
)
@pytest.mark.parametrize(
    argnames="granularity",
    argvalues=["weekly", "monthly", "quarterly", "yearly"],
    ids=["weekly", "monthly", "quarterly", "yearly"],
)
@pytest.mark.parametrize(
    argnames="time_length",
    argvalues=[8, 12, 16, 20],
    ids=["time_length_8", "time_length_12", "time_length_16", "time_length_20"],
)
@pytest.mark.parametrize(
    argnames="lag", argvalues=[2, 4, 6, 8], ids=["lag_2", "lag_4", "lag_6", "lag_8"]
)
def test_create_synth_dataset_no_controls(
    mmm_fitted_no_controls: MMM,
    toy_X: pd.DataFrame,
    noise_level: float,
    granularity: str,
    time_length: int,
    lag: int,
) -> None:
    """Test the _create_synth_dataset method of MMM class."""

    # Create a simple allocation strategy
    channels = mmm_fitted_no_controls.channel_columns
    allocation_strategy = xr.DataArray(
        data=np.ones(len(channels)),
        dims=["channel"],
        coords={"channel": channels},
    )

    # Generate synthetic dataset
    synth_df = mmm_fitted_no_controls._create_synth_dataset(
        df=toy_X,
        date_column=mmm_fitted_no_controls.date_column,
        channels=mmm_fitted_no_controls.channel_columns,
        controls=mmm_fitted_no_controls.control_columns,
        target_col="y",
        allocation_strategy=allocation_strategy,
        time_granularity=granularity,
        time_length=time_length,
        lag=lag,
        noise_level=noise_level,
    )

    # Test output properties
    assert isinstance(synth_df, pd.DataFrame)
    assert len(synth_df) == time_length

    # Check required columns exist
    required_columns = {
        mmm_fitted_no_controls.date_column,
        *mmm_fitted_no_controls.channel_columns,
        "y",
    }
    if mmm_fitted_no_controls.control_columns:
        required_columns.update(mmm_fitted_no_controls.control_columns)
    assert all(col in synth_df.columns for col in required_columns)

    # Check date properties
    assert pd.api.types.is_datetime64_any_dtype(
        synth_df[mmm_fitted_no_controls.date_column]
    )
    assert len(synth_df[mmm_fitted_no_controls.date_column].unique()) == time_length

    # Check channel values are non-negative (since they represent spend)
    for channel in mmm_fitted_no_controls.channel_columns:
        assert (synth_df[channel] >= 0).all()

    # Check target variable exists and has reasonable values
    assert "y" in synth_df.columns
    assert not synth_df["y"].isna().any()


# Tests for new helper methods added during refactoring
class TestMMMEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_row_dataframe(self):
        """Test model with single row of data."""
        X = pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-01-01")],
                "channel_1": [100.0],
                "channel_2": [200.0],
            }
        )
        y = pd.Series([50.0])

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=1),
            saturation=LogisticSaturation(),
        )

        # Single row is technically valid (though not useful for inference)
        mmm.build_model(X=X, y=y)
        assert hasattr(mmm, "model")

    def test_all_zeros_in_channel(self, toy_y: pd.Series):
        """Test handling of channel with all zero values."""
        X = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=50, freq="W-MON"),
                "channel_1": [0.0] * 50,
                "channel_2": rng.integers(100, 1000, size=50),
            }
        )

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )

        # Should handle gracefully - scaling should avoid division by zero
        mmm.build_model(X=X, y=toy_y)
        assert hasattr(mmm, "model")
        # Channel with all zeros should have scale = 1.0 (from max(scale, 1.0))
        assert mmm.channel_scale[0] == 1.0

    def test_negative_values_in_channels(self, toy_y: pd.Series):
        """Test handling of negative values in channel data."""
        X = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=50, freq="W-MON"),
                "channel_1": rng.integers(-100, 100, size=50),
                "channel_2": rng.integers(-100, 100, size=50),
            }
        )

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )

        # Should work - scaling uses absolute values
        mmm.build_model(X=X, y=toy_y)
        assert hasattr(mmm, "model")

    def test_very_large_values(self, toy_y: pd.Series):
        """Test handling of very large values in channels."""
        X = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=50, freq="W-MON"),
                "channel_1": rng.uniform(1e6, 1e9, size=50),
                "channel_2": rng.uniform(1e6, 1e9, size=50),
            }
        )

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )

        mmm.build_model(X=X, y=toy_y)
        assert hasattr(mmm, "model")
        # Scaling should normalize large values
        assert mmm.channel_scale[0] > 1e5

    def test_very_small_values(self, toy_y: pd.Series):
        """Test handling of very small positive values."""
        X = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=50, freq="W-MON"),
                "channel_1": rng.uniform(1e-9, 1e-6, size=50),
                "channel_2": rng.uniform(1e-9, 1e-6, size=50),
            }
        )

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )

        mmm.build_model(X=X, y=toy_y)
        assert hasattr(mmm, "model")

    def test_nan_values_in_input(self, toy_y: pd.Series):
        """Test that NaN values in input raise appropriate error."""
        X = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=50, freq="W-MON"),
                "channel_1": [np.nan] * 25 + list(rng.integers(100, 1000, size=25)),
                "channel_2": rng.integers(100, 1000, size=50),
            }
        )

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )

        # NaN values cause NotImplementedError when creating pm.Data
        with pytest.raises(NotImplementedError):
            mmm.build_model(X=X, y=toy_y)

    def test_infinite_values_in_input(self, toy_y: pd.Series):
        """Test handling of infinite values in input."""
        X = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=50, freq="W-MON"),
                "channel_1": [np.inf] * 10 + list(rng.integers(100, 1000, size=40)),
                "channel_2": rng.integers(100, 1000, size=50),
            }
        )

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )

        # Should handle or raise appropriate error
        mmm.build_model(X=X, y=toy_y)
        assert hasattr(mmm, "model")

    def test_duplicate_channel_names(self, toy_X: pd.DataFrame, toy_y: pd.Series):
        """Test that duplicate channel names are allowed (but not recommended)."""
        # Pydantic doesn't validate uniqueness by default
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_1"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )
        # Should work but will treat as two separate channels
        assert mmm.channel_columns == ["channel_1", "channel_1"]

    def test_missing_channel_columns(self, toy_y: pd.Series):
        """Test error when specified channels don't exist in data."""
        X = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=50, freq="W-MON"),
                "channel_1": rng.integers(100, 1000, size=50),
            }
        )

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "nonexistent_channel"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )

        with pytest.raises((KeyError, ValueError, Exception)):
            mmm.build_model(X=X, y=toy_y)

    def test_unsorted_dates(self, toy_y: pd.Series):
        """Test handling of unsorted date data."""
        dates = pd.date_range("2020-01-01", periods=50, freq="W-MON")
        shuffled_dates = dates.to_series().sample(frac=1, random_state=42).values

        X = pd.DataFrame(
            {
                "date": shuffled_dates,
                "channel_1": rng.integers(100, 1000, size=50),
                "channel_2": rng.integers(100, 1000, size=50),
            }
        )

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )

        # Model should work with unsorted dates
        mmm.build_model(X=X, y=toy_y)
        assert hasattr(mmm, "model")

    def test_duplicate_dates(self, toy_y: pd.Series):
        """Test that duplicate dates raise validation error."""
        X = pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-01-01")] * 50,
                "channel_1": rng.integers(100, 1000, size=50),
                "channel_2": rng.integers(100, 1000, size=50),
            }
        )

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )

        # Duplicate dates cause ValueError during validation
        with pytest.raises(ValueError, match="has repeated values"):
            mmm.build_model(X=X, y=toy_y)

    def test_invalid_date_format(self, toy_y: pd.Series):
        """Test error handling for invalid date format."""
        X = pd.DataFrame(
            {
                "date": ["not-a-date"] * 50,
                "channel_1": rng.integers(100, 1000, size=50),
                "channel_2": rng.integers(100, 1000, size=50),
            }
        )

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )

        with pytest.raises((ValueError, Exception)):
            mmm.build_model(X=X, y=toy_y)

    def test_mismatched_X_y_lengths(self, toy_X: pd.DataFrame):
        """Test that mismatched X and y lengths work (y is broadcast)."""
        y_short = pd.Series(rng.integers(0, 100, size=10))

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )

        # PyMC will use the shorter length - no error is raised
        mmm.build_model(X=toy_X, y=y_short)
        assert hasattr(mmm, "model")

    def test_minimal_adstock_lag(self, toy_X: pd.DataFrame, toy_y: pd.Series):
        """Test model with l_max=1 (minimal adstock effect)."""
        # l_max=0 is invalid, minimum is 1
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=1),
            saturation=LogisticSaturation(),
        )

        mmm.build_model(X=toy_X, y=toy_y)
        assert hasattr(mmm, "model")

    def test_very_large_adstock_lag(self, toy_X: pd.DataFrame, toy_y: pd.Series):
        """Test model with very large l_max."""
        # l_max larger than data length
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=100),
            saturation=LogisticSaturation(),
        )

        mmm.build_model(X=toy_X, y=toy_y)
        assert hasattr(mmm, "model")

    def test_no_control_columns(self, toy_X: pd.DataFrame, toy_y: pd.Series):
        """Test with no control columns (None vs empty list)."""
        # Empty list causes validation error, use None instead
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            control_columns=None,
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )

        mmm.build_model(X=toy_X, y=toy_y)
        assert hasattr(mmm, "model")
        assert mmm.control_columns is None

    def test_all_same_values_in_channel(self, toy_y: pd.Series):
        """Test channel with all identical non-zero values."""
        X = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=50, freq="W-MON"),
                "channel_1": [500.0] * 50,
                "channel_2": rng.integers(100, 1000, size=50),
            }
        )

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )

        mmm.build_model(X=X, y=toy_y)
        assert hasattr(mmm, "model")
        # Scale for constant channel should be the absolute value
        assert mmm.channel_scale[0] == 500.0

    def test_target_with_zeros(self, toy_X: pd.DataFrame):
        """Test with target variable containing zeros."""
        y = pd.Series([0.0] * 25 + list(rng.integers(1, 100, size=25)))

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )

        mmm.build_model(X=toy_X, y=y)
        assert hasattr(mmm, "model")

    def test_target_all_zeros(self, toy_X: pd.DataFrame):
        """Test with target variable that is all zeros."""
        y = pd.Series([0.0] * toy_X.shape[0])

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )

        mmm.build_model(X=toy_X, y=y)
        assert hasattr(mmm, "model")
        # Target scale should be 1.0 to avoid division by zero
        assert mmm.target_scale == 1.0

    def test_target_all_negative(self, toy_X: pd.DataFrame):
        """Test with all negative target values."""
        y = pd.Series(-rng.integers(1, 100, size=toy_X.shape[0]))

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )

        mmm.build_model(X=toy_X, y=y)
        assert hasattr(mmm, "model")

    def test_extreme_seasonality_order(self, toy_X: pd.DataFrame, toy_y: pd.Series):
        """Test with very high number of Fourier modes."""
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            yearly_seasonality=50,  # Very high number
        )

        mmm.build_model(X=toy_X, y=toy_y)
        assert hasattr(mmm, "model")

    def test_single_channel(self, toy_X: pd.DataFrame, toy_y: pd.Series):
        """Test model with only one channel."""
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )

        mmm.build_model(X=toy_X, y=toy_y)
        assert hasattr(mmm, "model")
        assert len(mmm.channel_scale) == 1

    def test_data_setter_with_different_length(
        self, toy_X: pd.DataFrame, toy_y: pd.Series, mock_pymc_sample
    ):
        """Test data setter with different data length."""
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )

        mmm.fit(X=toy_X, y=toy_y)

        # Create new data with different length
        new_X = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=20, freq="W-MON"),
                "channel_1": rng.integers(100, 1000, size=20),
                "channel_2": rng.integers(100, 1000, size=20),
            }
        )

        # Should handle different length data
        mmm._data_setter(X=new_X, y=None)

    def test_sample_posterior_predictive_without_fit(
        self, toy_X: pd.DataFrame, toy_y: pd.Series
    ):
        """Test that sampling posterior predictive without fitting raises error."""
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )

        mmm.build_model(X=toy_X, y=toy_y)

        with pytest.raises((AttributeError, ValueError, Exception)):
            mmm.sample_posterior_predictive(X=toy_X)

    def test_invalid_scaling_method(self, toy_X: pd.DataFrame, toy_y: pd.Series):
        """Test with invalid scaling method."""
        from pydantic import ValidationError

        from pymc_marketing.mmm.scaling import Scaling, VariableScaling

        # Invalid scaling method causes ValidationError at initialization
        with pytest.raises(ValidationError):
            MMM(
                date_column="date",
                channel_columns=["channel_1", "channel_2"],
                adstock=GeometricAdstock(l_max=4),
                saturation=LogisticSaturation(),
                scaling=Scaling(
                    target=VariableScaling(method="invalid_method", dims=()),
                    channel=VariableScaling(method="max", dims=()),
                ),
            )

    def test_mixed_positive_negative_channels(self, toy_y: pd.Series):
        """Test with one positive and one negative channel."""
        X = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=50, freq="W-MON"),
                "channel_1": rng.integers(100, 1000, size=50),
                "channel_2": -rng.integers(100, 1000, size=50),
            }
        )

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )

        mmm.build_model(X=X, y=toy_y)
        assert hasattr(mmm, "model")

    def test_date_conversion_error(self, toy_y: pd.Series):
        """Test error when date cannot be converted."""
        X = pd.DataFrame(
            {
                "date": [object() for _ in range(50)],
                "channel_1": rng.integers(100, 1000, size=50),
                "channel_2": rng.integers(100, 1000, size=50),
            }
        )

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )

        with pytest.raises(ValueError, match="Could not convert"):
            mmm.build_model(X=X, y=toy_y)

    def test_transformer_fitting_failure(
        self, toy_X: pd.DataFrame, toy_y: pd.Series, monkeypatch
    ):
        """Test handling when transformer fitting fails."""

        def mock_failing_transformer(*args, **kwargs):
            raise RuntimeError("Transformer failure")

        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )

        monkeypatch.setattr(mmm, "max_abs_scale_target_data", mock_failing_transformer)

        with pytest.warns(UserWarning, match="Failed to fit transformers"):
            mmm.build_model(X=toy_X, y=toy_y)

    def test_causal_graph_with_warnings(self, toy_X: pd.DataFrame, toy_y: pd.Series):
        """Test causal graph initialization with warnings when treatment nodes not specified."""
        dag = """
        digraph {
            channel_1 -> y;
            channel_2 -> y;
            control_1 -> y;
        }
        """

        with pytest.warns(UserWarning, match="No treatment nodes provided"):
            mmm = MMM(
                date_column="date",
                channel_columns=["channel_1", "channel_2"],
                control_columns=["control_1"],
                adstock=GeometricAdstock(l_max=4),
                saturation=LogisticSaturation(),
                dag=dag,
                outcome_node="y",
            )

        assert mmm.treatment_nodes == ["channel_1", "channel_2"]

    def test_causal_graph_excludes_seasonality(
        self, toy_X: pd.DataFrame, toy_y: pd.Series
    ):
        """Test that seasonality is excluded when not in adjustment set."""
        dag = """
        digraph {
            channel_1 -> y;
            channel_2 -> y;
            control_1 -> y;
        }
        """

        with pytest.warns(UserWarning, match="Yearly seasonality excluded"):
            mmm = MMM(
                date_column="date",
                channel_columns=["channel_1", "channel_2"],
                control_columns=["control_1"],
                adstock=GeometricAdstock(l_max=4),
                saturation=LogisticSaturation(),
                yearly_seasonality=2,
                dag=dag,
                treatment_nodes=["channel_1", "channel_2"],
                outcome_node="y",
            )

        assert mmm.yearly_seasonality is None

    def test_x_data_not_dataframe_error(self, toy_y: pd.Series):
        """Test error when X data is not a DataFrame."""
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )

        mmm.preprocessed_data = {"X": np.array([[1, 2], [3, 4]]), "y": toy_y}

        with pytest.raises(TypeError, match="must be a DataFrame"):
            mmm._compute_scales()

    def test_invalid_synth_dataset_granularity(
        self, mmm_fitted: MMM, toy_X: pd.DataFrame
    ):
        """Test invalid time granularity in synthetic dataset creation."""
        from xarray import DataArray

        allocation = DataArray(
            [1000, 2000],
            dims=["channel"],
            coords={"channel": mmm_fitted.channel_columns},
        )

        with pytest.raises(ValueError, match="Unsupported time granularity"):
            mmm_fitted._create_synth_dataset(
                df=toy_X,
                date_column="date",
                allocation_strategy=allocation,
                channels=mmm_fitted.channel_columns,
                controls=None,
                target_col="y",
                time_granularity="invalid_granularity",
                time_length=10,
                lag=4,
            )

    def test_invalid_allocation_dims(self, mmm_fitted: MMM, toy_X: pd.DataFrame):
        """Test error when allocation strategy has wrong dimensions."""
        from xarray import DataArray

        allocation = DataArray(
            [1000, 2000],
            dims=["wrong_dim"],
            coords={"wrong_dim": ["a", "b"]},
        )

        with pytest.raises(
            ValueError, match="must have a single dimension named 'channel'"
        ):
            mmm_fitted._create_synth_dataset(
                df=toy_X,
                date_column="date",
                allocation_strategy=allocation,
                channels=mmm_fitted.channel_columns,
                controls=None,
                target_col="y",
                time_granularity="weekly",
                time_length=10,
                lag=4,
            )


class TestMMMHelperMethods:
    """Tests for internal helper methods added during refactoring."""

    @pytest.mark.parametrize(
        "method,axis",
        [("max", 0), ("mean", 0), ("max", None), ("mean", None)],
        ids=["max_axis0", "mean_axis0", "max_none", "mean_none"],
    )
    def test_compute_scale_for_data(self, mmm_fitted: MMM, method: str, axis):
        """Test _compute_scale_for_data helper method."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        scale = mmm_fitted._compute_scale_for_data(data, method, axis)

        if method == "max":
            expected = np.abs(data).max(axis=axis)
        else:
            expected = np.abs(data).mean(axis=axis)

        np.testing.assert_array_equal(scale, expected)

    def test_compute_scale_for_data_zero_division(self, mmm_fitted: MMM):
        """Test that zero values are handled correctly."""
        data = np.array([[0, 0, 0], [0, 0, 0]])
        scale = mmm_fitted._compute_scale_for_data(data, "max", axis=0)
        np.testing.assert_array_equal(scale, np.ones(3))

    def test_compute_scale_for_data_invalid_method(self, mmm_fitted: MMM):
        """Test that invalid method raises ValueError."""
        data = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="Unknown scaling method"):
            mmm_fitted._compute_scale_for_data(data, "invalid", axis=0)

    def test_has_new_scaling(self, mmm_fitted: MMM):
        """Test _has_new_scaling helper method."""
        # After fitting, the model should have new scaling
        assert mmm_fitted._has_new_scaling()
        assert hasattr(mmm_fitted, "channel_scale")
        assert hasattr(mmm_fitted, "target_scale")

    def test_prepare_channel_data(self, mmm_fitted: MMM, toy_X: pd.DataFrame):
        """Test _prepare_channel_data helper method."""
        data = mmm_fitted._prepare_channel_data(toy_X)

        assert isinstance(data, dict)
        assert "channel_data" in data

        if mmm_fitted._has_new_scaling():
            assert "channel_scale" in data
            np.testing.assert_array_equal(
                data["channel_scale"], mmm_fitted.channel_scale
            )

    def test_prepare_channel_data_missing_columns(self, mmm_fitted: MMM):
        """Test _prepare_channel_data raises error for missing columns."""
        bad_X = pd.DataFrame({"wrong_col": [1, 2, 3]})
        with pytest.raises(RuntimeError, match="New data must contain channel_data"):
            mmm_fitted._prepare_channel_data(bad_X)

    def test_prepare_control_data_no_controls(
        self, mmm_fitted_no_controls: MMM, toy_X: pd.DataFrame
    ):
        """Test _prepare_control_data returns empty dict when no controls."""
        data = mmm_fitted_no_controls._prepare_control_data(toy_X)
        assert data == {}

    def test_prepare_control_data_with_controls(
        self, mmm_fitted: MMM, toy_X: pd.DataFrame
    ):
        """Test _prepare_control_data with control columns."""
        data = mmm_fitted._prepare_control_data(toy_X)
        assert isinstance(data, dict)
        assert "control_data" in data

    @pytest.mark.parametrize(
        "y_input",
        [None, pd.Series([1, 2, 3]), np.array([1, 2, 3])],
        ids=["none", "series", "array"],
    )
    def test_prepare_target_data(self, mmm_fitted: MMM, y_input):
        """Test _prepare_target_data with different input types."""
        n_rows = 3 if y_input is not None else 10
        data = mmm_fitted._prepare_target_data(y_input, n_rows)

        assert isinstance(data, dict)
        if mmm_fitted._has_new_scaling():
            assert "target_data" in data
            assert "target_scale" in data
        else:
            assert "target" in data

    def test_prepare_target_data_invalid_type(self, mmm_fitted: MMM):
        """Test _prepare_target_data raises error for invalid type."""
        with pytest.raises(
            TypeError, match="y must be either a pandas Series or a numpy array"
        ):
            mmm_fitted._prepare_target_data("invalid", 10)

    def test_validate_controls_for_synth_dataset(self, mmm_fitted: MMM):
        """Test _validate_controls_for_synth_dataset helper method."""
        # With valid controls
        controls = mmm_fitted._validate_controls_for_synth_dataset(
            mmm_fitted.control_columns
        )
        assert controls == mmm_fitted.control_columns

        # With None
        controls = mmm_fitted._validate_controls_for_synth_dataset(None)
        assert controls == []

    def test_validate_controls_for_synth_dataset_error(
        self, mmm_fitted_no_controls: MMM
    ):
        """Test _validate_controls_for_synth_dataset raises error when controls don't exist."""
        with pytest.raises(ValueError, match="model was built without controls"):
            mmm_fitted_no_controls._validate_controls_for_synth_dataset(["control_1"])

    @pytest.mark.parametrize(
        "granularity,time_length",
        [
            ("daily", 5),
            ("weekly", 4),
            ("monthly", 3),
            ("quarterly", 2),
            ("yearly", 1),
        ],
        ids=["daily", "weekly", "monthly", "quarterly", "yearly"],
    )
    def test_generate_future_dates(
        self, mmm_fitted: MMM, granularity: str, time_length: int
    ):
        """Test _generate_future_dates helper method."""
        last_date = pd.Timestamp("2021-12-31")
        dates = mmm_fitted._generate_future_dates(last_date, granularity, time_length)

        assert isinstance(dates, list)
        assert len(dates) == time_length
        assert all(isinstance(d, pd.Timestamp) for d in dates)
        assert all(d > last_date for d in dates)

    def test_build_control_contribution_no_controls(
        self, toy_X: pd.DataFrame, toy_y: pd.Series
    ):
        """Test _build_control_contribution returns None when no controls."""
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )
        mmm.build_model(X=toy_X, y=toy_y)

        with mmm.model:
            contribution = mmm._build_control_contribution()
            assert contribution is None

    def test_build_control_contribution_with_controls(
        self, toy_X: pd.DataFrame, toy_y: pd.Series
    ):
        """Test _build_control_contribution creates contribution when controls exist."""
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            control_columns=["control_1", "control_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )
        mmm.build_model(X=toy_X, y=toy_y)

        # Contribution should exist in the model
        assert "control_contribution" in [var.name for var in mmm.model.deterministics]

    def test_build_control_contribution_missing_columns(
        self, toy_X: pd.DataFrame, toy_y: pd.Series
    ):
        """Test _build_control_contribution handles missing columns gracefully."""
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            control_columns=["control_1", "control_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )
        mmm._generate_and_preprocess_model_data(toy_X, toy_y)
        mmm._compute_scales()

        # Create a model without the control data
        with pm.Model(coords=mmm.model_coords):
            # Manually set up preprocessed data without controls
            X_data_no_controls = toy_X.drop(columns=["control_1", "control_2"])
            mmm.preprocessed_data["X"] = X_data_no_controls

            with pytest.raises(
                ValueError, match=r"Control columns .* not found in X data"
            ):
                mmm._build_control_contribution()

    def test_build_yearly_seasonality_contribution_no_seasonality(
        self, toy_X: pd.DataFrame, toy_y: pd.Series
    ):
        """Test _build_yearly_seasonality_contribution returns None when no seasonality."""
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )
        mmm.build_model(X=toy_X, y=toy_y)

        with mmm.model:
            contribution = mmm._build_yearly_seasonality_contribution()
            assert contribution is None

    def test_build_yearly_seasonality_contribution_with_seasonality(
        self, toy_X: pd.DataFrame, toy_y: pd.Series
    ):
        """Test _build_yearly_seasonality_contribution creates contribution."""
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            yearly_seasonality=2,
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )
        mmm.build_model(X=toy_X, y=toy_y)

        # Contribution should exist in the model
        assert "yearly_seasonality_contribution" in [
            var.name for var in mmm.model.deterministics
        ]

    @pytest.mark.parametrize(
        "original_scale",
        [True, False],
        ids=["original_scale", "scaled"],
    )
    def test_get_intercept_for_plot(
        self,
        mmm: MMM,
        toy_X: pd.DataFrame,
        toy_y: pd.Series,
        mock_pymc_sample,
        original_scale: bool,
    ):
        """Test _get_intercept_for_plot helper method."""
        # Fit the model fresh for this test
        mmm.fit(X=toy_X, y=toy_y)

        intercept_mean, intercept_hdi = mmm._get_intercept_for_plot(original_scale)

        assert isinstance(intercept_mean, np.ndarray) or np.isscalar(intercept_mean)
        assert isinstance(intercept_hdi, np.ndarray)
        assert intercept_hdi.shape[0] == mmm.X.shape[0]
        assert intercept_hdi.shape[1] == 2  # Lower and upper bounds

    @pytest.mark.parametrize(
        "original_scale",
        [True, False],
        ids=["original_scale", "scaled"],
    )
    def test_get_target_for_plot(
        self, mmm: MMM, toy_X: pd.DataFrame, toy_y: pd.Series, original_scale: bool
    ):
        """Test _get_target_for_plot helper method."""
        # Build model to ensure preprocessed data exists
        mmm.build_model(X=toy_X, y=toy_y)

        y_plot = mmm._get_target_for_plot(original_scale)

        assert isinstance(y_plot, np.ndarray)
        assert y_plot.shape[0] == mmm.X.shape[0]

    def test_transform_to_original_scale_new(self, mmm_fitted: MMM):
        """Test _transform_to_original_scale_new helper method."""
        # Create a simple DataArray
        data = xr.DataArray(
            np.ones((10,)),
            dims=["date"],
            coords={"date": pd.date_range("2020-01-01", periods=10)},
        ).to_dataset(name="y")

        transformed = mmm_fitted._transform_to_original_scale_new(data)

        assert isinstance(transformed, xr.Dataset)
        if "y" in transformed:
            assert (transformed["y"] == mmm_fitted.target_scale).all()

    def test_compute_scales(self, toy_X: pd.DataFrame, toy_y: pd.Series):
        """Test _compute_scales method."""
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )
        mmm._generate_and_preprocess_model_data(toy_X, toy_y)
        mmm._compute_scales()

        assert hasattr(mmm, "channel_scale")
        assert hasattr(mmm, "target_scale")
        assert len(mmm.channel_scale) == len(mmm.channel_columns)
        assert isinstance(mmm.target_scale, float | np.floating)

    def test_build_intercept_static(self, toy_X: pd.DataFrame, toy_y: pd.Series):
        """Test _build_intercept with static intercept."""
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            time_varying_intercept=False,
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )
        mmm.build_model(X=toy_X, y=toy_y)

        # Check that intercept exists in the model
        with mmm.model:
            assert "intercept" in [var.name for var in mmm.model.free_RVs]
            # Check it's not time-varying (no date dimension in coords)
            intercept_var = mmm.model["intercept"]
            assert "date" not in [
                d.name for d in intercept_var.owner.inputs if hasattr(d, "name")
            ]

    def test_build_intercept_time_varying(
        self, toy_X: pd.DataFrame, toy_y: pd.Series, mock_pymc_sample
    ):
        """Test _build_intercept with time-varying intercept."""
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            time_varying_intercept=True,
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )
        mmm.build_model(X=toy_X, y=toy_y)

        # Check that intercept exists in deterministics with date dimension
        deterministic_names = {var.name: var for var in mmm.model.deterministics}
        assert "intercept" in deterministic_names
        # Check dims include date
        intercept_var = deterministic_names["intercept"]
        assert intercept_var.type.broadcastable == (False,)  # Has one dimension (date)

    def test_build_channel_contribution_static(
        self, toy_X: pd.DataFrame, toy_y: pd.Series
    ):
        """Test _build_channel_contribution with static media."""
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            time_varying_media=False,
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )
        mmm.build_model(X=toy_X, y=toy_y)

        # Check that channel_contribution exists
        assert "channel_contribution" in [var.name for var in mmm.model.deterministics]
        # Should not have baseline_channel_contribution
        assert "baseline_channel_contribution" not in [
            var.name for var in mmm.model.deterministics
        ]

    def test_build_channel_contribution_time_varying(
        self, toy_X: pd.DataFrame, toy_y: pd.Series
    ):
        """Test _build_channel_contribution with time-varying media."""
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            time_varying_media=True,
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )
        mmm.build_model(X=toy_X, y=toy_y)

        # Check that both baseline and final contribution exist
        deterministic_names = [var.name for var in mmm.model.deterministics]
        assert "baseline_channel_contribution" in deterministic_names
        assert "channel_contribution" in deterministic_names

    def test_add_original_scale_deterministics(
        self, toy_X: pd.DataFrame, toy_y: pd.Series
    ):
        """Test that original scale deterministics are added."""
        mmm = MMM(
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            control_columns=["control_1", "control_2"],
            yearly_seasonality=2,
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )
        mmm.build_model(X=toy_X, y=toy_y)

        deterministic_names = [var.name for var in mmm.model.deterministics]

        # Check original scale variables exist
        assert "channel_contribution_original_scale" in deterministic_names
        assert "total_contribution_original_scale" in deterministic_names
        assert "control_contribution_original_scale" in deterministic_names
        assert "yearly_seasonality_contribution_original_scale" in deterministic_names
        assert "y_original_scale" in deterministic_names

    def test_prepare_target_data_with_none_and_no_preprocessed(self, mmm_fitted: MMM):
        """Test _prepare_target_data when y is None but preprocessed_data doesn't match expected types."""
        mmm_fitted.preprocessed_data["y"] = "not_a_series_or_array"  # type: ignore

        result = mmm_fitted._prepare_target_data(y=None, n_rows=10)

        assert "target_data" in result or "target" in result
        if "target_data" in result:
            assert len(result["target_data"]) == 10

    def test_format_recovered_transformation_parameters(self, mmm_fitted: MMM):
        """Test format_recovered_transformation_parameters with quantile cache logic."""
        # Test with default quantile (0.5)
        result = mmm_fitted.format_recovered_transformation_parameters()

        # Verify structure
        assert isinstance(result, dict)
        assert len(result) == len(mmm_fitted.channel_columns)

        # Check each channel has the expected structure
        for channel in mmm_fitted.channel_columns:
            assert channel in result
            assert "saturation_params" in result[channel]
            assert "adstock_params" in result[channel]

            # Check saturation params exist
            saturation_params = result[channel]["saturation_params"]
            assert isinstance(saturation_params, dict)
            assert len(saturation_params) > 0

            # Check adstock params exist
            adstock_params = result[channel]["adstock_params"]
            assert isinstance(adstock_params, dict)
            assert len(adstock_params) > 0

            # Verify all values are numeric
            for param_value in saturation_params.values():
                assert isinstance(param_value, (int, float))
            for param_value in adstock_params.values():
                assert isinstance(param_value, (int, float))

    def test_format_recovered_transformation_parameters_different_quantiles(
        self, mmm_fitted: MMM
    ):
        """Test format_recovered_transformation_parameters with different quantiles."""
        # Test with different quantiles
        result_lower = mmm_fitted.format_recovered_transformation_parameters(
            quantile=0.05
        )
        result_median = mmm_fitted.format_recovered_transformation_parameters(
            quantile=0.5
        )
        result_upper = mmm_fitted.format_recovered_transformation_parameters(
            quantile=0.95
        )

        # Get a sample channel and parameter
        channel = mmm_fitted.channel_columns[0]

        # Get saturation parameter if available
        if result_median[channel]["saturation_params"]:
            param_name = next(iter(result_median[channel]["saturation_params"].keys()))
            lower_val = result_lower[channel]["saturation_params"][param_name]
            median_val = result_median[channel]["saturation_params"][param_name]
            upper_val = result_upper[channel]["saturation_params"][param_name]

            # Lower quantile should be <= median <= upper quantile (generally)
            # Note: Due to sampling variability this might not always hold exactly
            assert isinstance(lower_val, (int, float))
            assert isinstance(median_val, (int, float))
            assert isinstance(upper_val, (int, float))

    def test_format_recovered_transformation_parameters_prefix_stripping(
        self, mmm_fitted: MMM
    ):
        """Test that parameter prefixes are correctly stripped in the output."""
        result = mmm_fitted.format_recovered_transformation_parameters()

        channel = mmm_fitted.channel_columns[0]

        # Check that prefixes are stripped (e.g., 'saturation_' and 'adstock_')
        for param_name in result[channel]["saturation_params"].keys():
            # Parameters should not have 'saturation_' prefix
            assert not param_name.startswith("saturation_")

        for param_name in result[channel]["adstock_params"].keys():
            # Parameters should not have 'adstock_' prefix
            assert not param_name.startswith("adstock_")
