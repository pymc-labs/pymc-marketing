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
import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytest
from matplotlib import pyplot as plt

from pymc_marketing.mmm.components.adstock import GeometricAdstock
from pymc_marketing.mmm.components.saturation import (
    LogisticSaturation,
    MichaelisMentenSaturation,
)
from pymc_marketing.mmm.mmm import MMM, BaseMMM
from pymc_marketing.mmm.preprocessing import MaxAbsScaleTarget

seed: int = sum(map(ord, "pymc_marketing"))
rng: np.random.Generator = np.random.default_rng(seed=seed)


@pytest.fixture(scope="module", params=[0, 42], ids=["seed_0", "seed_42"])
def toy_X(request) -> pd.DataFrame:
    local_rng: np.random.Generator = np.random.default_rng(seed=request.param)
    date_data: pd.DatetimeIndex = pd.date_range(
        start="2019-06-01", end="2021-12-31", freq="W-MON"
    )

    n: int = date_data.size

    return pd.DataFrame(
        data={
            "date": date_data,
            "channel_1": local_rng.integers(low=0, high=400, size=n),
            "channel_2": local_rng.integers(low=0, high=50, size=n),
            "control_1": local_rng.gamma(shape=1000, scale=500, size=n),
            "control_2": local_rng.normal(loc=0, scale=2, size=n),
            "other_column_1": local_rng.integers(low=0, high=100, size=n),
            "other_column_2": local_rng.normal(loc=0, scale=1, size=n),
        }
    )


@pytest.fixture(scope="module")
def toy_y(toy_X) -> pd.Series:
    return pd.Series(rng.integers(low=0, high=100, size=toy_X.shape[0]))


def mock_fit_base(model, X: pd.DataFrame, y: np.ndarray, **kwargs):
    model.build_model(X=X, y=y)
    with model.model:
        idata = pm.sample_prior_predictive(random_seed=rng, **kwargs)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="The group fit_data is not defined in the InferenceData scheme",
        )
        idata.add_groups(
            {
                "posterior": idata.prior,
                "fit_data": pd.concat(
                    [X, pd.Series(y, index=X.index, name="y")], axis=1
                ).to_xarray(),
            }
        )
    model.idata = idata
    model.set_idata_attrs(idata=idata)

    return model


class TestBasePlotting:
    @pytest.fixture(
        scope="module",
        params=[
            "without_controls-default_transform",
            "with_controls-default_transform",
            "without_controls-target_transform",
            "with_controls-target_transform",
        ],
    )
    def plotting_mmm(self, request, toy_X, toy_y):
        control, transform = request.param.split("-")
        if transform == "default_transform":

            class ToyMMM(BaseMMM):
                pass

        elif transform == "target_transform":

            class ToyMMM(BaseMMM, MaxAbsScaleTarget):
                pass

        adstock = GeometricAdstock(l_max=4)
        saturation = LogisticSaturation()

        if control == "without_controls":
            mmm = ToyMMM(
                date_column="date",
                channel_columns=["channel_1", "channel_2"],
                adstock=adstock,
                saturation=saturation,
            )
        elif control == "with_controls":
            mmm = ToyMMM(
                date_column="date",
                control_columns=["control_1", "control_2"],
                channel_columns=["channel_1", "channel_2"],
                adstock=adstock,
                saturation=saturation,
            )

        for transform in [mmm.adstock, mmm.saturation]:
            for dist in transform.function_priors.values():
                dist.dims = "channel"

        # fit the model
        mmm = mock_fit_base(mmm, toy_X, toy_y)
        mmm.sample_prior_predictive(toy_X, toy_y, extend_idata=True, combined=True)
        mmm.sample_posterior_predictive(toy_X, extend_idata=True, combined=True)
        mmm._prior_predictive = mmm.prior_predictive
        mmm._fit_result = mmm.fit_result
        mmm._posterior_predictive = mmm.posterior_predictive

        return mmm

    @pytest.mark.parametrize(
        argnames="func_plot_name, kwargs_plot",
        argvalues=[
            ("_plot_group_predictive", {"group": "prior_predictive"}),
            ("_plot_group_predictive", {"group": "posterior_predictive"}),
            # Prior predictive
            ("plot_prior_predictive", {}),
            ("plot_prior_predictive", {"original_scale": True}),
            ("plot_prior_predictive", {"ax": plt.subplots()[1]}),
            (
                "plot_prior_predictive",
                {
                    "add_mean": True,
                    "original_scale": False,
                },
            ),
            (
                "plot_prior_predictive",
                {
                    "add_gradient": True,
                    "original_scale": True,
                },
            ),
            (
                "plot_prior_predictive",
                {
                    "original_scale": False,
                },
            ),
            (
                "plot_prior_predictive",
                {
                    "add_mean": True,
                    "original_scale": True,
                },
            ),
            (
                "plot_prior_predictive",
                {
                    "add_mean": True,
                    "add_gradient": True,
                    "original_scale": False,
                },
            ),
            (
                "plot_prior_predictive",
                {
                    "add_mean": True,
                    "add_gradient": True,
                    "original_scale": True,
                },
            ),
            (
                "plot_prior_predictive",
                {
                    "add_mean": False,
                    "add_gradient": True,
                    "original_scale": False,
                },
            ),
            ("plot_prior_predictive", {"hdi_list": None}),
            ("plot_prior_predictive", {"hdi_list": []}),
            ("plot_prior_predictive", {"hdi_list": [0.94]}),
            ("plot_prior_predictive", {"hdi_list": [0.94, 0.5]}),
            # Posterior predictive
            ("plot_posterior_predictive", {}),
            ("plot_posterior_predictive", {"original_scale": True}),
            ("plot_posterior_predictive", {"ax": plt.subplots()[1]}),
            (
                "plot_posterior_predictive",
                {
                    "add_mean": True,
                    "original_scale": False,
                },
            ),
            (
                "plot_posterior_predictive",
                {
                    "add_gradient": True,
                    "original_scale": True,
                },
            ),
            (
                "plot_posterior_predictive",
                {
                    "original_scale": False,
                },
            ),
            (
                "plot_posterior_predictive",
                {
                    "add_mean": True,
                    "original_scale": True,
                },
            ),
            (
                "plot_posterior_predictive",
                {
                    "add_mean": True,
                    "add_gradient": True,
                    "original_scale": False,
                },
            ),
            (
                "plot_posterior_predictive",
                {
                    "add_mean": True,
                    "add_gradient": True,
                    "original_scale": True,
                },
            ),
            (
                "plot_posterior_predictive",
                {
                    "add_mean": False,
                    "add_gradient": True,
                    "original_scale": False,
                },
            ),
            ("plot_posterior_predictive", {"hdi_list": None}),
            ("plot_posterior_predictive", {"hdi_list": []}),
            ("plot_posterior_predictive", {"hdi_list": [0.94]}),
            ("plot_posterior_predictive", {"hdi_list": [0.94, 0.5]}),
            # Other plots
            ("plot_errors", {}),
            ("plot_errors", {"original_scale": True}),
            ("plot_errors", {"ax": plt.subplots()[1]}),
            ("plot_waterfall_components_decomposition", {"original_scale": True}),
            ("plot_channel_contribution_share_hdi", {"hdi_prob": 0.95}),
            ("plot_grouped_contribution_breakdown_over_time", {}),
            (
                "plot_grouped_contribution_breakdown_over_time",
                {
                    "stack_groups": {"controls": ["control_1"]},
                    "original_scale": True,
                    "area_kwargs": {"alpha": 0.5},
                },
            ),
            ("plot_components_contributions", {}),
            ("plot_prior_vs_posterior", {"var_name": "adstock_alpha"}),
        ],
    )
    def test_plots(self, plotting_mmm, func_plot_name, kwargs_plot) -> None:
        func = plotting_mmm.__getattribute__(func_plot_name)
        assert isinstance(func(**kwargs_plot), plt.Figure)
        plt.close("all")


@pytest.fixture(
    scope="module",
    params=[LogisticSaturation(), MichaelisMentenSaturation()],
    ids=["LogisticSaturation", "MichaelisMentenSaturation"],
)
def saturation(request):
    return request.param


@pytest.fixture(scope="module")
def mock_mmm(saturation) -> MMM:
    adstock = GeometricAdstock(l_max=4)
    return MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        control_columns=["control_1", "control_2"],
        adstock=adstock,
        saturation=saturation,
    )


def mock_fit(model: MMM, X: pd.DataFrame, y: np.ndarray, **kwargs):
    model.build_model(X=X, y=y)

    with model.model:
        idata = pm.sample_prior_predictive(random_seed=rng, **kwargs)

    model.preprocess("X", X)
    model.preprocess("y", y)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="The group fit_data is not defined in the InferenceData scheme",
        )
        idata.add_groups(
            {
                "posterior": idata.prior,
                "fit_data": pd.concat(
                    [X, pd.Series(y, index=X.index, name="y")], axis=1
                ).to_xarray(),
            }
        )
    model.idata = idata
    model.set_idata_attrs(idata=idata)

    return model


@pytest.fixture(scope="module")
def mock_fitted_mmm(mock_mmm, toy_X, toy_y):
    return mock_fit(mock_mmm, toy_X, toy_y)


@pytest.mark.parametrize(
    argnames="func_plot_name, kwargs_plot",
    argvalues=[
        # Only part of MMM now
        ("plot_direct_contribution_curves", {}),
        ("plot_direct_contribution_curves", {"same_axes": True}),
        ("plot_direct_contribution_curves", {"channels": ["channel_2"]}),
        ("plot_channel_parameter", {"param_name": "adstock_alpha"}),
        ("plot_components_contributions", {}),
        ("plot_components_contributions", {"original_scale": True}),
    ],
)
def test_mmm_plots(mock_fitted_mmm, func_plot_name, kwargs_plot) -> None:
    func = mock_fitted_mmm.__getattribute__(func_plot_name)
    assert isinstance(func(**kwargs_plot), plt.Figure)
    plt.close("all")


@pytest.mark.parametrize(
    "channels, match",
    [
        (["invalid_channel"], "subset"),
        (["channel_1", "channel_1"], "unique"),
        ([], "Number of rows must be a positive"),
    ],
)
def test_plot_direct_contribution_curves_error(mock_fitted_mmm, channels, match):
    with pytest.raises(ValueError, match=match):
        mock_fitted_mmm.plot_direct_contribution_curves(channels=channels)
