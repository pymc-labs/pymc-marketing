import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from pymc_marketing.mmm.delayed_saturated_mmm import BaseDelayedSaturatedMMM
from pymc_marketing.mmm.preprocessing import MaxAbsScaleTarget

seed: int = sum(map(ord, "pymc_marketing"))
rng: np.random.Generator = np.random.default_rng(seed=seed)


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
def toy_y(toy_X) -> pd.Series:
    return pd.Series(rng.integers(low=0, high=100, size=toy_X.shape[0]))


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

            class ToyMMM(BaseDelayedSaturatedMMM):
                pass

        elif transform == "target_transform":

            class ToyMMM(BaseDelayedSaturatedMMM, MaxAbsScaleTarget):
                pass

        if control == "without_controls":
            mmm = ToyMMM(
                date_column="date",
                channel_columns=["channel_1", "channel_2"],
                adstock_max_lag=4,
            )
        elif control == "with_controls":
            mmm = ToyMMM(
                date_column="date",
                adstock_max_lag=4,
                control_columns=["control_1", "control_2"],
                channel_columns=["channel_1", "channel_2"],
            )
        # fit the model
        mmm.fit(
            X=toy_X,
            y=toy_y,
        )
        mmm.sample_prior_predictive(toy_X, toy_y, extend_idata=True, combined=True)
        mmm.sample_posterior_predictive(toy_X, extend_idata=True, combined=True)
        mmm._prior_predictive = mmm.prior_predictive
        mmm._fit_result = mmm.fit_result
        mmm._posterior_predictive = mmm.posterior_predictive

        return mmm

    @pytest.mark.parametrize(
        argnames="func_plot_name, kwargs_plot",
        argvalues=[
            ("plot_prior_predictive", {"samples": 3}),
            ("plot_posterior_predictive", {}),
            ("plot_posterior_predictive", {"original_scale": True}),
            ("plot_components_contributions", {}),
            ("plot_waterfall_components_decomposition", {"original_scale": True}),
            ("plot_channel_parameter", {"param_name": "alpha"}),
            ("plot_direct_contribution_curves", {}),
            ("plot_direct_contribution_curves", {"same_axes": True}),
            ("plot_direct_contribution_curves", {"channels": ["channel_2"]}),
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
        ],
    )
    def test_plots(self, plotting_mmm, func_plot_name, kwargs_plot) -> None:
        func = plotting_mmm.__getattribute__(func_plot_name)
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
    def test_plot_direct_contribution_curves_error(self, plotting_mmm, channels, match):
        with pytest.raises(ValueError, match=match):
            plotting_mmm.plot_direct_contribution_curves(channels=channels)
