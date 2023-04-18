from unittest.mock import patch

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib import pyplot as plt

from pymc_marketing.mmm.base import MMM
from pymc_marketing.mmm.preprocessing import MaxAbsScaleTarget, preprocessing_method
from pymc_marketing.mmm.validating import validation_method

seed: int = sum(map(ord, "pymc_marketing"))
rng: np.random.Generator = np.random.default_rng(seed=seed)
date_data: pd.DatetimeIndex = pd.date_range(
    start="2019-06-01", end="2021-12-31", freq="W-MON"
)

n: int = date_data.size

toy_df = pd.DataFrame(
    data={
        "date": date_data,
        "y": rng.integers(low=0, high=100, size=n),
        "channel_1": rng.integers(low=0, high=400, size=n),
        "channel_2": rng.integers(low=0, high=50, size=n),
        "control_1": rng.gamma(shape=1000, scale=500, size=n),
        "control_2": rng.gamma(shape=100, scale=5, size=n),
        "other_column_1": rng.integers(low=0, high=100, size=n),
        "other_column_2": rng.normal(loc=0, scale=1, size=n),
    }
)


@pytest.fixture(
    scope="module",
    params=[
        "without_controls-default_transform",
        "with_controls-default_transform",
        "without_controls-target_transform",
        "with_controls-target_transform",
    ],
)
def plotting_mmm(request):
    control, transform = request.param.split("-")
    if transform == "default_transform":

        class ToyMMM(MMM):
            def build_model(self, data, **kwargs):
                pass

    elif transform == "target_transform":

        class ToyMMM(MMM, MaxAbsScaleTarget):
            def build_model(self, data, **kwargs):
                pass

    mmm = ToyMMM(
        toy_df,
        target_column="y",
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
    )
    rng = np.random.default_rng(42)
    coords = {
        "chain": range(4),
        "draw": range(100),
        "channel": mmm.channel_columns,
        "date": toy_df.date,
    }
    likelihood_dims = ["chain", "draw", "date"]
    alpha_dims = ["chain", "draw", "channel"]
    channel_contrib_dims = ["chain", "draw", "date", "channel"]
    prior_post = xr.Dataset(
        {
            "intercept": xr.DataArray(
                rng.gamma(1, 1, size=(4, 100)),
                coords={k: v for k, v in coords.items() if k in ["chain", "draw"]},
                dims=["chain", "draw"],
            ),
            "alpha": xr.DataArray(
                rng.gamma(1, 1, size=(4, 100, 2)),
                coords={k: v for k, v in coords.items() if k in alpha_dims},
                dims=alpha_dims,
            ),
            "channel_contributions": xr.DataArray(
                rng.gamma(1, 1, size=(4, 100, len(toy_df), 2)),
                coords={k: v for k, v in coords.items() if k in channel_contrib_dims},
                dims=channel_contrib_dims,
            ),
        }
    )
    prior_post_pred = xr.Dataset(
        {
            "likelihood": xr.DataArray(
                rng.gamma(1, 1, size=(4, 100, len(toy_df))),
                coords={k: v for k, v in coords.items() if k in likelihood_dims},
                dims=likelihood_dims,
            )
        }
    )
    if control == "with_controls":
        mmm.control_columns = ["control_1", "control_2"]
        coords["control"] = mmm.control_columns
        control_contrib_dims = ["chain", "draw", "date", "control"]
        prior_post["control_contributions"] = xr.DataArray(
            rng.gamma(1, 1, size=(4, 100, len(toy_df), 2)),
            coords={k: v for k, v in coords.items() if k in control_contrib_dims},
            dims=control_contrib_dims,
        )
    mmm._prior_predictive = az.InferenceData(
        prior=prior_post,
        prior_predictive=prior_post_pred,
    )
    mmm._fit_result = az.InferenceData(
        posterior=prior_post,
        observed_data=xr.Dataset(
            {
                "likelihood": xr.DataArray(
                    toy_df.y.values,
                    coords={"date": coords["date"]},
                    dims=["date"],
                )
            }
        ),
    )
    mmm._posterior_predictive = az.InferenceData(
        posterior_predictive=prior_post_pred,
    )
    return mmm


class TestMMM:
    @patch("pymc_marketing.mmm.base.MMM.validate_target")
    @patch("pymc_marketing.mmm.base.MMM.validate_date_col")
    @patch("pymc_marketing.mmm.base.MMM.validate_channel_columns")
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
    def test_init(
        self,
        validate_channel_columns,
        validate_date_col,
        validate_target,
        channel_columns,
    ) -> None:
        validate_channel_columns.configure_mock(_tags={"validation": True})
        validate_date_col.configure_mock(_tags={"validation": True})
        validate_target.configure_mock(_tags={"validation": True})
        toy_validation_count = 0
        toy_preprocess_count = 0
        build_model_count = 0

        class ToyMMM(MMM):
            def build_model(*args, **kwargs):
                nonlocal build_model_count
                build_model_count += 1
                pd.testing.assert_frame_equal(kwargs["data"], toy_df)
                return None

            @validation_method
            def toy_validation(self, data):
                nonlocal toy_validation_count
                toy_validation_count += 1
                pd.testing.assert_frame_equal(data, toy_df)
                return None

            @preprocessing_method
            def toy_preprocessing(self, data):
                nonlocal toy_preprocess_count
                toy_preprocess_count += 1
                pd.testing.assert_frame_equal(data, toy_df)
                return data

        instance = ToyMMM(
            data=toy_df,
            target_column="y",
            date_column="date",
            channel_columns=channel_columns,
        )
        pd.testing.assert_frame_equal(instance.data, toy_df)
        pd.testing.assert_frame_equal(instance.preprocessed_data, toy_df)
        validate_target.assert_called_once_with(instance, toy_df)
        validate_date_col.assert_called_once_with(instance, toy_df)
        validate_channel_columns.assert_called_once_with(instance, toy_df)

        assert toy_validation_count == 1
        assert toy_preprocess_count == 1
        assert build_model_count == 1

    @pytest.mark.parametrize(
        argnames="func_plot_name, kwargs_plot",
        argvalues=[
            ("plot_prior_predictive", {"samples": 3}),
            ("plot_posterior_predictive", {}),
            ("plot_posterior_predictive", {"original_scale": True}),
            ("plot_components_contributions", {}),
            ("plot_channel_parameter", {"param_name": "alpha"}),
            ("plot_contribution_curves", {}),
            ("plot_channel_contribution_share_hdi", {"hdi_prob": 0.95}),
            ("plot_grouped_contribution_breakdown_over_time", {}),
            (
                "plot_grouped_contribution_breakdown_over_time",
                {
                    "stack_groups": {"controls": ["control_1"]},
                    "area_kwargs": {"alpha": 0.5},
                },
            ),
        ],
    )
    def test_plots(
        self,
        plotting_mmm,
        func_plot_name,
        kwargs_plot,
    ) -> None:
        func = plotting_mmm.__getattribute__(func_plot_name)
        assert isinstance(func(**kwargs_plot), plt.Figure)
