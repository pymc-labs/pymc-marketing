from typing import List, Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest

from pymc_marketing.mmm.delayed_saturated_mmm import DelayedSaturatedMMM

seed: int = sum(map(ord, "pymc_marketing"))
rng: np.random.Generator = np.random.default_rng(seed=seed)


@pytest.fixture(scope="class")
def toy_df() -> pd.DataFrame:
    date_data: pd.DatetimeIndex = pd.date_range(
        start="2019-06-01", end="2021-12-31", freq="W-MON"
    )

    n: int = date_data.size

    return pd.DataFrame(
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


@pytest.fixture(scope="class")
def mmm(toy_df: pd.DataFrame) -> DelayedSaturatedMMM:
    return DelayedSaturatedMMM(
        data=toy_df,
        target_column="y",
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        control_columns=["control_1", "control_2"],
    )


@pytest.fixture(scope="class")
def mmm_fitted(mmm: DelayedSaturatedMMM) -> DelayedSaturatedMMM:
    mmm.fit(target_accept=0.8, draws=3, chains=2)
    return mmm


class TestMMM:
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
        toy_df: pd.DataFrame,
        yearly_seasonality: Optional[int],
        channel_columns: List[str],
        control_columns: List[str],
        adstock_max_lag: int,
    ) -> None:
        mmm = DelayedSaturatedMMM(
            data=toy_df,
            target_column="y",
            date_column="date",
            channel_columns=channel_columns,
            control_columns=control_columns,
            adstock_max_lag=adstock_max_lag,
            yearly_seasonality=yearly_seasonality,
        )

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
            prior_predictive, group="prior", var_names=["beta_channel"], combined=True
        ).to_numpy().shape == (
            n_channel,
            samples,
        )
        assert az.extract(
            prior_predictive, group="prior", var_names=["alpha"], combined=True
        ).to_numpy().shape == (
            n_channel,
            samples,
        )
        assert az.extract(
            prior_predictive, group="prior", var_names=["lam"], combined=True
        ).to_numpy().shape == (
            n_channel,
            samples,
        )

    def test_fit(self, toy_df: pd.DataFrame) -> None:
        draws: int = 100
        chains: int = 2

        mmm = DelayedSaturatedMMM(
            data=toy_df,
            target_column="y",
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            control_columns=["control_1", "control_2"],
            adstock_max_lag=2,
            yearly_seasonality=2,
        )
        n_channel: int = len(mmm.channel_columns)
        mmm.fit(target_accept=0.81, draws=draws, chains=chains, random_seed=rng)
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

    @pytest.mark.parametrize(
        argnames="yearly_seasonality",
        argvalues=[None, 1, 2],
        ids=["no_yearly_seasonality", "yearly_seasonality=1", "yearly_seasonality=2"],
    )
    def test_get_fourier_models_data(
        self, toy_df: pd.DataFrame, yearly_seasonality: Optional[int]
    ) -> None:
        mmm = DelayedSaturatedMMM(
            data=toy_df,
            target_column="y",
            date_column="date",
            channel_columns=["channel_1", "channel_2"],
            control_columns=["control_1", "control_2"],
            adstock_max_lag=2,
            yearly_seasonality=yearly_seasonality,
        )

        fourier_modes_data: Optional[pd.DataFrame] = mmm._get_fourier_models_data()

        if yearly_seasonality is None:
            assert fourier_modes_data is None
        else:
            assert fourier_modes_data.shape == (
                toy_df.shape[0],
                2 * yearly_seasonality,
            )
            assert fourier_modes_data.max().max() <= 1
            assert fourier_modes_data.min().min() >= -1
