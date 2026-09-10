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
"""NestedCampaignMedia wired into the real MMM class (not the mock protocol).

The extra ``campaign`` dimension only exists when ``X`` is passed as an
``xr.Dataset``; the DataFrame API keeps channel/control columns only. The
decomposed channel ("social") is deliberately absent from ``channel_columns``
because the effect replaces its channel-level media term.
"""

import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.campaign_media import NestedCampaignMedia

seed = 20260825
CAMPAIGNS = ["social_a", "social_b", "social_c"]
MAPPING = {c: "social" for c in CAMPAIGNS}


@pytest.fixture(scope="module")
def campaign_mmm_data():
    n_dates = 24
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-06", freq="W-MON", periods=n_dates)
    media = np.abs(rng.normal(1.0, 0.35, size=(n_dates, 1))) + 0.2
    camp = np.array([0.7, 0.2, 0.1])[None, :] * rng.lognormal(0, 0.6, size=(n_dates, 3))
    target = np.abs(
        1.0 + 0.4 * media[:, 0] + 0.5 * camp.sum(1) + rng.normal(0, 0.1, n_dates)
    )

    X = xr.Dataset(
        {
            "media": xr.DataArray(media, dims=("date", "channel")),
            "campaign_data": xr.DataArray(camp, dims=("date", "campaign")),
        },
        coords={"date": dates, "channel": ["search"], "campaign": CAMPAIGNS},
    )
    y = xr.DataArray(target, dims=("date",), coords={"date": dates})
    X_df = pd.DataFrame({"date": dates, "search": media[:, 0]})
    return {"X": X, "y": y, "X_df": X_df, "y_series": pd.Series(target, name="y")}


def _make_mmm(effect=None):
    mmm = MMM(
        channel_columns=["search"],
        date_column="date",
        target_column="y",
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    )
    return mmm.add_mu_effect(
        effect
        if effect is not None
        else NestedCampaignMedia(campaign_to_channel=MAPPING)
    )


@pytest.fixture(scope="module")
def built_mmm(campaign_mmm_data):
    mmm = _make_mmm()
    mmm.build_model(X=campaign_mmm_data["X"], y=campaign_mmm_data["y"])
    return mmm


def test_build_registers_effect_variables(built_mmm):
    named = built_mmm.model.named_vars
    for name in [
        "campaign_data",
        "campaign_media_parent_idx",
        "campaign_media_channel_scale",
        "campaign_media_campaign_contribution",
        "campaign_media_channel_contribution",
        "campaign_media_effect_contribution",
    ]:
        assert name in named
    assert list(built_mmm.model.coords["campaign"]) == CAMPAIGNS
    assert list(built_mmm.model.coords["campaign_media_channel"]) == ["social"]
    # the decomposed channel is not a model channel
    assert list(built_mmm.model.coords["channel"]) == ["search"]


def test_prior_predictive_and_rollup(built_mmm):
    with built_mmm.model:
        idata = pm.sample_prior_predictive(draws=20, random_seed=seed)
    prior = idata.prior
    np.testing.assert_allclose(
        prior["campaign_media_campaign_contribution"].sum("campaign"),
        prior["campaign_media_effect_contribution"],
        rtol=1e-10,
    )
    assert prior["campaign_media_effect_contribution"].dims == (
        "chain",
        "draw",
        "date",
    )


def test_counterfactual_contributions_include_effect(campaign_mmm_data):
    """Prior draws stand in for a posterior, as the funnel fixtures do."""
    mmm = _make_mmm()
    mmm.build_model(X=campaign_mmm_data["X"], y=campaign_mmm_data["y"])
    with mmm.model:
        idata = pm.sample_prior_predictive(draws=20, random_seed=seed)
    idata["/posterior"] = idata["/prior"].to_dataset()
    idata["/fit_data"] = mmm.create_fit_data(
        campaign_mmm_data["X_df"], campaign_mmm_data["y_series"]
    )
    mmm.idata = idata
    mmm.set_idata_attrs(idata=idata)

    contributions = mmm.compute_counterfactual_contributions_dataset()
    assert "campaign_media_effect" in contributions.data_vars
    assert "campaign" not in contributions.dims


def test_idata_attrs_roundtrip_effect(campaign_mmm_data):
    from pymc_marketing.serialization import serialization

    effect = NestedCampaignMedia(campaign_to_channel=MAPPING, rho=0.5)
    payload = serialization.serialize(effect)
    restored = serialization.deserialize(payload)
    assert type(restored) is NestedCampaignMedia
    assert restored == effect


def test_double_count_warning(campaign_mmm_data):
    """Mapping campaigns onto a channel still in channel_columns must warn."""
    X = campaign_mmm_data["X"]
    mmm = _make_mmm(
        NestedCampaignMedia(campaign_to_channel=dict.fromkeys(CAMPAIGNS, "search"))
    )
    with pytest.warns(UserWarning, match="double-counts"):
        mmm.build_model(X=X, y=campaign_mmm_data["y"])


@pytest.mark.xfail(
    reason=(
        "MMM.load rebuilds X via fit_data as a DataFrame, which cannot carry "
        "the (date, campaign) variable, so create_data cannot find "
        "campaign_data on reload. Needs Dataset-aware fit_data or an "
        "idata_groups()-based supplementary-data route on the effect."
    ),
    strict=True,
)
def test_save_load_roundtrip(campaign_mmm_data, tmp_path):
    mmm = _make_mmm()
    mmm.build_model(X=campaign_mmm_data["X"], y=campaign_mmm_data["y"])
    with mmm.model:
        idata = pm.sample_prior_predictive(draws=10, random_seed=seed)
    idata["/posterior"] = idata["/prior"].to_dataset()
    idata["/fit_data"] = mmm.create_fit_data(
        campaign_mmm_data["X_df"], campaign_mmm_data["y_series"]
    )
    mmm.idata = idata
    mmm.set_idata_attrs(idata=idata)

    fname = tmp_path / "campaign_media_model.nc"
    mmm.save(str(fname))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = MMM.load(str(fname))
    assert isinstance(loaded.mu_effects[0], NestedCampaignMedia)


class TestGeoDims:
    """The effect with a real MMM carrying an extra model dim."""

    @pytest.fixture(scope="class")
    def geo_mmm(self):
        n_dates, geos = 16, ["north", "south"]
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2025-01-06", freq="W-MON", periods=n_dates)
        media = np.abs(rng.normal(1.0, 0.3, size=(n_dates, 2, 1))) + 0.2
        camp = np.array([0.7, 0.2, 0.1])[None, None, :] * rng.lognormal(
            0, 0.6, size=(n_dates, 2, 3)
        )
        target = np.abs(
            1.0
            + 0.4 * media[..., 0]
            + 0.5 * camp.sum(-1)
            + rng.normal(0, 0.1, size=(n_dates, 2))
        )
        X = xr.Dataset(
            {
                "media": xr.DataArray(media, dims=("date", "geo", "channel")),
                "campaign_data": xr.DataArray(camp, dims=("date", "geo", "campaign")),
            },
            coords={
                "date": dates,
                "geo": geos,
                "channel": ["search"],
                "campaign": CAMPAIGNS,
            },
        )
        y = xr.DataArray(
            target, dims=("date", "geo"), coords={"date": dates, "geo": geos}
        )
        mmm = MMM(
            channel_columns=["search"],
            date_column="date",
            target_column="y",
            dims=("geo",),
            adstock=GeometricAdstock(l_max=2),
            saturation=LogisticSaturation(),
        ).add_mu_effect(NestedCampaignMedia(campaign_to_channel=MAPPING))
        mmm.build_model(X=X, y=y)
        return mmm

    def test_contribution_dims(self, geo_mmm):
        named = geo_mmm.model.named_vars
        assert set(named["campaign_media_campaign_contribution"].type.dims) == {
            "date",
            "geo",
            "campaign",
        }
        assert set(named["campaign_media_effect_contribution"].type.dims) == {
            "date",
            "geo",
        }

    def test_prior_predictive_and_rollup(self, geo_mmm):
        with geo_mmm.model:
            idata = pm.sample_prior_predictive(draws=10, random_seed=seed)
        prior = idata.prior
        np.testing.assert_allclose(
            prior["campaign_media_campaign_contribution"].sum("campaign"),
            prior["campaign_media_effect_contribution"],
            rtol=1e-10,
        )
        # channel scale is computed over dates AND geos: one scale per channel
        assert geo_mmm.model["campaign_media_channel_scale"].get_value().shape == (1,)


def test_lift_test_on_real_mmm(campaign_mmm_data):
    mmm = _make_mmm()
    effect = mmm.mu_effects[0]
    mmm.build_model(X=campaign_mmm_data["X"], y=campaign_mmm_data["y"])
    df_lift = pd.DataFrame(
        {
            "campaign": ["social_c"],
            "x": [0.1],
            "delta_x": [0.1],
            "delta_y": [0.05],
            "sigma": [0.02],
        }
    )
    effect.add_lift_test_measurements(df_lift, mmm)
    assert "campaign_media_lift_measurements" in mmm.model.named_vars
    assert np.isfinite(mmm.model.compile_logp()(mmm.model.initial_point()))
