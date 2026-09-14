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
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from pymc_marketing.mmm.campaign_media import NestedCampaignMedia
from pymc_marketing.serialization import serialization

CAMPAIGNS = ["tv_brand", "tv_promo", "search_gen", "search_brand", "search_promo"]
MAPPING = {
    "tv_brand": "tv",
    "tv_promo": "tv",
    "search_gen": "search",
    "search_brand": "search",
    "search_promo": "search",
}


def _make_mock_mmm(seed=42, n_dates=30):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_dates, freq="W-MON")
    spend = rng.gamma(2.0, 1.0, size=(n_dates, len(CAMPAIGNS)))
    ds = xr.Dataset(
        {"campaign_data": (("date", "campaign"), spend)},
        coords={"date": dates, "campaign": CAMPAIGNS},
    )
    model = pm.Model(coords={"date": dates, "campaign": CAMPAIGNS})
    return type("MockMMM", (), {"dims": (), "model": model, "xarray_dataset": ds})()


def _build(effect=None):
    mmm = _make_mock_mmm()
    effect = effect or NestedCampaignMedia(campaign_to_channel=MAPPING)
    with mmm.model:
        effect.create_data(mmm)
        effect.create_effect(mmm)
    return mmm, effect


def test_create_data_registers_variables():
    mmm = _make_mock_mmm()
    effect = NestedCampaignMedia(campaign_to_channel=MAPPING)
    with mmm.model:
        effect.create_data(mmm)
    for name in [
        "campaign_data",
        "campaign_media_parent_idx",
        "campaign_media_parent_onehot",
        "campaign_media_channel_scale",
        "campaign_media_campaign_cap",
    ]:
        assert name in mmm.model.named_vars
    assert list(mmm.model.coords["campaign_media_channel"]) == ["tv", "search"]


def test_mapping_mismatch_raises():
    mmm = _make_mock_mmm()
    bad = {**MAPPING}
    bad.pop("tv_promo")
    bad["nonexistent"] = "tv"
    effect = NestedCampaignMedia(campaign_to_channel=bad)
    with mmm.model, pytest.raises(ValueError, match="must cover exactly"):
        effect.create_data(mmm)


def test_create_effect_contributions():
    mmm, effect = _build()
    named = mmm.model.named_vars
    assert "campaign_media_campaign_contribution" in named
    assert "campaign_media_channel_contribution" in named
    assert "campaign_media_effect_contribution" in named
    assert effect.contribution_var_name == "campaign_media_effect_contribution"

    total = named["campaign_media_effect_contribution"]
    assert set(total.type.dims) == {"date"}
    per_campaign = named["campaign_media_campaign_contribution"]
    assert set(per_campaign.type.dims) == {"date", "campaign"}
    per_channel = named["campaign_media_channel_contribution"]
    assert set(per_channel.type.dims) == {"date", "campaign_media_channel"}


def test_incrementality_spec_is_none():
    effect = NestedCampaignMedia(campaign_to_channel=MAPPING)
    assert effect.incrementality_spec() is None


def test_prior_predictive_and_channel_rollup():
    mmm, _ = _build()
    with mmm.model:
        idata = pm.sample_prior_predictive(draws=13, random_seed=1)
    prior = idata.prior
    contrib = prior["campaign_media_campaign_contribution"]
    assert (contrib >= 0).all()
    # channel roll-up sums to the total contribution
    np.testing.assert_allclose(
        prior["campaign_media_channel_contribution"].sum("campaign_media_channel"),
        prior["campaign_media_effect_contribution"],
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        contrib.sum("campaign"),
        prior["campaign_media_effect_contribution"],
        rtol=1e-10,
    )


def test_saturation_bounded_by_beta():
    # saturation shapes are bounded by 1: each campaign contribution is
    # bounded by its amplitude deterministic beta_campaign
    mmm, _ = _build()
    with mmm.model:
        idata = pm.sample_prior_predictive(draws=13, random_seed=2)
    prior = idata.prior
    contrib_max = prior["campaign_media_campaign_contribution"].max("date")
    beta_c = prior["campaign_media_beta_campaign"]
    assert ((contrib_max <= beta_c + 1e-12).all()).item()


def test_multipliers_weighted_zero_sum_within_channel():
    # log-multipliers satisfy the spend-share-weighted zero-sum constraint
    # within each channel, so beta_channel is the spend-weighted channel mean
    mmm, _ = _build()
    with mmm.model:
        idata = pm.sample_prior_predictive(draws=13, random_seed=4)
    prior = idata.prior
    spend = mmm.xarray_dataset["campaign_data"].values
    total = spend.sum(axis=0)
    for name in ["campaign_media_beta_multiplier", "campaign_media_lam_multiplier"]:
        log_mult = np.log(prior[name].values)  # (chain, draw, campaign)
        for channel in ["tv", "search"]:
            idx = [i for i, c in enumerate(CAMPAIGNS) if MAPPING[c] == channel]
            w = total[idx] / total[idx].sum()
            np.testing.assert_allclose(
                (log_mult[..., idx] * w).sum(-1), 0.0, atol=1e-10
            )


def test_zero_sum_multipliers_opt_out():
    mmm = _make_mock_mmm()
    effect = NestedCampaignMedia(
        campaign_to_channel=MAPPING, zero_sum_multipliers=False
    )
    with mmm.model:
        effect.create_data(mmm)
        effect.create_effect(mmm)
    assert "campaign_media_mult_basis" not in mmm.model.named_vars
    assert mmm.model["campaign_media_z_beta"].type.dims == ("campaign",)


def test_single_campaign_channel_fully_pooled():
    # a channel with one campaign has no free directions: multiplier == 1
    campaigns = ["solo_camp", "search_a", "search_b"]
    mapping = {"solo_camp": "tv", "search_a": "search", "search_b": "search"}
    rng = np.random.default_rng(11)
    dates = pd.date_range("2025-01-01", periods=20, freq="W-MON")
    ds = xr.Dataset(
        {"campaign_data": (("date", "campaign"), rng.gamma(2.0, 1.0, (20, 3)))},
        coords={"date": dates, "campaign": campaigns},
    )
    model = pm.Model(coords={"date": dates, "campaign": campaigns})
    mmm = type("MockMMM", (), {"dims": (), "model": model, "xarray_dataset": ds})()
    effect = NestedCampaignMedia(campaign_to_channel=mapping)
    with model:
        effect.create_data(mmm)
        effect.create_effect(mmm)
        idata = pm.sample_prior_predictive(draws=7, random_seed=5)
    mult = idata.prior["campaign_media_beta_multiplier"].sel(campaign="solo_camp")
    np.testing.assert_allclose(mult.values, 1.0, atol=1e-12)
    # only the two-campaign channel contributes a free direction
    assert len(model.coords["campaign_media_free"]) == 1


def test_serialization_roundtrip():
    effect = NestedCampaignMedia(
        campaign_to_channel=MAPPING, prefix="cm", rho=0.7, tau_beta_sigma=0.3
    )
    data = effect.to_dict()
    data["__type__"] = (
        f"{NestedCampaignMedia.__module__}.{NestedCampaignMedia.__qualname__}"
    )
    restored = serialization.deserialize(data)
    assert type(restored) is NestedCampaignMedia
    assert restored == effect


def _make_mock_mmm_with_covariates(seed=7, n_dates=30):
    rng = np.random.default_rng(seed)
    mmm = _make_mock_mmm(seed=seed, n_dates=n_dates)
    cov = rng.normal(size=(len(CAMPAIGNS), 2))
    mmm.xarray_dataset["covariates"] = xr.DataArray(
        cov,
        dims=("campaign", "covariate"),
        coords={"campaign": CAMPAIGNS, "covariate": ["log_impressions", "ctr"]},
    )
    return mmm


def test_covariates_registered_and_channel_centred():
    mmm = _make_mock_mmm_with_covariates()
    effect = NestedCampaignMedia(
        campaign_to_channel=MAPPING, covariate_var="covariates"
    )
    with mmm.model:
        effect.create_data(mmm)
        effect.create_effect(mmm)
    assert "campaign_media_covariates" in mmm.model.named_vars
    assert "campaign_media_gamma" in mmm.model.named_vars

    cov_centred = mmm.model["campaign_media_covariates"].values.eval()
    spend = mmm.xarray_dataset["campaign_data"].values
    total = spend.sum(axis=0)
    for channel in ["tv", "search"]:
        idx = [i for i, c in enumerate(CAMPAIGNS) if MAPPING[c] == channel]
        share = total[idx] / total[idx].sum()
        np.testing.assert_allclose(share @ cov_centred[idx], 0.0, atol=1e-12)


def test_covariate_prior_predictive():
    mmm = _make_mock_mmm_with_covariates()
    effect = NestedCampaignMedia(
        campaign_to_channel=MAPPING, covariate_var="covariates"
    )
    with mmm.model:
        effect.create_data(mmm)
        effect.create_effect(mmm)
        idata = pm.sample_prior_predictive(draws=9, random_seed=3)
    assert "campaign_media_campaign_contribution" in idata.prior


def test_covariate_bad_dims_raises():
    mmm = _make_mock_mmm_with_covariates()
    mmm.xarray_dataset["bad_cov"] = xr.DataArray(
        np.zeros((30, len(CAMPAIGNS))),
        dims=("date", "campaign"),
        coords={
            "date": mmm.xarray_dataset.coords["date"],
            "campaign": CAMPAIGNS,
        },
    )
    effect = NestedCampaignMedia(campaign_to_channel=MAPPING, covariate_var="bad_cov")
    with mmm.model, pytest.raises(ValueError, match="dims exactly"):
        effect.create_data(mmm)


def test_covariate_serialization_roundtrip():
    effect = NestedCampaignMedia(
        campaign_to_channel=MAPPING,
        covariate_var="covariates",
        gamma_sigma=0.25,
    )
    data = effect.to_dict()
    data["__type__"] = (
        f"{NestedCampaignMedia.__module__}.{NestedCampaignMedia.__qualname__}"
    )
    restored = serialization.deserialize(data)
    assert restored == effect


def test_lift_test_measurements():
    mmm, effect = _build()
    df_lift = pd.DataFrame(
        {
            "campaign": ["tv_promo", "search_gen"],
            "x": [2.0, 1.0],
            "delta_x": [1.0, 0.5],
            "delta_y": [0.15, 0.08],
            "sigma": [0.05, 0.03],
        }
    )
    effect.add_lift_test_measurements(df_lift, mmm)
    assert "campaign_media_lift_measurements" in mmm.model.named_vars
    logp = mmm.model.compile_logp()(mmm.model.initial_point())
    assert np.isfinite(logp)


def test_lift_test_requires_built_model():
    mmm = _make_mock_mmm()
    effect = NestedCampaignMedia(campaign_to_channel=MAPPING)
    df_lift = pd.DataFrame(
        {
            "campaign": ["tv_promo"],
            "x": [1.0],
            "delta_x": [1.0],
            "delta_y": [0.1],
            "sigma": [0.05],
        }
    )
    with pytest.raises(RuntimeError, match="has not been built"):
        effect.add_lift_test_measurements(df_lift, mmm)


def test_lift_test_unknown_campaign_raises():
    mmm, effect = _build()
    df_lift = pd.DataFrame(
        {
            "campaign": ["nope"],
            "x": [1.0],
            "delta_x": [1.0],
            "delta_y": [0.1],
            "sigma": [0.05],
        }
    )
    with pytest.raises(ValueError, match="Unknown campaigns"):
        effect.add_lift_test_measurements(df_lift, mmm)


def test_lift_test_moves_posterior_toward_truth():
    # a strong lift test on one campaign should move its beta multiplier
    # relative to an uncalibrated fit of the same prior-only model
    mmm, effect = _build()
    df_lift = pd.DataFrame(
        {
            "campaign": ["tv_promo"],
            "x": [1.0],
            "delta_x": [2.0],
            "delta_y": [1.2],
            "sigma": [0.02],
        }
    )
    effect.add_lift_test_measurements(df_lift, mmm)
    with mmm.model:
        idata = pm.sample(
            draws=150,
            tune=300,
            chains=2,
            cores=2,
            random_seed=7,
            progressbar=False,
            compute_convergence_checks=False,
        )
    calibrated = idata.posterior["campaign_media_beta_multiplier"].sel(
        campaign="tv_promo"
    )
    # the strong observed lift demands a large response from this campaign:
    # its multiplier posterior must sit clearly above the prior median of 1
    assert float(calibrated.median()) > 1.1


def _channel_contribution_at_initial_point(campaigns, mapping, spend, saturation=None):
    from pymc_marketing.mmm.campaign_media import NestedCampaignMedia as _Effect

    dates = pd.date_range("2025-01-01", periods=spend.shape[0], freq="W-MON")
    ds = xr.Dataset(
        {"campaign_data": (("date", "campaign"), spend)},
        coords={"date": dates, "campaign": campaigns},
    )
    model = pm.Model(coords={"date": dates, "campaign": campaigns})
    mmm = type("MockMMM", (), {"dims": (), "model": model, "xarray_dataset": ds})()
    kwargs = {} if saturation is None else {"saturation": saturation}
    effect = _Effect(campaign_to_channel=mapping, **kwargs)
    with model:
        effect.create_data(mmm)
        effect.create_effect(mmm)
    import pytensor

    (graph,) = model.replace_rvs_by_values(
        [model["campaign_media_channel_contribution"]]
    )
    fn = pytensor.function(model.value_vars, graph, on_unused_input="ignore")
    ip = model.initial_point()
    return fn(*(ip[v.name] for v in model.value_vars))


@pytest.mark.parametrize("saturation_cls", [None, "logistic"])
def test_split_invariance(saturation_cls):
    # splitting a campaign into two parts with the same total spend must not
    # change the channel-level contribution: channel capacity belongs to the
    # channel, not to the number of rows in the campaign mapping. Holds for
    # any saturation shape by construction.
    from pymc_marketing.mmm import LogisticSaturation

    saturation = None if saturation_cls is None else LogisticSaturation()
    rng = np.random.default_rng(3)
    spend_a = rng.gamma(2.0, 1.0, 30)
    spend_b = rng.gamma(2.0, 1.0, 30)

    whole = _channel_contribution_at_initial_point(
        ["a", "b"],
        {"a": "ch", "b": "ch"},
        np.column_stack([spend_a, spend_b]),
        saturation,
    )
    split = _channel_contribution_at_initial_point(
        ["a1", "a2", "b"],
        {"a1": "ch", "a2": "ch", "b": "ch"},
        np.column_stack([0.6 * spend_a, 0.4 * spend_a, spend_b]),
        saturation,
    )
    np.testing.assert_allclose(split, whole, rtol=1e-10)


def test_zero_spend_campaign_pinned():
    campaigns = ["live_a", "live_b", "dead"]
    mapping = dict.fromkeys(campaigns, "ch")
    rng = np.random.default_rng(9)
    spend = np.column_stack(
        [rng.gamma(2.0, 1.0, 25), rng.gamma(2.0, 1.0, 25), np.zeros(25)]
    )
    dates = pd.date_range("2025-01-01", periods=25, freq="W-MON")
    ds = xr.Dataset(
        {"campaign_data": (("date", "campaign"), spend)},
        coords={"date": dates, "campaign": campaigns},
    )
    model = pm.Model(coords={"date": dates, "campaign": campaigns})
    mmm = type("MockMMM", (), {"dims": (), "model": model, "xarray_dataset": ds})()
    effect = NestedCampaignMedia(campaign_to_channel=mapping)
    with model, pytest.warns(UserWarning, match="no spend"):
        effect.create_data(mmm)
        effect.create_effect(mmm)
        idata = pm.sample_prior_predictive(draws=7, random_seed=2)
    # only the two live campaigns contribute a free direction
    assert len(model.coords["campaign_media_free"]) == 1
    # the dead campaign's multiplier is pinned to the pooled value
    mult = idata.prior["campaign_media_beta_multiplier"].sel(campaign="dead")
    np.testing.assert_allclose(mult.values, 1.0, atol=1e-12)


def test_library_saturation_shapes():
    from pymc_marketing.mmm import HillSaturationSigmoid, LogisticSaturation

    for saturation in [LogisticSaturation(), HillSaturationSigmoid()]:
        mmm = _make_mock_mmm()
        effect = NestedCampaignMedia(campaign_to_channel=MAPPING, saturation=saturation)
        with mmm.model:
            effect.create_data(mmm)
            effect.create_effect(mmm)
            idata = pm.sample_prior_predictive(draws=5, random_seed=3)
        assert "campaign_media_campaign_contribution" in idata.prior
        # channel-level saturation params exist with the effect's channel dim
        for var_name in effect.saturation.variable_mapping.values():
            assert mmm.model[var_name].type.dims == ("campaign_media_channel",)
