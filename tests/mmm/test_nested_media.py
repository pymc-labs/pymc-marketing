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
import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytest
import xarray as xr
from pytensor.xtensor import as_xtensor

from pymc_marketing.mmm import MMM, GeometricAdstock
from pymc_marketing.mmm.components.saturation import (
    HillSaturationSigmoid,
    LogisticSaturation,
    MichaelisMentenSaturation,
    TanhSaturation,
)
from pymc_marketing.mmm.nested_media import NestedMediaEffect, _live_blocks
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
    effect = effect or NestedMediaEffect(child_to_parent=MAPPING)
    with mmm.model:
        effect.create_data(mmm)
        effect.create_effect(mmm)
    return mmm, effect


def test_create_data_registers_variables():
    mmm = _make_mock_mmm()
    effect = NestedMediaEffect(child_to_parent=MAPPING)
    with mmm.model:
        effect.create_data(mmm)
    for name in [
        "campaign_data",
        "nested_media_parent_idx",
        "nested_media_parent_onehot",
        "nested_media_channel_scale",
        "nested_media_campaign_cap",
    ]:
        assert name in mmm.model.named_vars
    assert list(mmm.model.coords["nested_media_channel"]) == ["tv", "search"]


def _channel_of(campaigns=CAMPAIGNS):
    return xr.DataArray(
        [MAPPING[c] for c in campaigns],
        dims="campaign",
        coords={"campaign": list(campaigns)},
        name="nested_media_channel",
    )


def test_prepare_saturation_builds_a_channel_level_copy():
    mmm = _make_mock_mmm()
    effect = NestedMediaEffect(child_to_parent=MAPPING)
    effect._prepare_saturation(mmm.model, "nested_media_channel", ["tv", "search"])

    assert list(mmm.model.coords["nested_media_channel"]) == ["tv", "search"]
    assert effect._saturation.prefix == "nested_media_saturation"
    for prior in effect._saturation.function_priors.values():
        assert prior.dims == ("nested_media_channel",)
    # the user's saturation is left as it was
    assert effect.saturation.prefix == "saturation"
    for prior in effect.saturation.function_priors.values():
        assert prior.dims is None


def test_scale_and_cap_aligns_and_warns_on_dead_campaigns():
    mmm = _make_mock_mmm()
    spend = mmm.xarray_dataset["campaign_data"].copy()
    spend.loc[{"campaign": "tv_promo"}] = 0.0
    effect = NestedMediaEffect(child_to_parent=MAPPING)

    with pytest.warns(UserWarning, match=r"\['tv_promo'\].*cap 0"):
        scale, cap = effect._scale_and_cap(mmm, spend, _channel_of(), ["search", "tv"])
    assert scale.dims == ("nested_media_channel",)
    assert list(scale["nested_media_channel"].values) == ["search", "tv"]
    assert cap.dims == ("campaign",)
    assert float(cap.sel(campaign="tv_promo")) == 0.0
    assert float(cap.sel(campaign="tv_brand")) > 0.0


def test_register_covariates():
    share = xr.DataArray(
        np.full(len(CAMPAIGNS), 0.5), dims="campaign", coords={"campaign": CAMPAIGNS}
    )
    live = share > 0

    mmm = _make_mock_mmm_with_covariates()
    effect = NestedMediaEffect(child_to_parent=MAPPING)
    with mmm.model:
        effect._register_covariates(mmm, mmm.model, share, live, _channel_of())
    assert "nested_media_covariates" not in mmm.model.named_vars

    effect = NestedMediaEffect(child_to_parent=MAPPING, covariate_var="covariates")
    with mmm.model:
        effect._register_covariates(mmm, mmm.model, share, live, _channel_of())
    assert mmm.model.named_vars_to_dims["nested_media_covariates"] == (
        "campaign",
        "covariate",
    )
    assert list(mmm.model.coords["covariate"]) == ["log_impressions", "ctr"]


def test_live_blocks():
    # search_gen is dead, but it still makes search the first channel seen
    campaigns = ["search_gen", "tv_brand", "tv_promo", "search_brand", "search_promo"]
    live = xr.DataArray(
        [False, True, True, True, True], dims="campaign", coords={"campaign": campaigns}
    )
    blocks = _live_blocks(campaigns, _channel_of(campaigns), live)
    assert blocks == {
        "search": ["search_brand", "search_promo"],
        "tv": ["tv_brand", "tv_promo"],
    }
    assert list(blocks) == ["search", "tv"]

    live[1] = False  # tv drops to one live campaign: no free direction, no block
    assert _live_blocks(campaigns, _channel_of(campaigns), live) == {
        "search": ["search_brand", "search_promo"]
    }


def test_parent_named_live_is_rejected():
    # "live" would share the reserved coordinate of all live campaigns
    with pytest.raises(ValueError, match='"live" is reserved'):
        NestedMediaEffect(child_to_parent={"a": "paid", "b": "live"})


def test_mapping_mismatch_raises():
    mmm = _make_mock_mmm()
    bad = {**MAPPING}
    bad.pop("tv_promo")
    bad["nonexistent"] = "tv"
    effect = NestedMediaEffect(child_to_parent=bad)
    with mmm.model, pytest.raises(ValueError, match="must cover exactly"):
        effect.create_data(mmm)


def test_create_effect_contributions():
    mmm, effect = _build()
    named = mmm.model.named_vars
    assert "nested_media_campaign_contribution" in named
    assert "nested_media_channel_contribution" in named
    assert "nested_media_effect_contribution" in named
    assert effect.contribution_var_name == "nested_media_effect_contribution"

    total = named["nested_media_effect_contribution"]
    assert set(total.type.dims) == {"date"}
    per_campaign = named["nested_media_campaign_contribution"]
    assert set(per_campaign.type.dims) == {"date", "campaign"}
    per_channel = named["nested_media_channel_contribution"]
    assert set(per_channel.type.dims) == {"date", "nested_media_channel"}


def test_incrementality_spec_is_none():
    effect = NestedMediaEffect(child_to_parent=MAPPING)
    assert effect.incrementality_spec() is None


def test_prior_predictive_and_channel_rollup():
    mmm, _ = _build()
    with mmm.model:
        idata = pm.sample_prior_predictive(draws=13, random_seed=1)
    prior = idata.prior
    contrib = prior["nested_media_campaign_contribution"]
    assert (contrib >= 0).all()
    # channel roll-up sums to the total contribution
    np.testing.assert_allclose(
        prior["nested_media_channel_contribution"].sum("nested_media_channel"),
        prior["nested_media_effect_contribution"],
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        contrib.sum("campaign"),
        prior["nested_media_effect_contribution"],
        rtol=1e-10,
    )


class _SubclassedLogistic(LogisticSaturation):
    pass


@pytest.mark.parametrize(
    "saturation",
    [
        MichaelisMentenSaturation(),
        LogisticSaturation(),
        HillSaturationSigmoid(),
        _SubclassedLogistic(),
    ],
    ids=lambda s: type(s).__name__,
)
def test_saturation_bounded_by_beta(saturation):
    # bounded saturation shapes top out at their amplitude parameter: each
    # campaign contribution is bounded by its amplitude deterministic
    mmm, _ = _build(NestedMediaEffect(child_to_parent=MAPPING, saturation=saturation))
    with mmm.model:
        idata = pm.sample_prior_predictive(draws=13, random_seed=2)
    prior = idata.prior
    contrib_max = prior["nested_media_campaign_contribution"].max("date")
    beta_c = prior["nested_media_beta_campaign"]
    assert ((contrib_max <= beta_c + 1e-12).all()).item()


def test_campaign_multiplier_normalises_the_covariate_shift():
    # m_c = exp(tau z_c + gamma cov_c) / sum_k w_k exp(tau z_k + gamma cov_k)
    # within each channel
    mmm = _make_mock_mmm_with_covariates()
    effect = NestedMediaEffect(
        child_to_parent=MAPPING,
        covariate_var="covariates",
        zero_sum_multipliers=False,
    )
    model = mmm.model
    with model:
        effect.create_data(mmm)
        mult = effect._campaign_multiplier(model)
    assert mult.type.dims == ("campaign",)

    names = ["tau_beta", "z_beta", "gamma"]
    m, tau, z, gamma = pm.draw(
        [mult, *(model[f"nested_media_{n}"] for n in names)], draws=4, random_seed=3
    )
    cov = model["nested_media_covariates"].values.eval()
    share = model["nested_media_spend_share"].get_value()
    raw = np.exp(tau[:, None] * z + gamma @ cov.T)
    expected = np.empty_like(raw)
    for channel in ["tv", "search"]:
        idx = [i for i, c in enumerate(CAMPAIGNS) if MAPPING[c] == channel]
        mean = raw[:, idx] @ share[idx] / share[idx].sum()
        expected[:, idx] = raw[:, idx] / mean[:, None]
    np.testing.assert_allclose(m, expected, rtol=1e-6)


def test_register_campaign_parameters():
    mmm = _make_mock_mmm()
    effect = NestedMediaEffect(child_to_parent=MAPPING)  # Michaelis-Menten
    model = mmm.model

    def const(value):
        return as_xtensor(np.full(len(CAMPAIGNS), value), dims=("campaign",))

    with model:
        effect.create_data(mmm)
        effect._built._create_distributions()
        effect._register_campaign_parameters(model, const(1.0), const(2.0), const(3.0))

    alpha, lam, beta_c, lam_c = pm.draw(
        [
            model["nested_media_saturation_alpha"],
            model["nested_media_saturation_lam"],
            model["nested_media_beta_campaign"],
            model["nested_media_lam_campaign"],
        ],
        random_seed=4,
    )
    parent = [["tv", "search"].index(MAPPING[c]) for c in CAMPAIGNS]
    np.testing.assert_allclose(beta_c, alpha[parent] * 2.0 * 3.0, rtol=1e-6)
    np.testing.assert_allclose(lam_c, lam[parent] * 2.0, rtol=1e-6)


def test_no_amplitude_deterministic_without_amplitude_param():
    mmm, _ = _build(
        NestedMediaEffect(child_to_parent=MAPPING, saturation=TanhSaturation())
    )
    assert "nested_media_beta_campaign" not in mmm.model.named_vars
    assert "nested_media_campaign_contribution" in mmm.model.named_vars


def test_multipliers_have_spend_weighted_mean_one_within_channel():
    # the multipliers' spend-weighted arithmetic mean is one in every channel,
    # which is all the channel total can see; the standardised deviations keep
    # their spend-share-weighted zero sum
    mmm, _ = _build()
    with mmm.model:
        idata = pm.sample_prior_predictive(draws=13, random_seed=4)
    prior = idata.prior
    spend = mmm.xarray_dataset["campaign_data"].values
    total = spend.sum(axis=0)
    mult = prior["nested_media_beta_multiplier"].values  # (chain, draw, campaign)
    z = prior["nested_media_z_beta"].values
    for channel in ["tv", "search"]:
        idx = [i for i, c in enumerate(CAMPAIGNS) if MAPPING[c] == channel]
        w = total[idx] / total[idx].sum()
        np.testing.assert_allclose((mult[..., idx] * w).sum(-1), 1.0, rtol=1e-10)
        np.testing.assert_allclose((z[..., idx] * w).sum(-1), 0.0, atol=1e-10)


def test_zero_sum_multipliers_opt_out():
    mmm = _make_mock_mmm()
    effect = NestedMediaEffect(child_to_parent=MAPPING, zero_sum_multipliers=False)
    with mmm.model:
        effect.create_data(mmm)
        effect.create_effect(mmm)
    assert "nested_media_z_beta_tv" not in mmm.model.named_vars
    assert "nested_media_live_index" not in mmm.model.named_vars
    assert mmm.model["nested_media_z_beta"].type.dims == ("campaign",)


def test_all_single_campaign_channels_stay_constrained():
    # zero_sum_multipliers=True with no channel holding two live campaigns
    # must pin every multiplier at 1, not fall back to free per-campaign
    # multipliers confounded with the channel amplitude
    campaigns = ["c1", "c2"]
    mapping = {"c1": "ch1", "c2": "ch2"}
    rng = np.random.default_rng(3)
    dates = pd.date_range("2025-01-01", periods=20, freq="W-MON")
    ds = xr.Dataset(
        {"campaign_data": (("date", "campaign"), rng.gamma(2.0, 1.0, (20, 2)))},
        coords={"date": dates, "campaign": campaigns},
    )
    model = pm.Model(coords={"date": dates, "campaign": campaigns})
    mmm = type("MockMMM", (), {"dims": (), "model": model, "xarray_dataset": ds})()
    effect = NestedMediaEffect(child_to_parent=mapping)
    with model:
        effect.create_data(mmm)
        effect.create_effect(mmm)
        idata = pm.sample_prior_predictive(draws=7, random_seed=5)
    assert not any("z_beta" in rv.name for rv in model.free_RVs)
    np.testing.assert_allclose(
        idata.prior["nested_media_beta_multiplier"].values, 1.0, atol=1e-12
    )


def test_campaign_dim_need_not_be_last():
    # the data constants are computed positionally; a dataset with the
    # campaign dim first must give the same scale, cap and shares
    def constants(ds):
        model = pm.Model(coords={"date": ds.coords["date"], "campaign": CAMPAIGNS})
        mmm = type("MockMMM", (), {"dims": (), "model": model, "xarray_dataset": ds})()
        effect = NestedMediaEffect(child_to_parent=MAPPING)
        with model:
            effect.create_data(mmm)
        return {
            name: model[f"nested_media_{name}"].get_value()
            for name in [
                "channel_scale",
                "campaign_cap",
                "spend_share",
                "live_index",
            ]
        }

    # square data so a wrong axis does not even raise
    ds = _make_mock_mmm(n_dates=len(CAMPAIGNS)).xarray_dataset
    expected = constants(ds)
    transposed = constants(ds.transpose("campaign", "date"))
    for name, value in expected.items():
        np.testing.assert_allclose(transposed[name], value)


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
    effect = NestedMediaEffect(child_to_parent=mapping)
    with model:
        effect.create_data(mmm)
        effect.create_effect(mmm)
        idata = pm.sample_prior_predictive(draws=7, random_seed=5)
    mult = idata.prior["nested_media_beta_multiplier"].sel(campaign="solo_camp")
    np.testing.assert_allclose(mult.values, 1.0, atol=1e-12)
    # only the two-campaign channel gets a constrained variable
    assert "nested_media_z_beta_tv" not in model.named_vars
    assert model["nested_media_z_beta_search"].type.dims == (
        "nested_media_search_campaign",
    )
    assert list(model.coords["nested_media_search_campaign"]) == [
        "search_a",
        "search_b",
    ]


def test_serialization_roundtrip():
    effect = NestedMediaEffect(
        child_to_parent=MAPPING, prefix="cm", rho=0.7, tau_beta_sigma=0.3
    )
    data = effect.to_dict()
    data["__type__"] = (
        f"{NestedMediaEffect.__module__}.{NestedMediaEffect.__qualname__}"
    )
    restored = serialization.deserialize(data)
    assert type(restored) is NestedMediaEffect
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
    effect = NestedMediaEffect(child_to_parent=MAPPING, covariate_var="covariates")
    with mmm.model:
        effect.create_data(mmm)
        effect.create_effect(mmm)
    assert "nested_media_covariates" in mmm.model.named_vars
    assert "nested_media_gamma" in mmm.model.named_vars

    cov_centred = mmm.model["nested_media_covariates"].values.eval()
    spend = mmm.xarray_dataset["campaign_data"].values
    total = spend.sum(axis=0)
    for channel in ["tv", "search"]:
        idx = [i for i, c in enumerate(CAMPAIGNS) if MAPPING[c] == channel]
        share = total[idx] / total[idx].sum()
        np.testing.assert_allclose(share @ cov_centred[idx], 0.0, atol=1e-12)


def test_covariates_finite_with_all_zero_channel():
    # an all-zero-spend channel has no share to centre by; it must not poison
    # the covariates of the live channels with NaN
    mmm = _make_mock_mmm_with_covariates()
    spend = mmm.xarray_dataset["campaign_data"]
    search = [c for c in CAMPAIGNS if MAPPING[c] == "search"]
    spend.loc[{"campaign": search}] = 0.0
    effect = NestedMediaEffect(child_to_parent=MAPPING, covariate_var="covariates")
    with mmm.model, pytest.warns(UserWarning, match="no spend"):
        effect.create_data(mmm)
        effect.create_effect(mmm)
    cov_centred = mmm.model["nested_media_covariates"].values.eval()
    assert np.isfinite(cov_centred).all()
    total = spend.values.sum(axis=0)
    tv = [i for i, c in enumerate(CAMPAIGNS) if MAPPING[c] == "tv"]
    share = total[tv] / total[tv].sum()
    np.testing.assert_allclose(share @ cov_centred[tv], 0.0, atol=1e-12)


def test_covariate_prior_predictive():
    mmm = _make_mock_mmm_with_covariates()
    effect = NestedMediaEffect(child_to_parent=MAPPING, covariate_var="covariates")
    with mmm.model:
        effect.create_data(mmm)
        effect.create_effect(mmm)
        idata = pm.sample_prior_predictive(draws=9, random_seed=3)
    assert "nested_media_campaign_contribution" in idata.prior


def test_covariate_prior_mean_is_the_trust_dial():
    # at the initial point (gamma = gamma_mu, z = 0) the multiplier is
    # exp(gamma_mu * covariate), normalised within its channel: a coefficient
    # of 1 follows the covariate one for one, 0 ignores it
    mmm = _make_mock_mmm_with_covariates()
    for gamma_mu in (0.0, 1.0):
        effect = NestedMediaEffect(
            child_to_parent=MAPPING,
            covariate_var="covariates",
            gamma_mu=gamma_mu,
            gamma_sigma=0.1,
        )
        with mmm.model:
            effect.create_data(mmm)
            effect.create_effect(mmm)
        model = mmm.model
        point = model.initial_point()
        (mult_graph,) = model.replace_rvs_by_values(
            [model["nested_media_beta_multiplier"]]
        )
        mult = model.compile_fn(
            mult_graph, inputs=model.value_vars, on_unused_input="ignore"
        )(point)
        cov = model["nested_media_covariates"].values.eval()
        share = model["nested_media_spend_share"].get_value()
        raw = np.exp(gamma_mu * cov.sum(axis=1))
        expected = raw.copy()
        for channel in ["tv", "search"]:
            idx = [i for i, c in enumerate(CAMPAIGNS) if MAPPING[c] == channel]
            expected[idx] = raw[idx] / (raw[idx] @ share[idx] / share[idx].sum())
        np.testing.assert_allclose(mult, expected, rtol=1e-6)
        mmm = _make_mock_mmm_with_covariates()


def test_zero_spend_campaign_stays_pinned_with_covariates():
    # a dead campaign keeps share 0, so its covariate row must be zero and its
    # multiplier one, whatever the platform export says about it
    mmm = _make_mock_mmm_with_covariates()
    mmm.xarray_dataset["campaign_data"].loc[{"campaign": "tv_promo"}] = 0.0
    effect = NestedMediaEffect(
        child_to_parent=MAPPING, covariate_var="covariates", gamma_mu=1.0
    )
    with mmm.model:
        effect.create_data(mmm)
        effect.create_effect(mmm)
    model = mmm.model
    cov = model["nested_media_covariates"].values.eval()
    dead = CAMPAIGNS.index("tv_promo")
    np.testing.assert_array_equal(cov[dead], 0.0)
    (mult_graph,) = model.replace_rvs_by_values([model["nested_media_beta_multiplier"]])
    mult = model.compile_fn(
        mult_graph, inputs=model.value_vars, on_unused_input="ignore"
    )(model.initial_point())
    assert mult[dead] == pytest.approx(1.0)


def test_field_bounds():
    with pytest.raises(ValueError):
        NestedMediaEffect(child_to_parent=MAPPING, tau_beta_sigma=-1.0)
    with pytest.raises(ValueError):
        NestedMediaEffect(child_to_parent=MAPPING, gamma_sigma=0.0)
    with pytest.raises(ValueError):
        NestedMediaEffect(child_to_parent=MAPPING, rho=-0.1)


def test_channel_scale_follows_mmm_scaling():
    from pymc_marketing.mmm.scaling import DataDerivedScaling, FixedScaling, Scaling

    def scale_with(channel_scaling):
        mmm = _make_mock_mmm()
        mmm.scaling = Scaling(
            target=DataDerivedScaling(method="max", dims=()), channel=channel_scaling
        )
        effect = NestedMediaEffect(child_to_parent=MAPPING)
        with mmm.model:
            effect.create_data(mmm)
        return mmm.model["nested_media_channel_scale"].get_value(), mmm

    spend = _make_mock_mmm().xarray_dataset["campaign_data"]
    totals = {
        ch: spend.sel(campaign=[c for c in CAMPAIGNS if MAPPING[c] == ch]).sum(
            "campaign"
        )
        for ch in ["tv", "search"]
    }
    scale_max, _ = scale_with(DataDerivedScaling(method="max", dims=()))
    np.testing.assert_allclose(scale_max, [totals["tv"].max(), totals["search"].max()])
    scale_mean, _ = scale_with(DataDerivedScaling(method="mean", dims=()))
    np.testing.assert_allclose(
        scale_mean, [totals["tv"].mean(), totals["search"].mean()]
    )
    # reducing over "channel" means over the effect's parents: one common scale
    scale_common, _ = scale_with(DataDerivedScaling(method="max", dims=("channel",)))
    np.testing.assert_allclose(
        scale_common, [max(t.max() for t in totals.values())] * 2
    )
    scale_fixed, _ = scale_with(FixedScaling(dims=(), value=1000.0))
    np.testing.assert_allclose(scale_fixed, [1000.0, 1000.0])
    with pytest.raises(ValueError, match="single number"):
        scale_with(FixedScaling(dims=(), value={"tv": 1.0, "search": 2.0}))


def test_unscaled_saturation_forces_scale_one():
    from pymc_marketing.mmm import LogSaturation

    mmm = _make_mock_mmm()
    effect = NestedMediaEffect(child_to_parent=MAPPING, saturation=LogSaturation())
    with mmm.model:
        effect.create_data(mmm)
    np.testing.assert_array_equal(
        mmm.model["nested_media_channel_scale"].get_value(), 1.0
    )


def test_build_does_not_mutate_saturation():
    mmm = _make_mock_mmm()
    effect = NestedMediaEffect(child_to_parent=MAPPING)
    before = effect.saturation.to_dict()
    with mmm.model:
        effect.create_data(mmm)
        effect.create_effect(mmm)
    assert effect.saturation.to_dict() == before
    assert effect._built.prefix == "nested_media_saturation"


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
    effect = NestedMediaEffect(child_to_parent=MAPPING, covariate_var="bad_cov")
    with mmm.model, pytest.raises(ValueError, match="dims exactly"):
        effect.create_data(mmm)


def test_covariate_serialization_roundtrip():
    effect = NestedMediaEffect(
        child_to_parent=MAPPING,
        covariate_var="covariates",
        gamma_mu=1.0,
        gamma_sigma=0.25,
    )
    data = effect.to_dict()
    data["__type__"] = (
        f"{NestedMediaEffect.__module__}.{NestedMediaEffect.__qualname__}"
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
    effect.add_campaign_lift_test_measurements(df_lift, mmm)
    assert "nested_media_lift_measurements" in mmm.model.named_vars
    logp = mmm.model.compile_logp()(mmm.model.initial_point())
    assert np.isfinite(logp)


def test_lift_test_requires_built_model():
    mmm = _make_mock_mmm()
    effect = NestedMediaEffect(child_to_parent=MAPPING)
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
        effect.add_campaign_lift_test_measurements(df_lift, mmm)


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
        effect.add_campaign_lift_test_measurements(df_lift, mmm)


def _lift_row(**overrides):
    row = {
        "campaign": ["tv_promo"],
        "x": [1.0],
        "delta_x": [2.0],
        "delta_y": [0.3],
        "sigma": [0.02],
    }
    return pd.DataFrame({**row, **overrides})


def test_validate_lift_table():
    mmm = _make_mock_mmm()
    effect = NestedMediaEffect(child_to_parent=MAPPING)
    with mmm.model:
        effect.create_data(mmm)
    with pytest.raises(RuntimeError, match="not been built"):
        effect._validate_lift_table(_lift_row(), mmm.model)

    mmm, effect = _build()
    effect._validate_lift_table(_lift_row(), mmm.model)
    with pytest.raises(KeyError, match="missing columns \\['sigma'\\]"):
        effect._validate_lift_table(_lift_row().drop(columns="sigma"), mmm.model)
    with pytest.raises(ValueError, match="Unknown campaigns"):
        effect._validate_lift_table(_lift_row(campaign=["radio"]), mmm.model)


def test_row_channel_scale_selects_by_channel_and_geo():
    effect = NestedMediaEffect(child_to_parent=MAPPING)
    coords = {"nested_media_channel": ["tv", "search"], "geo": ["north", "south"]}
    with pm.Model(coords=coords) as model:
        pmd.Data(
            "nested_media_channel_scale",
            np.array([[1.0, 2.0], [10.0, 20.0]]),
            dims=("geo", "nested_media_channel"),
        )
    rows = pd.DataFrame({"geo": ["south", "north", "south"]})
    row_channel = pd.Series(["tv", "search", "search"])

    scale = effect._row_channel_scale(rows, row_channel, model)
    np.testing.assert_allclose(scale, [10.0, 2.0, 20.0])
    with pytest.raises(KeyError, match="\\['geo'\\] needed to scale x/delta_x"):
        effect._row_channel_scale(rows.drop(columns="geo"), row_channel, model)


def test_row_target_scale():
    effect = NestedMediaEffect(child_to_parent=MAPPING)
    rows = pd.DataFrame({"geo": ["south", "north"]})

    def mmm_with(target):
        scalers = type("Scalers", (), {"_target": target})()
        return type("MockMMM", (), {"scalers": scalers})()

    no_scalers = type("MockMMM", (), {})()
    np.testing.assert_allclose(effect._row_target_scale(rows, no_scalers), [1, 1])
    np.testing.assert_allclose(effect._row_target_scale(rows, mmm_with(4.0)), [4, 4])
    per_geo = xr.DataArray([2.0, 3.0], dims="geo", coords={"geo": ["north", "south"]})
    np.testing.assert_allclose(
        effect._row_target_scale(rows, mmm_with(per_geo)), [3.0, 2.0]
    )
    with pytest.raises(KeyError, match="\\['geo'\\] needed to scale delta_y/sigma"):
        effect._row_target_scale(rows.drop(columns="geo"), mmm_with(per_geo))


def test_lift_test_likelihood_prior():
    from pymc_extras.prior import Prior

    mmm, effect = _build()
    df_lift = _lift_row(delta_y=[0.3])
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        effect.add_campaign_lift_test_measurements(
            df_lift, mmm, likelihood=Prior("StudentT", nu=5)
        )
    from pytensor.tensor.random.basic import StudentTRV

    observed = mmm.model["nested_media_lift_measurements"]
    assert isinstance(observed.owner.op.core_op, StudentTRV)
    assert np.isfinite(mmm.model.compile_logp()(mmm.model.initial_point()))


def test_lift_test_rejects_rows_outside_contract():
    # the row contract itself is tested in test_lift_test.py
    mmm, effect = _build()
    row = {
        "campaign": ["tv_promo"],
        "x": [1.0],
        "delta_x": [2.0],
        "delta_y": [0.3],
        "sigma": [0.0],
    }
    with pytest.raises(ValueError, match="offending rows: \\[0\\]"):
        effect.add_campaign_lift_test_measurements(pd.DataFrame(row), mmm)


def test_lift_test_accepts_null_result():
    # a measured lift of zero is a legitimate experiment outcome; the Normal
    # default scores it, so the hook must not reject it
    mmm, effect = _build()
    df_lift = pd.DataFrame(
        {
            "campaign": ["tv_promo"],
            "x": [1.0],
            "delta_x": [2.0],
            "delta_y": [0.0],
            "sigma": [0.02],
        }
    )
    effect.add_campaign_lift_test_measurements(df_lift, mmm)
    assert np.isfinite(mmm.model.compile_logp()(mmm.model.initial_point()))


def _estimated_lift(model, effect, ds, campaign, channel, x, delta_x):
    """Model lift of ``campaign`` between spend ``x`` and ``x + delta_x``.

    Mirrors the curve the lift-test hook conditions on, for the default
    Michaelis-Menten saturation, evaluated on parameter draws ``ds``.
    """
    p = effect.prefix
    campaigns = list(model.coords["campaign"])
    channels = list(model.coords[f"{p}_channel"])
    scale = float(model[f"{p}_channel_scale"].get_value()[channels.index(channel)])
    cap = float(model[f"{p}_campaign_cap"].get_value()[campaigns.index(campaign)])
    size = cap**effect.rho
    alpha = ds[f"{p}_saturation_alpha"].sel({f"{p}_channel": channel})
    lam = ds[f"{p}_saturation_lam"].sel({f"{p}_channel": channel})
    beta_mult = ds[f"{p}_beta_multiplier"].sel(campaign=campaign)

    def curve(spend):
        x_rel = (spend / scale) / size
        return size * beta_mult * alpha * x_rel / (x_rel + lam)

    return curve(x + delta_x) - curve(x)


def test_lift_test_calibrates_response_at_operating_point():
    # A lift test identifies the campaign's response between the two spend
    # levels, not any single parameter: the model can explain it through the
    # campaign multiplier, the channel amplitude or the channel half
    # saturation. So we check the quantity the observation pins down, the
    # model's estimated lift at the operating point. The lift value is well
    # beyond the prior predictive lift (median 0.14): the default likelihood
    # must still pull the posterior onto it rather than collapse the lift.
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
    effect.add_campaign_lift_test_measurements(df_lift, mmm)
    with mmm.model:
        prior = pm.sample_prior_predictive(draws=500, random_seed=7)
        idata = pm.sample(
            draws=150,
            tune=300,
            chains=2,
            cores=2,
            random_seed=7,
            progressbar=False,
            compute_convergence_checks=False,
        )
    lift_args = (mmm.model, effect, "tv_promo", "tv", 1.0, 2.0)
    prior_lift = _estimated_lift(*lift_args[:2], prior.prior, *lift_args[2:])
    posterior_lift = _estimated_lift(*lift_args[:2], idata.posterior, *lift_args[2:])
    # the posterior lift concentrates on the measurement, far more tightly
    # than under the prior
    assert abs(float(posterior_lift.median()) - 1.2) < 0.05
    assert float(posterior_lift.std()) < 0.25 * float(prior_lift.std())


def _channel_contribution_at_initial_point(
    campaigns, mapping, spend, saturation=None, point=None, **effect_kwargs
):
    """Channel contribution at the initial point, with ``point`` values overriding it.

    ``point`` maps value-variable names to values, or to a callable taking the
    campaign coordinate and returning the value.
    """
    from pymc_marketing.mmm.nested_media import NestedMediaEffect as _Effect

    dates = pd.date_range("2025-01-01", periods=spend.shape[0], freq="W-MON")
    ds = xr.Dataset(
        {"campaign_data": (("date", "campaign"), spend)},
        coords={"date": dates, "campaign": campaigns},
    )
    model = pm.Model(coords={"date": dates, "campaign": campaigns})
    mmm = type("MockMMM", (), {"dims": (), "model": model, "xarray_dataset": ds})()
    kwargs = {} if saturation is None else {"saturation": saturation}
    effect = _Effect(child_to_parent=mapping, **kwargs, **effect_kwargs)
    with model:
        effect.create_data(mmm)
        effect.create_effect(mmm)
    import pytensor

    (graph,) = model.replace_rvs_by_values([model["nested_media_channel_contribution"]])
    fn = pytensor.function(model.value_vars, graph, on_unused_input="ignore")
    ip = model.initial_point()
    for name, value in (point or {}).items():
        ip[name] = value(campaigns) if callable(value) else value
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


@pytest.mark.parametrize("saturation_cls", [None, "logistic"])
def test_split_invariance_away_from_unit_multipliers(saturation_cls):
    # the homogeneity argument holds for any common multiplier, not only at
    # m = 1: parts of a split campaign that share its multiplier add back to
    # it exactly
    from pymc_marketing.mmm import LogisticSaturation

    saturation = None if saturation_cls is None else LogisticSaturation()
    rng = np.random.default_rng(3)
    spend_a = rng.gamma(2.0, 1.0, 30)
    spend_b = rng.gamma(2.0, 1.0, 30)
    z_by_parent = {"a": 0.8, "b": -0.3}

    point = {
        "nested_media_tau_beta_log__": np.log(0.7),
        "nested_media_z_beta": lambda campaigns: np.array(
            [z_by_parent[c[0]] for c in campaigns]
        ),
    }
    whole = _channel_contribution_at_initial_point(
        ["a", "b"],
        {"a": "ch", "b": "ch"},
        np.column_stack([spend_a, spend_b]),
        saturation,
        point=point,
        zero_sum_multipliers=False,
    )
    split = _channel_contribution_at_initial_point(
        ["a1", "a2", "b"],
        {"a1": "ch", "a2": "ch", "b": "ch"},
        np.column_stack([0.6 * spend_a, 0.4 * spend_a, spend_b]),
        saturation,
        point=point,
        zero_sum_multipliers=False,
    )
    unit = _channel_contribution_at_initial_point(
        ["a", "b"],
        {"a": "ch", "b": "ch"},
        np.column_stack([spend_a, spend_b]),
        saturation,
    )
    assert not np.allclose(whole, unit)  # the multipliers are really away from 1
    np.testing.assert_allclose(split, whole, rtol=1e-10)


def _power_law_contribution(saturation, rho, z_beta):
    """Campaign contribution and the parts of its closed form, at a fixed point."""
    mmm, effect = _build(
        NestedMediaEffect(
            child_to_parent=MAPPING,
            saturation=saturation,
            rho=rho,
            zero_sum_multipliers=False,
        )
    )
    model = mmm.model
    p = "nested_media"
    point = model.initial_point()
    point[f"{p}_tau_beta_log__"] = np.log(0.6)
    point[f"{p}_z_beta"] = z_beta
    names = [
        f"{p}_campaign_contribution",
        f"{p}_beta_multiplier",
        *effect._built.variable_mapping.values(),
    ]
    graphs = model.replace_rvs_by_values([model[n] for n in names])
    values = model.compile_fn(
        graphs, inputs=model.value_vars, on_unused_input="ignore"
    )(point)
    out = {
        n: xr.DataArray(v, dims=model[n].type.dims)
        for n, v in zip(names, values, strict=True)
    }
    parent = model[f"{p}_parent_idx"].get_value()
    scale = model[f"{p}_channel_scale"].get_value()[parent]
    x = mmm.xarray_dataset["campaign_data"].values / scale
    cap = model[f"{p}_campaign_cap"].get_value()
    params = {
        name: out[var].values[parent]
        for name, var in effect._built.variable_mapping.items()
    }
    return model, out, x, cap, params


@pytest.mark.parametrize("saturation_cls", ["no", "root"])
def test_power_law_contribution(saturation_cls):
    from pymc_marketing.mmm import NoSaturation, RootSaturation

    saturation = NoSaturation() if saturation_cls == "no" else RootSaturation()
    z_beta = np.linspace(-1.0, 1.0, len(CAMPAIGNS))
    rho = 0.5
    _, out, x, cap, params = _power_law_contribution(saturation, rho, z_beta)
    p = "nested_media"
    m = out[f"{p}_beta_multiplier"].values
    size = cap**rho
    if saturation_cls == "no":
        expected = m * params["beta"] * x  # the size cancels
    else:
        expected = size * m * params["beta"] * (x / size) ** params["alpha"]
    np.testing.assert_allclose(
        out[f"{p}_campaign_contribution"].transpose("date", "campaign").values,
        expected,
        rtol=1e-10,
    )


@pytest.mark.parametrize("saturation_cls", ["michaelis_menten", "logistic", "no"])
def test_no_x_scale_multiplier(saturation_cls):
    # campaigns differ from the size-tied channel curve by their height only:
    # a second, x-scale multiplier ties the campaigns to the channel total
    # through the nonlinear curve and samples with divergences
    from pymc_marketing.mmm import (
        LogisticSaturation,
        MichaelisMentenSaturation,
        NoSaturation,
    )

    saturation = {
        "michaelis_menten": MichaelisMentenSaturation,
        "logistic": LogisticSaturation,
        "no": NoSaturation,
    }[saturation_cls]()
    mmm, _ = _build(NestedMediaEffect(child_to_parent=MAPPING, saturation=saturation))
    names = set(mmm.model.named_vars)
    assert not {n for n in names if "lam_multiplier" in n or "tau_lam" in n}
    assert not {n for n in names if "z_lam" in n}


def test_rho_has_no_effect_without_saturation():
    from pymc_marketing.mmm import NoSaturation

    z_beta = np.linspace(-1.0, 1.0, len(CAMPAIGNS))
    contributions = [
        _power_law_contribution(NoSaturation(), rho, z_beta)[1][
            "nested_media_campaign_contribution"
        ].values
        for rho in (0.0, 1.0)
    ]
    np.testing.assert_allclose(contributions[0], contributions[1], rtol=1e-12)


@pytest.mark.parametrize("rho", [0.0, 0.5, 1.0])
def test_rho_ties_the_curve_to_size(rho):
    # contribution_c = cap**rho * m * S(x / cap**rho), checked against a
    # closed form at a point where the multipliers are away from 1
    mmm, effect = _build(
        NestedMediaEffect(child_to_parent=MAPPING, rho=rho, zero_sum_multipliers=False)
    )
    model = mmm.model
    p = "nested_media"
    point = model.initial_point()
    point[f"{p}_tau_beta_log__"] = np.log(0.6)
    point[f"{p}_z_beta"] = np.linspace(-1.0, 1.0, len(CAMPAIGNS))
    names = [
        f"{p}_campaign_contribution",
        f"{p}_beta_multiplier",
        effect._built.variable_mapping["alpha"],
        effect._built.variable_mapping["lam"],
    ]
    graphs = model.replace_rvs_by_values([model[n] for n in names])
    values = model.compile_fn(
        graphs, inputs=model.value_vars, on_unused_input="ignore"
    )(point)
    contrib, m, alpha, lam = (
        xr.DataArray(v, dims=model[n].type.dims)
        for n, v in zip(names, values, strict=True)
    )

    parent = model[f"{p}_parent_idx"].get_value()
    channel = f"{p}_channel"
    scale = model[f"{p}_channel_scale"].get_value()[parent]
    cap = xr.DataArray(model[f"{p}_campaign_cap"].get_value(), dims="campaign")
    x = mmm.xarray_dataset["campaign_data"] / xr.DataArray(scale, dims="campaign")
    alpha_c = alpha.isel({channel: xr.DataArray(parent, dims="campaign")})
    lam_c = lam.isel({channel: xr.DataArray(parent, dims="campaign")})
    size = cap**rho
    x_rel = x.values / size.values
    expected = size.values * m.values * alpha_c.values * x_rel / (lam_c.values + x_rel)
    np.testing.assert_allclose(
        contrib.transpose("date", "campaign").values, expected, rtol=1e-10
    )


@pytest.mark.parametrize("saturation_cls", [None, "logistic"])
def test_flighting_split_is_nearly_invariant(saturation_cls):
    # a part that runs on alternate dates inherits its parent's intensity, so
    # the channel contribution barely moves; with a max-based size the part
    # that misses the peak shrinks and the total moves by several percent
    from pymc_marketing.mmm import LogisticSaturation

    saturation = None if saturation_cls is None else LogisticSaturation()
    rng = np.random.default_rng(3)
    spend_a = rng.gamma(2.0, 1.0, 60)
    spend_b = rng.gamma(2.0, 1.0, 60)
    odd = np.arange(60) % 2 == 1

    whole = _channel_contribution_at_initial_point(
        ["a", "b"],
        {"a": "ch", "b": "ch"},
        np.column_stack([spend_a, spend_b]),
        saturation,
    )
    split = _channel_contribution_at_initial_point(
        ["a1", "a2", "b"],
        {"a1": "ch", "a2": "ch", "b": "ch"},
        np.column_stack(
            [np.where(odd, spend_a, 0), np.where(~odd, spend_a, 0), spend_b]
        ),
        saturation,
    )
    np.testing.assert_allclose(split.sum(), whole.sum(), rtol=5e-3)


def test_single_campaign_channel_has_cap_one():
    mmm = _make_mock_mmm()
    mapping = {c: ("solo" if c == CAMPAIGNS[0] else "rest") for c in CAMPAIGNS}
    effect = NestedMediaEffect(child_to_parent=mapping)
    with mmm.model:
        effect.create_data(mmm)
    cap = mmm.model["nested_media_campaign_cap"].get_value()
    assert cap[0] == pytest.approx(1.0)


@pytest.mark.parametrize("rho", [1.0, 0.5])
def test_zero_spend_campaign_cannot_be_funded(rho):
    # cap 0: the dead campaign contributes nothing even when given spend, so a
    # forecast or the budget optimizer cannot route money to it. The live mask
    # is derived from the cap in the graph; a fractional rho is where a wrong
    # size for the dead campaign would show as a non-finite contribution
    campaigns = ["live_a", "live_b", "dead"]
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
    effect = NestedMediaEffect(child_to_parent=dict.fromkeys(campaigns, "ch"), rho=rho)
    with model, pytest.warns(UserWarning, match="cap 0"):
        effect.create_data(mmm)
        effect.create_effect(mmm)
    assert model["nested_media_campaign_cap"].get_value()[2] == 0.0
    with model:
        pm.set_data({"campaign_data": np.full_like(spend, 3.0)})
    (graph,) = model.replace_rvs_by_values(
        [model["nested_media_campaign_contribution"]]
    )
    contribution = model.compile_fn(
        graph, inputs=model.value_vars, on_unused_input="ignore"
    )(model.initial_point())
    axis = model.named_vars_to_dims["nested_media_campaign_contribution"].index(
        "campaign"
    )
    assert np.all(np.isfinite(contribution))
    np.testing.assert_array_equal(np.take(contribution, 2, axis=axis), 0.0)
    assert np.all(np.take(contribution, [0, 1], axis=axis) > 0)


def test_float_dust_spend_counts_as_no_spend():
    # a residue like 1e-12 where a zero was meant must not make a campaign
    # live: its cap would underflow and the contribution turn NaN
    campaigns = ["a", "b", "dust"]
    rng = np.random.default_rng(1)
    spend = np.column_stack(
        [rng.gamma(2.0, 1.0, 52) * 1e5, rng.gamma(2.0, 1.0, 52) * 1e5, np.zeros(52)]
    )
    spend[7, 2] = 1e-12
    dates = pd.date_range("2025-01-06", periods=52, freq="W-MON")
    ds = xr.Dataset(
        {"campaign_data": (("date", "campaign"), spend)},
        coords={"date": dates, "campaign": campaigns},
    )
    model = pm.Model(coords={"date": dates, "campaign": campaigns})
    mmm = type("MockMMM", (), {"dims": (), "model": model, "xarray_dataset": ds})()
    effect = NestedMediaEffect(child_to_parent=dict.fromkeys(campaigns, "ch"))
    with model, pytest.warns(UserWarning, match=r"\['dust'\] have no spend"):
        effect.create_data(mmm)
        effect.create_effect(mmm)
    assert model["nested_media_campaign_cap"].get_value()[2] == 0.0
    (graph,) = model.replace_rvs_by_values([model["nested_media_effect_contribution"]])
    total = model.compile_fn(graph, inputs=model.value_vars, on_unused_input="ignore")(
        model.initial_point()
    )
    assert np.all(np.isfinite(total))


def test_float_dust_campaign_is_dead_everywhere():
    # spend below float dust gives cap 0 but a positive spend share; the
    # zero-sum blocks and the covariates must follow the cap, not the share
    campaigns = ["a", "b", "dust"]
    rng = np.random.default_rng(1)
    spend = np.column_stack(
        [rng.gamma(2.0, 1.0, 52) * 1e5, rng.gamma(2.0, 1.0, 52) * 1e5, np.zeros(52)]
    )
    spend[7, 2] = 1e-12
    dates = pd.date_range("2025-01-06", periods=52, freq="W-MON")
    ds = xr.Dataset(
        {
            "campaign_data": (("date", "campaign"), spend),
            "covariates": (("campaign", "covariate"), [[0.0], [1.0], [4.0]]),
        },
        coords={"date": dates, "campaign": campaigns, "covariate": ["log_ctr"]},
    )
    model = pm.Model(coords={"date": dates, "campaign": campaigns})
    mmm = type("MockMMM", (), {"dims": (), "model": model, "xarray_dataset": ds})()
    effect = NestedMediaEffect(
        child_to_parent=dict.fromkeys(campaigns, "ch"), covariate_var="covariates"
    )
    with model, pytest.warns(UserWarning, match=r"\['dust'\] have no spend"):
        effect.create_data(mmm)
        effect.create_effect(mmm)
        idata = pm.sample_prior_predictive(draws=20, random_seed=4)

    assert model["nested_media_spend_share"].get_value()[2] > 0
    assert list(model.coords["nested_media_ch_campaign"]) == ["a", "b"]
    assert model["nested_media_covariates"].get_value()[2, 0] == 0.0
    mult = idata.prior["nested_media_beta_multiplier"]
    np.testing.assert_array_equal(mult.sel(campaign="dust"), 1.0)
    # the live campaigns' weighted mean is exactly one, not off by the dust term
    share = model["nested_media_spend_share"].get_value()[:2]
    live_mean = mult.sel(campaign=["a", "b"]).values @ share / share.sum()
    np.testing.assert_allclose(live_mean, 1.0, rtol=1e-12)


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
    effect = NestedMediaEffect(child_to_parent=mapping)
    with model, pytest.warns(UserWarning, match="no spend"):
        effect.create_data(mmm)
        effect.create_effect(mmm)
        idata = pm.sample_prior_predictive(draws=7, random_seed=2)
    # only the two live campaigns enter the channel's constraint
    assert "dead" not in model.coords["nested_media_ch_campaign"]
    assert len(model.coords["nested_media_ch_campaign"]) == 2
    # the dead campaign's multiplier is pinned to the pooled value
    mult = idata.prior["nested_media_beta_multiplier"].sel(campaign="dead")
    np.testing.assert_allclose(mult.values, 1.0, atol=1e-12)


def test_library_saturation_shapes():
    from pymc_marketing.mmm import HillSaturationSigmoid, LogisticSaturation

    for saturation in [LogisticSaturation(), HillSaturationSigmoid()]:
        mmm = _make_mock_mmm()
        effect = NestedMediaEffect(child_to_parent=MAPPING, saturation=saturation)
        with mmm.model:
            effect.create_data(mmm)
            effect.create_effect(mmm)
            idata = pm.sample_prior_predictive(draws=5, random_seed=3)
        assert "nested_media_campaign_contribution" in idata.prior
        # channel-level saturation params exist with the effect's channel dim
        for var_name in effect._built.variable_mapping.values():
            assert mmm.model[var_name].type.dims == ("nested_media_channel",)


def test_deserialize_without_prior_import():
    # MMM.load resolves effects by their registered type name, so the effect
    # must be registered as soon as pymc_marketing.mmm is imported
    import subprocess
    import sys

    code = (
        "import pymc_marketing.mmm\n"
        "from pymc_marketing.serialization import serialization\n"
        "data = {'__type__': 'pymc_marketing.mmm.nested_media.NestedMediaEffect',"
        " 'child_to_parent': {'a': 'ch'}}\n"
        "eff = serialization.deserialize(data)\n"
        "print(type(eff).__name__)\n"
    )
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "NestedMediaEffect"


seed = 20260825
SOCIAL_CAMPAIGNS = ["social_a", "social_b", "social_c"]
SOCIAL_MAPPING = {c: "social" for c in SOCIAL_CAMPAIGNS}


@pytest.fixture(scope="module")
def campaign_mmm_data():
    """Data for NestedMediaEffect wired into the real MMM class.

    The extra ``campaign`` dimension only exists when ``X`` is passed as an
    ``xr.Dataset``; the DataFrame API keeps channel/control columns only. The
    decomposed channel ("social") is deliberately absent from
    ``channel_columns`` because the effect replaces its channel-level media
    term.
    """
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
        coords={"date": dates, "channel": ["search"], "campaign": SOCIAL_CAMPAIGNS},
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
        else NestedMediaEffect(child_to_parent=SOCIAL_MAPPING)
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
        "nested_media_parent_idx",
        "nested_media_channel_scale",
        "nested_media_campaign_contribution",
        "nested_media_channel_contribution",
        "nested_media_effect_contribution",
    ]:
        assert name in named
    assert list(built_mmm.model.coords["campaign"]) == SOCIAL_CAMPAIGNS
    assert list(built_mmm.model.coords["nested_media_channel"]) == ["social"]
    # the decomposed channel is not a model channel
    assert list(built_mmm.model.coords["channel"]) == ["search"]


def test_prior_predictive_and_rollup(built_mmm):
    with built_mmm.model:
        idata = pm.sample_prior_predictive(draws=20, random_seed=seed)
    prior = idata.prior
    np.testing.assert_allclose(
        prior["nested_media_campaign_contribution"].sum("campaign"),
        prior["nested_media_effect_contribution"],
        rtol=1e-10,
    )
    assert prior["nested_media_effect_contribution"].dims == (
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
    assert "nested_media_effect" in contributions.data_vars
    assert "campaign" not in contributions.dims


def test_idata_attrs_roundtrip_effect(campaign_mmm_data):
    from pymc_marketing.serialization import serialization

    effect = NestedMediaEffect(child_to_parent=SOCIAL_MAPPING, rho=0.5)
    payload = serialization.serialize(effect)
    restored = serialization.deserialize(payload)
    assert type(restored) is NestedMediaEffect
    assert restored == effect


def test_double_count_warning(campaign_mmm_data):
    """Mapping campaigns onto a channel still in channel_columns must warn."""
    X = campaign_mmm_data["X"]
    mmm = _make_mmm(
        NestedMediaEffect(child_to_parent=dict.fromkeys(SOCIAL_CAMPAIGNS, "search"))
    )
    with pytest.warns(UserWarning, match="double-counts"):
        mmm.build_model(X=X, y=campaign_mmm_data["y"])


def test_budget_optimizer_allocates_to_campaigns(campaign_mmm_data):
    """Prior draws stand in for a posterior, as the funnel fixtures do."""
    from pydantic import ValidationError

    X = campaign_mmm_data["X"].copy(deep=True)
    X["campaign_data"].loc[{"campaign": "social_c"}] = 0.0  # a dead campaign
    mmm = _make_mmm()
    with pytest.warns(UserWarning, match=r"\['social_c'\] have no spend"):
        mmm.build_model(X=X, y=campaign_mmm_data["y"])
    with mmm.model:
        idata = pm.sample_prior_predictive(draws=20, random_seed=seed)
    idata["/posterior"] = idata["/prior"].to_dataset()
    idata["/fit_data"] = mmm.create_fit_data(
        campaign_mmm_data["X_df"], campaign_mmm_data["y_series"]
    )
    mmm.idata = idata
    mmm.set_idata_attrs(idata=idata)
    dates = X.indexes["date"]
    window = {
        "start_date": dates[-1] + pd.Timedelta(weeks=1),
        "end_date": dates[-1] + pd.Timedelta(weeks=4),
        "spend_vars": ["campaign_data"],
    }

    # the default objective is the channel contribution, which cannot see the
    # effect: refused at construction rather than starving the campaigns
    with pytest.raises(ValidationError, match="does not depend on spend_vars"):
        mmm.budget_optimizer(**window)

    optimizer = mmm.budget_optimizer(
        **window, response_variable="total_response_original_scale"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # default budget bounds
        result = optimizer.allocate_budget(total_budget=4.0)
    campaigns = result.optimized_vars["campaign_data"]
    assert list(campaigns["campaign"].values) == SOCIAL_CAMPAIGNS
    total = float(result.budgets.sum()) + float(campaigns.sum())
    np.testing.assert_allclose(total, 4.0, rtol=1e-6)
    # cap 0: no money on a campaign the model cannot fund (up to solver
    # round-off)
    assert float(campaigns.sel(campaign="social_c")) < 1e-9 * 4.0
    assert float(campaigns.sel(campaign=["social_a", "social_b"]).sum()) > 0.0


@pytest.mark.parametrize("explicit_scaling", [True, False])
def test_unscaled_saturation_warns_on_real_mmm(campaign_mmm_data, explicit_scaling):
    """A configured channel scaling is overridden for unscaled saturations, loudly."""
    from pymc_marketing.mmm import DataDerivedScaling, LogSaturation, Scaling

    scaling = (
        Scaling(
            target=DataDerivedScaling(dims=(), method="max"),
            channel=DataDerivedScaling(dims=(), method="mean"),
        )
        if explicit_scaling
        else None
    )
    mmm = MMM(
        channel_columns=["search"],
        date_column="date",
        target_column="y",
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        scaling=scaling,
    ).add_mu_effect(
        NestedMediaEffect(child_to_parent=SOCIAL_MAPPING, saturation=LogSaturation())
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mmm.build_model(X=campaign_mmm_data["X"], y=campaign_mmm_data["y"])
    messages = [str(w.message) for w in caught]
    fired = any("requires unscaled inputs" in m for m in messages)
    assert fired == explicit_scaling
    np.testing.assert_array_equal(
        mmm.model["nested_media_channel_scale"].get_value(), 1.0
    )


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

    fname = tmp_path / "nested_media_model.nc"
    mmm.save(str(fname))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = MMM.load(str(fname))
    assert isinstance(loaded.mu_effects[0], NestedMediaEffect)


@pytest.fixture(scope="module")
def geo_mmm():
    """A real MMM with an extra ``geo`` model dim and the effect added."""
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
            "campaign": SOCIAL_CAMPAIGNS,
        },
    )
    y = xr.DataArray(target, dims=("date", "geo"), coords={"date": dates, "geo": geos})
    mmm = MMM(
        channel_columns=["search"],
        date_column="date",
        target_column="y",
        dims=("geo",),
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    ).add_mu_effect(NestedMediaEffect(child_to_parent=SOCIAL_MAPPING))
    mmm.build_model(X=X, y=y)
    return mmm


class TestGeoDims:
    """The effect with a real MMM carrying an extra model dim."""

    def test_contribution_dims(self, geo_mmm):
        named = geo_mmm.model.named_vars
        assert set(named["nested_media_campaign_contribution"].type.dims) == {
            "date",
            "geo",
            "campaign",
        }
        assert set(named["nested_media_effect_contribution"].type.dims) == {
            "date",
            "geo",
        }

    def test_prior_predictive_and_rollup(self, geo_mmm):
        with geo_mmm.model:
            idata = pm.sample_prior_predictive(draws=10, random_seed=seed)
        prior = idata.prior
        np.testing.assert_allclose(
            prior["nested_media_campaign_contribution"].sum("campaign"),
            prior["nested_media_effect_contribution"],
            rtol=1e-10,
        )
        # the MMM's default channel scaling reduces over its model dims, so
        # the effect's channel scale is one number per channel, like the MMM's
        assert geo_mmm.model.named_vars_to_dims["nested_media_channel_scale"] == (
            "nested_media_channel",
        )
        assert geo_mmm.model["nested_media_channel_scale"].get_value().shape == (1,)


def _geo_dataset(rng):
    n_dates, geos = 16, ["north", "south"]
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
            "campaign": SOCIAL_CAMPAIGNS,
        },
    )
    y = xr.DataArray(target, dims=("date", "geo"), coords={"date": dates, "geo": geos})
    return X, y


def test_lift_test_with_per_geo_channel_scale_builds():
    # unpatched: the cap and channel scale carry the geo dim, and the lift
    # observation graph must index them by the row's geo
    from pymc_marketing.mmm import DataDerivedScaling, Scaling

    X, y = _geo_dataset(np.random.default_rng(seed))
    effect = NestedMediaEffect(child_to_parent=SOCIAL_MAPPING)
    mmm = MMM(
        channel_columns=["search"],
        date_column="date",
        target_column="y",
        dims=("geo",),
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        scaling=Scaling(
            target=DataDerivedScaling(dims=(), method="max"),
            channel=DataDerivedScaling(dims=(), method="max"),
        ),
    ).add_mu_effect(effect)
    mmm.build_model(X=X, y=y)
    df_lift = pd.DataFrame(
        {
            "campaign": ["social_c", "social_a"],
            "geo": ["south", "north"],
            "x": [0.1, 0.2],
            "delta_x": [0.1, 0.1],
            "delta_y": [0.05, 0.08],
            "sigma": [0.02, 0.03],
        }
    )
    effect.add_campaign_lift_test_measurements(df_lift, mmm)
    assert "nested_media_lift_measurements" in mmm.model.named_vars
    assert np.isfinite(mmm.model.compile_logp()(mmm.model.initial_point()))

    # the estimated lift of each row uses that row's geo for both the channel
    # scale and the campaign cap: check it by value, away from the initial
    # point, so a swapped geo or campaign index cannot pass
    model = mmm.model
    p = effect.prefix
    rng = np.random.default_rng(11)
    point = {
        k: v + rng.normal(0.0, 0.3, np.shape(v))
        for k, v in model.initial_point().items()
    }
    names = [
        f"{p}_lift_measurements_model_estimated_lift",
        f"{p}_beta_multiplier",
        effect._built.variable_mapping["alpha"],
        effect._built.variable_mapping["lam"],
    ]
    values = model.compile_fn(
        model.replace_rvs_by_values([model[n] for n in names]),
        inputs=model.value_vars,
        on_unused_input="ignore",
    )(point)
    lift, m, alpha, lam = (
        xr.DataArray(v, dims=model[n].type.dims, coords=_coords_of(model, n))
        for n, v in zip(names, values, strict=True)
    )
    scale = _data_array(model, f"{p}_channel_scale")
    cap = _data_array(model, f"{p}_campaign_cap")
    assert "geo" in scale.dims and "geo" in cap.dims

    expected = []
    for row in df_lift.itertuples():
        size = float(cap.sel(geo=row.geo, campaign=row.campaign)) ** effect.rho
        x_scale = float(scale.sel(geo=row.geo, nested_media_channel="social"))
        m_c = float(m.sel(campaign=row.campaign))
        a = float(alpha.sel(nested_media_channel="social"))
        k = float(lam.sel(nested_media_channel="social"))

        def curve(spend, size=size, x_scale=x_scale, m_c=m_c, a=a, k=k):
            x_rel = spend / x_scale / size
            return size * m_c * a * x_rel / (k + x_rel)

        expected.append(curve(row.x + row.delta_x) - curve(row.x))
    np.testing.assert_allclose(lift.values, expected, rtol=1e-10)

    with pytest.raises(KeyError, match="geo"):
        effect.add_campaign_lift_test_measurements(df_lift.drop(columns="geo"), mmm)


def _coords_of(model, name):
    """Labelled coordinates of a model variable; length-only dims are skipped."""
    return {
        d: list(model.coords[d])
        for d in model[name].type.dims
        if model.coords.get(d) is not None
    }


def _data_array(model, name):
    """A registered data variable as a labelled DataArray."""
    dims = tuple(model.named_vars_to_dims[name])
    return xr.DataArray(
        model[name].get_value(),
        dims=dims,
        coords={d: list(model.coords[d]) for d in dims},
    )


@pytest.mark.parametrize("reduce_dims", [(), ("geo",)])
def test_lift_test_scaling_with_model_dims(monkeypatch, reduce_dims):
    # the lift table is scaled by the campaign's channel scale (x, delta_x)
    # and by the fitted target scaler (delta_y, sigma), selecting on exactly
    # the dims the scaler carries. `DataDerivedScaling.dims` are the dims
    # reduced over: () keeps a per-geo scaler, ("geo",) gives a global one.
    import pymc_marketing.mmm.nested_media as nested_media
    from pymc_marketing.mmm import DataDerivedScaling, Scaling

    X, y = _geo_dataset(np.random.default_rng(seed))
    mmm = MMM(
        channel_columns=["search"],
        date_column="date",
        target_column="y",
        dims=("geo",),
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        scaling=Scaling(
            target=DataDerivedScaling(dims=reduce_dims, method="max"),
            channel=DataDerivedScaling(dims=reduce_dims, method="max"),
        ),
    ).add_mu_effect(NestedMediaEffect(child_to_parent=SOCIAL_MAPPING))
    mmm.build_model(X=X, y=y)
    target_scale = xr.DataArray(mmm.scalers._target)
    per_geo = "geo" in target_scale.dims
    assert per_geo == (reduce_dims == ())

    captured = {}
    monkeypatch.setattr(
        nested_media,
        "add_saturation_observations",
        lambda df, **kw: captured.update(df=df),
    )
    df_lift = pd.DataFrame(
        {
            "campaign": ["social_c", "social_a"],
            "geo": ["south", "north"],
            "x": [0.1, 0.2],
            "delta_x": [0.1, 0.1],
            "delta_y": [0.05, 0.08],
            "sigma": [0.02, 0.03],
        }
    )
    mmm.mu_effects[0].add_campaign_lift_test_measurements(df_lift, mmm)
    scaled = captured["df"]

    # the effect's channel scale follows the MMM's channel scaling: per geo
    # with dims=(), one number per channel with dims=("geo",)
    scale_dims = mmm.model.named_vars_to_dims["nested_media_channel_scale"]
    assert scale_dims == (
        ("geo", "nested_media_channel") if per_geo else ("nested_media_channel",)
    )
    scale_da = xr.DataArray(
        mmm.model["nested_media_channel_scale"].get_value(),
        dims=scale_dims,
        coords={d: list(mmm.model.coords[d]) for d in scale_dims},
    )
    if per_geo:
        channel_scale = np.array(
            [
                float(scale_da.sel(geo=g, nested_media_channel="social"))
                for g in df_lift["geo"]
            ]
        )
    else:
        channel_scale = np.full(
            len(df_lift), float(scale_da.sel(nested_media_channel="social"))
        )
    if per_geo:
        row_target = np.array([float(target_scale.sel(geo=g)) for g in df_lift["geo"]])
    else:
        row_target = np.full(len(df_lift), float(target_scale))
    np.testing.assert_allclose(scaled["x"], df_lift["x"] / channel_scale)
    np.testing.assert_allclose(scaled["delta_x"], df_lift["delta_x"] / channel_scale)
    np.testing.assert_allclose(scaled["delta_y"], df_lift["delta_y"] / row_target)
    np.testing.assert_allclose(scaled["sigma"], df_lift["sigma"] / row_target)
    assert list(scaled["nested_media_channel"]) == ["social", "social"]

    # a per-geo scaler needs the geo column; a global one does not
    no_geo = df_lift.drop(columns="geo")
    if per_geo:
        with pytest.raises(KeyError, match="geo"):
            mmm.mu_effects[0].add_campaign_lift_test_measurements(no_geo, mmm)
    else:
        mmm.mu_effects[0].add_campaign_lift_test_measurements(no_geo, mmm)


def test_set_data_without_campaign_data_zeroes_spend(built_mmm, campaign_mmm_data):
    # the DataFrame prediction route cannot carry campaign_data; the effect
    # must not reuse the training spend nor fail on a different window length
    effect = built_mmm.mu_effects[0]
    X_new = campaign_mmm_data["X"].isel(date=slice(0, 5)).drop_vars("campaign_data")
    ds = built_mmm._posterior_predictive_data_transformation(X=X_new)
    model = built_mmm._set_xarray_data(dataset_xarray=ds, model=built_mmm.model.copy())
    with pytest.warns(UserWarning, match="campaign_data"):
        effect.set_data(built_mmm, model, ds)
    value = model["campaign_data"].get_value()
    assert value.shape == (5, len(SOCIAL_CAMPAIGNS))
    np.testing.assert_array_equal(value, 0.0)
    # the training model is untouched
    assert built_mmm.model["campaign_data"].get_value().shape == (
        24,
        len(SOCIAL_CAMPAIGNS),
    )


def test_set_data_with_campaign_data_updates_spend(built_mmm, campaign_mmm_data):
    effect = built_mmm.mu_effects[0]
    X_new = campaign_mmm_data["X"].isel(date=slice(0, 5))
    ds = built_mmm._posterior_predictive_data_transformation(X=X_new)
    model = built_mmm._set_xarray_data(dataset_xarray=ds, model=built_mmm.model.copy())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        effect.set_data(built_mmm, model, ds)
    np.testing.assert_array_equal(
        model["campaign_data"].get_value(), X_new["campaign_data"].values
    )


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
    effect.add_campaign_lift_test_measurements(df_lift, mmm)
    assert "nested_media_lift_measurements" in mmm.model.named_vars
    assert np.isfinite(mmm.model.compile_logp()(mmm.model.initial_point()))
