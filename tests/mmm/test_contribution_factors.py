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
"""Statistical and lifecycle contract for fixed-training contribution factors."""

import copy
import json
import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from pymc_extras.prior import Prior
from scipy.stats import norm

from pymc_marketing.mmm import MMM, GeometricAdstock
from pymc_marketing.mmm.scaling import DataDerivedScaling, Scaling
from tests.mmm.contribution_factors_helpers import data as data
from tests.mmm.contribution_factors_helpers import (
    evaluate,
    make_model,
)
from tests.mmm.contribution_factors_helpers import labeled_panel as labeled_panel


@pytest.mark.parametrize("panel", [False, True])
def test_aggregation_logdensity_and_unchanged_likelihood(data, panel):
    X, y = data
    if panel:
        X = pd.concat(
            [X.assign(country="A"), X.assign(country="B", search=X.search * 2)],
            ignore_index=True,
        )
        y = pd.concat([y, y * 7], ignore_index=True)
    m = make_model(
        dims=("country",) if panel else (),
        scaling=Scaling(
            target=DataDerivedScaling(method="max", dims=()),
            channel=DataDerivedScaling(method="max", dims=()),
        ),
    )
    m.build_model(X, y)
    point = m.model.initial_point()
    base_logp = m.model.compile_logp()(point)
    likelihood = evaluate(m.model, [m.model.logp(vars=m.model.observed_RVs)], point)[0]
    m.add_original_scale_contribution_variable(["channel_contribution"])
    contributions = evaluate(
        m.model, [m.model["channel_contribution_original_scale"]], point
    )[0]
    if panel:
        mask = (
            X.assign(observed=y.notna().to_numpy())
            .set_index(["date", "country"])["observed"]
            .to_xarray()
            .transpose("date", "country")
            .values
        )
    else:
        mask = y.notna().to_numpy()
    expected_channel = np.where(mask[..., None], contributions, 0).sum(
        axis=tuple(range(contributions.ndim - 1))
    )
    denominator = np.nansum(y)
    for name, channels, scale in [
        ("all_share", ["search", "tiktok"], "share"),
        ("individual", ["tiktok"], "total"),
        ("nested", ["tiktok"], "share"),
    ]:
        m.add_contribution_factor(
            name, channels, scale, Prior("Normal", mu=0.3, sigma=2)
        )
    actual = evaluate(
        m.model, [m.model[n] for n in ["all_share", "individual", "nested"]], point
    )
    expected = [
        expected_channel.sum() / denominator,
        expected_channel[1],
        expected_channel[1] / denominator,
    ]
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(
        m.model.compile_logp()(point) - base_logp, norm.logpdf(expected, 0.3, 2).sum()
    )
    np.testing.assert_allclose(
        evaluate(m.model, [m.model.logp(vars=m.model.observed_RVs)], point)[0],
        likelihood,
    )
    # No extra RV, repeated likelihood, renormalization, or duplicated full array.
    assert len(m.model.potentials) == 3
    assert len(m.model.observed_RVs) == 1


def test_frozen_reference_prediction_optimization_and_rebuild(data):
    X, y = data
    m = make_model()
    m.build_model(X, y)
    m.add_contribution_factor(
        "belief", ["search"], "share", Prior("Normal", mu=0.3, sigma=0.1)
    )
    before = evaluate(
        m.model, [m.model["belief"], m.model.logp(vars=m.model.potentials)]
    )
    reference = copy.deepcopy(m._contribution_factors.fingerprint)
    # Change both dates and data, as prediction and optimizer windows do.
    with m.model:
        pm.set_data(
            {"channel_data": np.full((8, 2), 1000.0), "target_data": np.full(8, 999.0)},
            coords={"date": pd.date_range("2026-01-01", periods=8)},
        )
    after = evaluate(
        m.model, [m.model["belief"], m.model.logp(vars=m.model.potentials)]
    )
    np.testing.assert_allclose(after, before)
    assert m._contribution_factors.fingerprint == reference
    # Adding another component after prediction must still use training inputs.
    m.add_contribution_factor(
        "second", ["search"], "share", Prior("Normal", mu=0.3, sigma=0.1)
    )
    np.testing.assert_allclose(evaluate(m.model, [m.model["second"]])[0], before[0])
    m.build_model(X, y)
    np.testing.assert_allclose(evaluate(m.model, [m.model["belief"]])[0], before[0])
    assert len(m.model.potentials) == 2
    original_model = m.model
    with pytest.raises(ValueError, match="original training"):
        m.build_model(X.assign(search=X.search * 2), y)
    assert m.model is original_model
    with pytest.raises(ValueError, match="original training"):
        m.fit(X.assign(search=X.search * 2), y)
    assert m.model is original_model


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"name": "bad name"}, "identifier"),
        ({"name": "mu"}, "already in use"),
        ({"name": "channel_contribution_original_scale"}, "already in use"),
        ({"channels": []}, "nonempty"),
        ({"channels": "search"}, "nonempty"),
        ({"channels": ["search", "search"]}, "distinct"),
        ({"channels": ["missing"]}, "unknown"),
        ({"quantity": "average"}, "quantity"),
        ({"prior": Prior("Normal", mu=[0.2, 0.3], sigma=1)}, "fixed finite scalars"),
        (
            {"prior": Prior("Normal", mu=Prior("Normal"), sigma=1)},
            "fixed finite scalars",
        ),
        ({"prior": Prior("Normal", dims="channel")}, "scalar Prior"),
        ({"prior": Prior("Normal", transform="exp")}, "scalar Prior"),
        ({"prior": Prior("Uniform", lower=0, upper=0.05)}, "hard caps"),
        ({"prior": Prior("Beta", alpha=2, beta=5)}, "including Beta"),
        ({"prior": Prior("Poisson", mu=2)}, "LogNormal/HalfNormal"),
        ({"prior": Prior("Gamma", alpha=0.5, beta=1)}, "LogNormal/HalfNormal"),
        ({"prior": Prior("Gamma", alpha=2, beta=1)}, "LogNormal/HalfNormal"),
        ({"prior": Prior("Weibull", alpha=0.5, beta=1)}, "LogNormal/HalfNormal"),
        ({"prior": Prior("Normal", mu=0.2, sigma=-1)}, "sigma"),
    ],
)
def test_invalid_input_does_not_mutate_graph(data, kwargs, match):
    m = make_model()
    m.build_model(*data)
    names = set(m.model.named_vars)
    values = {
        "name": "belief",
        "channels": ["search"],
        "quantity": "share",
        "prior": Prior("Normal", mu=0.3, sigma=0.1),
    } | kwargs
    with pytest.raises(ValueError, match=match):
        m.add_contribution_factor(**values)
    assert set(m.model.named_vars) == names
    assert m._contribution_factors.specs == []


def test_duplicates_and_unsupported_models(data):
    m = make_model()
    with pytest.raises(ValueError, match="built"):
        m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    m.build_model(*data)
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    names = set(m.model.named_vars)
    with pytest.raises(ValueError, match="already in use"):
        m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    assert set(m.model.named_vars) == names
    from pymc_marketing.mmm import LinearTrend
    from pymc_marketing.mmm.additive_effect import LinearTrendEffect

    with pytest.raises(ValueError, match="before adding"):
        m.add_mu_effect(
            LinearTrendEffect(trend=LinearTrend(n_changepoints=2), prefix="trend")
        )
    log_model = make_model(link="log")
    log_model.build_model(data[0], data[1].fillna(1))
    with pytest.raises(ValueError, match="identity-link"):
        log_model.add_contribution_factor("other", ["search"], "share", Prior("Normal"))


@pytest.mark.parametrize("target", [np.zeros(5), -np.ones(5), np.full(5, np.nan)])
def test_invalid_denominator(data, target):
    m = make_model()
    m.build_model(data[0], pd.Series(target))
    names = set(m.model.named_vars)
    with pytest.raises(ValueError, match=r"finite positive|finite observed"):
        m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    assert set(m.model.named_vars) == names


@pytest.mark.parametrize(
    "prior", [Prior("Normal", mu=0.3, sigma=0.1), Prior("LogNormal", mu=-1, sigma=0.5)]
)
def test_save_load_restore_original_missingness(data, tmp_path, prior):
    X, y = data
    m = make_model()
    m.build_model(X, y)
    m.add_contribution_factor("belief", ["search"], "share", prior)
    before = evaluate(
        m.model, [m.model["belief"], m.model.logp(vars=m.model.potentials)]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.sample_prior_predictive(X, y, samples=3, random_seed=123)
    # Serialization needs a posterior group; forward draws suffice for this graph round-trip.
    m.idata["posterior"] = m.idata.prior.dataset
    m.idata["fit_data"] = m.create_fit_data(X, y.copy())
    # Simulate the zero-filled fit_data case too: saved mask must win.
    m.idata["fit_data"]["y"] = m.idata.fit_data["y"].fillna(0)
    m.save(tmp_path / "belief.nc")
    loaded = MMM.load(tmp_path / "belief.nc")
    assert loaded._contribution_factors.specs == m._contribution_factors.specs
    assert (
        loaded._contribution_factors.fingerprint == m._contribution_factors.fingerprint
    )
    np.testing.assert_allclose(
        evaluate(
            loaded.model,
            [loaded.model["belief"], loaded.model.logp(vars=loaded.model.potentials)],
        ),
        before,
    )
    loaded.build_from_idata(loaded.idata)
    assert len(loaded.model.potentials) == 1
    assert np.isfinite(loaded.model.compile_dlogp()(loaded.model.initial_point())).all()
    assert not bool(loaded.idata.contribution_factor_data.observed_mask.isel(date=2))


def test_xarray_two_dimensions_and_missing_cells(labeled_panel):
    """Label alignment, rather than axis position, defines pooled training cells."""
    ds, _, _ = labeled_panel
    m = make_model(dims=("region", "product"))
    m.build_model(ds)
    m.add_contribution_factor("pooled", ["search"], "share", Prior("Normal"))
    source = m.model["channel_contribution"] * m.model["target_scale"]
    contributions = xr.DataArray(
        evaluate(m.model, [source])[0],
        dims=source.dims,
        coords={d: np.asarray(m.model.coords[d]) for d in source.dims},
    )
    expected = (
        contributions.sel(channel="search").where(np.isfinite(ds["_target"]), 0).sum()
        / ds["_target"].sum()
    )
    np.testing.assert_allclose(evaluate(m.model, [m.model["pooled"]])[0], expected)


@pytest.mark.parametrize(
    "layout",
    [
        ("date", "region", "product", "channel"),
        ("region", "channel", "date", "product"),
        ("channel", "product", "date", "region"),
    ],
)
@pytest.mark.parametrize("permute_labels", [False, True])
def test_equivalent_named_axes_rebuild_fit_and_persist(
    labeled_panel,
    layout,
    permute_labels,
    tmp_path,
    mock_pymc_sample,
):
    """Both reference identity and frozen input values respect dimension names."""
    ds, X, y = labeled_panel
    if permute_labels:
        ds = ds.sel(region=["B", "A"], product=["q", "p"])
    original = ds.transpose(*layout)
    m = make_model(dims=("region", "product"))
    m.build_model(original)
    m.add_contribution_factor(
        "belief", ["search"], "share", Prior("Normal", mu=0.3, sigma=0.1)
    )
    reference = copy.deepcopy(m._contribution_factors.fingerprint)
    expected = evaluate(
        m.model, [m.model["belief"], m.model.logp(vars=m.model.potentials)]
    )
    m.build_model(original.transpose("product", "date", "region", "channel"))
    assert m._contribution_factors.fingerprint == reference
    np.testing.assert_allclose(
        evaluate(m.model, [m.model["belief"], m.model.logp(vars=m.model.potentials)]),
        expected,
    )
    # Sample generation is mocked; the valid wide fit_data, graph round-trip,
    # missing-cell mask and named frozen inputs are persisted and restored for real.
    m.fit(X, y)
    path = tmp_path / "named-axes.nc"
    m.save(path)
    loaded = MMM.load(path)
    assert loaded._contribution_factors.fingerprint == reference
    assert len(loaded.model.potentials) == 1
    np.testing.assert_allclose(
        evaluate(
            loaded.model,
            [loaded.model["belief"], loaded.model.logp(vars=loaded.model.potentials)],
        ),
        expected,
    )
    loaded.build_model(original)
    assert loaded._contribution_factors.fingerprint == reference
    assert len(loaded.model.potentials) == 1
    np.testing.assert_allclose(
        evaluate(loaded.model, [loaded.model["belief"]])[0], expected[0]
    )
    assert np.isfinite(loaded.model.compile_dlogp()(loaded.model.initial_point())).all()
    with pytest.raises(ValueError, match="original training"):
        loaded.build_model(original.assign(_channel=original._channel + 1))


@pytest.mark.parametrize(
    "name", ["date", "channel", "y_original_scale", "belief_factor"]
)
def test_reserved_names_leave_graph_unchanged(data, name):
    m = make_model()
    m.build_model(*data)
    names = set(m.model.named_vars)
    with pytest.raises(ValueError, match=r"in use|reserved"):
        m.add_contribution_factor(name, ["search"], "share", Prior("Normal"))
    assert set(m.model.named_vars) == names
    assert not m.model.potentials


@pytest.mark.parametrize("channels", [np.array(["search"]), pd.Index(["search"])])
def test_channel_collections(data, channels):
    m = make_model()
    m.build_model(*data)
    m.add_contribution_factor("belief", channels, "share", Prior("Normal"))
    assert np.isfinite(evaluate(m.model, [m.model["belief"]])[0])
    assert "channel_contribution_original_scale" not in m.model.named_vars


@pytest.mark.parametrize(
    "prior",
    [
        Prior("Normal", mu=0.3, sigma=0.1),
        Prior("StudentT", nu=4, mu=0.3, sigma=0.1),
        Prior("Laplace", mu=0.3, b=0.1),
        Prior("SkewNormal", mu=0.3, sigma=0.1, alpha=2),
        Prior("AsymmetricLaplace", mu=0.3, b=10, kappa=1.5),
        Prior("LogNormal", mu=-1, sigma=0.5),
        Prior("HalfNormal", sigma=0.5),
    ],
)
def test_density_gradient_matches_finite_differences(data, prior):
    from pymc.blocking import DictToArrayBijection, RaveledVars

    m = make_model()
    m.build_model(*data)
    original = m.model.compile_logp()
    m.add_contribution_factor("belief", ["search"], "share", prior)
    logp = m.model.compile_logp()
    gradient = m.model.compile_dlogp()
    point = m.model.initial_point()
    point = {v.name: point[v.name] for v in m.model.value_vars}
    flat = DictToArrayBijection.map(point)
    for shift in (-0.3, 0.2):
        values = flat.data.copy() + shift
        current = DictToArrayBijection.rmap(RaveledVars(values, flat.point_map_info))
        expected = []
        for j in range(len(values)):
            plus = values.copy()
            plus[j] += 1e-5
            minus = values.copy()
            minus[j] -= 1e-5
            expected.append(
                (
                    logp(
                        DictToArrayBijection.rmap(
                            RaveledVars(plus, flat.point_map_info)
                        )
                    )
                    - logp(
                        DictToArrayBijection.rmap(
                            RaveledVars(minus, flat.point_map_info)
                        )
                    )
                )
                / 2e-5
            )
        np.testing.assert_allclose(gradient(current), expected, rtol=2e-5, atol=2e-5)
        share = evaluate(m.model, [m.model["belief"]], current)[0]
        expected_logp = pm.logp(
            prior.pymc_distribution.dist(**prior.parameters), share
        ).eval()
        np.testing.assert_allclose(logp(current) - original(current), expected_logp)


@pytest.mark.parametrize("existing_factor", [False, True])
def test_nonnegative_factor_rejects_incompatible_zero_without_mutation(
    data, existing_factor
):
    X, y = data
    m = make_model()
    m.build_model(X.assign(tiktok=0.0), y)
    if existing_factor:
        m.add_contribution_factor("search_belief", ["search"], "share", Prior("Normal"))
    names = set(m.model.named_vars)
    state = copy.deepcopy(m._contribution_factors.specs)
    source = m._contribution_factors.frozen_graph
    fingerprint = m._contribution_factors.fingerprint
    with pytest.raises(ValueError, match="nonzero historical channel inputs"):
        m.add_contribution_factor(
            "zero_share", ["tiktok"], "share", Prior("LogNormal", mu=-1, sigma=0.5)
        )
    assert set(m.model.named_vars) == names
    assert m._contribution_factors.specs == state
    assert m._contribution_factors.frozen_graph is source
    assert m._contribution_factors.fingerprint == fingerprint


@pytest.mark.parametrize("quantity", ["share", "total"])
def test_nonnegative_factor_uses_density_without_upper_clipping(data, quantity):
    from scipy.stats import lognorm

    m = make_model()
    m.build_model(*data)
    original = m.model.compile_logp()
    point = m.model.initial_point()
    # Shares relative to observed revenue can exceed one: retain that value.
    point["saturation_beta_log__"] = np.full_like(
        point["saturation_beta_log__"], np.log(10.0)
    )
    m.add_contribution_factor(
        "belief", ["search", "tiktok"], quantity, Prior("LogNormal", mu=-1, sigma=0.5)
    )
    aggregate = evaluate(m.model, [m.model["belief"]], point)[0]
    assert aggregate > 1
    np.testing.assert_allclose(
        m.model.compile_logp()(point) - original(point),
        lognorm.logpdf(aggregate, s=0.5, scale=np.exp(-1)),
    )


def test_failed_factor_restoration_rolls_back(data, monkeypatch):
    from pymc_marketing.mmm.contribution_factors import ContributionFactorState

    m = make_model()
    m.build_model(*data)
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    original_model = m.model
    before = evaluate(m.model, [m.model["belief"], m.model.logp()])
    with monkeypatch.context() as patch:
        patch.setattr(
            ContributionFactorState,
            "_register_factor",
            lambda *a: (_ for _ in ()).throw(RuntimeError("restore failed")),
        )
        with pytest.raises(RuntimeError, match="restore failed"):
            m.build_model(*data)
    assert m.model is original_model
    np.testing.assert_allclose(
        evaluate(m.model, [m.model["belief"], m.model.logp()]), before
    )
    m.build_model(*data)
    np.testing.assert_allclose(
        evaluate(m.model, [m.model["belief"], m.model.logp()]), before
    )


def test_post_fit_addition_rejected(data, mock_pymc_sample):
    m = make_model()
    m.fit(*data)
    names = set(m.model.named_vars)
    with pytest.raises(ValueError, match="before fitting"):
        m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    assert set(m.model.named_vars) == names


def test_trend_effect_supported_before_factors(data):
    from pymc_marketing.mmm import LinearTrend
    from pymc_marketing.mmm.additive_effect import LinearTrendEffect

    m = make_model()
    m.add_mu_effect(
        LinearTrendEffect(trend=LinearTrend(n_changepoints=2), prefix="trend")
    )
    m.build_model(*data)
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    value = evaluate(m.model, [m.model["belief"]])[0]
    m.build_model(*data)
    np.testing.assert_allclose(evaluate(m.model, [m.model["belief"]])[0], value)


@pytest.mark.parametrize(
    "prefix, match",
    [
        ("indirect", r"contribution node .* was not found"),
        (None, "inspectable MuEffects"),
    ],
)
def test_unregistered_custom_effect_rejected_without_mutation(data, prefix, match):
    from pymc_marketing.mmm.additive_effect import MuEffect

    class DirectSpendEffect(MuEffect):
        prefix: str | None = "indirect"

        def create_data(self, mmm):
            pass

        def create_effect(self, mmm):
            return 0.2 * mmm.model["channel_data"].sum("channel")

        def set_data(self, mmm, model, X):
            pass

    m = make_model()
    m.add_mu_effect(DirectSpendEffect(prefix=prefix))
    m.build_model(*data)
    names = set(m.model.named_vars)
    before = evaluate(m.model, [m.model.logp()])
    with pytest.raises(ValueError, match=match):
        m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    assert set(m.model.named_vars) == names
    assert m._contribution_factors.specs == []
    np.testing.assert_allclose(evaluate(m.model, [m.model.logp()]), before)


def test_incompatible_belief_warning(data):
    m = make_model()
    m.build_model(*data)
    with pytest.warns(UserWarning, match="less than 0.1%"):
        m.add_contribution_factor(
            "belief", ["search"], "share", Prior("Normal", mu=-0.2, sigma=0.05)
        )


def test_time_slice_cv_rejects_rebinding_historical_factors(data):
    from pymc_marketing.mmm import TimeSliceCrossValidator

    m = make_model()
    m.build_model(*data)
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    cv = TimeSliceCrossValidator(n_init=3, forecast_horizon=1, date_column="date")
    with pytest.raises(ValueError, match="time-slice cross-validation"):
        cv.run(*data, mmm=m)
    assert "belief" in m.model.named_vars


def test_spend_dependent_effect_rejected(funnel_identity_fitted_mmm):
    m = copy.deepcopy(funnel_identity_fitted_mmm)
    m.idata = None
    names = set(m.model.named_vars)
    with pytest.raises(ValueError, match="spend-dependent"):
        m.add_contribution_factor("belief", ["channel_1"], "share", Prior("Normal"))
    assert set(m.model.named_vars) == names


@pytest.mark.parametrize("corruption", ["missing_mask", "unknown_version"])
def test_failed_load_restoration_preserves_existing_model(data, corruption):
    m = make_model()
    m.build_model(*data)
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    model = m.model
    before = evaluate(model, [model["belief"], model.logp()])
    broken = xr.DataTree.from_dict({"fit_data": m.create_fit_data(*data)})
    broken.attrs.update(m.create_idata_attrs())
    error: type[Exception] = AttributeError
    match = "contribution_factor_data"
    if corruption == "unknown_version":
        metadata = json.loads(broken.attrs["contribution_factors"])
        metadata["version"] = 2
        broken.attrs["contribution_factors"] = json.dumps(metadata)
        error, match = (
            ValueError,
            "Unsupported contribution-factor serialization version",
        )
    with pytest.raises(error, match=match):
        m.build_from_idata(broken)
    assert m.model is model
    assert m.idata is None
    np.testing.assert_allclose(
        evaluate(m.model, [m.model["belief"], m.model.logp()]), before
    )


@pytest.mark.parametrize("family", ["LogNormal", "HalfNormal"])
@pytest.mark.parametrize("quantity", ["share", "total"])
def test_positive_factors_reject_signed_effect_support_atomically(
    data, family, quantity
):
    m = make_model(
        model_config={
            "saturation_beta": Prior("Normal", mu=0.3, sigma=0.5, dims="channel")
        }
    )
    m.build_model(*data)
    m.add_contribution_factor("existing", ["search"], "share", Prior("Normal"))
    names = set(m.model.named_vars)
    before = copy.deepcopy(m._contribution_factors.specs)
    with pytest.raises(ValueError, match="nonnegative support"):
        m.add_contribution_factor(
            "positive", ["tiktok"], quantity, Prior(family, sigma=0.05)
        )
    assert set(m.model.named_vars) == names
    assert m._contribution_factors.specs == before


@pytest.mark.parametrize("name", ["chain", "draw", "sample"])
def test_sampling_dimension_names_are_reserved(data, name):
    m = make_model()
    m.build_model(*data)
    names = set(m.model.named_vars)
    with pytest.raises(ValueError, match="reserved"):
        m.add_contribution_factor(name, ["search"], "share", Prior("Normal"))
    assert set(m.model.named_vars) == names
    assert not m._contribution_factors.specs


def test_total_rejects_zero_target_scaling(data):
    m = make_model()
    m.build_model(data[0], np.zeros(len(data[0])))
    names = set(m.model.named_vars)
    with pytest.raises(ValueError, match="positive target scaling"):
        m.add_contribution_factor("total", ["search"], "total", Prior("Normal"))
    assert set(m.model.named_vars) == names


def test_factor_rejects_positive_placeholder_target(data):
    from pymc_marketing.special_priors import LogNormalPrior

    m = make_model(model_config={"likelihood": LogNormalPrior(std=0.1)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        m.sample_prior_predictive(data[0], samples=2, random_seed=123)
    assert m._target_is_placeholder
    with pytest.raises(ValueError, match="placeholder target"):
        m.add_contribution_factor("total", ["search"], "total", Prior("Normal"))


@pytest.mark.parametrize("extra_potential", [False, True])
def test_predictive_warning_is_specific_to_unclassified_potentials(
    data, mock_pymc_sample, extra_potential
):
    m = make_model()
    m.build_model(*data)
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    if extra_potential:
        with m.model:
            pm.Potential("unclassified", pm.math.constant(0.0))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        m.sample_prior_predictive(*data, samples=2, random_seed=123)
    messages = [str(w.message) for w in caught]
    assert sum("not the effective prior" in msg for msg in messages) == 1
    assert any("The effect of Potentials" in msg for msg in messages) == extra_potential
    m.fit(*data)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        m.sample_posterior_predictive(data[0], extend_idata=False, random_seed=123)
    assert (
        any("The effect of Potentials" in str(w.message) for w in caught)
        == extra_potential
    )


def test_failed_factor_rebuild_isolates_mutable_configuration(data, monkeypatch):
    m = make_model()
    m.build_model(*data)
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    model = m.model
    effects = m.mu_effects
    configuration = copy.deepcopy(m.model_config)
    original = MMM._build_graph

    def broken(working, X, y):
        original(working, X, y)
        working.mu_effects.append(object())
        working.model_config["saturation_beta"] = Prior("Normal")
        raise RuntimeError("reject after in-place mutations")

    monkeypatch.setattr(MMM, "_build_graph", broken)
    with pytest.raises(RuntimeError, match="in-place mutations"):
        m.build_model(*data)
    assert m.model is model
    assert m.mu_effects is effects
    assert m.mu_effects == []
    assert m.model_config == configuration


@pytest.mark.parametrize("family", ["LogNormal", "HalfNormal"])
def test_positive_factor_rebuild_rejects_new_signed_effects(data, family):
    m = make_model()
    m.build_model(*data)
    m.add_contribution_factor("belief", ["search"], "share", Prior(family, sigma=0.1))
    model = m.model
    m.model_config["saturation_beta"] = Prior(
        "Normal", mu=0.3, sigma=0.5, dims="channel"
    )
    with pytest.raises(ValueError, match="nonnegative support"):
        m.build_model(*data)
    assert m.model is model
    assert len(m._contribution_factors.specs) == 1


@pytest.mark.parametrize("family", ["LogNormal", "HalfNormal"])
@pytest.mark.parametrize("parameter", ["saturation_lam", "adstock_alpha"])
def test_positive_factor_rejects_signed_shape_parameters(data, family, parameter):
    m = make_model(
        model_config={parameter: Prior("Normal", mu=0.5, sigma=1, dims="channel")}
    )
    m.build_model(*data)
    names = set(m.model.named_vars)
    with pytest.raises(ValueError, match=r"nonnegative|sign"):
        m.add_contribution_factor(
            "belief", ["search"], "share", Prior(family, sigma=0.1)
        )
    assert set(m.model.named_vars) == names
    assert not m._contribution_factors.specs


@pytest.mark.parametrize("family", ["Normal", "HalfNormal"])
def test_hierarchical_positive_effect_prior_accepts_factor(data, family):
    m = make_model(
        model_config={
            "saturation_beta": Prior(
                "HalfNormal", sigma=Prior("HalfNormal", sigma=1), dims="channel"
            )
        }
    )
    m.build_model(*data)
    m.add_contribution_factor("belief", ["search"], "share", Prior(family, sigma=0.1))
    assert np.isfinite(m.model.compile_dlogp()(m.model.initial_point())).all()


@pytest.mark.parametrize("effect_prior", ["special", "michaelis_menten"])
def test_positive_factor_accepts_supported_positive_amplitudes(data, effect_prior):
    from pymc_marketing.mmm import MichaelisMentenSaturation
    from pymc_marketing.special_priors import LogNormalPrior

    if effect_prior == "special":
        m = make_model(
            model_config={
                "saturation_beta": LogNormalPrior(mean=0.5, std=0.2, dims="channel")
            }
        )
    else:
        m = MMM(
            date_column="date",
            channel_columns=["search", "tiktok"],
            adstock=GeometricAdstock(l_max=2),
            saturation=MichaelisMentenSaturation(),
        )
    m.build_model(*data)
    m.add_contribution_factor(
        "belief", ["search"], "share", Prior("HalfNormal", sigma=0.1)
    )
    assert np.isfinite(m.model.compile_dlogp()(m.model.initial_point())).all()


def test_positive_factor_rejects_signed_media_multiplier(data):
    from pymc_marketing.mmm.hsgp import HSGP

    m = make_model(time_varying_media=HSGP(m=3, L=5, dims="date", eta=1, ls=2))
    m.build_model(*data)
    names = set(m.model.named_vars)
    with pytest.raises(ValueError, match=r"nonnegative|multiplier|sign"):
        m.add_contribution_factor(
            "belief", ["search"], "share", Prior("HalfNormal", sigma=0.1)
        )
    assert set(m.model.named_vars) == names


@pytest.mark.parametrize("locked", [False, True])
def test_factor_rebuild_preserves_user_components(data, locked):
    from threading import Lock
    from typing import Any

    from pydantic import Field

    from pymc_marketing.mmm import LinearTrend
    from pymc_marketing.mmm.additive_effect import LinearTrendEffect

    class ResourceTrend(LinearTrendEffect):
        resource: Any = Field(default=None, exclude=True)

    effect = ResourceTrend(
        trend=LinearTrend(n_changepoints=2),
        prefix="trend",
        resource=Lock() if locked else None,
    )
    m = make_model()
    m.add_mu_effect(effect)
    m.build_model(*data)
    config, effects = m.model_config, m.mu_effects
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    m.build_model(*data)
    assert m.mu_effects is effects
    assert m.mu_effects[0] is effect
    assert m.model_config is config
    assert np.isfinite(m.model.compile_dlogp()(m.model.initial_point())).all()


def test_factor_mask_order_matches_denominator_on_load(
    labeled_panel, tmp_path, mock_pymc_sample
):
    ds, X, y = labeled_panel
    m = make_model(dims=("region", "product"))
    m.build_model(ds.sel(region=["B", "A"], product=["q", "p"]))
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    expected = evaluate(m.model, [m.model["belief"]])[0]
    m.fit(X, y)
    path = tmp_path / "without-posterior.nc"
    m.save(str(path), groups=["fit_data", "contribution_factor_data"])
    loaded = MMM.load(str(path))
    np.testing.assert_allclose(
        evaluate(loaded.model, [loaded.model["belief"]])[0], expected
    )


@pytest.mark.parametrize("inspection", ["graphviz", "table", "networkx"])
def test_documented_xtensor_potential_inspection_limitation(data, inspection):
    from pytensor.graph.utils import MissingInputError

    m = make_model()
    m.build_model(*data)
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    inspect = {
        "graphviz": m.graphviz,
        "table": m.table,
        "networkx": lambda: pm.model_to_networkx(m.model),
    }
    with pytest.raises(MissingInputError):
        inspect[inspection]()
    point = m.model.initial_point()
    m.model.check_start_vals(point)
    assert np.isfinite(m.model.compile_logp()(point))
    assert np.isfinite(m.model.compile_dlogp()(point)).all()


def test_debug_limitation_also_occurs_with_plain_pymc_potential(data, capsys):
    with pm.Model() as plain:
        value = pm.Normal("value")
        pm.Potential("belief", pm.logp(pm.Normal.dist(), value))
    with pytest.raises(TypeError, match="does not support iteration"):
        plain.debug()
    m = make_model()
    m.build_model(*data)
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    with pytest.raises(TypeError, match="does not support iteration"):
        m.model.debug()
    capsys.readouterr()


def test_positive_factor_rejects_negative_historical_inputs(data):
    X, y = data
    m = make_model()
    m.build_model(X.assign(search=-X.search), y)
    names = set(m.model.named_vars)
    with pytest.raises(ValueError, match="nonnegative"):
        m.add_contribution_factor(
            "belief", ["search"], "share", Prior("HalfNormal", sigma=0.1)
        )
    assert set(m.model.named_vars) == names


@pytest.mark.parametrize(
    "saturation_name",
    [
        "LogisticSaturation",
        "InverseScaledLogisticSaturation",
        "MichaelisMentenSaturation",
        "TanhSaturation",
        "TanhSaturationBaselined",
        "HillSaturation",
        "HillSaturationSigmoid",
        "RootSaturation",
        "LogSaturation",
        "NoSaturation",
    ],
)
def test_positive_factor_with_builtin_saturations(data, saturation_name):
    from pymc_marketing.mmm import components

    saturation = getattr(components.saturation, saturation_name)()
    m = MMM(
        date_column="date",
        channel_columns=["search", "tiktok"],
        adstock=GeometricAdstock(l_max=2),
        saturation=saturation,
    )
    m.build_model(*data)
    m.add_contribution_factor(
        "belief", ["search"], "share", Prior("HalfNormal", sigma=0.5)
    )
    assert np.isfinite(m.model.compile_dlogp()(m.model.initial_point())).all()


@pytest.mark.parametrize(
    "adstock_name",
    [
        "BinomialAdstock",
        "GeometricAdstock",
        "DelayedAdstock",
        "NoAdstock",
        "WeibullPDFAdstock",
        "WeibullCDFAdstock",
    ],
)
def test_positive_factor_with_builtin_adstocks(data, adstock_name):
    from pymc_marketing.mmm import LogisticSaturation, components

    adstock = getattr(components.adstock, adstock_name)(l_max=2)
    m = MMM(
        date_column="date",
        channel_columns=["search", "tiktok"],
        adstock=adstock,
        saturation=LogisticSaturation(),
    )
    m.build_model(*data)
    m.add_contribution_factor(
        "belief", ["search"], "share", Prior("HalfNormal", sigma=0.5)
    )
    assert np.isfinite(m.model.compile_dlogp()(m.model.initial_point())).all()


def test_positive_factor_rejects_custom_saturation_override(data):
    from pymc_marketing.mmm import LogisticSaturation

    class SignedSaturation(LogisticSaturation):
        def function(self, x, lam, beta, *, dim=None):
            return -super().function(x, lam, beta, dim=dim)

    m = MMM(
        date_column="date",
        channel_columns=["search", "tiktok"],
        adstock=GeometricAdstock(l_max=2),
        saturation=SignedSaturation(),
    )
    m.build_model(*data)
    with pytest.raises(ValueError, match="SignedSaturation"):
        m.add_contribution_factor(
            "belief", ["search"], "share", Prior("HalfNormal", sigma=0.5)
        )
    m.add_contribution_factor("signed_belief", ["search"], "share", Prior("Normal"))
    assert np.isfinite(m.model.compile_dlogp()(m.model.initial_point())).all()


def test_factor_rebuild_preserves_hsgp_registration_on_failure(data, monkeypatch):
    from pymc_marketing.mmm.hsgp import SoftPlusHSGP

    factory = SoftPlusHSGP(m=3, L=5, dims="date", eta=1, ls=2)
    m = make_model(time_varying_media=factory)
    m.build_model(*data)
    m.add_contribution_factor(
        "belief", ["search"], "share", Prior("HalfNormal", sigma=0.5)
    )
    old_model, old_data = m.model, factory.X
    original = MMM._build_graph

    def broken(working, X, y):
        original(working, X, y)
        raise RuntimeError("failed staged registration")

    with monkeypatch.context() as context:
        context.setattr(MMM, "_build_graph", broken)
        with pytest.raises(RuntimeError, match="failed staged registration"):
            m.build_model(*data)
    assert m.model is old_model
    assert m.time_varying_media is factory
    assert factory.X is old_data
    m.build_model(*data)
    assert m.time_varying_media is factory
    assert factory.X is m.model["time_index"]
