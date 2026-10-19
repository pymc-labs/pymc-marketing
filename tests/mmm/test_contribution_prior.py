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

from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.scaling import DataDerivedScaling, Scaling


def make_model(dims=(), **kwargs):
    return MMM(
        date_column="date",
        channel_columns=["search", "tiktok"],
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        dims=dims,
        **kwargs,
    )


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=5),
            "search": [1.0, 3.0, 2.0, 4.0, 5.0],
            "tiktok": [2.0, 4.0, 3.0, 5.0, 6.0],
        }
    )
    return X, pd.Series([10.0, 20.0, np.nan, 30.0, 40.0])


def evaluate(model, expressions, point=None):
    expressions = [v.values if hasattr(v, "values") else v for v in expressions]
    fn = model.compile_fn(
        model.replace_rvs_by_values(expressions),
        inputs=model.value_vars,
        on_unused_input="ignore",
    )
    return fn(model.initial_point() if point is None else point)


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


def test_save_load_restore_original_missingness(data, tmp_path):
    X, y = data
    m = make_model()
    m.build_model(X, y)
    m.add_contribution_factor(
        "belief", ["search"], "share", Prior("Normal", mu=0.3, sigma=0.1)
    )
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


def test_effective_prior_removes_sales_likelihood_retains_factors(data):
    m = make_model()
    m.build_model(*data)
    m.add_contribution_factor(
        "belief", ["search"], "share", Prior("Normal", mu=0.3, sigma=0.1)
    )
    from unittest.mock import patch

    def inspect_model(**kwargs):
        prior_model = pm.modelcontext(None)
        assert not prior_model.observed_RVs
        point = prior_model.initial_point()
        assert np.isfinite(prior_model.compile_dlogp()(point)).all()
        actual = prior_model.compile_logp()(point)
        parameter_logp = evaluate(
            prior_model, [prior_model.logp(vars=prior_model.free_RVs)], point
        )[0]
        share = evaluate(prior_model, [prior_model["belief"]], point)[0]
        np.testing.assert_allclose(
            actual - parameter_logp, norm.logpdf(share, 0.3, 0.1)
        )
        return xr.DataTree.from_dict(
            {"posterior": xr.Dataset({"belief": (("chain", "draw"), [[share]])})}
        )

    with patch("pymc.sample", side_effect=inspect_model):
        result = m.sample_effective_prior()
    assert "prior" in result and "posterior" not in result
    with pytest.warns(UserWarning, match="not the effective prior"):
        m.sample_prior_predictive(*data, samples=2, random_seed=123)


@pytest.fixture
def labeled_panel():
    dates = pd.date_range("2025-01-01", periods=5)
    coords = {
        "date": dates,
        "region": ["A", "B"],
        "product": ["p", "q"],
        "channel": ["tiktok", "search"],
    }
    ds = xr.Dataset(
        {
            "_channel": xr.DataArray(
                np.arange(40).reshape(2, 2, 5, 2) + 1.0,
                dims=("channel", "product", "date", "region"),
                coords=coords,
            ),
            "_target": xr.DataArray(
                np.arange(20).reshape(2, 5, 2) + 20.0,
                dims=("region", "date", "product"),
                coords={d: coords[d] for d in ["region", "date", "product"]},
            ),
        }
    )
    ds["_target"].loc[{"region": "B", "product": "q", "date": dates[2]}] = np.nan
    channels = ds["_channel"].to_series().unstack("channel").reset_index()
    targets = ds["_target"].to_series().rename("y").reset_index()
    frame = channels.merge(targets, on=["date", "region", "product"])
    return ds, frame.drop(columns="y"), frame["y"]


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
def test_equivalent_named_axes_rebuild_fit_and_persist(
    labeled_panel,
    layout,
    tmp_path,
    mock_pymc_sample,
):
    """Both reference identity and frozen input values respect dimension names."""
    ds, X, y = labeled_panel
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


def test_incompatible_belief_warning(data):
    m = make_model()
    m.build_model(*data)
    with pytest.warns(UserWarning, match="less than 0.1%"):
        m.add_contribution_factor(
            "belief", ["search"], "share", Prior("Normal", mu=-0.2, sigma=0.05)
        )


def test_public_prediction_optimizer_and_effective_prior(data, mock_pymc_sample):
    m = make_model()
    m.build_model(*data)
    m.add_contribution_factor(
        "belief", ["search"], "share", Prior("Normal", mu=0.3, sigma=0.1)
    )
    m.fit(*data)
    point = m.model.initial_point()
    before = evaluate(
        m.model, [m.model["belief"], m.model.logp(vars=m.model.potentials)], point
    )
    posterior = m.idata.posterior.dataset.copy(deep=True)
    future = data[0].assign(
        date=pd.date_range("2026-01-01", periods=5), search=100.0, tiktok=200.0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        m.sample_posterior_predictive(future, extend_idata=False, random_seed=123)
        result = m.budget_optimizer("2026-01-01", "2026-01-03").allocate_budget(
            total_budget=30.0
        )
    assert result is not None
    np.testing.assert_allclose(
        evaluate(
            m.model, [m.model["belief"], m.model.logp(vars=m.model.potentials)], point
        ),
        before,
    )
    posterior = m.idata.posterior.dataset.copy(deep=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        draws = m.sample_effective_prior(
            draws=3,
            tune=3,
            chains=1,
            cores=1,
            random_seed=123,
            progressbar=False,
            compute_convergence_checks=False,
        )
    assert "prior" in draws
    xr.testing.assert_equal(
        m.idata.posterior.dataset[list(posterior.data_vars)], posterior
    )


def test_factor_metadata_is_small(data):
    m = make_model()
    m.build_model(*data)
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    metadata = json.loads(m.create_idata_attrs()["contribution_factors"])
    assert set(metadata) == {"version", "specs", "fingerprint"}
    assert len(json.dumps(metadata)) < 1000


def test_time_slice_cv_rejects_rebinding_historical_factors(data):
    from pymc_marketing.mmm import TimeSliceCrossValidator

    m = make_model()
    m.build_model(*data)
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    cv = TimeSliceCrossValidator(n_init=3, forecast_horizon=1, date_column="date")
    with pytest.raises(ValueError, match="time-slice cross-validation"):
        cv.run(*data, mmm=m)
    assert "belief" in m.model.named_vars


def test_effective_prior_panel_target_and_predictive_parameter_preservation(
    labeled_panel,
):
    from unittest.mock import patch

    ds, _, _ = labeled_panel
    m = make_model(dims=("region", "product"))
    m.build_model(ds)
    m.add_original_scale_contribution_variable(["channel_contribution"])
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    point = m.model.initial_point()
    before = evaluate(
        m.model, [m.model["belief"], m.model.logp(vars=m.model.potentials)], point
    )

    def check_sampling_target(**kwargs):
        sampling = pm.modelcontext(None)
        assert not sampling.observed_RVs
        np.testing.assert_allclose(
            evaluate(
                sampling,
                [sampling["belief"], sampling.logp(vars=sampling.potentials)],
                point,
            ),
            before,
        )
        # Forward draws here supply test values, not an effective-prior approximation.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            prior = pm.sample_prior_predictive(draws=4, random_seed=123).prior.dataset
        return xr.DataTree.from_dict({"posterior": prior})

    with patch("pymc.sample", side_effect=check_sampling_target):
        result = m.sample_effective_prior()
    with m.model, warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        predictive = pm.sample_posterior_predictive(
            result.prior.dataset, var_names=["y", "belief"], random_seed=123
        )
    xr.testing.assert_allclose(
        predictive.posterior_predictive.belief, result.prior.belief
    )


def test_effective_prior_rejects_target_override(data):
    m = make_model()
    m.build_model(*data)
    with pytest.raises(ValueError, match="controls its model"):
        m.sample_effective_prior(model=m.model)


def test_spend_dependent_effect_rejected(funnel_identity_fitted_mmm):
    m = copy.deepcopy(funnel_identity_fitted_mmm)
    m.idata = None
    names = set(m.model.named_vars)
    with pytest.raises(ValueError, match="spend-dependent"):
        m.add_contribution_factor("belief", ["channel_1"], "share", Prior("Normal"))
    assert set(m.model.named_vars) == names


def test_legacy_metadata_never_silently_drops_factors():
    from pymc_marketing.mmm.contribution_factors import ContributionFactorState

    idata = xr.DataTree(
        xr.Dataset(attrs={"contribution_priors": '[{"name":"belief"}]'})
    )
    with pytest.raises(ValueError, match="Legacy contribution-prior"):
        ContributionFactorState.from_idata(idata)


def test_effective_prior_preserves_user_parameters_and_deterministics(data):
    from unittest.mock import patch

    import pymc.dims as pmd

    m = make_model()
    m.build_model(*data)
    with m.model:
        extra = pmd.Normal("extra_parameter", mu=1, sigma=2)
        pmd.Deterministic("extra_squared", extra**2)
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))

    def inspect(**kwargs):
        sampling = pm.modelcontext(None)
        assert "extra_parameter" in {v.name for v in sampling.free_RVs}
        assert "extra_squared" in sampling.named_vars
        assert not sampling.observed_RVs
        return xr.DataTree.from_dict({"posterior": xr.Dataset()})

    with patch("pymc.sample", side_effect=inspect):
        m.sample_effective_prior()


def test_failed_load_restoration_preserves_existing_model(data):
    m = make_model()
    m.build_model(*data)
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    model = m.model
    before = evaluate(model, [model["belief"], model.logp()])
    broken = xr.DataTree.from_dict({"fit_data": m.create_fit_data(*data)})
    broken.attrs.update(m.create_idata_attrs())
    # Missing the required labeled mask must fail without replacing this model.
    with pytest.raises(AttributeError, match="contribution_factor_data"):
        m.build_from_idata(broken)
    assert m.model is model
    assert m.idata is None
    np.testing.assert_allclose(
        evaluate(m.model, [m.model["belief"], m.model.logp()]), before
    )
