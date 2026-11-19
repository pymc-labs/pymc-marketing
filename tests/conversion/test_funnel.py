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
import pytest
import xarray as xr
from pymc_extras.prior import Prior

from pymc_marketing.conversion import FunnelModel, pooled_p
from pymc_marketing.decision import prob_best
from pymc_marketing.terms import Parameter
from tests.conversion.conftest import (
    FIT_KWARGS,
    STAGES,
    STEP_PROBABILITIES,
    simulate_stage_counts,
)

STEPS = STAGES[1:]
SAMPLE_DIMS = ("chain", "draw")


def segment_dataset() -> xr.Dataset:
    """Simulate an A/B test on a closed three-step funnel, split into eight segments.

    The segments differ a lot in size. The smallest one converts nobody at the
    last step in either variant, which is what small segments look like in
    practice and what partial pooling is for.
    """
    rng = np.random.default_rng(0)
    steps = ["signup", "trial", "paid"]
    variants = ["control", "treatment"]
    segments = np.arange(8)
    entrants = np.geomspace(60, 6_000, num=len(segments)).astype(int)
    step_logit = np.array([[-0.6, -0.4], [0.0, 0.0], [-1.1, -0.9]])
    segment_effect = rng.normal(0, 0.4, size=(3, len(segments)))
    probability = 1 / (1 + np.exp(-(step_logit[:, :, None] + segment_effect[:, None])))

    entered = np.empty((3, 2, len(segments)), dtype=int)
    converted = np.empty_like(entered)
    at_risk = np.broadcast_to(entrants, (2, len(segments)))
    for k in range(3):
        entered[k] = at_risk
        converted[k] = rng.binomial(at_risk, probability[k])
        at_risk = converted[k]
    converted[2, :, 0] = 0
    return xr.Dataset(
        {
            "entered": (("step", "variant", "segment"), entered),
            "converted": (("step", "variant", "segment"), converted),
        },
        coords={"step": steps, "variant": variants, "segment": segments},
    )


def test_default_posterior_matches_conjugate_beta(fitted_model, ab_dataset) -> None:
    exact = (1 + ab_dataset["converted"]) / (2 + ab_dataset["entered"])

    posterior_mean = fitted_model.step_conversion().mean(SAMPLE_DIMS)

    xr.testing.assert_allclose(posterior_mean, exact, atol=5e-3, rtol=0)


def test_output_dims(fitted_model) -> None:
    by_step = (*SAMPLE_DIMS, "step", "variant")

    assert fitted_model.step_conversion().dims == by_step
    assert fitted_model.cumulative_conversion().dims == by_step
    assert fitted_model.share_lost().dims == by_step
    assert fitted_model.overall_conversion().dims == (*SAMPLE_DIMS, "variant")
    assert fitted_model.value_per_entrant(40.0).dims == (*SAMPLE_DIMS, "variant")
    assert fitted_model.prob_largest_leak().dims == ("step", "variant")
    assert list(fitted_model.step_conversion()["step"].values) == STEPS


def test_cumulative_conversion_is_the_running_product(fitted_model) -> None:
    step = fitted_model.step_conversion()
    cumulative = fitted_model.cumulative_conversion()

    xr.testing.assert_allclose(cumulative.isel(step=0), step.isel(step=0))
    xr.testing.assert_allclose(
        cumulative.isel(step=2, drop=True), step.isel(step=slice(0, 3)).prod("step")
    )
    xr.testing.assert_allclose(
        cumulative.isel(step=-1, drop=True), fitted_model.overall_conversion()
    )


def test_share_lost_is_the_share_of_all_entrants_lost_at_each_step(
    fitted_model,
) -> None:
    step = fitted_model.step_conversion()
    share_lost = fitted_model.share_lost()

    xr.testing.assert_allclose(share_lost.isel(step=0), 1 - step.isel(step=0))
    np.testing.assert_allclose(
        share_lost.isel(step=2),
        step.isel(step=0) * step.isel(step=1) * (1 - step.isel(step=2)),
    )
    np.testing.assert_allclose(
        share_lost.sum("step") + fitted_model.overall_conversion(), 1.0, atol=1e-12
    )


def test_prob_largest_leak_matches_conjugate_posterior(
    fitted_model, ab_dataset
) -> None:
    rng = np.random.default_rng(0)
    converted = ab_dataset["converted"].values
    entered = ab_dataset["entered"].values
    step = rng.beta(1 + converted, 1 + entered - converted, size=(20_000, 4, 2))
    reached = np.concatenate(
        [np.ones((20_000, 1, 2)), np.cumprod(step, axis=1)[:, :-1]], axis=1
    )
    largest = np.argmax(reached * (1 - step), axis=1)
    expected = np.stack([(largest == k).mean(axis=0) for k in range(4)])

    leak = fitted_model.prob_largest_leak()

    np.testing.assert_allclose(leak.sum("step"), 1.0)
    np.testing.assert_allclose(leak.transpose("step", "variant"), expected, atol=0.05)


def test_treatment_is_detected_as_the_better_variant(fitted_model) -> None:
    best = prob_best(fitted_model.overall_conversion(), "variant")

    assert best.sel(variant="treatment").item() > 0.99


def test_value_per_entrant(fitted_model) -> None:
    overall = fitted_model.overall_conversion()
    value = xr.DataArray(
        [40.0, 50.0], dims="variant", coords={"variant": ["control", "treatment"]}
    )

    xr.testing.assert_allclose(fitted_model.value_per_entrant(40.0), 40.0 * overall)
    xr.testing.assert_allclose(
        fitted_model.value_per_entrant(value).sel(variant="treatment"),
        50.0 * overall.sel(variant="treatment"),
    )


def test_value_per_entrant_with_draws_of_the_value(fitted_model) -> None:
    overall = fitted_model.overall_conversion()
    rng = np.random.default_rng(0)
    value = xr.DataArray(
        rng.normal(40.0, 2.0, size=(overall.sizes["chain"], overall.sizes["draw"])),
        dims=("chain", "draw"),
        coords={"chain": overall["chain"], "draw": overall["draw"]},
    )

    result = fitted_model.value_per_entrant(value)

    assert result.dims == (*SAMPLE_DIMS, "variant")
    np.testing.assert_allclose(result, overall.values * value.values[:, :, None])


def test_value_per_entrant_rejects_draws_that_do_not_line_up(fitted_model) -> None:
    value = xr.DataArray(
        np.full((1, 500), 40.0),
        dims=("chain", "draw"),
        coords={"chain": [7], "draw": np.arange(500)},
    )

    with pytest.raises(ValueError, match="chain"):
        fitted_model.value_per_entrant(value)


def test_funnel_with_a_single_step(ab_dataset) -> None:
    model = FunnelModel()
    model.fit(ab_dataset.isel(step=[0]), **FIT_KWARGS)

    leak = model.prob_largest_leak()

    assert leak.dims == ("step", "variant")
    np.testing.assert_array_equal(leak, 1.0)
    xr.testing.assert_allclose(
        model.share_lost().isel(step=0, drop=True), 1 - model.overall_conversion()
    )


def test_outputs_before_fit_raise() -> None:
    with pytest.raises(RuntimeError, match="hasn't been fit"):
        FunnelModel().step_conversion()


def test_build_model_without_data_raises() -> None:
    with pytest.raises(ValueError, match="No data"):
        FunnelModel().build_model()


def test_build_model_accepts_a_dataframe_of_stage_counts(ab_dataset) -> None:
    counts = simulate_stage_counts(STEP_PROBABILITIES, n_entrants=8_000, seed=0)
    counts["stage"] = pd.Categorical(counts["stage"], categories=STAGES, ordered=True)

    model = FunnelModel()
    model.build_model(counts)

    xr.testing.assert_identical(model.data, ab_dataset)


def test_save_and_load_roundtrip(fitted_model, tmp_path) -> None:
    path = str(tmp_path / "funnel.nc")
    fitted_model.save(path)

    loaded = FunnelModel.load(path)

    assert loaded.id == fitted_model.id
    xr.testing.assert_identical(
        loaded.overall_conversion(), fitted_model.overall_conversion()
    )
    xr.testing.assert_identical(loaded.data, fitted_model.data)


def test_refit_on_other_data_rebuilds(ab_dataset) -> None:
    model = FunnelModel()
    model.fit(ab_dataset, **FIT_KWARGS)

    model.fit(ab_dataset.isel(step=slice(0, 2), variant=0, drop=True), **FIT_KWARGS)

    assert model.step_conversion().dims == (*SAMPLE_DIMS, "step")
    assert list(model.step_conversion()["step"].values) == STEPS[:2]


def test_step_that_nobody_entered_returns_the_prior(ab_dataset) -> None:
    ds = ab_dataset.copy(deep=True)
    ds["converted"].loc[{"step": "checkout", "variant": "control"}] = 0
    for name in ("entered", "converted"):
        ds[name].loc[{"step": "purchase", "variant": "control"}] = 0

    model = FunnelModel()
    model.fit(ds, **FIT_KWARGS)

    empty_step = model.step_conversion().sel(step="purchase", variant="control")
    assert empty_step.mean().item() == pytest.approx(0.5, abs=0.05)
    assert empty_step.std().item() == pytest.approx(12**-0.5, abs=0.03)
    # The step before it converted nobody, so the funnel as a whole still
    # reports a conversion close to zero.
    overall = model.overall_conversion().sel(variant="control")
    assert overall.mean().item() < 1e-3


def test_integer_group_labels_roundtrip(tmp_path) -> None:
    ds = segment_dataset().sel(variant="control", drop=True).isel(segment=slice(5, 8))
    model = FunnelModel()
    model.fit(ds, **FIT_KWARGS)
    path = str(tmp_path / "funnel.nc")
    model.save(path)

    loaded = FunnelModel.load(path)

    overall = loaded.overall_conversion()
    assert list(overall["segment"].values) == [5, 6, 7]
    assert np.issubdtype(overall["segment"].dtype, np.integer)
    xr.testing.assert_identical(overall, model.overall_conversion())


def test_prior_shared_across_variants(ab_dataset) -> None:
    prior = Prior("Beta", alpha=1, beta=1, dims="step")
    model = FunnelModel(model_config={"p": prior})
    model.fit(ab_dataset, **FIT_KWARGS)
    pooled = ab_dataset.sum("variant")
    exact = (1 + pooled["converted"]) / (2 + pooled["entered"])

    step = model.step_conversion()

    assert step.dims == (*SAMPLE_DIMS, "step")
    assert model.overall_conversion().dims == SAMPLE_DIMS
    xr.testing.assert_allclose(step.mean(SAMPLE_DIMS), exact, atol=5e-3, rtol=0)


def test_prior_shared_across_steps_is_broadcast_to_steps(ab_dataset) -> None:
    prior = Prior("Beta", alpha=1, beta=1, dims="variant")
    model = FunnelModel(model_config={"p": prior})
    model.fit(ab_dataset, **FIT_KWARGS)

    step = model.step_conversion()

    assert step.dims == (*SAMPLE_DIMS, "step", "variant")
    xr.testing.assert_allclose(
        model.overall_conversion(), step.isel(step=0, drop=True) ** 4
    )


def test_default_prior_is_not_mutated(ab_dataset) -> None:
    prior = Prior("Beta", alpha=1, beta=1)
    model = FunnelModel(model_config={"p": prior})

    model.build_model(ab_dataset)

    assert model.model_config["p"] is prior
    assert prior.dims is None


def test_p_with_a_dim_the_data_lacks_raises(ab_dataset) -> None:
    prior = Prior("Beta", alpha=1, beta=1, dims=("step", "region"))
    model = FunnelModel(model_config={"p": prior})

    with pytest.raises(ValueError, match="'region'"):
        model.build_model(ab_dataset)


def test_recipe_named_other_than_p_raises(ab_dataset) -> None:
    recipe = Parameter("q", Prior("Beta", alpha=1, beta=1, dims="step"))
    model = FunnelModel(model_config={"p": recipe})

    with pytest.raises(ValueError, match="named 'q'"):
        model.build_model(ab_dataset)


def test_recipe_with_a_dim_the_data_lack_says_which_dims_the_data_have(
    ab_dataset,
) -> None:
    model = FunnelModel(model_config={"p": pooled_p("segment")})

    with pytest.raises(ValueError, match="dimensions \\('step', 'variant'\\)"):
        model.build_model(ab_dataset)


def test_pooled_p_shrinks_a_small_segment_and_keeps_variants_apart(tmp_path) -> None:
    ds = segment_dataset()
    recipe = pooled_p("segment", fixed_dims=["variant"])
    model = FunnelModel(model_config={"p": recipe})
    model.fit(ds, **(FIT_KWARGS | {"target_accept": 0.95}))

    posterior = model.fit_result
    step = model.step_conversion()

    assert posterior["step_logit"].dims == (*SAMPLE_DIMS, "step", "variant")
    assert posterior["segment_effect"].dims == (*SAMPLE_DIMS, "step", "segment")
    assert step.dims == (*SAMPLE_DIMS, "step", "variant", "segment")

    # Segment 0 converted nobody at the last step, so its raw rate is zero.
    # The pooled estimate lies between that and the rate of a typical segment.
    last = {"step": "paid", "variant": "control"}
    typical = 1 / (1 + np.exp(-posterior["step_logit"].sel(last)))
    pooled = step.sel(last).sel(segment=0)
    assert 0.02 < pooled.mean().item() < typical.mean().item()

    path = str(tmp_path / "pooled.nc")
    model.save(path)
    loaded = FunnelModel.load(path)
    assert loaded.id == model.id
    xr.testing.assert_identical(loaded.step_conversion(), step)


def test_pooled_p_without_fixed_dims_shares_the_baseline(ab_dataset) -> None:
    model = FunnelModel(model_config={"p": pooled_p("variant")})

    model.build_model(ab_dataset)

    dims = model.model.named_vars_to_dims
    assert dims["step_logit"] == ("step",)
    assert dims["variant_effect"] == ("step", "variant")
