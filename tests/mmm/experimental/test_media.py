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
from pymc_extras.prior import Prior, Scaled

from pymc_marketing.mmm.components.adstock import GeometricAdstock, NoAdstock
from pymc_marketing.mmm.components.saturation import (
    LogisticSaturation,
    LogSaturation,
    NoSaturation,
)
from pymc_marketing.mmm.experimental._graph import Binding, BuildContext, Equation, walk
from pymc_marketing.mmm.experimental._media import Media
from pymc_marketing.mmm.transformers import ConvMode
from pymc_marketing.terms import Parameter


def _adstock_reference(values: np.ndarray, alpha: float, length: int) -> np.ndarray:
    weights = alpha ** np.arange(length)
    weights = weights / weights.sum()
    result = np.zeros_like(values)
    for lag, weight in enumerate(weights):
        result[lag:] += weight * values[: len(values) - lag]
    return result


@pytest.mark.parametrize(
    "adstock_first", [True, False], ids=["adstock-first", "saturation-first"]
)
def test_pipeline_matches_numeric_reference_with_labeled_scales(
    adstock_first: bool,
) -> None:
    dates = pd.date_range("2025-01-01", periods=5)
    ds = xr.Dataset(
        {
            "tv": (("geo", "date"), [[3, 1, 5, 0, 2], [2, 4, 1, 3, 0]]),
            "search": (("date", "geo"), [[4, 2], [1, 3], [0, 1], [2, 5], [3, 4]]),
        },
        coords={"date": dates, "geo": ["west", "east"]},
    )
    adstock = GeometricAdstock(l_max=3, priors={"alpha": 0.4})
    saturation = LogisticSaturation(priors={"lam": 1.3, "beta": 2.1})
    transforms = (adstock, saturation) if adstock_first else (saturation, adstock)
    media = Media(["tv", "search"], transforms, dims=("geo",))
    media.channel_scale = xr.DataArray(
        [[8.0, 4.0], [2.0, 5.0]],
        dims=("channel", "geo"),
        coords={"channel": ["search", "tv"], "geo": ["east", "west"]},
    )
    raw = np.stack(
        [ds[name].transpose("date", "geo").values for name in media], axis=-1
    )
    scale = (
        media.channel_scale.sel(channel=list(media), geo=ds.geo)
        .transpose("geo", "channel")
        .values
    )
    values = raw / scale
    if adstock_first:
        expected = 2.1 * np.tanh(1.3 * _adstock_reference(values, 0.4, 3) / 2)
    else:
        expected = _adstock_reference(2.1 * np.tanh(1.3 * values / 2), 0.4, 3)
    with pm.Model(coords={"date": dates, "geo": ds.geo.values, "channel": list(media)}):
        context = BuildContext(ds, default_dims=("date", "geo"))
        total = context.build(media.total)
        actual = total.eval()
        individual = [context.build(media[name].contribution).eval() for name in media]
    np.testing.assert_allclose(actual, expected.sum(axis=-1))
    for index, contribution in enumerate(individual):
        np.testing.assert_allclose(contribution, expected[..., index])


def test_joint_and_individual_contributions_share_parameter_draws() -> None:
    ds = xr.Dataset(
        {"tv": ("date", [1.0, 2.0, 0.0]), "search": ("date", [0.0, 3.0, 4.0])},
        coords={"date": pd.date_range("2025-01-01", periods=3)},
    )
    media = Media(["tv", "search"], (GeometricAdstock(l_max=2), LogisticSaturation()))
    with pm.Model(coords={"date": ds.date.values, "channel": list(media)}):
        context = BuildContext(ds)
        individual = context.build(
            media["tv"].contribution + media["search"].contribution
        )
        total = context.build(media.total)
        difference = pm.draw(total - individual, draws=4, random_seed=71)
    np.testing.assert_allclose(difference, 0.0, atol=1e-12)


def test_explicit_scalar_prior_remains_shared_across_channels() -> None:
    ds = xr.Dataset(
        {"tv": ("date", [1.0, 2.0]), "search": ("date", [2.0, 4.0])},
        coords={"date": pd.date_range("2025-01-01", periods=2)},
    )
    media = Media(
        ["tv", "search"],
        (
            NoAdstock(l_max=1),
            NoSaturation(priors={"beta": Prior("HalfNormal", sigma=1, dims=())}),
        ),
    )
    with pm.Model(coords={"date": ds.date.values, "channel": list(media)}):
        context = BuildContext(ds)
        proportional_difference = context.build(
            2 * media["tv"].contribution - media["search"].contribution
        )
        actual = pm.draw(proportional_difference, draws=4, random_seed=24)
    np.testing.assert_allclose(actual, 0.0, atol=1e-12)


def test_upstream_individual_contribution_does_not_depend_on_endogenous_peer() -> None:
    ds = xr.Dataset(
        {"tv": ("date", [1.0, 2.0, 3.0]), "search": ("date", [2.0, 4.0, 6.0])},
        coords={"date": pd.date_range("2025-01-01", periods=3)},
    )
    media = Media(
        ["tv", "search"], (NoAdstock(l_max=1), NoSaturation(priors={"beta": 2.0}))
    )
    equation = Equation(
        mu=media["tv"].contribution, likelihood=Prior("Normal", sigma=0.1)
    )
    media["search"].equation = equation
    list(walk(media.total))
    with pm.Model(coords={"date": ds.date.values, "channel": list(media)}) as model:
        context = BuildContext(
            ds,
            bindings={id(equation): Binding("search_observation", "search", ("date",))},
        )
        context.build(media.total)
        actual = model.compile_logp()(model.initial_point())
    expected = -len(ds.date) * np.log(0.1 * np.sqrt(2 * np.pi))
    np.testing.assert_allclose(actual, expected)


def test_channel_value_stays_raw_and_existing_handles_follow_equation_replacement() -> (
    None
):
    ds = xr.Dataset(
        {"tv": ("date", [10.0, 20.0]), "other": ("date", [4.0, 8.0])},
        coords={"date": pd.date_range("2025-01-01", periods=2)},
    )
    media = Media(["tv"], (NoAdstock(l_max=1), NoSaturation(priors={"beta": 3.0})))
    media.channel_scale = xr.DataArray(
        [10.0], dims="channel", coords={"channel": ["tv"]}
    )
    raw = media["tv"].value
    contribution = media["tv"].contribution
    equation = Equation(
        mu=0.0,
        likelihood=Prior("Normal", sigma=1.0),
        name="spend",
        observed="other",
        dims=("date",),
    )
    media["tv"].equation = equation
    with pm.Model(coords={"date": ds.date.values, "channel": list(media)}):
        context = BuildContext(ds, prediction=True, condition_on=("spend",))
        raw_values = context.build(raw).eval()
        transformed = context.build(contribution).eval()
    np.testing.assert_allclose(raw_values, [4.0, 8.0])
    np.testing.assert_allclose(transformed, [1.2, 2.4])


@pytest.mark.parametrize("adstock_first", [True, False])
def test_raw_required_saturation_bypasses_channel_normalization(
    adstock_first: bool,
) -> None:
    ds = xr.Dataset(
        {"tv": ("date", [0.0, 10.0, 100.0])},
        coords={"date": pd.date_range("2025-01-01", periods=3)},
    )
    adstock = NoAdstock(l_max=1)
    saturation = LogSaturation(priors={"beta": 2.0})
    transforms = (adstock, saturation) if adstock_first else (saturation, adstock)
    media = Media(["tv"], transforms)
    media.channel_scale = 100.0
    with pm.Model(coords={"date": ds.date.values, "channel": list(media)}):
        actual = BuildContext(ds).build(media.total).eval()
    np.testing.assert_allclose(actual, 2 * np.log1p(ds.tv.values))


def test_parameter_names_cannot_reuse_unrelated_graph_variables() -> None:
    ds = xr.Dataset(
        {"tv": ("date", [1.0, 2.0])},
        coords={"date": pd.date_range("2025-01-01", periods=2)},
    )
    media = Media(["tv"], (GeometricAdstock(l_max=2), LogisticSaturation()))
    unrelated = Parameter(
        "adstock_alpha", Prior("Beta", alpha=2, beta=2, dims="channel")
    )
    with pm.Model(coords={"date": ds.date.values, "channel": list(media)}):
        context = BuildContext(ds)
        context.build(unrelated)
        with pytest.raises(ValueError, match="already owned"):
            context.build(media.total)


def test_configured_transforms_can_be_reused_without_cross_model_changes() -> None:
    location = Scaled(
        Prior(
            "Dirichlet",
            a=xr.DataArray(np.ones(2), dims="channel"),
            dims="channel",
            core_dims="channel",
        ),
        factor=2,
    )
    adstock = GeometricAdstock(l_max=2)
    saturation = LogisticSaturation(
        priors={"beta": Prior("Normal", mu=location, sigma=1, dims=("channel",))}
    )
    ds = xr.Dataset(
        {"tv": ("date", [1.0, 2.0]), "search": ("date", [3.0, 4.0])},
        coords={"date": pd.date_range("2026-01-01", periods=2)},
    )

    def draw_recipe():
        media = Media(["tv", "search"], (adstock, saturation))
        with pm.Model(
            coords={
                "date": ds.date.values,
                "channel": ["tv", "search"],
            }
        ):
            return pm.draw(BuildContext(ds).build(media.total), draws=4, random_seed=57)

    original = draw_recipe()
    changed = Media(["tv", "search"], (adstock, saturation))
    changed.transforms[0].function_priors["alpha"].parameters["alpha"] = 8
    copied_location = changed.transforms[1].function_priors["beta"].parameters["mu"]
    copied_location.dist.parameters["a"][0] = 3
    np.testing.assert_allclose(draw_recipe(), original)


@pytest.mark.parametrize("mode", [ConvMode.Before, ConvMode.Overlap])
def test_noncausal_history_is_rejected(mode: ConvMode) -> None:
    media = Media(["tv"], (GeometricAdstock(l_max=3, mode=mode), LogisticSaturation()))
    with pytest.raises(ValueError, match="causal"):
        _ = media.required_history


@pytest.mark.parametrize(
    "transforms",
    [
        (GeometricAdstock(l_max=2), GeometricAdstock(l_max=3)),
        (LogisticSaturation(), LogisticSaturation()),
        (GeometricAdstock(l_max=2),),
    ],
)
def test_transform_recipe_requires_one_component_of_each_kind(
    transforms: tuple,
) -> None:
    with pytest.raises(TypeError, match="exactly one adstock and one saturation"):
        Media(["tv"], transforms)


def test_readonly_factory_dimensions_cannot_silently_become_shared_parameters():
    beta = Scaled(Prior("HalfNormal"), factor=2)
    transforms = (NoAdstock(l_max=1), NoSaturation(priors={"beta": beta}))
    with pytest.raises(ValueError):
        Media(["tv", "search"], transforms)
    beta.dist.dims = ("channel",)
    media = Media(["tv", "search"], transforms)
    ds = xr.Dataset(
        {"tv": ("date", [1.0]), "search": ("date", [1.0])},
        coords={"date": pd.date_range("2026-01-01", periods=1)},
    )
    with pm.Model(coords={"date": ds.date.values, "channel": ["tv", "search"]}):
        context = BuildContext(ds)
        difference = context.build(media["tv"].contribution) - context.build(
            media["search"].contribution
        )
        draws = pm.draw(difference, draws=5, random_seed=54)
    assert not np.allclose(draws, 0)
