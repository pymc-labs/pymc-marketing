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
import pytest
import xarray as xr

from pymc_marketing.decision import contrast, expected_loss, prob_best


@pytest.fixture
def hand_example() -> xr.DataArray:
    """Four draws of three arms with a two-way and a three-way tie."""
    return xr.DataArray(
        [
            [0.10, 0.20, 0.15],
            [0.30, 0.10, 0.30],
            [0.20, 0.25, 0.05],
            [0.40, 0.40, 0.40],
        ],
        dims=("sample", "variant"),
        coords={"variant": ["a", "b", "c"]},
    )


@pytest.fixture
def draws() -> xr.DataArray:
    rng = np.random.default_rng(0)
    return xr.DataArray(
        rng.normal(loc=[[0.0], [0.1], [0.2], [-0.1], [0.05]], size=(2, 300, 5, 3)),
        dims=("chain", "draw", "variant", "segment"),
        coords={"variant": list("abcde"), "segment": [10, 20, 30]},
    )


def test_hand_example(hand_example) -> None:
    np.testing.assert_allclose(
        prob_best(hand_example, "variant"), [5 / 24, 14 / 24, 5 / 24], rtol=1e-12
    )
    np.testing.assert_allclose(
        expected_loss(hand_example, "variant"), [0.0375, 0.05, 0.0625], rtol=1e-12
    )


def test_hand_example_lower_is_better(hand_example) -> None:
    np.testing.assert_allclose(
        prob_best(hand_example, "variant", higher_is_better=False),
        [1 / 3, 1 / 3, 1 / 3],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        expected_loss(hand_example, "variant", higher_is_better=False),
        [0.0875, 0.075, 0.0625],
        rtol=1e-12,
    )


def test_two_arms_closed_form() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, size=2_000)
    b = rng.normal(0.3, 1.0, size=2_000)
    samples = xr.DataArray(
        np.stack([a, b], axis=-1),
        dims=("sample", "variant"),
        coords={"variant": ["a", "b"]},
    )

    best = prob_best(samples, "variant")
    loss = expected_loss(samples, "variant")

    assert best.sel(variant="b").item() == pytest.approx((b > a).mean())
    assert loss.sel(variant="a").item() == pytest.approx(np.clip(b - a, 0, None).mean())
    assert loss.sel(variant="b").item() == pytest.approx(np.clip(a - b, 0, None).mean())


def test_extra_dims_are_kept(draws) -> None:
    assert prob_best(draws, "variant").dims == ("variant", "segment")
    assert expected_loss(draws, "variant").dims == ("variant", "segment")
    assert contrast(draws, "variant", "a").dims == draws.dims
    assert list(prob_best(draws, "variant").coords["segment"].values) == [10, 20, 30]


def test_stacked_and_unstacked_draws_agree(draws) -> None:
    stacked = draws.stack(sample=("chain", "draw"))

    xr.testing.assert_allclose(
        prob_best(stacked, "variant"), prob_best(draws, "variant")
    )
    xr.testing.assert_allclose(
        expected_loss(stacked, "variant"), expected_loss(draws, "variant")
    )


def test_contrast_difference_and_ratio(draws) -> None:
    difference = contrast(draws, "variant", "b")
    ratio = contrast(draws, "variant", "b", kind="ratio")

    assert list(difference.coords["variant"].values) == ["a", "c", "d", "e"]
    np.testing.assert_allclose(
        difference.sel(variant="d"), draws.sel(variant="d") - draws.sel(variant="b")
    )
    np.testing.assert_allclose(
        ratio.sel(variant="d"), draws.sel(variant="d") / draws.sel(variant="b")
    )
    assert difference.attrs == {"reference": "b", "kind": "difference"}
    assert ratio.attrs == {"reference": "b", "kind": "ratio"}


@pytest.mark.parametrize(
    "samples, dim, match",
    [
        pytest.param(
            xr.DataArray(np.zeros((4, 2)), dims=("iteration", "variant")),
            "variant",
            "chain.*draw.*sample",
            id="no sample dims",
        ),
        pytest.param(
            xr.DataArray(np.zeros((4, 2)), dims=("sample", "variant")),
            "segment",
            "'segment' is not a dimension",
            id="dim missing",
        ),
        pytest.param(
            xr.DataArray(np.zeros((2, 4, 2)), dims=("chain", "draw", "variant")),
            "draw",
            "'draw' is a sample dimension",
            id="dim is a sample dim",
        ),
        pytest.param(
            xr.DataArray(np.zeros((4, 1)), dims=("sample", "variant")),
            "variant",
            "at least two",
            id="one label",
        ),
        pytest.param(
            xr.DataArray(np.full((4, 2), np.nan), dims=("sample", "variant")),
            "variant",
            "NaN",
            id="nan",
        ),
    ],
)
@pytest.mark.parametrize("func", [prob_best, expected_loss])
def test_summary_errors(func, samples, dim, match) -> None:
    with pytest.raises(ValueError, match=match):
        func(samples, dim)


def test_contrast_unknown_reference(hand_example) -> None:
    with pytest.raises(ValueError, match="'z' is not a label of 'variant'"):
        contrast(hand_example, "variant", "z")


def test_contrast_unknown_kind(hand_example) -> None:
    with pytest.raises(ValueError, match="kind must be 'difference' or 'ratio'"):
        contrast(hand_example, "variant", "a", kind="odds")  # type: ignore[arg-type]
