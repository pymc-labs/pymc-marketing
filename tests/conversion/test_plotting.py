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
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from matplotlib.figure import Figure

from pymc_marketing.conversion import plot_funnel

STEPS = ["signup", "trial", "paid", "renewal"]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def samples() -> xr.DataArray:
    """Draws of a conversion for four steps, two variants and three segments."""
    rng = np.random.default_rng(0)
    return xr.DataArray(
        rng.beta(8, 12, size=(2, 250, 4, 2, 3)),
        dims=("chain", "draw", "step", "variant", "segment"),
        coords={
            "step": STEPS,
            "variant": ["control", "treatment"],
            "segment": ["new", "returning", "lapsed"],
        },
    )


def bar_widths(ax) -> np.ndarray:
    return np.array([patch.get_width() for patch in ax.patches])


def interval_ends(ax) -> np.ndarray:
    """Return the left and right ends of every interval drawn on ``ax``."""
    segments = np.concatenate(
        [collection.get_segments() for collection in ax.collections]
    )
    return segments[:, :, 0]


def test_one_panel_with_one_colour_per_variant(samples) -> None:
    fig, axes = plot_funnel(samples.sel(segment="new"))

    assert isinstance(fig, Figure)
    assert axes.shape == (1,)
    assert [label.get_text() for label in axes[0].get_yticklabels()] == STEPS
    assert len(axes[0].patches) == 4 * 2
    assert [text.get_text() for text in fig.legends[0].get_texts()] == [
        "control",
        "treatment",
    ]


def test_first_step_is_at_the_top(samples) -> None:
    _, axes = plot_funnel(samples.sel(segment="new", variant="control"))

    assert axes[0].yaxis_inverted()


def test_no_group_dims_gives_one_series_without_legend(samples) -> None:
    fig, axes = plot_funnel(samples.sel(segment="new", variant="control"))

    assert len(axes[0].patches) == 4
    assert fig.legends == []


def test_one_panel_per_label_of_the_other_group_dim(samples) -> None:
    _, axes = plot_funnel(samples)

    assert axes.shape == (3,)
    assert [ax.get_title() for ax in axes] == [
        "segment = new",
        "segment = returning",
        "segment = lapsed",
    ]


def test_hue_selects_the_dim_that_gets_the_colours(samples) -> None:
    _, axes = plot_funnel(samples, hue="segment")

    assert axes.shape == (2,)
    assert len(axes[0].patches) == 4 * 3


def test_bars_are_posterior_means_and_intervals_are_hdis(samples) -> None:
    one_series = samples.sel(segment="new", variant="control")

    _, axes = plot_funnel(one_series, hdi_prob=0.8)

    hdi = az.hdi(one_series, prob=0.8, dim=["chain", "draw"])
    np.testing.assert_allclose(bar_widths(axes[0]), one_series.mean(("chain", "draw")))
    np.testing.assert_allclose(
        interval_ends(axes[0]),
        np.column_stack([hdi.sel(ci_bound="lower"), hdi.sel(ci_bound="upper")]),
    )


def test_stacked_draws_give_the_same_plot(samples) -> None:
    one_panel = samples.sel(segment="new")

    _, unstacked = plot_funnel(one_panel)
    _, stacked = plot_funnel(one_panel.stack(sample=("chain", "draw")))

    np.testing.assert_allclose(bar_widths(stacked[0]), bar_widths(unstacked[0]))
    np.testing.assert_allclose(interval_ends(stacked[0]), interval_ends(unstacked[0]))


@pytest.mark.parametrize(
    "select, kwargs, match",
    [
        pytest.param({"step": "paid"}, {}, "'step' dimension", id="no step dim"),
        pytest.param({}, {"hue": "region"}, "'region' is not", id="unknown hue"),
        pytest.param({}, {"hue": "step"}, "'step' is not", id="hue is the step dim"),
        pytest.param({"chain": 0}, {}, "chain.*draw.*sample", id="no sample dims"),
    ],
)
def test_errors(samples, select, kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        plot_funnel(samples.sel(select), **kwargs)


@pytest.mark.parametrize("quantity", ["cumulative", "step"])
def test_model_method_plots_the_requested_quantity(fitted_model, quantity) -> None:
    draws = {
        "cumulative": fitted_model.cumulative_conversion(),
        "step": fitted_model.step_conversion(),
    }[quantity]

    _, axes = fitted_model.plot_funnel(quantity=quantity)

    expected = draws.mean(("chain", "draw")).transpose("variant", "step")
    np.testing.assert_allclose(bar_widths(axes[0]), expected.values.ravel())
    assert axes[0].get_xlabel() != ""


def test_model_method_rejects_an_unknown_quantity(fitted_model) -> None:
    with pytest.raises(ValueError, match="quantity must be 'cumulative' or 'step'"):
        fitted_model.plot_funnel(quantity="overall")
