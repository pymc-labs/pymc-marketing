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
"""Tests for the movement of mediator nodes, and for the spend a counterfactual induces.

A movement is compared with the node itself, read with
``sample_posterior_predictive`` under actual spend and under the counterfactual.
"""

import warnings

import numpy as np
import pandas as pd
import pymc.dims as pmd
import pytest
import xarray as xr

from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.additive_effect import (
    IncrementalitySpec,
    MediatorSpec,
    MuEffect,
)
from pymc_marketing.mmm.estimand_label import EstimandLabel
from tests.mmm.conftest import (
    FUNNEL_L_MAX,
    FunnelEffect,
    _build_funnel_mmm,
    _build_simple_effect_mmm,
    _funnel_effect,
    _media_transformation,
    mock_fit,
    seed,
)
from tests.mmm.test_path_estimands import (
    HubEffect,
    assert_matches,
    moved_spend,
    oracle,
    response,
)

LF_SPEND = "funnel_lf_spend"
DEMAND = "funnel_demand"


class IndicatorEffect(MuEffect):
    """An effect that measures a spend-driven indicator and adds nothing spend moves.

    Its term does not read spend, so the model stays separable: one
    all-channels counterfactual serves every channel's contribution.  The
    indicator has no column per channel, so its movement cannot be read that
    way.
    """

    prefix: str
    is_spend: bool = False

    def create_data(self, mmm) -> None:
        """No data of its own."""

    def set_data(self, mmm, model, X) -> None:
        """No data of its own."""

    def incrementality_spec(self) -> IncrementalitySpec:
        """Declare the indicator, reported in searches."""
        return IncrementalitySpec(
            mediators=(
                MediatorSpec(
                    name="searches",
                    variable=f"{self.prefix}_searches",
                    units="searches",
                    scale=250.0,
                    is_spend=self.is_spend,
                ),
            )
        )

    def create_effect(self, mmm):
        """Register the indicator and add a constant level."""
        spend = mmm.channel_data_scaled.sum(dim="channel")
        pmd.Deterministic(f"{self.prefix}_searches", pmd.math.tanh(spend))
        level = pmd.Normal(f"{self.prefix}_level", sigma=0.1)
        ones = pmd.Data(
            f"{self.prefix}_ones", np.ones(len(mmm.model.coords["date"])), dims="date"
        )
        return pmd.Deterministic(f"{self.prefix}_effect_contribution", level * ones)


class TwoHubsEffect(MuEffect):
    """A mediator with a hub that can be reported and one that cannot.

    The second hub keeps the ``channel`` dimension, and a movement per channel
    of a node that already has one would have two meanings.
    """

    prefix: str

    def create_data(self, mmm) -> None:
        """No data of its own."""

    def set_data(self, mmm, model, X) -> None:
        """No data of its own."""

    def incrementality_spec(self) -> IncrementalitySpec:
        """Declare both hubs."""
        return IncrementalitySpec(
            role="mediator",
            mediators=(
                MediatorSpec(name="pooled", variable=f"{self.prefix}_pooled"),
                MediatorSpec(name="split", variable=f"{self.prefix}_split"),
            ),
        )

    def create_effect(self, mmm):
        """Add both hubs, each with its own coefficient."""
        spend = mmm.channel_data_scaled
        pooled = pmd.Deterministic(
            f"{self.prefix}_pooled", pmd.math.tanh(spend.sum(dim="channel"))
        )
        split = pmd.Deterministic(f"{self.prefix}_split", pmd.math.tanh(spend))
        beta = pmd.HalfNormal(f"{self.prefix}_beta", sigma=1.0)
        return pmd.Deterministic(
            f"{self.prefix}_effect_contribution",
            beta * (pooled + split.sum(dim="channel")),
        )


class ShrinkingHubEffect(HubEffect):
    """A spend mediator that more channel spend makes smaller, by a lot."""

    def create_effect(self, mmm):
        """Add ``beta * hub`` with a hub that falls steeply with spend."""
        spend = mmm.channel_data_scaled.sum(dim="channel")
        hub = pmd.Deterministic(f"{self.prefix}_hub", -1e3 * pmd.math.tanh(spend))
        beta = pmd.HalfNormal(f"{self.prefix}_beta", sigma=1e-3)
        return pmd.Deterministic(f"{self.prefix}_effect_contribution", beta * hub)


class MemoryNodeFunnelEffect(FunnelEffect):
    """A funnel effect with one more declared node, which remembers another series.

    The node adds a twelve-date adstock of ``lf_budget`` to demand.  On a window
    shorter than that memory it takes another value than on the full date axis
    with spend unchanged.
    """

    def incrementality_spec(self) -> IncrementalitySpec:
        """Add the node to the funnel's declaration."""
        spec = super().incrementality_spec()
        index = MediatorSpec(name="index", variable=f"{self.prefix}_index")
        return spec.model_copy(update={"mediators": (*spec.mediators, index)})

    def create_effect(self, mmm):
        """Register the node next to the funnel's term."""
        term = super().create_effect(mmm)
        memory = _media_transformation("memory", (), 12)
        pmd.Deterministic(
            f"{self.prefix}_index",
            mmm.model[f"{self.prefix}_demand"]
            + memory(mmm.model["lf_budget"], dim="date"),
        )
        return term


class FullAxisMemoryNodeFunnelEffect(MemoryNodeFunnelEffect):
    """The same effect, declared as needing the whole date axis."""

    def incrementality_spec(self) -> IncrementalitySpec:
        """Add ``evaluation_mode="full"`` to the declaration."""
        return (
            super().incrementality_spec().model_copy(update={"evaluation_mode": "full"})
        )


def movement_oracle(
    mmm, variable, *, frequency, factor=0.0, scope="per_channel", scale=1.0
) -> xr.DataArray:
    """The node's movement by brute force, laid out as the module's."""
    incr = mmm.incrementality
    dates = incr.data.dates
    periods = incr._create_period_groups(dates[0], dates[-1], frequency)
    channels = list(mmm.channel_columns)
    keys = range(len(channels)) if scope == "per_channel" else [None]
    factual = response(mmm, mmm.model["channel_data"].get_value(), var=variable)
    sign = scale if factor <= 1.0 else -scale

    stacked = []
    for period in periods:
        moves = [
            sign
            * (
                factual
                - response(mmm, moved_spend(mmm, period, key, factor), var=variable)
            ).sum("date")
            for key in keys
        ]
        stacked.append(
            xr.concat(moves, dim=pd.Index(channels, name="channel"))
            if scope == "per_channel"
            else moves[0]
        )
    if frequency == "all_time":
        return stacked[0]
    ends = pd.DatetimeIndex([end for _, end in periods], name="date")
    return xr.concat(stacked, dim=ends)


def expected_complete(mmm, frequency, max_lag) -> list[bool]:
    """Per period, whether its last spend date plus *max_lag* dates is still fitted."""
    incr = mmm.incrementality
    dates = incr.data.dates
    periods = incr._create_period_groups(dates[0], dates[-1], frequency)
    step = dates[1] - dates[0]
    return [
        bool(dates[dates <= end][-1] + max_lag * step <= dates[-1])
        for _, end in periods
    ]


@pytest.fixture(scope="module")
def indicator_fitted_mmm(funnel_mmm_data):
    """Separable identity-link MMM with a measurement-only node."""
    return _build_simple_effect_mmm(
        funnel_mmm_data, IndicatorEffect(prefix="ind"), link="identity"
    )


@pytest.fixture(scope="module")
def scaled_hub_fitted_mmm(funnel_mmm_data):
    """Identity-link MMM whose hub is reported in impressions, 250 per model unit."""
    return _build_simple_effect_mmm(
        funnel_mmm_data,
        HubEffect(prefix="halo", units="impressions", scale=250.0),
        link="identity",
    )


@pytest.fixture(scope="module")
def two_hubs_fitted_mmm(funnel_mmm_data):
    """Identity-link MMM with one reportable hub and one that keeps ``channel``."""
    return _build_simple_effect_mmm(
        funnel_mmm_data, TwoHubsEffect(prefix="two"), link="identity"
    )


def _panel_mmm(panel_mmm_data, *effects) -> MMM:
    """Mock-fitted panel MMM over two countries carrying *effects*."""
    X = panel_mmm_data["X"].copy()
    for column in X.columns[X.columns.str.startswith("channel_")]:
        X[column] = X[column].astype(np.float64)
    mmm = MMM(
        channel_columns=["channel_1", "channel_2"],
        date_column="date",
        target_column="target",
        dims=("country",),
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    )
    for effect in effects:
        mmm.add_mu_effect(effect)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mock_fit(mmm, X, panel_mmm_data["y"], random_seed=seed)
    return mmm


@pytest.fixture(scope="module")
def panel_hubs_fitted_mmm(panel_mmm_data):
    """Panel MMM with a spend hub per country and a national one."""
    return _panel_mmm(
        panel_mmm_data,
        HubEffect(prefix="local", units="USD", is_spend=True),
        HubEffect(prefix="national", units="USD", is_spend=True, national=True),
    )


class TestIncrementalMediator:
    """How far a spend counterfactual moves a declared node, in its reported units."""

    @pytest.mark.parametrize(
        "frequency, scope, factor",
        [
            ("all_time", "per_channel", 0.0),
            ("monthly", "per_channel", 0.0),
            ("monthly", "joint", 0.0),
            ("all_time", "per_channel", 1.5),
        ],
    )
    def test_a_movement_is_the_node_under_the_counterfactual(
        self, funnel_log_link_fitted_mmm, frequency, scope, factor
    ):
        """Summed over each period's evaluated dates, per channel or for all at once.

        More spend makes more demand in this model, so the movement is positive
        whichever way spend is moved: above a factor of one it is counterfactual
        minus actual.
        """
        mmm = funnel_log_link_fitted_mmm

        moved = mmm.incrementality.compute_incremental_mediator(
            frequency=frequency,
            scope=scope,
            counterfactual_spend_factor=factor,
            mediators=[LF_SPEND],
        )

        expected = movement_oracle(
            mmm, LF_SPEND, frequency=frequency, factor=factor, scope=scope
        )
        assert list(moved.data_vars) == [LF_SPEND]
        assert_matches(moved[LF_SPEND], expected)
        assert (moved[LF_SPEND] > 0).all()

    def test_every_readable_mediator_is_reported_by_default(
        self, funnel_log_link_fitted_mmm
    ):
        """One variable per declared node, each saying which node it is."""
        moved = funnel_log_link_fitted_mmm.incrementality.compute_incremental_mediator(
            frequency="all_time"
        )

        assert set(moved.data_vars) == {DEMAND, LF_SPEND}
        assert moved.attrs == {}
        label = EstimandLabel.read(moved[LF_SPEND])
        assert label.quantity == "incremental_mediator"
        assert (label.mediator, label.variable) == (LF_SPEND, LF_SPEND)
        assert (label.units, label.is_spend) == ("USD", True)
        assert (label.paths, label.central_tendency) == (None, None)
        assert label.definition == "M(x) - M(x')"
        assert EstimandLabel.read(moved[DEMAND]).is_spend is False

    def test_a_movement_is_in_the_reported_units(self, scaled_hub_fitted_mmm):
        """The declared scale converts model units before anything is summed."""
        mmm = scaled_hub_fitted_mmm

        moved = mmm.incrementality.compute_incremental_mediator(frequency="monthly")

        expected = movement_oracle(mmm, "halo_hub", frequency="monthly", scale=250.0)
        assert_matches(moved["halo"], expected)
        assert EstimandLabel.read(moved["halo"]).units == "impressions"

    def test_a_separable_model_still_moves_one_channel_at_a_time(
        self, indicator_fitted_mmm
    ):
        """A node has no column per channel, so each channel gets its own row.

        Read off the one all-channels counterfactual a separable model
        evaluates, every channel would be given the joint movement.
        """
        mmm = indicator_fitted_mmm

        moved = mmm.incrementality.compute_incremental_mediator(frequency="all_time")

        expected = movement_oracle(
            mmm, "ind_searches", frequency="all_time", scale=250.0
        )
        assert_matches(moved["searches"], expected)
        per_channel = moved["searches"].mean(("chain", "draw"))
        assert not np.isclose(per_channel[0], per_channel[1], rtol=1e-3)

    def test_a_node_keeps_its_own_dimensions(self, panel_hubs_fitted_mmm):
        """A national node in a panel model has no ``country`` to report."""
        mmm = panel_hubs_fitted_mmm

        moved = mmm.incrementality.compute_incremental_mediator(frequency="all_time")

        assert moved["national"].dims == ("chain", "draw", "channel")
        assert moved["local"].dims == ("chain", "draw", "channel", "country")
        for name in ("national", "local"):
            assert_matches(
                moved[name],
                movement_oracle(mmm, f"{name}_hub", frequency="all_time"),
            )

    def test_periods_the_data_cuts_short_are_marked(self, funnel_log_link_fitted_mmm):
        """``complete`` is false where a node's carryover runs past the fitted dates.

        Lower-funnel spend follows the model's adstock, whose kernel covers
        ``FUNNEL_L_MAX`` dates.  Only the last month's spend still moves it
        after the data ends.
        """
        mmm = funnel_log_link_fitted_mmm
        incr = mmm.incrementality
        reach = FUNNEL_L_MAX - 1

        monthly = incr.compute_incremental_mediator(
            frequency="monthly", mediators=[LF_SPEND]
        )
        without = incr.compute_incremental_mediator(
            frequency="monthly", mediators=[LF_SPEND], include_carryover=False
        )
        all_time = incr.compute_incremental_mediator(
            frequency="all_time", mediators=[LF_SPEND]
        )

        assert monthly[LF_SPEND]["complete"].dims == ("date",)
        assert monthly[LF_SPEND]["complete"].values.tolist() == expected_complete(
            mmm, "monthly", reach
        )
        assert monthly[LF_SPEND]["complete"].values.tolist() == [True] * 5 + [False]
        assert without[LF_SPEND]["complete"].all()
        assert all_time[LF_SPEND]["complete"].item() is False

    def test_a_node_that_differs_on_a_window_is_named(self, funnel_mmm_data):
        """Its movement on a window is not the effect of spend alone.

        The refusal names the node and the fix, and the fix works: declared as
        needing the full axis, the node is reported and matches the
        intervention.  A request that does not read the node is answered.
        """
        windowed = _build_funnel_mmm(
            funnel_mmm_data,
            link="log",
            effects=[_funnel_effect(MemoryNodeFunnelEffect)],
        )
        full = _build_funnel_mmm(
            funnel_mmm_data,
            link="log",
            effects=[_funnel_effect(FullAxisMemoryNodeFunnelEffect)],
        )

        with pytest.raises(ValueError, match=r"(?s)'index'.*evaluation_mode='full'"):
            windowed.incrementality.compute_incremental_mediator(frequency="monthly")
        others = windowed.incrementality.compute_incremental_mediator(
            frequency="monthly", mediators=[LF_SPEND]
        )
        moved = full.incrementality.compute_incremental_mediator(
            frequency="monthly", mediators=["index"]
        )

        assert_matches(
            others[LF_SPEND], movement_oracle(windowed, LF_SPEND, frequency="monthly")
        )
        assert_matches(
            moved["index"], movement_oracle(full, "funnel_index", frequency="monthly")
        )

    def test_a_node_that_cannot_be_reported_is_named(self, two_hubs_fitted_mmm):
        """Left out of the default with a warning, and refused when asked for."""
        incr = two_hubs_fitted_mmm.incrementality

        with pytest.warns(UserWarning, match=r"'split'.*'channel' dimension"):
            moved = incr.compute_incremental_mediator(frequency="all_time")
        assert list(moved.data_vars) == ["pooled"]

        with pytest.raises(ValueError, match=r"'split'.*'channel' dimension"):
            incr.compute_incremental_mediator(frequency="all_time", mediators=["split"])

    @pytest.mark.parametrize(
        "mediators, complaint",
        [
            pytest.param(["nope"], r"'nope'.*declared", id="a name nobody declared"),
            pytest.param(LF_SPEND, "sequence of names", id="a bare string"),
            pytest.param([], "at least one", id="an empty list"),
            pytest.param(
                [LF_SPEND, LF_SPEND], "more than once", id="a name given twice"
            ),
        ],
    )
    def test_a_request_for_no_usable_mediator_is_refused(
        self, funnel_log_link_fitted_mmm, mediators, complaint
    ):
        """Refused before anything is compiled."""
        with pytest.raises(ValueError, match=complaint):
            funnel_log_link_fitted_mmm.incrementality.compute_incremental_mediator(
                frequency="all_time", mediators=mediators
            )

    def test_a_model_that_declares_no_mediator_has_none_to_report(
        self, simple_fitted_mmm
    ):
        """Only declared nodes are reported."""
        with pytest.raises(ValueError, match="declares no mediator"):
            simple_fitted_mmm.incrementality.compute_incremental_mediator(
                frequency="all_time"
            )


class TestInducedSpend:
    """Spend a counterfactual induces downstream, next to the spend it moves."""

    def test_a_roas_over_all_incremental_spend(self, funnel_log_link_fitted_mmm):
        """Numerator and denominator are each the intervention's, on the same draws."""
        mmm = funnel_log_link_fitted_mmm
        incr = mmm.incrementality
        spend = incr._aggregate_spend("all_time")

        roas = incr.contribution_over_spend(
            frequency="all_time", induced_spend=[LF_SPEND]
        )

        total = oracle(mmm, frequency="all_time")["total"]
        induced = movement_oracle(mmm, LF_SPEND, frequency="all_time")
        assert_matches(roas, total / (spend + induced))
        assert bool((roas < incr.contribution_over_spend(frequency="all_time")).all())
        label = EstimandLabel.read(roas)
        assert (label.induced_spend, label.induced_spend_units) == ((LF_SPEND,), "USD")
        assert label.definition == "[Y(x) - Y(x')] / [S(x) - S(x') + M(x) - M(x')]"

    def test_the_reciprocal_and_the_marginal_ratio_take_it_too(
        self, funnel_log_link_fitted_mmm
    ):
        """Each method adds the induced movement to its own denominator."""
        incr = funnel_log_link_fitted_mmm.incrementality
        spend = incr._aggregate_spend("monthly")
        raised = {"frequency": "monthly", "counterfactual_spend_factor": 1.05}
        total = incr.compute_incremental_contribution(**raised)
        induced = incr.compute_incremental_mediator(**raised, mediators=[LF_SPEND])

        marginal = incr.marginal_contribution_over_spend(
            frequency="monthly", spend_increase_pct=0.05, induced_spend=[LF_SPEND]
        )
        roas = incr.contribution_over_spend(
            frequency="monthly", induced_spend=[LF_SPEND]
        )
        cac = incr.spend_over_contribution(
            frequency="monthly", induced_spend=[LF_SPEND]
        )

        assert_matches(marginal, total / (0.05 * spend + induced[LF_SPEND]), rtol=1e-9)
        xr.testing.assert_allclose(cac, 1.0 / roas, rtol=1e-12)
        assert EstimandLabel.read(marginal).definition == (
            "[Y(x') - Y(x)] / [S(x') - S(x) + M(x') - M(x)]"
        )
        assert EstimandLabel.read(cac).induced_spend == (LF_SPEND,)

    def test_a_ratio_is_marked_where_the_data_cuts_it_short(
        self, funnel_log_link_fitted_mmm
    ):
        """The numerator's carryover reaches further than the mediator's.

        The mediated path chains the funnel's adstock behind the model's, so a
        period's response still moves two kernels after its spend.
        """
        mmm = funnel_log_link_fitted_mmm
        reach = 2 * (FUNNEL_L_MAX - 1)

        roas = mmm.incrementality.contribution_over_spend(
            frequency="monthly", induced_spend=[LF_SPEND]
        )

        assert roas["complete"].values.tolist() == expected_complete(
            mmm, "monthly", reach
        )
        assert roas["complete"].values.tolist() == [True] * 4 + [False] * 2

    @pytest.mark.parametrize(
        "call, complaint",
        [
            pytest.param(
                lambda incr: incr.contribution_over_spend(
                    frequency="all_time", paths="direct", induced_spend=[LF_SPEND]
                ),
                "paths='total'",
                id="under a direct path",
            ),
            pytest.param(
                lambda incr: incr.contribution_over_spend(
                    frequency="all_time", induced_spend=["nope"]
                ),
                r"'nope'.*declared",
                id="a name nobody declared",
            ),
            pytest.param(
                lambda incr: incr.contribution_over_spend(
                    frequency="all_time", induced_spend=[DEMAND]
                ),
                r"'funnel_demand'.*is_spend",
                id="a mediator that is not spend",
            ),
            pytest.param(
                lambda incr: incr.marginal_contribution_over_spend(
                    frequency="all_time", induced_spend=LF_SPEND
                ),
                "sequence of names",
                id="a bare string",
            ),
            pytest.param(
                lambda incr: incr.spend_over_contribution(
                    frequency="all_time", induced_spend=[]
                ),
                "at least one",
                id="an empty list",
            ),
        ],
    )
    def test_what_cannot_enter_a_denominator_is_refused(
        self, funnel_log_link_fitted_mmm, call, complaint
    ):
        """Refused before anything is compiled, by name."""
        with pytest.raises(ValueError, match=complaint):
            call(funnel_log_link_fitted_mmm.incrementality)

    def test_a_spend_node_that_feeds_no_term_is_refused(self, funnel_mmm_data):
        """Its return cannot be in the numerator, so the ratio would count only its cost."""
        mmm = _build_simple_effect_mmm(
            funnel_mmm_data, IndicatorEffect(prefix="ind", is_spend=True), "identity"
        )

        with pytest.raises(ValueError, match=r"'searches'.*feeds no term"):
            mmm.incrementality.contribution_over_spend(
                frequency="all_time", induced_spend=["searches"]
            )

    def test_spend_in_two_currencies_is_not_added(self, panel_mmm_data):
        """The named mediators are summed, so they have to agree on their units."""
        mmm = _panel_mmm(
            panel_mmm_data,
            HubEffect(prefix="dollars", units="USD", is_spend=True),
            HubEffect(prefix="euros", units="EUR", is_spend=True),
        )

        with pytest.raises(ValueError, match=r"units.*'EUR'.*'USD'"):
            mmm.incrementality.contribution_over_spend(
                frequency="all_time", induced_spend=["dollars", "euros"]
            )

    def test_a_national_mediator_is_not_spread_over_the_geos(
        self, panel_hubs_fitted_mmm
    ):
        """A ratio per geo has no place for it; the decomposition returns it as it is.

        Added to a per-geo denominator, a national movement would be counted
        once for every geo.
        """
        incr = panel_hubs_fitted_mmm.incrementality

        with pytest.raises(ValueError, match=r"'national'.*\('country',\)"):
            incr.contribution_over_spend(
                frequency="all_time", induced_spend=["national"]
            )
        with pytest.raises(ValueError, match="share their dimensions"):
            incr.decompose_incremental_contribution(
                frequency="all_time", induced_spend=["national", "local"]
            )
        parts = incr.decompose_incremental_contribution(
            frequency="all_time", induced_spend=["national"]
        )
        moved = incr.compute_incremental_mediator(
            frequency="all_time", mediators=["national"]
        )

        assert parts["induced_spend"].dims == ("chain", "draw", "channel")
        assert_matches(parts["induced_spend"], moved["national"], rtol=1e-12)
        label = EstimandLabel.read(parts["induced_spend"])
        assert (label.quantity, label.induced_spend) == ("induced_spend", ("national",))
        assert label.induced_spend_units == "USD"
        # Each part summed over its own dimensions before anything is divided.
        cells = ("channel", "country")
        roas = parts["total"].sum(cells) / (
            parts["spend_change"].sum(cells) + parts["induced_spend"].sum("channel")
        )
        assert roas.dims == ("chain", "draw")
        assert bool(np.isfinite(roas).all())

    def test_the_decomposition_marks_every_variable(self, funnel_log_link_fitted_mmm):
        """One mark for the request, from the furthest reach of what it reads."""
        parts = funnel_log_link_fitted_mmm.incrementality.decompose_incremental_contribution(
            frequency="monthly", induced_spend=[LF_SPEND]
        )

        assert set(parts.data_vars) == {
            "total",
            "direct",
            "mediated",
            "spend_change",
            "induced_spend",
        }
        assert parts["complete"].values.tolist() == [True] * 4 + [False] * 2
        for name in parts.data_vars:
            assert EstimandLabel.read(parts[name]).frequency == "monthly"

    def test_a_denominator_that_is_not_positive_is_not_divided_by(
        self, funnel_mmm_data
    ):
        """Such a draw is ``NaN``, and the call says how many there are."""
        mmm = _build_simple_effect_mmm(
            funnel_mmm_data,
            ShrinkingHubEffect(prefix="shrink", units="USD", is_spend=True),
            link="identity",
        )

        with pytest.warns(UserWarning, match=r"100\.0% of the draws"):
            roas = mmm.incrementality.contribution_over_spend(
                frequency="all_time", induced_spend=["shrink"]
            )

        assert roas.isnull().all()


class TestMediatorsTable:
    """The declared nodes of a model, and what is wrong with each."""

    def test_one_row_per_declared_node(self, funnel_log_link_fitted_mmm):
        """What a user can pass as ``mediators`` or ``induced_spend``."""
        table = funnel_log_link_fitted_mmm.incrementality.mediators

        assert list(table.columns) == [
            "name",
            "variable",
            "effects",
            "units",
            "scale",
            "is_spend",
            "feeds_term",
            "readable",
            "problem",
            "unreadable",
        ]
        assert table["name"].tolist() == [DEMAND, LF_SPEND]
        assert table["is_spend"].tolist() == [False, True]
        # Missing text is None in every column, with or without text beside it.
        assert table["units"].tolist() == [None, "USD"]
        assert table["problem"].tolist() == [None, None]
        assert table["readable"].all()
        assert table["effects"].tolist() == [("FunnelEffect",)] * 2

    def test_a_fault_is_a_row_and_not_an_exception(self, two_hubs_fitted_mmm):
        """The table lists a node that cannot be reported, with the reason."""
        table = two_hubs_fitted_mmm.incrementality.mediators.set_index("name")

        assert bool(table.loc["pooled", "readable"])
        assert not table.loc["split", "readable"]
        assert "'channel' dimension" in table.loc["split", "unreadable"]
        assert table.loc["split", "problem"] is None

    def test_a_model_without_declarations_has_an_empty_table(self, simple_fitted_mmm):
        """The columns are there, with nothing in them."""
        table = simple_fitted_mmm.incrementality.mediators

        assert len(table) == 0
        assert "readable" in table.columns
