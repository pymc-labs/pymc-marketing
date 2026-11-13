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
"""Tests for the paths of an increment: direct, mediated and total.

The numbers are compared with an oracle that shares neither the batched
evaluator nor ``ConditionedModel`` with the module.  The model is cloned, the
mediator nodes are turned into inputs that hold their posterior draws, spend is
set with ``pm.set_data`` and the response is read with
``sample_posterior_predictive``.  Holding the nodes is the intervention itself,
``do(spend := x', mediators := M(x))``, while the module gets there by leaving
terms out of a sum.
"""

import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytest
import xarray as xr

from pymc_marketing.mmm import MMM
from pymc_marketing.mmm.additive_effect import (
    IncrementalitySpec,
    MediatorSpec,
    MuEffect,
)
from pymc_marketing.mmm.estimand_label import LABEL_ATTR, EstimandLabel
from pymc_marketing.mmm.link import LinkFunction
from pymc_marketing.model_graph import deterministics_to_flat
from tests.mmm.conftest import (
    FunnelEffect,
    NegativeEffect,
    _build_funnel_mmm,
    _build_simple_effect_mmm,
    _funnel_effect,
    _media_transformation,
)

FUNNEL_TERM = "funnel_effect_contribution"
FUNNEL_NODES = ("funnel_demand",)
"""Demand cuts every route from spend into the funnel's term, so holding it holds the term."""


class HubEffect(MuEffect):
    """Spend moves a hub and the hub moves the target: a mediator with one node.

    ``national`` sums the hub over every dimension of a panel but ``date``, so
    the node carries fewer dimensions than the model.
    """

    prefix: str
    role: str | None = "mediator"
    units: str | None = None
    scale: float = 1.0
    is_spend: bool = False
    national: bool = False

    def create_data(self, mmm) -> None:
        """No data of its own."""

    def set_data(self, mmm, model, X) -> None:
        """No data of its own."""

    def incrementality_spec(self) -> IncrementalitySpec:
        """Declare the hub, under the effect's prefix as its name."""
        return IncrementalitySpec(
            role=self.role,
            mediators=(
                MediatorSpec(
                    name=self.prefix,
                    variable=f"{self.prefix}_hub",
                    units=self.units,
                    scale=self.scale,
                    is_spend=self.is_spend,
                ),
            ),
        )

    def create_effect(self, mmm):
        """Add ``beta * hub`` with ``hub`` a saturating function of total spend."""
        spend = mmm.channel_data_scaled
        summed = (
            [dim for dim in spend.dims if dim != "date"] if self.national else "channel"
        )
        hub = pmd.Deterministic(
            f"{self.prefix}_hub", pmd.math.tanh(spend.sum(dim=summed))
        )
        beta = pmd.HalfNormal(f"{self.prefix}_beta", sigma=1.0)
        return pmd.Deterministic(f"{self.prefix}_effect_contribution", beta * hub)


class DirectSynergyEffect(NegativeEffect):
    """A product of the channels' spend, declared as one more direct function of spend."""

    def incrementality_spec(self) -> IncrementalitySpec:
        """Claim the direct path."""
        return IncrementalitySpec(role="direct")


class CentredControlEffect(MuEffect):
    """A control centred on its own mean over the fitted dates, inside the graph.

    It does not read spend, so it is no accounted term, and its value on a
    window is not its value on the full axis.  Compared against the full-axis
    baseline, the predictor's move on a window would be off by the difference.
    """

    prefix: str

    def create_data(self, mmm) -> None:
        """Reads ``lf_budget``, which the funnel effect registers."""

    def set_data(self, mmm, model, X) -> None:
        """No data of its own."""

    def create_effect(self, mmm):
        """Add ``beta * (lf_budget - mean(lf_budget))``."""
        series = mmm.model["lf_budget"]
        beta = pmd.Normal(f"{self.prefix}_beta", sigma=1.0)
        return pmd.Deterministic(
            f"{self.prefix}_effect_contribution",
            beta * (series - series.mean(dim="date")),
        )


class MemoryFunnelEffect(FunnelEffect):
    """A funnel effect whose term also adds a long-memory transform of another series.

    The term is accounted, and on a window shorter than that memory it takes
    another value than on the full axis with spend unchanged.
    """

    def create_effect(self, mmm):
        """Add the funnel's term plus a twelve-date adstock of ``lf_budget``."""
        funnel = super().create_effect(mmm)
        memory = _media_transformation("memory", (), 12)
        return pmd.Deterministic(
            f"{self.prefix}_total_contribution",
            funnel + memory(mmm.model["lf_budget"], dim="date"),
        )

    @property
    def contribution_var_name(self) -> str:
        """The sum of both parts is what enters the linear predictor."""
        return f"{self.prefix}_total_contribution"


class FullAxisMemoryFunnelEffect(MemoryFunnelEffect):
    """The same effect, declared as needing the whole date axis."""

    def incrementality_spec(self) -> IncrementalitySpec:
        """Add ``evaluation_mode="full"`` to the funnel's declaration."""
        return (
            super().incrementality_spec().model_copy(update={"evaluation_mode": "full"})
        )


def response(mmm, spend, *, held=(), var=None) -> xr.DataArray:
    """Evaluate the response, or *var*, per draw and date under *spend*.

    Nodes named in *held* keep their stored posterior draws whatever spend is.
    """
    names = [*mmm.frozen_deterministics, *held]
    model = (
        deterministics_to_flat(mmm.model, names=names) if names else mmm.model.copy()
    )
    log_link = mmm.link == LinkFunction.LOG
    name = var or ("mu_original_scale" if log_link else "mu")
    with model:
        pm.set_data({"channel_data": spend.astype(model["channel_data"].type.dtype)})
        values = pm.sample_posterior_predictive(
            mmm.idata, var_names=[name], progressbar=False
        ).posterior_predictive[name]
    if var is None and not log_link:
        values = values * mmm.incrementality.data.get_target_scale()
    return values


def moved_spend(mmm, period, channel, factor) -> np.ndarray:
    """Actual spend with *period* scaled by *factor*, for one channel or for all."""
    spend = mmm.model["channel_data"].get_value().copy()
    dates = mmm.incrementality.data.dates
    in_period = np.asarray((dates >= period[0]) & (dates <= period[1]))
    if channel is None:
        spend[in_period] = spend[in_period] * factor
    else:
        spend[in_period, ..., channel] = spend[in_period, ..., channel] * factor
    return spend


def oracle(mmm, *, frequency, factor=0.0, scope="per_channel", held=()) -> dict:
    """Total, direct and mediated increments by brute force, laid out as the module's.

    Each difference is summed over every fitted date.  The models here are
    causal filters of spend, so the difference is zero before a period and past
    its carryover, and the sum equals the module's sum over its window.
    """
    incr = mmm.incrementality
    dates = incr.data.dates
    periods = incr._create_period_groups(dates[0], dates[-1], frequency)
    channels = list(mmm.channel_columns)
    keys = range(len(channels)) if scope == "per_channel" else [None]
    factual = response(mmm, mmm.model["channel_data"].get_value())
    sign = 1.0 if factor <= 1.0 else -1.0

    stacked: dict[str, list[xr.DataArray]] = {"total": [], "direct": [], "mediated": []}
    for period in periods:
        parts: dict[str, list[xr.DataArray]] = {name: [] for name in stacked}
        for key in keys:
            spend = moved_spend(mmm, period, key, factor)
            moved = response(mmm, spend)
            moved_held = response(mmm, spend, held=held) if held else moved
            parts["total"].append(sign * (factual - moved).sum("date"))
            parts["direct"].append(sign * (factual - moved_held).sum("date"))
            parts["mediated"].append(sign * (moved_held - moved).sum("date"))
        for name, values in parts.items():
            stacked[name].append(
                xr.concat(values, dim=pd.Index(channels, name="channel"))
                if scope == "per_channel"
                else values[0]
            )
    ends = pd.DatetimeIndex([end for _, end in periods], name="date")
    return {
        name: values[0] if frequency == "all_time" else xr.concat(values, dim=ends)
        for name, values in stacked.items()
    }


def assert_matches(result: xr.DataArray, expected: xr.DataArray, rtol=1e-6) -> None:
    """Assert *result* equals *expected* per draw, whatever order the dimensions are in.

    The ``complete`` mark of either side is left out of the comparison.
    """
    result = result.drop_vars("complete", errors="ignore")
    expected = expected.drop_vars("complete", errors="ignore")
    scale = float(np.abs(expected).max())
    xr.testing.assert_allclose(
        result, expected.transpose(*result.dims), rtol=rtol, atol=1e-9 * scale
    )


@pytest.fixture(scope="module")
def two_mediators_declared_mmm(funnel_mmm_data):
    """Log-link funnel MMM with a second mediator effect that has its own hub."""
    return _build_funnel_mmm(
        funnel_mmm_data,
        link="log",
        effects=[_funnel_effect(), HubEffect(prefix="halo")],
    )


@pytest.fixture(scope="module")
def direct_synergy_fitted_mmm(funnel_mmm_data):
    """Identity-link MMM whose one effect is declared as a direct function of spend."""
    return _build_simple_effect_mmm(
        funnel_mmm_data, DirectSynergyEffect(prefix="synergy"), link="identity"
    )


class TestDirectAndMediated:
    """The two parts of a total, against the intervention that defines them."""

    @pytest.mark.parametrize(
        "fixture_name", ["funnel_log_link_fitted_mmm", "funnel_identity_fitted_mmm"]
    )
    @pytest.mark.parametrize(
        "frequency, factor",
        [("all_time", 0.0), ("monthly", 0.0), ("monthly", 1.01), ("all_time", 0.5)],
    )
    def test_the_decomposition_matches_holding_the_mediator(
        self, request, fixture_name, frequency, factor
    ):
        """``direct`` is the response with spend moved and demand where it was.

        Checked per draw, period and channel, under both links, when spend is
        removed, cut in half and raised.  The factor above one pins the sign:
        the result is then counterfactual minus actual.
        """
        mmm = request.getfixturevalue(fixture_name)

        parts = mmm.incrementality.decompose_incremental_contribution(
            frequency=frequency, counterfactual_spend_factor=factor
        )

        expected = oracle(mmm, frequency=frequency, factor=factor, held=FUNNEL_NODES)
        for name in ("total", "direct", "mediated"):
            assert_matches(parts[name], expected[name])
        xr.testing.assert_allclose(
            parts["direct"] + parts["mediated"], parts["total"], rtol=1e-9
        )
        assert float(np.abs(parts["mediated"]).max()) > 1e-3 * float(
            np.abs(parts["total"]).max()
        )

    def test_the_joint_scope_moves_every_channel_at_once(
        self, funnel_log_link_fitted_mmm
    ):
        """One number per period, with the same two parts."""
        mmm = funnel_log_link_fitted_mmm

        parts = mmm.incrementality.decompose_incremental_contribution(
            frequency="monthly", scope="joint"
        )

        expected = oracle(mmm, frequency="monthly", scope="joint", held=FUNNEL_NODES)
        assert "channel" not in parts["direct"].dims
        for name in ("total", "direct", "mediated"):
            assert_matches(parts[name], expected[name])

    def test_every_mediator_effect_is_held(self, two_mediators_declared_mmm):
        """``direct`` holds the nodes of both effects, and its label lists both terms."""
        mmm = two_mediators_declared_mmm

        parts = mmm.incrementality.decompose_incremental_contribution(
            frequency="all_time"
        )

        expected = oracle(mmm, frequency="all_time", held=(*FUNNEL_NODES, "halo_hub"))
        for name in ("total", "direct", "mediated"):
            assert_matches(parts[name], expected[name])
        assert EstimandLabel.read(parts["direct"]).held == (
            FUNNEL_TERM,
            "halo_effect_contribution",
        )

    def test_under_an_additive_link_direct_is_the_channel_contribution(
        self, funnel_identity_fitted_mmm, funnel_log_link_fitted_mmm
    ):
        """What the notebooks assemble by hand is right under the identity link only.

        Removing a channel removes its contribution, and with nothing else
        moving, the response drops by exactly that.  Under the log link the
        same contribution is a factor on everything else, so the two differ.
        """

        def direct_and_contribution(mmm):
            direct = mmm.incrementality.compute_incremental_contribution(
                frequency="all_time", paths="direct"
            )
            stored = mmm.idata.posterior["channel_contribution_original_scale"]
            return direct, stored.sum("date").transpose(*direct.dims)

        direct, contribution = direct_and_contribution(funnel_identity_fitted_mmm)
        xr.testing.assert_allclose(direct, contribution, rtol=1e-8)

        direct, contribution = direct_and_contribution(funnel_log_link_fitted_mmm)
        assert not np.allclose(direct, contribution, rtol=1e-2)


class TestPathsOnThePublicMethods:
    """``paths=`` on the increments and the ratios, and what each result says of itself."""

    @pytest.mark.parametrize("paths", ["direct", "mediated"])
    def test_an_increment_is_the_decomposition_s_variable(
        self, funnel_log_link_fitted_mmm, paths
    ):
        """Asking for one part alone gives the number the decomposition gives."""
        incr = funnel_log_link_fitted_mmm.incrementality

        alone = incr.compute_incremental_contribution(frequency="monthly", paths=paths)
        together = incr.decompose_incremental_contribution(frequency="monthly")

        xr.testing.assert_allclose(alone, together[paths], rtol=1e-12)
        assert EstimandLabel.read(alone) == EstimandLabel.read(together[paths])

    def test_the_joint_increment_takes_paths_too(self, funnel_log_link_fitted_mmm):
        """The joint method and the joint decomposition agree."""
        incr = funnel_log_link_fitted_mmm.incrementality

        alone = incr.compute_joint_incremental_contribution(
            frequency="all_time", paths="mediated"
        )
        together = incr.decompose_incremental_contribution(
            frequency="all_time", scope="joint"
        )

        xr.testing.assert_allclose(alone, together["mediated"], rtol=1e-12)
        assert EstimandLabel.read(alone).scope == "joint"

    @pytest.mark.parametrize(
        "factor, definition",
        [(0.0, "Y(x) - Y(x', M(x))"), (1.01, "Y(x', M(x)) - Y(x)")],
    )
    def test_a_direct_increment_says_what_was_held(
        self, funnel_log_link_fitted_mmm, factor, definition
    ):
        """The label carries the request, the held terms and the sign."""
        direct = (
            funnel_log_link_fitted_mmm.incrementality.compute_incremental_contribution(
                frequency="monthly", paths="direct", counterfactual_spend_factor=factor
            )
        )

        label = EstimandLabel.read(direct)

        assert list(direct.attrs) == [LABEL_ATTR]
        assert (label.quantity, label.paths) == ("incremental_contribution", "direct")
        assert (label.scope, label.frequency) == ("per_channel", "monthly")
        assert (label.held, label.held_at) == ((FUNNEL_TERM,), "factual")
        assert label.accounting_verified is True
        assert label.definition == definition
        assert (label.start_date, label.end_date) == (
            "2023-01-02T00:00:00",
            "2023-06-12T00:00:00",
        )

    def test_a_ratio_divides_the_part_by_the_same_spend(
        self, funnel_log_link_fitted_mmm
    ):
        """The ratio methods keep their own denominators."""
        incr = funnel_log_link_fitted_mmm.incrementality
        direct = incr.compute_incremental_contribution(
            frequency="all_time", paths="direct"
        )
        spend = incr._aggregate_spend("all_time")

        roas = incr.contribution_over_spend(frequency="all_time", paths="direct")
        cac = incr.spend_over_contribution(frequency="all_time", paths="direct")

        xr.testing.assert_allclose(roas, direct / spend, rtol=1e-12)
        xr.testing.assert_allclose(
            cac, (spend / direct).transpose(*cac.dims), rtol=1e-12
        )
        assert EstimandLabel.read(roas).definition == (
            "[Y(x) - Y(x', M(x))] / [S(x) - S(x')]"
        )
        assert EstimandLabel.read(cac).quantity == "spend_over_contribution"

    def test_a_marginal_ratio_of_a_part(self, funnel_log_link_fitted_mmm):
        """The mediated share of a marginal return, per unit of added spend."""
        incr = funnel_log_link_fitted_mmm.incrementality
        mediated = incr.compute_incremental_contribution(
            frequency="all_time", paths="mediated", counterfactual_spend_factor=1.05
        )
        spend = incr._aggregate_spend("all_time")

        marginal = incr.marginal_contribution_over_spend(
            frequency="all_time", spend_increase_pct=0.05, paths="mediated"
        )

        xr.testing.assert_allclose(marginal, mediated / (0.05 * spend), rtol=1e-12)
        label = EstimandLabel.read(marginal)
        assert label.counterfactual_spend_factor == 1.05
        assert label.definition == "[Y(x') - Y(x', M(x))] / [S(x') - S(x)]"

    def test_spend_over_a_mediated_share_is_not_offered(
        self, funnel_log_link_fitted_mmm
    ):
        """A share of the total can be zero, and its reciprocal says nothing."""
        with pytest.raises(ValueError, match="paths='mediated'"):
            funnel_log_link_fitted_mmm.incrementality.spend_over_contribution(
                frequency="all_time", paths="mediated"
            )

    @pytest.mark.parametrize(
        "call, complaint",
        [
            pytest.param(
                lambda incr: incr.compute_incremental_contribution(
                    frequency="all_time", paths="indirect"
                ),
                "paths must be one of",
                id="an unknown path",
            ),
            pytest.param(
                lambda incr: incr.contribution_over_spend(
                    frequency="all_time", paths="Direct"
                ),
                "paths must be one of",
                id="an unknown path on a ratio",
            ),
            pytest.param(
                lambda incr: incr.compute_incremental_contribution(
                    frequency="all_time", counterfactual_spend_factor=float("inf")
                ),
                "counterfactual_spend_factor",
                id="an infinite spend factor",
            ),
            pytest.param(
                lambda incr: incr.compute_incremental_contribution(
                    frequency="all_time", counterfactual_spend_factor=float("nan")
                ),
                "counterfactual_spend_factor",
                id="a spend factor that is not a number",
            ),
            pytest.param(
                lambda incr: incr.marginal_contribution_over_spend(
                    frequency="all_time", spend_increase_pct=1e-20
                ),
                "too small to change spend",
                id="an increase too small to change spend",
            ),
        ],
    )
    def test_a_request_that_makes_no_sense_is_refused(
        self, funnel_log_link_fitted_mmm, call, complaint
    ):
        """Refused before anything is compiled."""
        with pytest.raises(ValueError, match=complaint):
            call(funnel_log_link_fitted_mmm.incrementality)


class TestNothingToHold:
    """Models without a mediator answer every path, so one loop covers them all."""

    @pytest.mark.parametrize(
        "fixture_name", ["direct_synergy_fitted_mmm", "simple_fitted_mmm"]
    )
    def test_direct_is_the_total_and_mediated_is_zero(self, request, fixture_name):
        """A direct effect only, or no effect at all: nothing is on the mediated path."""
        incr = request.getfixturevalue(fixture_name).incrementality

        total = incr.compute_incremental_contribution(frequency="monthly")
        direct = incr.compute_incremental_contribution(
            frequency="monthly", paths="direct"
        )
        mediated = incr.compute_incremental_contribution(
            frequency="monthly", paths="mediated"
        )

        xr.testing.assert_allclose(direct, total, rtol=1e-12)
        assert not mediated.any()
        label = EstimandLabel.read(mediated)
        assert (label.held, label.held_at) == ((), "factual")
        assert label.accounting_verified is True

    def test_an_effect_with_no_role_is_refused_and_its_total_is_not(
        self, negative_effect_fitted_mmm
    ):
        """Opting in to totals does not say which path a term is on."""
        incr = negative_effect_fitted_mmm.incrementality

        with pytest.raises(NotImplementedError, match="NegativeEffect"):
            incr.compute_incremental_contribution(frequency="all_time", paths="direct")
        with pytest.raises(NotImplementedError, match="NegativeEffect"):
            incr.decompose_incremental_contribution(frequency="all_time")
        assert (
            incr.compute_incremental_contribution(frequency="all_time").notnull().all()
        )


class TestDefaultTotal:
    """A request that names no path returns the total, with a label on it."""

    def test_the_values_are_the_core_s_and_the_label_is_the_only_attribute(
        self, funnel_log_link_fitted_mmm
    ):
        """The label is attached to the numbers the core computes, unchanged."""
        incr = funnel_log_link_fitted_mmm.incrementality
        request = {
            "frequency": "monthly",
            "start_date": None,
            "end_date": None,
            "include_carryover": True,
            "num_samples": None,
            "random_state": None,
            "counterfactual_spend_factor": 0.0,
            "central_tendency": "median",
        }
        core = incr._stack_periods(
            incr._compute_increments(scope="per_channel", **request), "monthly"
        )

        total = incr.compute_incremental_contribution(frequency="monthly")

        assert core.attrs == {}
        np.testing.assert_array_equal(total.values, core.values)
        assert list(total.attrs) == [LABEL_ATTR]
        label = EstimandLabel.read(total)
        assert (label.paths, label.held, label.held_at) == ("total", (), None)
        assert label.accounting_verified is True
        assert label.definition == "Y(x) - Y(x')"

    def test_a_total_reads_no_role_and_no_node(self, funnel_mmm_data):
        """A spec that only a path request would read is never asked for.

        The effect does not read spend, so a total skips it without calling
        ``incrementality_spec``.  The requests that list mediator nodes do call
        it, and its error is theirs.
        """

        class Unreadable(CentredControlEffect):
            def create_effect(self, mmm):
                beta = pmd.Normal(f"{self.prefix}_beta", sigma=1.0)
                return pmd.Deterministic(
                    f"{self.prefix}_effect_contribution", beta * mmm.model["lf_budget"]
                )

            def incrementality_spec(self):
                raise RuntimeError("this spec was not meant to be read")

        mmm = _build_funnel_mmm(
            funnel_mmm_data,
            link="log",
            effects=[_funnel_effect(), Unreadable(prefix="plain")],
        )
        incr = mmm.incrementality

        assert (
            incr.compute_incremental_contribution(frequency="all_time").notnull().all()
        )
        with pytest.raises(RuntimeError, match="not meant to be read"):
            incr.compute_incremental_contribution(frequency="all_time", paths="direct")

    def test_a_total_the_probe_could_not_check_says_so(
        self, funnel_mmm_data, monkeypatch
    ):
        """``accounting_verified`` is false only under the warning that says why."""
        mmm = _build_funnel_mmm(funnel_mmm_data, link="log")
        monkeypatch.setattr(type(mmm), "frozen_deterministics", ["mu"])

        with pytest.warns(UserWarning, match="could not be recovered"):
            total = mmm.incrementality.compute_incremental_contribution(
                frequency="all_time"
            )

        assert EstimandLabel.read(total).accounting_verified is False

    def test_a_model_without_effects_has_no_accounting_left_open(self, funnel_mmm_data):
        """Spend reaches its mean through ``channel_contribution`` alone.

        Where no probe can be placed its total is warned that the reach was
        not measured, and neither the warning nor the label says the
        accounting is unverified.
        """
        data = TestAccountingOnRows._no_probe_data(funnel_mmm_data)
        mmm = _build_funnel_mmm(data, link="log", effects=[])

        with pytest.warns(UserWarning, match="could not be measured") as caught:
            total = mmm.incrementality.compute_incremental_contribution(
                frequency="monthly"
            )

        assert not any("unverified" in str(warning.message) for warning in caught)
        assert EstimandLabel.read(total).accounting_verified is True


class TestAccountingOnRows:
    """A path request checks, on its own rows, that its terms carry the whole move."""

    def test_a_term_that_does_not_read_spend_may_differ_on_a_window(
        self, funnel_mmm_data
    ):
        """The predictor is compared with itself on the same window.

        A control centred on its date mean takes another value on a window.
        It cancels between a perturbed row and the factual row of the same
        window, which the full-axis baseline could not make it do.
        """
        mmm = _build_funnel_mmm(
            funnel_mmm_data,
            link="log",
            effects=[_funnel_effect(), CentredControlEffect(prefix="centred")],
        )

        direct = mmm.incrementality.compute_incremental_contribution(
            frequency="monthly", paths="direct"
        )

        expected = oracle(mmm, frequency="monthly", held=FUNNEL_NODES)
        assert_matches(direct, expected["direct"])

    def test_an_accounted_term_that_differs_on_a_window_is_named(self, funnel_mmm_data):
        """On a window the term moves without spend moving, and the refusal names it.

        The fix it names works: declared as needing the full axis, the same
        effect is answered, and matches the intervention.
        """
        windowed = _build_funnel_mmm(
            funnel_mmm_data, link="log", effects=[_funnel_effect(MemoryFunnelEffect)]
        )
        full = _build_funnel_mmm(
            funnel_mmm_data,
            link="log",
            effects=[_funnel_effect(FullAxisMemoryFunnelEffect)],
        )

        with pytest.raises(
            ValueError, match=r"(?s)'funnel_total_contribution'.*evaluation_mode='full'"
        ):
            windowed.incrementality.compute_incremental_contribution(
                frequency="monthly", paths="direct"
            )
        direct = full.incrementality.compute_incremental_contribution(
            frequency="monthly", paths="direct"
        )

        expected = oracle(full, frequency="monthly", held=FUNNEL_NODES)
        assert_matches(direct, expected["direct"])

    @staticmethod
    def _no_probe_data(funnel_mmm_data) -> dict:
        """Funnel data whose second channel spends on the last date alone."""
        data = {key: value.copy(deep=True) for key, value in funnel_mmm_data.items()}
        media = data["X"]["media"].values
        media[:-1, 1] = 0.0
        data["X_df"]["channel_2"] = media[:, 1]
        return data

    def test_a_model_the_probe_cannot_place_is_still_checked(self, funnel_mmm_data):
        """The rows need no probe, so a path is verified where a total is not.

        Both are warned that the reach could not be measured.  Only the total
        is told that its accounting goes unverified.
        """
        mmm = _build_funnel_mmm(self._no_probe_data(funnel_mmm_data), link="log")
        incr = mmm.incrementality

        with pytest.warns(UserWarning, match="goes unverified"):
            total = incr.compute_incremental_contribution(frequency="monthly")
        with pytest.warns(UserWarning, match="could not be measured") as caught:
            direct = incr.compute_incremental_contribution(
                frequency="monthly", paths="direct"
            )

        assert not any("unverified" in str(warning.message) for warning in caught)
        assert EstimandLabel.read(total).accounting_verified is False
        assert EstimandLabel.read(direct).accounting_verified is True

    def test_a_path_the_terms_do_not_carry_is_refused_on_the_rows(
        self, funnel_mmm_data
    ):
        """Without a probe a total answers with a warning; a path request raises.

        The effect adds its term to the predictor twice and reports it once, so
        half of the mediated move is in no accounted term.
        """

        class HalfReportedFunnelEffect(FunnelEffect):
            def create_effect(self, mmm):
                reported = super().create_effect(mmm)
                return 2.0 * reported

        mmm = _build_funnel_mmm(
            self._no_probe_data(funnel_mmm_data),
            link="log",
            effects=[_funnel_effect(HalfReportedFunnelEffect)],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(NotImplementedError, match="Some path from spend"):
                mmm.incrementality.compute_incremental_contribution(
                    frequency="monthly", paths="direct"
                )

    def test_a_frozen_predictor_leaves_nothing_to_check_against(
        self, funnel_mmm_data, monkeypatch
    ):
        """A path is never returned unverified, so the request is refused."""
        mmm = _build_funnel_mmm(funnel_mmm_data, link="log")
        monkeypatch.setattr(type(mmm), "frozen_deterministics", ["mu"])

        with pytest.raises(NotImplementedError, match="linear predictor"):
            mmm.incrementality.compute_incremental_contribution(
                frequency="all_time", paths="direct"
            )


class TestTheDecomposition:
    """The three paths and the spend change, from one evaluation on one set of draws."""

    def test_every_variable_is_computed_on_the_same_draws(
        self, funnel_log_link_fitted_mmm
    ):
        """A ``Generator`` is consumed once, so the parts add up draw for draw."""
        parts = funnel_log_link_fitted_mmm.incrementality.decompose_incremental_contribution(
            frequency="monthly", num_samples=8, random_state=np.random.default_rng(3)
        )

        assert parts["total"].sizes["draw"] == 8
        xr.testing.assert_allclose(
            parts["direct"] + parts["mediated"], parts["total"], rtol=1e-9
        )
        assert parts.attrs == {}
        assert {
            name: EstimandLabel.read(parts[name]).quantity for name in parts.data_vars
        } == {
            "total": "incremental_contribution",
            "direct": "incremental_contribution",
            "mediated": "incremental_contribution",
            "spend_change": "spend_change",
        }

    @pytest.mark.parametrize("factor, share", [(0.0, 1.0), (0.5, 0.5), (1.01, 0.01)])
    @pytest.mark.parametrize("scope", ["per_channel", "joint"])
    def test_spend_change_is_the_spend_the_counterfactual_moves(
        self, funnel_log_link_fitted_mmm, factor, share, scope
    ):
        """The denominator of any ratio of the parts, on the same periods."""
        incr = funnel_log_link_fitted_mmm.incrementality
        spend = incr._aggregate_spend("monthly")

        parts = incr.decompose_incremental_contribution(
            frequency="monthly", counterfactual_spend_factor=factor, scope=scope
        )

        expected = share * (spend if scope == "per_channel" else spend.sum("channel"))
        xr.testing.assert_allclose(parts["spend_change"], expected, rtol=1e-9)
        assert "chain" not in parts["spend_change"].dims
        label = EstimandLabel.read(parts["spend_change"])
        assert label.definition == ("S(x') - S(x)" if factor > 1 else "S(x) - S(x')")

    def test_a_ratio_of_the_parts_is_the_ratio_method(self, funnel_log_link_fitted_mmm):
        """Arithmetic on one result reproduces a ratio, to rounding."""
        incr = funnel_log_link_fitted_mmm.incrementality

        parts = incr.decompose_incremental_contribution(frequency="all_time")
        roas = incr.contribution_over_spend(frequency="all_time")

        xr.testing.assert_allclose(
            parts["total"] / parts["spend_change"], roas, rtol=1e-9
        )


def test_a_declaring_effect_survives_save_and_load(tmp_path, funnel_mmm_data):
    """The declaration is rebuilt from what the effect saves, and gives the same numbers."""
    mmm = _build_simple_effect_mmm(funnel_mmm_data, HubEffect(prefix="halo"))
    before = mmm.incrementality.decompose_incremental_contribution(frequency="all_time")
    path = tmp_path / "mmm.nc"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mmm.save(str(path))
        loaded = MMM.load(str(path))
    after = loaded.incrementality.decompose_incremental_contribution(
        frequency="all_time"
    )

    assert float(np.abs(before["mediated"]).max()) > 0
    for name in ("total", "direct", "mediated"):
        xr.testing.assert_allclose(after[name], before[name], rtol=1e-9)
    assert EstimandLabel.read(after["direct"]).held == ("halo_effect_contribution",)
