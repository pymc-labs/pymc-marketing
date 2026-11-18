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
"""A tiered price schedule: checking its brackets, and settling a budget allocation on its breakpoints.

:class:`~pymc_marketing.mmm.price_response.TieredPriceResponse` prices each bracket of money at its own rate. This
module holds what the declaration and the optimizer need beyond the price map itself.

**Checking the brackets.** The declaration checks its schedule as two ``nan``-padded arrays, breakpoints and rate
multipliers, with one row per label or cell and the brackets along the columns, once at construction and again on
the cells it resolves to. :class:`ScheduleRows` holds those rows, names them in messages and refuses what is not a
schedule.

**Settling on the breakpoints.** A schedule changes slope at its breakpoints, and a gradient solver steps across
such a kink badly. :meth:`~pymc_marketing.mmm.budget_optimizer.BudgetOptimizer.allocate_budget` therefore solves
once with every breakpoint inside the bounds blended, and uses that solve only to find the linear piece of the map
each entry is on. :class:`BreakpointSettler` then solves on those pieces, where no breakpoint is left to cross: an
entry a pass leaves at a breakpoint end of its piece is pinned there, and :func:`entries_leaving_breakpoints` decides
whether a pinned entry would gain by leaving it. Each :meth:`BreakpointSettler.settle` call keeps its state in a
:class:`SettleRun`, and a pass's plan is a :class:`PiecewisePlan`. :func:`objective_scale_from_gradient` is the
objective scale every solve of ``allocate_budget`` uses, here because the passes measure their progress on it.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.optimize import OptimizeResult, linprog

from pymc_marketing.mmm.spend_reach import _PKG_PREFIX
from pymc_marketing.names.mmm import Dim

if TYPE_CHECKING:
    # Only an annotation: optimization_variables imports price_response, which imports this module.
    from pymc_marketing.mmm.optimization_variables import OptimizationVariables

__all__ = [
    "Bounds",
    "BreakpointSettler",
    "BreakpointStatus",
    "PiecewisePlan",
    "ScheduleRows",
    "SettleRun",
    "entries_leaving_breakpoints",
    "objective_scale_from_gradient",
]

type Bounds = list[tuple[float | None, float | None]]
"""SciPy bounds of every flat entry, ``None`` where open."""
BreakpointStatus = Literal["certified", "settled", "unsettled", "skipped"]
"""How the passes of :class:`BreakpointSettler` ended, recorded as ``scipy_result["breakpoint_status"]``."""


@dataclass(frozen=True)
class ScheduleRows:
    """A schedule as ``nan``-padded rows, one per label or cell, and the name of row ``i`` for messages.

    Attributes
    ----------
    breakpoints : np.ndarray or None
        Shape ``(rows, K)``. ``None`` where the two fields could not be laid over the same cells.
    multipliers : np.ndarray
        Shape ``(rows, K + 1)``, with no columns where only the breakpoints are checked.
    name : Callable[[int], object]
        The name of row ``i`` in messages.
    label : str
        Prefix of every message.
    """

    breakpoints: np.ndarray | None
    multipliers: np.ndarray
    name: Callable[[int], object]
    label: str

    def refuse_invalid_schedule(
        self, rate_tolerance: float, reference_rate: float | None
    ) -> None:
        """Raise if the rows are not a schedule.

        Run at construction and again on the resolved cells: ``frozen`` stops a field being reassigned, but the
        declaration holds the caller's dict or ``DataArray``, which can still be mutated in place. Counts are
        checked where both arrays are given, on the rows that have multipliers.

        Parameters
        ----------
        rate_tolerance : float
            How far from 1 a multiplier may be, as a factor either way, and still read as a ratio of two rates.
        reference_rate : float or None
            The rate the base price stands for, quoted in the message about a far multiplier; ``None`` when the
            declaration has none.

        Raises
        ------
        ValueError
            If a row has ``nan`` before a value, a multiplier that is not positive and finite, breakpoints that are
            not non-negative, finite and strictly ascending, a multiplier count that is not its breakpoint count
            plus one, or a multiplier further from 1 than ``rate_tolerance``.
        """
        self._refuse_inner_nan()
        self._refuse_bad_multipliers()
        self._refuse_bad_breakpoints()
        self._refuse_mismatched_counts()
        self._refuse_far_multipliers(rate_tolerance, reference_rate)

    def bracket_counts(self) -> np.ndarray:
        """Return the number of brackets per row.

        Returns
        -------
        np.ndarray
            Shape ``(rows,)``: the count of non-``nan`` multipliers in each row.
        """
        return (~np.isnan(self.multipliers)).sum(axis=1)

    def _format_row_names(self, bad: np.ndarray) -> str:
        """Return the names of the first five flagged rows for a message, with ``...`` when there are more."""
        rows = np.flatnonzero(bad)
        return (
            f"{[self.name(int(i)) for i in rows[:5]]}{', ...' if rows.size > 5 else ''}"
        )

    def _refuse_inner_nan(self) -> None:
        """Raise if a row of either array has ``nan`` before a value."""
        for what, values in (
            ("breakpoints", self.breakpoints),
            ("rate_multipliers", self.multipliers),
        ):
            if values is None or values.shape[1] < 2:
                continue
            present = ~np.isnan(values)
            gap = (~present[:, :-1] & present[:, 1:]).any(axis=1)
            if gap.any():
                raise ValueError(
                    f"{self.label}: {what} on {self._format_row_names(gap)} has nan before a value along "
                    f"{Dim.BRACKET!r}. nan pads a cell with fewer brackets, so nan may only trail."
                )

    def _refuse_bad_multipliers(self) -> None:
        """Raise if a multiplier is not positive and finite."""
        r = self.multipliers
        bad = (~np.isnan(r) & ~(np.isfinite(r) & (r > 0.0))).any(axis=1)
        if bad.any():
            raise ValueError(
                f"{self.label}: rate_multipliers on {self._format_row_names(bad)} must be positive and finite: "
                "each is the price of a unit in its bracket relative to the base price."
            )

    def _refuse_bad_breakpoints(self) -> None:
        """Raise if breakpoints are not non-negative, finite and strictly ascending."""
        b = self.breakpoints
        if b is None:
            return
        present = ~np.isnan(b)
        bad = (present & ~(np.isfinite(b) & (b >= 0.0))).any(axis=1)
        bad |= (present[:, 1:] & ~(b[:, 1:] > b[:, :-1])).any(axis=1)
        if bad.any():
            raise ValueError(
                f"{self.label}: breakpoints on {self._format_row_names(bad)} must be non-negative, finite and "
                "strictly ascending: each is where a bracket ends, in budget units."
            )

    def _refuse_mismatched_counts(self) -> None:
        """Raise if a row with multipliers does not have one more multiplier than breakpoints."""
        if self.breakpoints is None:
            return
        n_breakpoints = (~np.isnan(self.breakpoints)).sum(axis=1)
        n_multipliers = self.bracket_counts()
        wrong = (n_multipliers > 0) & (n_multipliers != n_breakpoints + 1)
        if wrong.any():
            first = int(np.flatnonzero(wrong)[0])
            raise ValueError(
                f"{self.label}: one multiplier per bracket: {self._format_row_names(wrong)} have breakpoints and "
                f"multipliers that do not match ({n_breakpoints[first]} breakpoints and {n_multipliers[first]} "
                "multipliers on the first). len(rate_multipliers) must be len(breakpoints) + 1."
            )

    def _refuse_far_multipliers(
        self, rate_tolerance: float, reference_rate: float | None
    ) -> None:
        """Raise if a multiplier is further from 1 than ``rate_tolerance`` allows a ratio of two rates to be."""
        r = self.multipliers
        far = ~np.isnan(r) & ((r > rate_tolerance) | (r < 1.0 / rate_tolerance))
        rows = far.any(axis=1)
        if not rows.any():
            return
        first = int(np.flatnonzero(rows)[0])
        offending = r[first][far[first]]
        worst = float(offending[np.argmax(np.abs(np.log(offending)))])
        division = (
            ""
            if reference_rate is None
            else f" With reference_rate {reference_rate:g}, the ratio is {worst:g} / {reference_rate:g} = "
            f"{worst / reference_rate:g}."
        )
        raise ValueError(
            f"{self.label}: rate_multipliers on {self._format_row_names(rows)} include {worst:g}, which is too far "
            f"from 1 to be a ratio of two rates (rate_tolerance={rate_tolerance:g}). Multipliers are ratios to the "
            "base price, cost_per_unit: an absolute rate goes in cost_per_unit, or is divided by the rate the base "
            f"stands for.{division} Raise rate_tolerance if the ratio is meant."
        )


def objective_scale_from_gradient(grad: np.ndarray) -> float:
    """Return the largest absolute entry of ``grad``, or one if it is not larger than one.

    ``allocate_budget`` divides the objective by it, so that SLSQP's absolute ``ftol`` is relative to the steepest
    marginal response; the passes measure their progress against the same scale.

    Parameters
    ----------
    grad : np.ndarray
        The objective's gradient at the start of the solve.

    Returns
    -------
    float
        The largest absolute entry of ``grad`` when it is finite and above one, else 1.0.
    """
    steepest = float(np.max(np.abs(grad), initial=0.0))
    return steepest if np.isfinite(steepest) and steepest > 1.0 else 1.0


def entries_leaving_breakpoints(
    *,
    right: np.ndarray,
    left: np.ndarray,
    x: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    pinned: np.ndarray,
    equalities: Sequence[np.ndarray],
    active_inequalities: Sequence[np.ndarray],
    size: np.ndarray,
    rtol: float = 1e-9,
    tolerance: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return which pinned entries a first-order improving direction moves off their breakpoint, up and down.

    A cell on a breakpoint has two slopes, one each way, and belongs there when the common marginal return of the
    free entries lies between them; with constraints beyond the budget that is not a comparison of two numbers, so
    this asks directly whether a feasible direction improves the objective to first order. A linear program with
    two variables per entry, up and down, priced at the entry's ``right`` and ``left`` slope of the objective (the
    minimiser's sign) and charged ``tolerance`` of that slope, so a cell leaves only for a relative gain above
    about twice it and solver noise between free cells trades nothing:

    - an entry within ``rtol * size`` of one of its own bounds (``low``, ``high``) cannot move past it; a pinned
      entry may move one unit either way; any other open direction up to ``4 n`` units, so the program is bounded
      and zero is feasible;
    - directions satisfy the linearised ``equalities`` and the ``active_inequalities`` (``jac @ d >= 0``);
    - costs are divided by the largest absolute cost among the open directions that improve, so a steep entry
      held on its bound does not shrink the others' costs below HiGHS's tolerances.

    Parameters
    ----------
    right : np.ndarray
        Shape ``(n,)``: the objective's gradient with every entry on the piece above it, the slope of moving up.
    left : np.ndarray
        Shape ``(n,)``: the objective's gradient with every entry on the piece below it, the slope of moving down.
    x : np.ndarray
        Shape ``(n,)``: the flat plan the directions start from.
    low : np.ndarray
        Shape ``(n,)``: each entry's lower bound, ``-inf`` where open.
    high : np.ndarray
        Shape ``(n,)``: each entry's upper bound, ``inf`` where open.
    pinned : np.ndarray
        Shape ``(n,)``, bool: the entries held on a breakpoint, the only ones the answer can name.
    equalities : sequence of np.ndarray
        The Jacobian row of every equality constraint at ``x``; a direction keeps each at zero.
    active_inequalities : sequence of np.ndarray
        The Jacobian row of every inequality constraint active at ``x``; a direction may only grow each.
    size : np.ndarray
        Shape ``(n,)``: each entry's size, against which closeness to a bound is measured.
    rtol : float, default 1e-9
        How close, relative to ``size``, an entry is to a bound it cannot move past.
    tolerance : float, default 1e-4
        The share of its own slope each direction is charged, so only a relative gain above about twice it counts.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray) or None
        Two bool arrays of shape ``(n,)``: the pinned entries that would leave their breakpoint upwards, and those
        that would leave it downwards; both all ``False`` when no direction improves. ``None`` when HiGHS reports
        anything but an optimum.
    """
    n = x.size
    # HiGHS's feasibility tolerances, and the optimum below which the program reports a gain.
    lp_tolerance = 1e-10
    # The program is homogeneous and a pinned entry moves at most 1, so a move above a millionth of that is a
    # direction, not HiGHS rounding a zero.
    released = 1e-6
    caps = _direction_caps(x, low, high, pinned, rtol * size)
    cost = _direction_costs(right, left, tolerance)
    improving = (caps > 0.0) & (cost < 0.0)
    scale = float(np.max(np.abs(cost[improving]), initial=0.0)) or 1.0
    program = linprog(
        cost / scale,
        **_direction_constraints(equalities, active_inequalities),
        bounds=list(zip(np.zeros(2 * n), caps, strict=True)),
        method="highs",
        options={
            "primal_feasibility_tolerance": lp_tolerance,
            "dual_feasibility_tolerance": lp_tolerance,
        },
    )
    if program.status != 0:
        return None
    if program.fun >= -lp_tolerance:
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    moves = program.x > released
    return pinned & moves[:n], pinned & moves[n:]


def _direction_caps(
    x: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    pinned: np.ndarray,
    reach: np.ndarray,
) -> np.ndarray:
    """Return how far each entry may move up, then down: 1 if pinned, ``4 n`` if free, 0 within ``reach`` of a bound."""
    free_cap = np.where(pinned, 1.0, 4.0 * x.size)
    return np.concatenate(
        [
            np.where(x >= high - reach, 0.0, free_cap),
            np.where(x <= low + reach, 0.0, free_cap),
        ]
    )


def _direction_costs(
    right: np.ndarray, left: np.ndarray, tolerance: float
) -> np.ndarray:
    """Return the cost of moving each entry up, then down: its one-sided slope, charged ``tolerance`` of itself."""
    return np.concatenate(
        [right + tolerance * np.abs(right), -left + tolerance * np.abs(left)]
    )


def _direction_constraints(
    equalities: Sequence[np.ndarray], active_inequalities: Sequence[np.ndarray]
) -> dict[str, np.ndarray | None]:
    """``linprog``'s constraint arguments for the directions, where a Jacobian row acts on ``up - down``.

    An equality stays put to first order; an active inequality may only grow, ``-jac @ d <= 0``.
    """
    arguments: dict[str, np.ndarray | None] = dict.fromkeys(
        ("A_eq", "b_eq", "A_ub", "b_ub")
    )
    if equalities:
        arguments["A_eq"] = np.array(
            [np.concatenate([row, -row]) for row in equalities]
        )
        arguments["b_eq"] = np.zeros(len(equalities))
    if active_inequalities:
        arguments["A_ub"] = -np.array(
            [np.concatenate([row, -row]) for row in active_inequalities]
        )
        arguments["b_ub"] = np.zeros(len(active_inequalities))
    return arguments


@dataclass(frozen=True)
class PiecewisePlan:
    """A solve's plan, with the linear piece of the price map each entry is on.

    A failed solve has ``value`` ``inf`` and a zero ``grad``, and is never compared.

    Attributes
    ----------
    result : OptimizeResult
        The solve the plan comes from.
    x : np.ndarray
        The flat plan, with every pinned entry on its knot.
    piece_low : np.ndarray
        The low end of each entry's piece; a pinned entry's piece is its knot at both ends.
    piece_high : np.ndarray
        The high end of each entry's piece.
    new_arrivals : np.ndarray
        Bool: the entries this solve left at a knot without holding them there, which a closing solve holds with
        both bounds.
    value : float
        The objective at ``x`` on the exact map.
    grad : np.ndarray
        The objective's gradient at ``x`` on the exact map.
    """

    result: OptimizeResult
    x: np.ndarray
    piece_low: np.ndarray
    piece_high: np.ndarray
    new_arrivals: np.ndarray
    value: float
    grad: np.ndarray

    @property
    def all_arrivals_held(self) -> bool:
        """Whether every entry the solve left at a knot was held there by its bounds."""
        return not self.new_arrivals.any()

    @property
    def held_on_knot(self) -> np.ndarray:
        """Which entries are held on a knot: their piece is a single point."""
        return self.piece_low == self.piece_high

    def spans_whole_box(self, low: np.ndarray, high: np.ndarray) -> bool:
        """Return whether every piece is the whole box, so no breakpoint is in any entry's way.

        Parameters
        ----------
        low : np.ndarray
            Each entry's lower bound, ``-inf`` where open.
        high : np.ndarray
            Each entry's upper bound, ``inf`` where open.

        Returns
        -------
        bool
            ``True`` when every piece reaches both of its entry's bounds.
        """
        return not np.any((self.piece_low > low) | (self.piece_high < high))

    def piece_bounds(self) -> Bounds:
        """Return the pieces as SciPy bounds.

        Returns
        -------
        Bounds
            One ``(low, high)`` pair per entry, ``None`` where the piece is open.
        """
        return [
            (
                float(lo) if np.isfinite(lo) else None,
                float(hi) if np.isfinite(hi) else None,
            )
            for lo, hi in zip(self.piece_low, self.piece_high, strict=True)
        ]


@dataclass
class SettleRun:
    """The state of one :meth:`BreakpointSettler.settle` call.

    Parameters
    ----------
    first : OptimizeResult
        The solve with breakpoints blended that the run settles.
    max_passes : int
        The settler's ``max_passes``, quoted in the warning of an unsettled run.

    Attributes
    ----------
    solves : list of OptimizeResult
        Every solve, the first one first, for the summed counts.
    best : PiecewisePlan or None
        The successful pass with the lowest objective on the exact map, ``None`` before the first.
    previous : float
        The objective the progress rule last compared with, ``inf`` before the first comparison.
    """

    first: OptimizeResult
    max_passes: int
    solves: list[OptimizeResult] = field(init=False)
    best: PiecewisePlan | None = None
    previous: float = np.inf

    def __post_init__(self) -> None:
        """Start the list of solves with the first one."""
        self.solves = [self.first]

    def keep_if_best(self, plan: PiecewisePlan) -> PiecewisePlan:
        """Keep a pass if its objective is the lowest so far, and return the best pass.

        Only successful passes get here: :meth:`BreakpointSettler._solve_on_pieces` and ``_snap_held_plan_to_knots``
        return ``None`` for a failed solve, whose ``value`` is ``inf``.

        Parameters
        ----------
        plan : PiecewisePlan
            A successful pass.

        Returns
        -------
        PiecewisePlan
            The pass with the lowest objective so far, ``plan`` included.
        """
        best = self.best
        if best is None or plan.value < best.value:
            self.best = best = plan
        return best

    def progress_stalled(self, plan: PiecewisePlan, tolerance: float) -> bool:
        """Return whether letting entries go since the last check bought less than ``tolerance``.

        This is the progress rule; it records ``plan``'s objective as the one the next check compares with.

        Parameters
        ----------
        plan : PiecewisePlan
            The pass just checked.
        tolerance : float
            The objective decrease below which the passes made no progress.

        Returns
        -------
        bool
            ``True`` when ``plan``'s objective is above the previous one less ``tolerance``.
        """
        stalled = plan.value > self.previous - tolerance
        self.previous = plan.value
        return stalled

    def finish_run(
        self, status: BreakpointStatus, plan: PiecewisePlan | None = None
    ) -> OptimizeResult:
        """Return the plan's result, else the best pass's, else the first solve's, recorded with ``status``.

        Without a plan the run did not settle, and a successful result says so with a warning.

        Parameters
        ----------
        status : BreakpointStatus
            How the passes ended.
        plan : PiecewisePlan, optional
            The settled plan; ``None`` when the run did not settle.

        Returns
        -------
        OptimizeResult
            The chosen result, as :meth:`with_run_record` records it.
        """
        chosen = plan or self.best
        result = self.first if chosen is None else chosen.result
        if plan is None and result.success:
            warnings.warn(
                "allocate_budget: the plan was not settled on its breakpoints: a pass failed, the check could "
                f"not be solved, or {self.max_passes} passes did not settle it. The best plan found comes back; "
                "a cell near a breakpoint may sit off it.",
                UserWarning,
                skip_file_prefixes=(_PKG_PREFIX,),
            )
        return self.with_run_record(result, status)

    def with_run_record(
        self, result: OptimizeResult, status: BreakpointStatus
    ) -> OptimizeResult:
        """Return a copy of ``result`` with the counts summed over every solve and its ``breakpoint_status``.

        Every result the passes return goes through here, so this is the one place the status is written. The
        solves themselves are left as they were.

        Parameters
        ----------
        result : OptimizeResult
            The result to return.
        status : BreakpointStatus
            How the passes ended, written as ``breakpoint_status``.

        Returns
        -------
        OptimizeResult
            A copy of ``result`` whose ``nit``, ``nfev`` and ``njev`` are summed over every solve when there was
            more than one.
        """
        record = OptimizeResult(dict(result))
        if len(self.solves) > 1:
            for key in ("nit", "nfev", "njev"):
                record[key] = sum(int(solve.get(key, 0)) for solve in self.solves)
        record["breakpoint_status"] = status
        return record


@dataclass
class BreakpointSettler:
    """Solve again on each entry's linear piece until no entry on a breakpoint would gain by leaving it.

    A settler is configuration only; each :meth:`settle` call keeps its own :class:`SettleRun`.

    Parameters
    ----------
    variables : OptimizationVariables
        The decision variables; their price maps answer which piece an entry is on.
    objective_and_grad : callable
        The objective and its gradient at a flat plan, on whatever map the variables are conditioned on.
    constraints : sequence of dict
        The compiled SciPy constraints, with ``"type"``, ``"fun"`` and ``"jac"``.
    solve : callable
        ``solve(start, bounds, index)`` runs one solve of the optimizer over ``bounds``; ``index`` tags it in
        ``callback_info``.
    low, high : np.ndarray
        The caller's bounds of every flat entry, ``-inf`` and ``inf`` where open.
    total_budget : float
        The size of a monetary entry, against which closeness to a bound is measured.
    ftol : float
        The objective change below which letting entries go bought nothing, on the scale of
        :func:`objective_scale_from_gradient`
        when ``rescale`` and in objective units otherwise, as SLSQP reads it.
    rescale : bool
        Whether the solves divide the objective by :func:`objective_scale_from_gradient`.
    max_passes : int
        Solves after the first, closing solves included; a run still closing on the last pass, after the progress
        rule fires or with new arrivals, may add one closing solve.
        Default 20.
    rtol : float
        How close, relative, an entry is to the bound or breakpoint that holds it, and a constraint value to
        zero, when SLSQP leaves them; it leaves about ``8e-10``. Default ``1e-9``.

    Attributes
    ----------
    size : np.ndarray
        Each flat entry's size, computed at construction: ``total_budget`` for money, and for a lever the largest
        absolute value among its finite bounds, at least 1.
    """

    variables: OptimizationVariables
    objective_and_grad: Callable[[np.ndarray], tuple[Any, Any]]
    constraints: Sequence[dict]
    solve: Callable[[np.ndarray, Bounds, int], OptimizeResult]
    low: np.ndarray
    high: np.ndarray
    total_budget: float
    ftol: float
    rescale: bool
    max_passes: int = 20
    rtol: float = 1e-9
    size: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Measure every entry's size once: closeness to a bound and constraint slack are relative to it."""
        self.size = self._entry_sizes()

    def record_skipped_passes(
        self, first: OptimizeResult, method: object
    ) -> OptimizeResult:
        """Return ``first`` recorded as ``"skipped"``, with a warning: the passes run under SLSQP only.

        The plan is the first solve's, on a map with its breakpoints blended, so a cell near a breakpoint may sit
        off it.

        Parameters
        ----------
        first : OptimizeResult
            The solve with breakpoints blended.
        method : object
            The ``method`` the caller's ``minimize_kwargs`` ask for, quoted in the warning.

        Returns
        -------
        OptimizeResult
            A copy of ``first`` with ``breakpoint_status`` ``"skipped"``.
        """
        warnings.warn(
            "allocate_budget: the plan was not settled on its breakpoints. The exact passes that return a cell on "
            f"its breakpoint run under SLSQP only, and minimize_kwargs asks for method={method!r}: the plan is the "
            "first solve's, on a map with its breakpoints blended, and a cell near a breakpoint may sit off it.",
            UserWarning,
            skip_file_prefixes=(_PKG_PREFIX,),
        )
        return SettleRun(first, self.max_passes).with_run_record(first, "skipped")

    def settle(self, first: OptimizeResult) -> OptimizeResult:
        """Settle ``first``, the solve with breakpoints blended, and return the plan with its status.

        Every exit records ``result["breakpoint_status"]``: ``"certified"`` when no entry on a breakpoint would
        gain by leaving it, ``"settled"`` when letting entries go stopped buying anything, ``"unsettled"`` when a
        pass failed, the check could not be solved, the passes ran out, a plan held on every knot breaks a constraint
        there, or no solve succeeded. An unsettled plan is the best successful pass, else ``first``, with a warning
        when it succeeded. ``nit``, ``nfev`` and ``njev`` are summed over every solve.

        Parameters
        ----------
        first : OptimizeResult
            The solve with every breakpoint inside the bounds blended.

        Returns
        -------
        OptimizeResult
            The settled plan's result, or the best one found, with ``breakpoint_status`` and the summed counts.
        """
        run = SettleRun(first, self.max_passes)
        plan = self._plan_from_solve(
            first, *self.variables.linear_piece_bounds(first.x, self.low, self.high)
        )
        if plan.spans_whole_box(self.low, self.high):
            # Nothing was blended and no entry is on a breakpoint: the first solve was on the exact map.
            return run.finish_run("certified" if first.success else "unsettled", plan)
        for index in range(1, self.max_passes + 1):
            step = self._run_pass_and_check(run, plan, index)
            if isinstance(step, OptimizeResult):
                return step
            plan = step
        return run.finish_run("unsettled")

    def _run_pass_and_check(
        self, run: SettleRun, plan: PiecewisePlan, index: int
    ) -> PiecewisePlan | OptimizeResult:
        """Run one pass and its check: return the plan the next pass solves, or the finished result.

        When nothing would leave but the pass left new arrivals, the next pass holds them with both bounds and
        checks again; on the last pass that hold runs here instead, one solve past ``max_passes``.
        """
        solved = self._solve_on_pieces(run, plan, index)
        if solved is None:
            return run.finish_run("unsettled")
        best = run.keep_if_best(solved)
        leaving = self._entries_that_would_leave(solved)
        if leaving is None:
            return run.finish_run("unsettled")
        up, down = leaving
        if not (up | down).any():
            if solved.all_arrivals_held:
                return run.finish_run("certified", solved)
            if index < self.max_passes:
                return solved
            return self._certify_held_last_pass(run, solved, index + 1)
        if run.progress_stalled(solved, self._progress_tolerance(solved.grad)):
            held = self._solve_until_arrivals_held(run, best, index + 1)
            return (
                run.finish_run("settled", held) if held else run.finish_run("unsettled")
            )
        return self._release_entries(solved, up, down)

    def _certify_held_last_pass(
        self, run: SettleRun, plan: PiecewisePlan, index: int
    ) -> OptimizeResult:
        """Hold the new arrivals of the last pass and certify the held plan if nothing would leave it."""
        held = self._solve_until_arrivals_held(run, plan, index)
        if held is None:
            return run.finish_run("unsettled")
        run.keep_if_best(held)
        leaving = self._entries_that_would_leave(held)
        if leaving is None or (leaving[0] | leaving[1]).any():
            return run.finish_run("unsettled")
        return run.finish_run("certified", held)

    def _solve_on_pieces(
        self, run: SettleRun, plan: PiecewisePlan, index: int
    ) -> PiecewisePlan | None:
        """Solve on the plan's pieces, or put every entry on its knot when all are held; ``None`` on failure."""
        if plan.held_on_knot.all():
            return self._snap_held_plan_to_knots(plan)
        start = np.clip(plan.x, plan.piece_low, plan.piece_high)
        with self.variables.conditioned_on(plan.piece_low, plan.piece_high):
            result = self.solve(start, plan.piece_bounds(), index)
        run.solves.append(result)
        if not result.success:
            return None
        return self._plan_from_solve(result, plan.piece_low, plan.piece_high)

    def _snap_held_plan_to_knots(self, plan: PiecewisePlan) -> PiecewisePlan | None:
        """Return the plan on its knots when every entry is held, or ``None`` if the knots are infeasible.

        Every entry is held, so no solve is left. SciPy answers bounds that fix every variable without running
        SLSQP, and accepts them only if every constraint holds to the last bit, so the knots are checked here within
        ``rtol``. When they fail, the run ends unsettled: a check on the knots would certify a point the result does
        not hold. The result keeps the last solve's ``multipliers`` and counts.
        """
        knots = plan.piece_low.copy()
        if not plan.result.success or not self._constraints_hold_at(knots):
            return None
        result = OptimizeResult({**plan.result, "x": knots})
        value, grad = self._exact_objective_and_gradient(result)
        return replace(
            plan,
            result=result,
            x=knots,
            new_arrivals=np.zeros(knots.size, dtype=bool),
            value=value,
            grad=grad,
        )

    def _solve_until_arrivals_held(
        self, run: SettleRun, plan: PiecewisePlan, index: int
    ) -> PiecewisePlan | None:
        """Hold the entries a plan's solve left at a knot, solving again until none is left.

        It may use one solve past ``max_passes``, so a run whose progress rule fires on the last pass, or whose last
        pass leaves new arrivals, still closes.
        """
        while not plan.all_arrivals_held and index <= self.max_passes + 1:
            following = self._solve_on_pieces(run, plan, index)
            if following is None:
                return None
            plan, index = following, index + 1
        return plan if plan.all_arrivals_held else None

    def _plan_from_solve(
        self, result: OptimizeResult, piece_low: np.ndarray, piece_high: np.ndarray
    ) -> PiecewisePlan:
        """Build the plan of a solve on the given pieces, pinning the entries it left at a breakpoint end."""
        x = np.asarray(result.x, dtype="float64")
        pin, knot = self._entries_at_breakpoint_ends(x, piece_low, piece_high)
        value, grad = self._exact_objective_and_gradient(result)
        return PiecewisePlan(
            result=result,
            x=np.where(pin, knot, x),
            piece_low=np.where(pin, knot, piece_low),
            piece_high=np.where(pin, knot, piece_high),
            new_arrivals=pin,
            value=value,
            grad=grad,
        )

    def _entries_at_breakpoint_ends(
        self, x: np.ndarray, piece_low: np.ndarray, piece_high: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return which entries sit at an end of their piece that is a breakpoint, and that end.

        The end is a breakpoint inside the entry's bounds, or one of its own bounds when that bound is a
        breakpoint; breakpoints are positive, so "at the end" is relative. The top end wins a tie.
        """
        is_breakpoint = self.variables.is_breakpoint
        open_piece = piece_low < piece_high
        top = (
            open_piece
            & ((piece_high < self.high) | is_breakpoint(piece_high))
            & (x >= piece_high * (1.0 - self.rtol))
        )
        bottom = (
            open_piece
            & ~top
            & ((piece_low > self.low) | is_breakpoint(piece_low))
            & (x <= piece_low * (1.0 + self.rtol))
        )
        return top | bottom, np.where(top, piece_high, piece_low)

    def _exact_objective_and_gradient(
        self, result: OptimizeResult
    ) -> tuple[float, np.ndarray]:
        """Return the objective and gradient at a result's plan on the exact map; ``inf`` for a failed solve."""
        if not result.success:
            return np.inf, np.zeros(np.size(result.x))
        value, grad = self.objective_and_grad(result.x)
        return float(value), np.asarray(grad, dtype="float64")

    def _progress_tolerance(self, grad: np.ndarray) -> float:
        """Return the objective change below which letting entries go bought nothing, at a plan's gradient."""
        return self.ftol * (
            objective_scale_from_gradient(grad) if self.rescale else 1.0
        )

    def _entries_that_would_leave(
        self, plan: PiecewisePlan
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Ask :func:`entries_leaving_breakpoints` which held entries would gain by leaving their breakpoint."""
        right, left = self._one_sided_gradients(plan.x)
        equalities, active = self._equality_and_active_jacobians(plan.x)
        return entries_leaving_breakpoints(
            right=right,
            left=left,
            x=plan.x,
            low=self.low,
            high=self.high,
            pinned=plan.held_on_knot & (self.low < self.high),
            equalities=equalities,
            active_inequalities=active,
            size=self.size,
            rtol=self.rtol,
        )

    def _release_entries(
        self, plan: PiecewisePlan, up: np.ndarray, down: np.ndarray
    ) -> PiecewisePlan:
        """Give the entries the check lets go the piece on the side that gains."""
        new_low, new_high = self.variables.linear_piece_bounds(
            plan.x, self.low, self.high, from_below=down
        )
        moved = up | down
        return replace(
            plan,
            piece_low=np.where(moved, new_low, plan.piece_low),
            piece_high=np.where(moved, new_high, plan.piece_high),
        )

    def _one_sided_gradients(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the objective's gradient at ``x`` with every entry on the piece above it, and below it.

        They differ only for an entry exactly on a breakpoint, whose slope changes there.
        """
        variables = self.variables
        with variables.conditioned_on(
            *variables.linear_piece_bounds(x, self.low, self.high)
        ):
            right = np.asarray(self.objective_and_grad(x)[1], dtype="float64")
        below = np.ones(x.size, dtype=bool)
        with variables.conditioned_on(
            *variables.linear_piece_bounds(x, self.low, self.high, from_below=below)
        ):
            left = np.asarray(self.objective_and_grad(x)[1], dtype="float64")
        return right, left

    def _equality_and_active_jacobians(
        self, x: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return the Jacobians of every equality, and of every inequality active at ``x``.

        An inequality is active when its value is below ``rtol`` of what moving every entry across its size would
        move it by. A ``Constraint`` compiles to a scalar, so each is one row.
        """
        equalities, active = [], []
        for constraint in self.constraints:
            jacobian = np.asarray(constraint["jac"](x), dtype="float64").reshape(-1)
            if constraint["type"] == "eq":
                equalities.append(jacobian)
            elif float(constraint["fun"](x)) <= self._constraint_tolerance(jacobian):
                active.append(jacobian)
        return equalities, active

    def _constraints_hold_at(self, x: np.ndarray) -> bool:
        """Return whether every constraint holds at ``x`` to within :meth:`_constraint_tolerance`."""
        for constraint in self.constraints:
            jacobian = np.asarray(constraint["jac"](x), dtype="float64").reshape(-1)
            value = np.asarray(constraint["fun"](x), dtype="float64")
            slack = self._constraint_tolerance(jacobian)
            violated = (
                np.abs(value) > slack if constraint["type"] == "eq" else value < -slack
            )
            if np.any(violated):
                return False
        return True

    def _constraint_tolerance(self, jacobian: np.ndarray) -> float:
        """Return ``rtol`` of what moving every entry across its size would move a constraint by."""
        return self.rtol * float(np.abs(jacobian) @ self.size)

    def _entry_sizes(self) -> np.ndarray:
        """Return each flat entry's size: ``total_budget`` for money, and for a lever its largest finite bound.

        A lever's bounds are in its own units, so the budget is no measure of how far it can move; its size is the
        largest absolute value among its finite bounds, at least 1, and 1 when it has none.
        """
        low, high = self.low, self.high
        size = np.maximum(
            1.0,
            np.maximum(
                np.where(np.isfinite(low), np.abs(low), 0.0),
                np.where(np.isfinite(high), np.abs(high), 0.0),
            ),
        )
        for variable in self.variables.spending_variables():
            size[self.variables.slices[variable.name]] = float(self.total_budget)
        return size
