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
"""Settle a budget allocation on the breakpoints of a price schedule.

A schedule such as :class:`~pymc_marketing.mmm.price_response.TieredPriceResponse` changes slope at its
breakpoints, and a gradient solver steps across such a kink badly.
:meth:`~pymc_marketing.mmm.budget_optimizer.BudgetOptimizer.allocate_budget` therefore solves once with every
breakpoint inside the bounds blended, and uses that solve only to find the linear piece of the map each entry is
on. :class:`BreakpointSettler` then solves on those pieces, where no breakpoint is left to cross: an entry a pass
leaves at a breakpoint end of its piece is pinned there, and :func:`leaving_breakpoints` decides whether a pinned
entry would gain by leaving it.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

import numpy as np
from scipy.optimize import OptimizeResult, linprog

from pymc_marketing.mmm.optimization_variables import OptimizationVariables
from pymc_marketing.mmm.spend_reach import _VALIDATION_WARNING_SKIP

Bounds = list[tuple[float | None, float | None]]
BreakpointStatus = Literal["certified", "settled", "unsettled", "skipped"]


def leaving_breakpoints(
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

    Returns ``None`` when HiGHS reports anything but an optimum.
    """
    n = x.size
    # HiGHS's feasibility tolerances, and the optimum below which the program reports a gain.
    lp_tolerance = 1e-10
    # The program is homogeneous and a pinned entry moves at most 1, so a move above a millionth of that is a
    # direction, not HiGHS rounding a zero.
    released = 1e-6
    reach = rtol * size
    free_cap = np.where(pinned, 1.0, 4.0 * n)
    caps = np.concatenate(
        [
            np.where(x >= high - reach, 0.0, free_cap),
            np.where(x <= low + reach, 0.0, free_cap),
        ]
    )
    cost = np.concatenate(
        [right + tolerance * np.abs(right), -left + tolerance * np.abs(left)]
    )
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
    moves = (
        program.x > released
        if program.fun < -lp_tolerance
        else np.zeros(2 * n, dtype=bool)
    )
    return pinned & moves[:n], pinned & moves[n:]


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
class _Candidate:
    """A solve's plan, with the linear piece of the map each entry is on.

    ``x`` is the plan with every pinned entry on its knot; a pinned entry's piece is that knot at both ends.
    ``newly`` marks the entries this solve left at a knot without holding them there, which a closing solve holds
    with both bounds. ``value`` and ``grad`` are the objective's on the exact map; a failed solve has ``inf``
    and a zero gradient, and is never compared.
    """

    result: OptimizeResult
    x: np.ndarray
    piece_low: np.ndarray
    piece_high: np.ndarray
    newly: np.ndarray
    value: float
    grad: np.ndarray

    @property
    def closed(self) -> bool:
        return not self.newly.any()

    @property
    def held(self) -> np.ndarray:
        return self.piece_low == self.piece_high

    def bounds(self) -> Bounds:
        return [
            (
                float(lo) if np.isfinite(lo) else None,
                float(hi) if np.isfinite(hi) else None,
            )
            for lo, hi in zip(self.piece_low, self.piece_high, strict=True)
        ]


@dataclass
class BreakpointSettler:
    """Solve again on each entry's linear piece until no entry on a breakpoint would gain by leaving it.

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
    progress_tolerance : callable
        The absolute objective change, given the gradient at the plan, below which letting entries go bought
        nothing.
    max_passes : int
        Solves after the first, closing solves included. Default 20.
    rtol : float
        How close, relative, an entry is to the bound or breakpoint that holds it, and a constraint value to
        zero, when SLSQP leaves them; it leaves about ``8e-10``. Default ``1e-9``.
    """

    variables: OptimizationVariables
    objective_and_grad: Callable[[np.ndarray], tuple[Any, Any]]
    constraints: Sequence[dict]
    solve: Callable[[np.ndarray, Bounds, int], OptimizeResult]
    low: np.ndarray
    high: np.ndarray
    total_budget: float
    progress_tolerance: Callable[[np.ndarray], float]
    max_passes: int = 20
    rtol: float = 1e-9
    size: np.ndarray = field(init=False)
    _solves: list[OptimizeResult] = field(init=False, default_factory=list)
    _best: _Candidate | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Measure every entry's size once: closeness to a bound and constraint slack are relative to it."""
        self.size = self._entry_sizes()

    def settle(self, first: OptimizeResult) -> OptimizeResult:
        """Settle ``first``, the solve with breakpoints blended, and return the plan with its status.

        Every exit records ``result["breakpoint_status"]``: ``"certified"`` when no entry on a breakpoint would
        gain by leaving it, ``"settled"`` when letting entries go stopped buying anything, ``"unsettled"`` when a
        pass failed, the check could not be solved or the passes ran out. An unsettled plan is the best successful
        pass, else ``first``, with a warning when it succeeded. ``nit``, ``nfev`` and ``njev`` are summed over
        every solve.
        """
        self._solves, self._best = [first], None
        current = self._arrive(
            first, *self.variables.linear_pieces(first.x, self.low, self.high)
        )
        if not np.any(
            (current.piece_low > self.low) | (current.piece_high < self.high)
        ):
            # Nothing was blended and no entry is on a breakpoint: the first solve was on the exact map.
            return self._finish("certified" if first.success else "unsettled", current)
        previous = np.inf
        for index in range(1, self.max_passes + 1):
            following = self._next(current, index)
            if following is None:
                return self._finish("unsettled")
            current = following
            self._track(current)
            leaving = self._leaving(current)
            if leaving is None:
                return self._finish("unsettled")
            up, down = leaving
            if not (up | down).any():
                if current.closed:
                    return self._finish("certified", current)
                continue  # a closing pass holds the new arrivals with both bounds
            if current.value > previous - self.progress_tolerance(current.grad):
                closed = self._close(cast(_Candidate, self._best), index + 1)
                return (
                    self._finish("settled", closed)
                    if closed
                    else self._finish("unsettled")
                )
            previous = current.value
            current = self._released(current, up, down)
        return self._finish("unsettled")

    def _next(self, candidate: _Candidate, index: int) -> _Candidate | None:
        """Solve on the candidate's pieces, or put every entry on its knot when all are held; ``None`` on failure."""
        if candidate.held.all():
            return self._held(candidate)
        start = np.clip(candidate.x, candidate.piece_low, candidate.piece_high)
        with self.variables.conditioned(candidate.piece_low, candidate.piece_high):
            result = self.solve(start, candidate.bounds(), index)
        self._solves.append(result)
        if not result.success:
            return None
        return self._arrive(result, candidate.piece_low, candidate.piece_high)

    def _held(self, candidate: _Candidate) -> _Candidate | None:
        """Every entry is held, so no solve is left: return the plan on the knots when the constraints hold there.

        SciPy answers bounds that fix every variable without running SLSQP, and accepts them only if every
        constraint holds to the last bit, so the knots are checked here within ``rtol``. When they do not hold,
        the last solve's plan stands, within ``rtol`` of its knots.
        """
        if not candidate.result.success:
            return None
        knots = candidate.piece_low
        result = candidate.result
        if self._feasible(knots):
            result = OptimizeResult({**result, "x": knots.copy()})
        value, grad = self.objective_and_grad(result.x)
        return replace(
            candidate,
            result=result,
            x=knots.copy(),
            newly=np.zeros(knots.size, dtype=bool),
            value=float(value),
            grad=np.asarray(grad, dtype="float64"),
        )

    def _close(self, candidate: _Candidate, index: int) -> _Candidate | None:
        """Hold the entries a candidate's solve left at a knot, solving again until none is left."""
        while not candidate.closed and index <= self.max_passes:
            following = self._next(candidate, index)
            if following is None:
                return None
            candidate, index = following, index + 1
        return candidate if candidate.closed else None

    def _arrive(
        self, result: OptimizeResult, piece_low: np.ndarray, piece_high: np.ndarray
    ) -> _Candidate:
        """Pin the entries a solve left at an end of their piece that is a breakpoint, and put them on it.

        The end is a breakpoint inside the entry's bounds, or one of its own bounds when that bound is a
        breakpoint; breakpoints are positive, so "at the end" is relative. The move is for the check and the next
        solve; the plan a solve returned is not moved.
        """
        x = np.asarray(result.x, dtype="float64")
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
        pin = top | bottom
        knot = np.where(top, piece_high, piece_low)
        value, grad = (
            self.objective_and_grad(result.x)
            if result.success
            else (np.inf, np.zeros(x.size))
        )
        return _Candidate(
            result=result,
            x=np.where(pin, knot, x),
            piece_low=np.where(pin, knot, piece_low),
            piece_high=np.where(pin, knot, piece_high),
            newly=pin,
            value=float(value),
            grad=np.asarray(grad, dtype="float64"),
        )

    def _track(self, candidate: _Candidate) -> None:
        """Keep the successful pass with the lowest objective on the exact map."""
        if candidate.result.success and (
            self._best is None or candidate.value < self._best.value
        ):
            self._best = candidate

    def _leaving(self, candidate: _Candidate) -> tuple[np.ndarray, np.ndarray] | None:
        """Ask :func:`leaving_breakpoints` which held entries would gain by leaving their breakpoint."""
        right, left = self._one_sided_gradients(candidate.x)
        equalities, active = self._constraint_rows(candidate.x)
        return leaving_breakpoints(
            right=right,
            left=left,
            x=candidate.x,
            low=self.low,
            high=self.high,
            pinned=candidate.held & (self.low < self.high),
            equalities=equalities,
            active_inequalities=active,
            size=self.size,
            rtol=self.rtol,
        )

    def _released(
        self, candidate: _Candidate, up: np.ndarray, down: np.ndarray
    ) -> _Candidate:
        """Give the entries the check lets go the piece on the side that gains."""
        new_low, new_high = self.variables.linear_pieces(
            candidate.x, self.low, self.high, from_below=down
        )
        moved = up | down
        return replace(
            candidate,
            piece_low=np.where(moved, new_low, candidate.piece_low),
            piece_high=np.where(moved, new_high, candidate.piece_high),
        )

    def _finish(
        self, status: BreakpointStatus, candidate: _Candidate | None = None
    ) -> OptimizeResult:
        """Return the candidate's plan; without one, the best pass, else the first solve, warned if it succeeded."""
        chosen = candidate or self._best
        result = self._solves[0] if chosen is None else chosen.result
        if candidate is None and result.success:
            warnings.warn(
                "allocate_budget: the plan was not settled on its breakpoints: a pass failed, the check could "
                f"not be solved, or {self.max_passes} passes did not settle it. The best plan found comes back; "
                "a cell near a breakpoint may sit off it.",
                UserWarning,
                skip_file_prefixes=_VALIDATION_WARNING_SKIP,
            )
        if len(self._solves) > 1:
            for key in ("nit", "nfev", "njev"):
                result[key] = sum(int(solve.get(key, 0)) for solve in self._solves)
        result["breakpoint_status"] = status
        return result

    def _one_sided_gradients(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the objective's gradient at ``x`` with every entry on the piece above it, and below it.

        They differ only for an entry exactly on a breakpoint, whose slope changes there.
        """
        variables = self.variables
        with variables.conditioned(*variables.linear_pieces(x, self.low, self.high)):
            right = np.asarray(self.objective_and_grad(x)[1], dtype="float64")
        below = np.ones(x.size, dtype=bool)
        with variables.conditioned(
            *variables.linear_pieces(x, self.low, self.high, from_below=below)
        ):
            left = np.asarray(self.objective_and_grad(x)[1], dtype="float64")
        return right, left

    def _constraint_rows(
        self, x: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Jacobians of every equality, and of every inequality active at ``x``.

        An inequality is active when its value is below ``rtol`` of what moving every entry across its size would
        move it by. A ``Constraint`` compiles to a scalar, so each is one row.
        """
        equalities, active = [], []
        for constraint in self.constraints:
            jacobian = np.asarray(constraint["jac"](x), dtype="float64").reshape(-1)
            if constraint["type"] == "eq":
                equalities.append(jacobian)
            elif float(constraint["fun"](x)) <= self.rtol * float(
                np.abs(jacobian) @ self.size
            ):
                active.append(jacobian)
        return equalities, active

    def _feasible(self, x: np.ndarray) -> bool:
        """Whether every constraint holds at ``x`` to within ``rtol`` of what moving every entry would move it by."""
        for constraint in self.constraints:
            jacobian = np.asarray(constraint["jac"](x), dtype="float64").reshape(-1)
            slack = self.rtol * float(np.abs(jacobian) @ self.size)
            value = np.asarray(constraint["fun"](x), dtype="float64")
            violated = (
                np.abs(value) > slack if constraint["type"] == "eq" else value < -slack
            )
            if np.any(violated):
                return False
        return True

    def _entry_sizes(self) -> np.ndarray:
        """Each flat entry's size: ``total_budget`` for money, and for a lever its largest finite bound.

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
