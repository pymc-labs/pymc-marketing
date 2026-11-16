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
"""Validation rules for a tiered price schedule laid out as rows.

:class:`~pymc_marketing.mmm.price_response.TieredPriceResponse` checks its schedule as two ``nan``-padded arrays,
breakpoints and rate multipliers, with one row per label or cell and the brackets along the columns, once at
construction and again on the cells it resolves to. :class:`ScheduleRows` holds those rows, names them in
messages and refuses what is not a schedule. The module is private: nothing outside the price response reads it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pymc_marketing.names.mmm import Dim


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

    def refuse_invalid(
        self, rate_tolerance: float, reference_rate: float | None
    ) -> None:
        """Refuse what is not a schedule.

        Run at construction and again on the resolved cells: ``frozen`` stops a field being reassigned, but the
        declaration holds the caller's dict or ``DataArray``, which can still be mutated in place. Counts are
        checked where both arrays are given, on the rows that have multipliers.
        """
        self._refuse_inner_nan()
        self._refuse_bad_multipliers()
        self._refuse_bad_breakpoints()
        self._refuse_mismatched_counts()
        self._refuse_far_multipliers(rate_tolerance, reference_rate)

    def bracket_counts(self) -> np.ndarray:
        """Return the number of brackets per row."""
        return (~np.isnan(self.multipliers)).sum(axis=1)

    def _cells(self, bad: np.ndarray) -> str:
        rows = np.flatnonzero(bad)
        return (
            f"{[self.name(int(i)) for i in rows[:5]]}{', ...' if rows.size > 5 else ''}"
        )

    def _refuse_inner_nan(self) -> None:
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
                    f"{self.label}: {what} on {self._cells(gap)} has nan before a value along {Dim.BRACKET!r}. "
                    "nan pads a cell with fewer brackets, so nan may only trail."
                )

    def _refuse_bad_multipliers(self) -> None:
        r = self.multipliers
        bad = (~np.isnan(r) & ~(np.isfinite(r) & (r > 0.0))).any(axis=1)
        if bad.any():
            raise ValueError(
                f"{self.label}: rate_multipliers on {self._cells(bad)} must be positive and finite: each is the "
                "price of a unit in its bracket relative to the base price."
            )

    def _refuse_bad_breakpoints(self) -> None:
        b = self.breakpoints
        if b is None:
            return
        present = ~np.isnan(b)
        bad = (present & ~(np.isfinite(b) & (b >= 0.0))).any(axis=1)
        bad |= (present[:, 1:] & ~(b[:, 1:] > b[:, :-1])).any(axis=1)
        if bad.any():
            raise ValueError(
                f"{self.label}: breakpoints on {self._cells(bad)} must be non-negative, finite and strictly "
                "ascending: each is where a bracket ends, in budget units."
            )

    def _refuse_mismatched_counts(self) -> None:
        if self.breakpoints is None:
            return
        n_breakpoints = (~np.isnan(self.breakpoints)).sum(axis=1)
        n_multipliers = self.bracket_counts()
        wrong = (n_multipliers > 0) & (n_multipliers != n_breakpoints + 1)
        if wrong.any():
            first = int(np.flatnonzero(wrong)[0])
            raise ValueError(
                f"{self.label}: one multiplier per bracket: {self._cells(wrong)} have breakpoints and "
                f"multipliers that do not match ({n_breakpoints[first]} breakpoints and {n_multipliers[first]} "
                "multipliers on the first). len(rate_multipliers) must be len(breakpoints) + 1."
            )

    def _refuse_far_multipliers(
        self, rate_tolerance: float, reference_rate: float | None
    ) -> None:
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
            f"{self.label}: rate_multipliers on {self._cells(rows)} include {worst:g}, which is too far from 1 "
            f"to be a ratio of two rates (rate_tolerance={rate_tolerance:g}). Multipliers are ratios to the base "
            "price, cost_per_unit: an absolute rate goes in cost_per_unit, or is divided by the rate the base "
            f"stands for.{division} Raise rate_tolerance if the ratio is meant."
        )
