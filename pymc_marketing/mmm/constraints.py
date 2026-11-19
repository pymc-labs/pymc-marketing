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

"""Constraints for the BudgetOptimizer."""

import math
import numbers
from collections.abc import Callable
from typing import Any, Literal

import pytensor.tensor as pt
from pytensor import function
from pytensor.graph import rewrite_graph
from pytensor.xtensor.type import XTensorVariable


class Constraint:
    """
    Represents a constraint for the BudgetOptimizer.

    SLSQP compares the summed violation of the constraints with its ``ftol``. With the default tolerance,
    :meth:`~pymc_marketing.mmm.budget_optimizer.BudgetOptimizer.allocate_budget` divides each constraint by a
    scale, so that a residual in large units, such as revenue, is held to a relative accuracy rather than to
    ``1e-9`` of its own units. With a ``ftol`` or ``tol`` of your own, every constraint reaches SLSQP as written.

    Parameters
    ----------
    key : str
        Identifier for the constraint.
    constraint_type : {"eq", "ineq"}
        ``"eq"`` for an equality, ``"ineq"`` for an inequality that holds when the residual is nonnegative.
    constraint_fun : Callable[[XTensorVariable, XTensorVariable, Any], XTensorVariable]
        Function that computes the symbolic scalar residual, taking ``budgets_sym``, ``total_budget_sym`` and
        ``optimizer``.
    scale : float, optional
        Positive divisor of the residual under the default tolerance. ``None`` (default) lets ``allocate_budget``
        take the largest change one entry makes in the residual moving across its size, at least 1. Pass ``1.0``
        to keep the residual's own units, as the default budget constraint does.

    Raises
    ------
    ValueError
        If ``scale`` is not a positive finite number.
    """

    def __init__(
        self,
        key: str,
        constraint_type: Literal["eq", "ineq"],
        constraint_fun: Callable[
            [XTensorVariable, XTensorVariable, Any], XTensorVariable
        ],
        scale: float | None = None,
    ):
        if scale is not None and (
            isinstance(scale, bool)
            or not isinstance(scale, numbers.Real)
            or not (math.isfinite(scale) and scale > 0)
        ):
            raise ValueError(
                f"Constraint {key!r}: scale must be a positive finite number, got {scale!r}."
            )
        self.key = key
        self.constraint_type = constraint_type
        self.constraint_fun = constraint_fun
        self.scale = None if scale is None else float(scale)


def build_default_sum_constraint(key: str = "default") -> Constraint:
    """Return a Constraint enforcing sum(budgets) == total_budget.

    Its scale is 1: the sum is linear in money and SLSQP meets it to rounding, so its tolerance stays ``ftol`` in
    money and the budgets add up to ``total_budget``.
    """

    def _constraint_fun(
        budgets_sym: XTensorVariable, total_budget_sym: XTensorVariable, optimizer
    ) -> XTensorVariable:
        # Total every variable that spends from the pot, not the media tensor
        # alone, so a second monetary variable shares the budget rather than
        # being optimized for free.
        contributions = optimizer.optimization_variables.budget_contributions()
        total = contributions[0].sum()
        for contribution in contributions[1:]:
            total = total + contribution.sum()
        return total - total_budget_sym

    return Constraint(
        key=key,
        constraint_type="eq",
        constraint_fun=_constraint_fun,
        scale=1.0,
    )


def compile_constraints_for_scipy(constraints: list[Constraint] | dict, optimizer):
    """Compile constraints for scipy."""
    compiled_constraints = []

    budgets = optimizer._budgets
    budgets_flat = optimizer._budgets_flat
    total_budget = optimizer._total_budget

    if isinstance(constraints, dict):
        constraints = list(constraints.values())

    if not constraints:
        raise ValueError("No constraints provided for compilation.")

    for constraint in constraints:
        if not isinstance(constraint, Constraint):
            raise TypeError(
                f"Expected an instance of Constraint, but received {type(constraint)}. "
                "Ensure all constraints are created using the Constraint class."
            )

        # Pass the required arguments to constraint_fun
        constraint_fun_output = constraint.constraint_fun(
            budgets, total_budget, optimizer
        )
        # Gradients not implement for XTensorVariables
        constraint_fun_output_tensor = rewrite_graph(
            constraint_fun_output.values,
            include=("lower_xtensor", "canonicalize", "stabilize"),
        )
        sym_jac_output = pt.grad(constraint_fun_output_tensor, budgets_flat)

        # Compile symbolic => python callables
        compiled_fun = function(
            inputs=[budgets_flat],
            outputs=constraint_fun_output,
            on_unused_input="ignore",
        )
        compiled_jac = function(
            inputs=[budgets_flat],
            outputs=sym_jac_output,
            on_unused_input="ignore",
        )

        compiled_constraints.append(
            {
                "key": constraint.key,
                "type": constraint.constraint_type,
                "fun": compiled_fun,
                "jac": compiled_jac,
                "scale": constraint.scale,
            }
        )
    return compiled_constraints
