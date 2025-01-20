#   Copyright 2022 - 2025 The PyMC Labs Developers
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

from collections.abc import Callable
from typing import Any, Literal

import pytensor.tensor as pt
from pymc.pytensorf import rewrite_pregrad
from pytensor import function


class Constraint:
    """
    Represents a constraint for the BudgetOptimizer.

    Attributes
    ----------
        key (str): Identifier for the constraint.
        constraint_type (Literal["eq", "ineq"]): Type of the constraint ("eq" for equality, "ineq" for inequality).
        constraint_fun (Callable[[pt.TensorVariable, pt.TensorVariable, Any], pt.TensorVariable]):
            Function that computes the symbolic constraint, taking `budgets_sym`, `total_budget_sym`, and `optimizer`.
    """

    def __init__(
        self,
        key: str,
        constraint_type: Literal["eq", "ineq"],
        constraint_fun: Callable[
            [pt.TensorVariable, pt.TensorVariable, Any], pt.TensorVariable
        ],
    ):
        self.key = key
        self.constraint_type = constraint_type
        self.constraint_fun = constraint_fun


def build_default_sum_constraint(key: str = "default") -> Constraint:
    """Return a Constraint enforcing sum(budgets) == total_budget."""

    def _constraint_fun(
        budgets_sym: pt.TensorVariable, total_budget_sym: pt.TensorVariable, optimizer
    ) -> pt.TensorVariable:
        return pt.sum(budgets_sym) - total_budget_sym

    return Constraint(
        key=key,
        constraint_type="eq",
        constraint_fun=_constraint_fun,
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
        sym_jac_output = pt.grad(rewrite_pregrad(constraint_fun_output), budgets_flat)

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
                "type": constraint.constraint_type,
                "fun": compiled_fun,
                "jac": compiled_jac,
            }
        )
    return compiled_constraints
