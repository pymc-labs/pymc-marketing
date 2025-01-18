#   Copyright 2025 The PyMC Labs Developers
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

import pytensor.tensor as pt
from pymc.pytensorf import rewrite_pregrad
from pytensor import function


def build_default_sum_constraint(key: str = "default"):
    """Return a constraint dict that enforces sum(budgets) == total_budget."""

    def _constraint_fun(budgets_sym, total_budget_sym, optimizer):
        return pt.sum(budgets_sym) - total_budget_sym

    return dict(
        key=key,
        constraint_type="eq",
        constraint_fun=_constraint_fun,
    )  # type: ignore


def compile_constraints_for_scipy(constraints, optimizer):
    """Compile constraints for scipy."""
    compiled_constraints = []

    budgets = optimizer._budgets
    budgets_flat = optimizer._budgets_flat
    total_budget = optimizer._total_budget
    for c in constraints.values() if isinstance(constraints, dict) else constraints:
        ctype = c["constraint_type"]
        sym_fun_output = c["constraint_fun"](budgets, total_budget, optimizer)
        sym_jac_output = pt.grad(rewrite_pregrad(sym_fun_output), budgets_flat)

        # Compile symbolic => python callables
        compiled_fun = function(
            inputs=[budgets_flat],
            outputs=sym_fun_output,
            on_unused_input="ignore",
        )
        compiled_jac = function(
            inputs=[budgets_flat],
            outputs=sym_jac_output,
            on_unused_input="ignore",
        )

        compiled_constraints.append(
            {
                "type": ctype,
                "fun": compiled_fun,
                "jac": compiled_jac,
            }
        )
    return compiled_constraints
