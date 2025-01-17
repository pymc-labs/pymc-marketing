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

"""Constraints for marketing mix model budget optimization.

This module defines functions for building and compiling constraints,
for marketing mix model budget optimization in PyMC-Marketing.
"""

import pytensor.tensor as pt
from pymc.pytensorf import rewrite_pregrad
from pytensor import function
from pytensor.graph.basic import get_var_by_name


def auto_jacobian(
    constraint_fun,
):
    """Auto jacobian for scipy.

    Given a symbolic constraint function constraint_fun(budgets_sym, total_budget_sym, optimizer),
    return a symbolic jacobian function that depends on the same variables.
    """

    def _jac(budgets_sym, total_budget_sym, optimizer):
        _fun = constraint_fun(budgets_sym, total_budget_sym, optimizer)
        [budgets_flat] = get_var_by_name([budgets_sym], "budgets_flat")
        return pt.grad(rewrite_pregrad(_fun), budgets_flat)

    return _jac


def build_constraint(
    key: str,
    constraint_type: str,
    constraint_fun,
    constraint_jac=None,
):
    """Build a constraint for scipy.

    Return a dictionary of the form:
      {
          "key": key,
          "type": constraint_type,
          "sym_fun": ...,
          "sym_jac": ...,
      }
    `constraint_fun` is a python callable returning a PyTensor expression.
    `constraint_jac` is optional; if None, we derive it automatically.
    """
    if constraint_jac is None:
        constraint_jac = auto_jacobian(constraint_fun)

    return {
        "key": key,
        "type": constraint_type,
        "sym_fun": constraint_fun,
        "sym_jac": constraint_jac,
    }


def build_default_sum_constraint(key: str = "default"):
    """Return a constraint dict that enforces sum(budgets) == total_budget."""

    def _constraint_fun(budgets_sym, total_budget_sym, optimizer):
        return pt.sum(budgets_sym) - total_budget_sym

    return build_constraint(
        key=key,
        constraint_type="eq",
        constraint_fun=_constraint_fun,
        constraint_jac=None,
    )


def compile_constraints_for_scipy(constraints, optimizer):
    """Compile constraints for scipy."""
    compiled_constraints = []

    budgets = optimizer._create_budget_variable()
    [budgets_flat] = get_var_by_name([budgets], "budgets_flat")
    total_budget = pt.scalar("total_budget")

    for c in constraints.values() if isinstance(constraints, dict) else constraints:
        ctype = c["type"]
        sym_fun = c["sym_fun"]
        sym_jac = c["sym_jac"]

        # Compile symbolic => python callables
        compiled_fun = function(
            inputs=[budgets_flat, total_budget],
            outputs=sym_fun(budgets, total_budget, optimizer),
            on_unused_input="ignore",
        )
        compiled_jac = function(
            inputs=[budgets_flat, total_budget],
            outputs=sym_jac(budgets, total_budget, optimizer),
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
