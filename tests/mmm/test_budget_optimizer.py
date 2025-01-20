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
from unittest.mock import patch

import arviz as az
import numpy as np
import pandas as pd
import pytest

from pymc_marketing.mmm import MMM
from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer, MinimizeException
from pymc_marketing.mmm.components.adstock import GeometricAdstock
from pymc_marketing.mmm.components.saturation import LogisticSaturation


@pytest.fixture(scope="module")
def dummy_df():
    n = 10
    # Data is not needed for optimization of this model
    df = pd.DataFrame(
        data={
            "date_week": pd.date_range(start=pd.Timestamp.today(), periods=n, freq="W"),
            "channel_1": np.linspace(0, 1, num=n),
            "channel_2": np.linspace(0, 1, num=n),
            "event_1": np.concatenate([np.zeros(n - 1), [1]]),
            "event_2": np.concatenate([[1], np.zeros(n - 1)]),
            "t": range(n),
        }
    )

    y = np.ones(n)

    df_kwargs = {
        "date_column": "date_week",
        "channel_columns": ["channel_1", "channel_2"],
        "control_columns": ["event_1", "event_2", "t"],
    }

    return df_kwargs, df, y


@pytest.mark.parametrize(
    argnames="total_budget, budget_bounds, parameters, minimize_kwargs, expected_optimal, expected_response",
    argvalues=[
        (
            100,
            {"channel_1": (0, 50), "channel_2": (0, 50)},
            {
                "saturation_params": {
                    "lam": np.array(
                        [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]
                    ),  # dims: chain, draw, channel
                    "beta": np.array(
                        [[[0.5, 1.0], [0.5, 1.0]], [[0.5, 1.0], [0.5, 1.0]]]
                    ),  # dims: chain, draw, channel
                },
                "adstock_params": {
                    "alpha": np.array(
                        [[[0.5, 0.7], [0.5, 0.7]], [[0.5, 0.7], [0.5, 0.7]]]
                    )  # dims: chain, draw, channel
                },
            },
            None,
            {"channel_1": 50.0, "channel_2": 50.0},
            48.8,
        ),
        (
            100,
            {"channel_1": (0, 50), "channel_2": (0, 50)},
            {
                "saturation_params": {
                    "lam": np.array(
                        [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]
                    ),  # dims: chain, draw, channel
                    "beta": np.array(
                        [[[0.5, 1.0], [0.5, 1.0]], [[0.5, 1.0], [0.5, 1.0]]]
                    ),  # dims: chain, draw, channel
                },
                "adstock_params": {
                    "alpha": np.array(
                        [[[0.5, 0.7], [0.5, 0.7]], [[0.5, 0.7], [0.5, 0.7]]]
                    )  # dims: chain, draw, channel
                },
            },
            {
                "method": "SLSQP",
                "options": {"ftol": 1e-8, "maxiter": 1_002},
            },
            {"channel_1": 50.0, "channel_2": 50.0},
            48.8,
        ),
        # Zero budget case
        (
            0,
            {"channel_1": (0, 50), "channel_2": (0, 50)},
            {
                "saturation_params": {
                    "lam": np.array(
                        [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]
                    ),  # dims: chain, draw, channel
                    "beta": np.array(
                        [[[0.5, 1.0], [0.5, 1.0]], [[0.5, 1.0], [0.5, 1.0]]]
                    ),  # dims: chain, draw, channel
                },
                "adstock_params": {
                    "alpha": np.array(
                        [[[0.5, 0.7], [0.5, 0.7]], [[0.5, 0.7], [0.5, 0.7]]]
                    )  # dims: chain, draw, channel
                },
                "channels": ["channel_1", "channel_2"],
            },
            None,
            {"channel_1": 0.0, "channel_2": 7.94e-13},
            2.38e-10,
        ),
    ],
    ids=["default_minimizer_kwargs", "custom_minimizer_kwargs", "zero_total_budget"],
)
def test_allocate_budget(
    total_budget,
    budget_bounds,
    parameters,
    minimize_kwargs,
    expected_optimal,
    expected_response,
    dummy_df,
):
    df_kwargs, X_dummy, y_dummy = dummy_df

    mmm = MMM(
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        **df_kwargs,
    )

    mmm.build_model(X=X_dummy, y=y_dummy)

    # Only these parameters are needed for the optimizer
    mmm.idata = az.from_dict(
        posterior={
            "saturation_lam": parameters["saturation_params"]["lam"],
            "saturation_beta": parameters["saturation_params"]["beta"],
            "adstock_alpha": parameters["adstock_params"]["alpha"],
        }
    )

    # Create BudgetOptimizer Instance
    optimizer = BudgetOptimizer(
        model=mmm,
        num_periods=30,
    )

    # Allocate Budget
    match = "Using default equality constraint"
    with pytest.warns(UserWarning, match=match):
        optimal_budgets, optimization_res = optimizer.allocate_budget(
            total_budget=total_budget,
            budget_bounds=budget_bounds,
            minimize_kwargs=minimize_kwargs,
        )

    # Assert Results
    assert optimal_budgets.to_dataframe(name="_").to_dict()["_"] == pytest.approx(
        expected_optimal, abs=1e-12
    )
    assert -optimization_res.fun == pytest.approx(expected_response, abs=1e-2, rel=1e-2)


@patch("pymc_marketing.mmm.budget_optimizer.minimize")
def test_allocate_budget_custom_minimize_args(minimize_mock, dummy_df) -> None:
    df_kwargs, X_dummy, y_dummy = dummy_df

    mmm = MMM(
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        **df_kwargs,
    )
    mmm.build_model(X=X_dummy, y=y_dummy)
    mmm.idata = az.from_dict(
        posterior={
            "saturation_lam": [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]],
            "saturation_beta": [[[0.5, 1.0], [0.5, 1.0]], [[0.5, 1.0], [0.5, 1.0]]],
            "adstock_alpha": [[[0.5, 0.7], [0.5, 0.7]], [[0.5, 0.7], [0.5, 0.7]]],
        }
    )

    optimizer = BudgetOptimizer(
        model=mmm,
        num_periods=30,
    )

    total_budget = 100
    budget_bounds = {"channel_1": (0.0, 50.0), "channel_2": (0.0, 50.0)}
    minimize_kwargs = {
        "method": "SLSQP",
        "options": {"ftol": 1e-8, "maxiter": 1_002},
    }

    match = "Using default equality constraint"
    with pytest.warns(UserWarning, match=match):
        # Uninteresting exception raised when we try to convert the mocked result to a DataArray
        with pytest.raises(ValueError, match="conflicting sizes for dimension"):
            optimizer.allocate_budget(
                total_budget, budget_bounds, minimize_kwargs=minimize_kwargs
            )

    kwargs = minimize_mock.call_args_list[0].kwargs

    np.testing.assert_array_equal(x=kwargs["x0"], y=np.array([50.0, 50.0]))
    assert kwargs["bounds"] == [(0.0, 50.0), (0.0, 50.0)]
    # default constraint constraints = {"type": "eq", "fun": lambda x: np.sum(x) - total_budget}
    assert kwargs["constraints"]["type"] == "eq"
    assert (
        kwargs["constraints"]["fun"](np.array([total_budget / 2, total_budget / 2]))
        == 0.0
    )
    assert kwargs["constraints"]["fun"](np.array([100.0, 0.0])) == 0.0
    assert kwargs["constraints"]["fun"](np.array([0.0, 0.0])) == -total_budget
    assert kwargs["method"] == minimize_kwargs["method"]
    assert kwargs["options"] == minimize_kwargs["options"]


@pytest.mark.parametrize(
    "total_budget, budget_bounds, parameters, custom_constraints",
    [
        (
            100,
            {"channel_1": (0, 50), "channel_2": (0, 50)},
            {
                "saturation_params": {
                    "lam": np.array(
                        [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]
                    ),  # dims: chain, draw, channel
                    "beta": np.array(
                        [[[0.5, 1.0], [0.5, 1.0]], [[0.5, 1.0], [0.5, 1.0]]]
                    ),  # dims: chain, draw, channel
                },
                "adstock_params": {
                    "alpha": np.array(
                        [[[0.5, 0.7], [0.5, 0.7]], [[0.5, 0.7], [0.5, 0.7]]]
                    )  # dims: chain, draw, channel
                },
                "channels": ["channel_1", "channel_2"],
            },
            {
                "type": "ineq",
                "fun": lambda x: x[0] - 60,
            },  # channel_1 must be >= 60, which is infeasible
        ),
    ],
)
def test_allocate_budget_infeasible_constraints(
    total_budget, budget_bounds, parameters, custom_constraints, dummy_df
):
    df_kwargs, X_dummy, y_dummy = dummy_df

    mmm = MMM(
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        **df_kwargs,
    )

    mmm.build_model(X=X_dummy, y=y_dummy)

    # Only these parameters are needed for the optimizer
    mmm.idata = az.from_dict(
        posterior={
            "saturation_lam": parameters["saturation_params"]["lam"],
            "saturation_beta": parameters["saturation_params"]["beta"],
            "adstock_alpha": parameters["adstock_params"]["alpha"],
        }
    )

    optimizer = BudgetOptimizer(
        model=mmm,
        num_periods=30,
    )

    with pytest.raises(MinimizeException, match="Optimization failed"):
        optimizer.allocate_budget(total_budget, budget_bounds, custom_constraints)
