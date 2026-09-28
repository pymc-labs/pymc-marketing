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
r"""Synthetic markets with endogenous media spend.

:func:`simulate_endogenous_spend_market` generates weekly data for two
channels, TV and Digital, in which TV's budget is assigned by one of several
mechanisms. All scenarios share the same sales equation

.. math::

    y_t = 50 + g_{TV}(a_{TV,t}) + g_{D}(a_{D,t})
          + 2.5\, d_t + 1.5\, c_t + \varepsilon_t,

where :math:`a_{\cdot,t}` is normalised geometric adstock
(:math:`\alpha = 0.1`), :math:`g` is Michaelis-Menten saturation, :math:`c_t`
an annual calendar wave, and :math:`d_t` latent demand. Demand has a part
driven by recorded variables (the calendar, inflation and unemployment) and a
persistent AR(1) shock nobody records. What differs between scenarios is how
TV spend responds to that demand:

``"forecast"``
    Budgets follow an unrecorded forecast of demand, and a shared policy
    shock moves TV and Digital together. Spend is procyclical, so a plain MMM
    overstates TV.
``"target_chasing"``
    Budgets react to last week's sales gap against plan: the team spends more
    when it is missing target. Spend is countercyclical, so a plain MMM
    understates TV.
``"search"``
    Spend follows demand mechanically, as query volume times cost per click,
    up to a weekly cap. Endogeneity is contemporaneous, strong, and nonlinear
    once the cap binds.
``"observed_only"``
    Spend follows demand only through the recorded drivers. The backdoor path
    is closed by the MMM's controls, so a plain MMM is unbiased: a negative
    control for endogeneity diagnostics.

After ``n_history`` weeks a TV experiment runs for ``n_test`` weeks. For the
``"search"`` scenario it is a holdout that sets spend to zero; otherwise spend
is shifted by ``test_delta`` relative to what the budget rule would have
chosen. The untested path is simulated from the same random draws, so the
reported lift includes any feedback the test induces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "EndogenousSpendMarket",
    "simulate_endogenous_spend_market",
    "tv_marginal_return",
]

EndogenousSpendScenario = Literal[
    "forecast", "target_chasing", "search", "observed_only"
]

TV_BETA, TV_LAM = 38.0, 5.0
DIGITAL_BETA, DIGITAL_LAM = 22.0, 3.0
ADSTOCK_ALPHA, ADSTOCK_L_MAX = 0.1, 8


@dataclass(frozen=True)
class EndogenousSpendMarket:
    """A simulated market and its answer key.

    Attributes
    ----------
    scenario : str
        The spend-assignment mechanism used.
    data : pd.DataFrame
        Columns ``date``, ``tv``, ``digital``, ``inflation``, ``unemployment``
        (both standardised) and ``y``. Spend and sales are in the same
        currency units.
    design : pd.DataFrame
        The TV experiment in the format of
        :func:`~pymc_marketing.mmm.budget_model.lift_test_design`.
    lift_test : pd.DataFrame
        The same experiment summarised for
        :meth:`~pymc_marketing.mmm.MMM.add_lift_test_measurements`, with a
        noisy lift estimate.
    n_history : int
        Number of weeks before the experiment starts.
    operating_point : float
        Mean business-as-usual TV spend over the history.
    true_marginal_return : float
        TV's true marginal return at ``operating_point``.
    demand_shock : np.ndarray
        The unrecorded AR(1) demand shock.
    """

    scenario: str
    data: pd.DataFrame
    design: pd.DataFrame
    lift_test: pd.DataFrame
    n_history: int
    operating_point: float
    true_marginal_return: float
    demand_shock: np.ndarray


def _michaelis_menten(x: float, beta: float, lam: float) -> float:
    return beta * x / (lam + x)


def _adstock_at(path: np.ndarray, t: int, weights: np.ndarray) -> float:
    window = path[max(0, t - ADSTOCK_L_MAX + 1) : t + 1][::-1]
    return float(weights[: len(window)] @ window)


def tv_marginal_return(spend: float) -> float:
    """Compute the true marginal return of TV at a steady weekly spend level.

    Parameters
    ----------
    spend : float
        Weekly TV spend. Adstock is normalised, so steady spend equals
        steady adstocked spend.

    Returns
    -------
    float
        Derivative of TV's weekly contribution with respect to spend.
    """
    return TV_BETA * TV_LAM / (TV_LAM + spend) ** 2


def simulate_endogenous_spend_market(
    scenario: EndogenousSpendScenario = "forecast",
    *,
    random_seed: int | np.random.Generator | None = None,
    n_history: int = 104,
    n_test: int = 4,
    n_followup: int = 12,
    test_delta: float = -1.5,
    search_cap: float = 6.5,
    lift_sigma: float = 1.0,
) -> EndogenousSpendMarket:
    """Simulate a weekly two-channel market in which TV spend is endogenous.

    Parameters
    ----------
    scenario : {"forecast", "target_chasing", "search", "observed_only"}
        How TV budgets respond to demand; see the module docstring.
    random_seed : int or np.random.Generator, optional
        Seed or generator for reproducibility.
    n_history, n_test, n_followup : int
        Weeks before, during and after the TV experiment.
    test_delta : float, default -1.5
        Designed change in weekly TV spend for ``"shift"`` experiments.
    search_cap : float, default 6.5
        Weekly spend cap in the ``"search"`` scenario.
    lift_sigma : float, default 1.0
        Standard deviation of the noise added to the true lift, and the
        ``sigma`` reported in ``lift_test``.

    Returns
    -------
    EndogenousSpendMarket
        The observed data, the experiment in two formats, and the answer key.

    Examples
    --------
    .. code-block:: python

        from pymc_marketing.mmm import MMM, BudgetModelEffect
        from pymc_marketing.mmm.synthetic_data import simulate_endogenous_spend_market

        market = simulate_endogenous_spend_market("target_chasing", random_seed=1)
        X, y = market.data.drop(columns="y"), market.data["y"]
        mmm = MMM(...).add_mu_effect(BudgetModelEffect(design=market.design))
        mmm.fit(X, y)
    """
    valid = ("forecast", "target_chasing", "search", "observed_only")
    if scenario not in valid:
        raise ValueError(f"scenario must be one of {valid}, got {scenario!r}.")

    rng = np.random.default_rng(random_seed)
    n = n_history + n_test + n_followup
    t = np.arange(n)
    dates = pd.date_range("2022-01-03", periods=n, freq="W-MON")
    in_test = np.zeros(n, dtype=bool)
    in_test[n_history : n_history + n_test] = True

    calendar = np.sin(2 * np.pi * t / 52 - np.pi / 3)
    calendar -= calendar.mean()
    inflation = 3.0 + 0.015 * t + rng.normal(0, 0.20, n)
    unemployment = 5.0 + 1.2 * np.sin(2 * np.pi * t / 104) + rng.normal(0, 0.15, n)
    inflation_z = (inflation - inflation.mean()) / inflation.std()
    unemployment_z = (unemployment - unemployment.mean()) / unemployment.std()

    innovations = rng.normal(0, 1, n)
    shock = np.zeros(n)
    shock[0] = innovations[0]
    for i in range(1, n):
        shock[i] = 0.8 * shock[i - 1] + np.sqrt(1 - 0.8**2) * innovations[i]
    predictable = 3.0 * calendar + 2.0 * inflation_z - 2.0 * unemployment_z
    demand = predictable + shock

    forecast = 0.7 * demand + 0.5 * calendar + rng.normal(0, 0.4, n)
    policy = rng.normal(0, 0.2, n)
    if scenario == "forecast":
        digital = (
            3.0
            + 0.1 * forecast
            + 0.42 * policy
            + 0.10 * calendar
            + rng.normal(0, 0.25, n)
        )
    else:
        digital = 3.0 + 0.10 * calendar + rng.normal(0, 0.4, n)
    digital = np.clip(digital, 0.1, None)

    spend_noise = rng.normal(0, 1, n)
    cost_per_click = np.exp(rng.normal(0, 0.10, n))
    sales_noise = rng.normal(0, 1.5, n)
    weights = ADSTOCK_ALPHA ** np.arange(ADSTOCK_L_MAX)
    weights /= weights.sum()
    # The plan a target-chasing team compares last week's sales against.
    plan = 50.0 + 19.0 + 11.0 + 2.5 * predictable + 1.5 * calendar

    def run(with_test: bool) -> tuple[np.ndarray, np.ndarray]:
        tv = np.zeros(n)
        y = np.zeros(n)
        for i in range(n):
            if scenario == "forecast":
                chosen = (
                    5.0
                    + 0.5 * forecast[i]
                    + 0.6 * policy[i]
                    + 0.15 * calendar[i]
                    + 0.3 * spend_noise[i]
                )
            elif scenario == "target_chasing":
                gap = y[i - 1] - plan[i - 1] if i else 0.0
                chosen = 5.0 - 0.15 * gap + 0.15 * calendar[i] + 0.3 * spend_noise[i]
            elif scenario == "search":
                chosen = min(
                    search_cap, 5.0 * np.exp(0.12 * demand[i]) * cost_per_click[i]
                )
            else:
                chosen = (
                    5.0
                    + 0.25 * predictable[i]
                    + 0.15 * calendar[i]
                    + 0.3 * spend_noise[i]
                )
            chosen = max(chosen, 0.1)
            if with_test and in_test[i]:
                tv[i] = 0.0 if scenario == "search" else max(chosen + test_delta, 0.1)
            else:
                tv[i] = chosen
            y[i] = (
                50.0
                + _michaelis_menten(_adstock_at(tv, i, weights), TV_BETA, TV_LAM)
                + _michaelis_menten(
                    _adstock_at(digital, i, weights), DIGITAL_BETA, DIGITAL_LAM
                )
                + 2.5 * demand[i]
                + 1.5 * calendar[i]
                + sales_noise[i]
            )
        return tv, y

    tv, y = run(with_test=True)
    tv_untested, y_untested = run(with_test=False)

    data = pd.DataFrame(
        {
            "date": dates,
            "tv": tv,
            "digital": digital,
            "inflation": inflation_z,
            "unemployment": unemployment_z,
            "y": y,
        }
    )
    mode = "set" if scenario == "search" else "shift"
    design = pd.DataFrame(
        {
            "channel": ["tv"],
            "start_date": [dates[n_history]],
            "end_date": [dates[n_history + n_test - 1]],
            "mode": [mode],
            "delta_x": [np.nan if mode == "set" else test_delta],
        }
    )
    lift_test = pd.DataFrame(
        {
            "channel": ["tv"],
            "x": [float(tv_untested[in_test].mean())],
            "delta_x": [float((tv - tv_untested)[in_test].mean())],
            "delta_y": [
                float((y - y_untested)[in_test].mean() + rng.normal(0, lift_sigma))
            ],
            "sigma": [lift_sigma],
        }
    )
    operating_point = float(tv_untested[:n_history].mean())
    return EndogenousSpendMarket(
        scenario=scenario,
        data=data,
        design=design,
        lift_test=lift_test,
        n_history=n_history,
        operating_point=operating_point,
        true_marginal_return=tv_marginal_return(operating_point),
        demand_shock=shock,
    )
