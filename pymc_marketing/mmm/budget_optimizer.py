#   Copyright 2024 The PyMC Labs Developers
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
"""Budget optimization module."""

import inspect
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import minimize

from pymc_marketing.mmm.components.adstock import AdstockTransformation
from pymc_marketing.mmm.components.saturation import SaturationTransformation


class MinimizeException(Exception):
    """Custom exception for optimization failure."""

    def __init__(self, message: str):
        super().__init__(message)


class BudgetOptimizer(BaseModel):
    """A class for optimizing budget allocation in a marketing mix model.

    The goal of this optimization is to maximize the total expected response
    by allocating the given budget across different marketing channels. The
    optimization is performed using the Sequential Least Squares Quadratic
    Programming (SLSQP) method, which is a gradient-based optimization algorithm
    suitable for solving constrained optimization problems.

    For more information on the SLSQP algorithm, refer to the documentation:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    Parameters
    ----------
    adstock : AdstockTransformation
        The adstock class.
    saturation : SaturationTransformation
        The saturation class.
    num_periods : int
        The number of time units.
    parameters : dict
        A dictionary of parameters for each channel.
    adstock_first : bool, optional
        Whether to apply adstock transformation first or saturation transformation first.
        Default is True.
    objective_function : Callable[[np.ndarray], float], optional
        The objective function to maximize. Default is the mean of the response distribution.
    objective_function_kwargs : dict, optional
        Additional keyword arguments for the objective function. Default is an empty dictionary.

    """

    adstock: AdstockTransformation = Field(
        ..., description="The adstock transformation class."
    )
    saturation: SaturationTransformation = Field(
        ..., description="The saturation transformation class."
    )
    num_periods: int = Field(
        ...,
        gt=0,
        description="The number of time units at time granularity which the budget is to be allocated.",
    )
    parameters: dict[str, Any] = Field(
        ..., description="A dictionary of parameters for each channel."
    )
    scales: np.ndarray = Field(
        ..., description="The scale parameter for each channel variable"
    )
    adstock_first: bool = Field(
        True,
        description="Whether to apply adstock transformation first or saturation transformation first.",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)
    objective_function: Callable[[np.ndarray], float] = Field(
        default=np.mean,
        description="Objective function to maximize.",
    )
    objective_function_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments for the objective function.",
    )

    def _estimate_response(self, budgets: list[float]) -> np.ndarray:
        """Calculate the total response during a period of time given the budgets.

        It considers the saturation and adstock transformations.

        Parameters
        ----------
        budgets : list[float]
            The budgets for each channel.

        Returns
        -------
        float
            The negative total response value.

        """
        first_transform, second_transform = (
            (self.adstock, self.saturation)
            if self.adstock_first
            else (self.saturation, self.adstock)
        )

        budget = budgets / self.scales  # dim: (channels)

        first_params = (
            self.parameters["adstock_params"]
            if self.adstock_first
            else self.parameters["saturation_params"]
        )
        second_params = (
            self.parameters["saturation_params"]
            if self.adstock_first
            else self.parameters["adstock_params"]
        )

        spend = np.tile(budget, (self.num_periods, 1))  # dim: (periods, channels)
        spend_extended = np.vstack(
            [spend, np.zeros((self.adstock.l_max, spend.shape[1]))]
        )  # dim: (periods + l_max, channels)

        _response = first_transform.function(x=spend_extended, **first_params)

        # Dynamically reshape parameters in `second_params` to match `_response`
        for param_name, param_value in second_params.items():
            if param_value.ndim == 3:  # Check if it lacks the `time` dimension
                # Reshape by adding a singleton dimension for broadcasting over time
                second_params[param_name] = param_value.reshape(
                    param_value.shape[0], param_value.shape[1], 1, param_value.shape[2]
                )

        # Apply the second transformation with the reshaped `second_params`
        response = second_transform.function(x=_response, **second_params).eval()
        return response.sum(axis=(2, 3)).flatten()  # dim: (chain * draw)

    def objective(self, budgets):
        """Objective function for the budget optimization."""
        response_distribution = self._estimate_response(budgets=budgets)

        # Inspect the signature of the objective function
        sig = inspect.signature(self.objective_function)
        params = sig.parameters

        # Prepare arguments
        function_kwargs = self.objective_function_kwargs.copy()
        if "assets" in params:
            function_kwargs["assets"] = budgets

        if "capital" in params:
            function_kwargs["capital"] = np.sum(budgets)

        return -self.objective_function(response_distribution, **function_kwargs)

    def allocate_budget(
        self,
        total_budget: float,
        budget_bounds: dict[str, tuple[float, float]] | None = None,
        custom_constraints: dict[Any, Any] | None = None,
        minimize_kwargs: dict[str, Any] | None = None,
    ) -> tuple[dict[str, float], float]:
        """Allocate the budget based on the total budget, budget bounds, and custom constraints.

        The default budget bounds are (0, total_budget) for each channel.

        The default constraint is the sum of all budgets should be equal to the total budget.

        The optimization is done using the Sequential Least Squares Quadratic Programming (SLSQP) method
        and it's constrained such that:
        1. The sum of budgets across all channels equals the total available budget.
        2. The budget allocated to each individual channel lies within its specified range.

        The purpose is to maximize the total expected objective based on the inequality
        and equality constraints.

        Parameters
        ----------
        total_budget : float
            The total budget.
        budget_bounds : dict[str, tuple[float, float]], optional
            The budget bounds for each channel. Default is None.
        custom_constraints : dict, optional
            Custom constraints for the optimization. Default is None.
        minimize_kwargs : dict, optional
            Additional keyword arguments for the `scipy.optimize.minimize` function. If None, default values are used.
            Method is set to "SLSQP", ftol is set to 1e-9, and maxiter is set to 1_000.

        Returns
        -------
        tuple[dict[str, float], float]
            The optimal budgets for each channel and the negative total response value.

        Raises
        ------
        Exception
            If the optimization fails, an exception is raised with the reason for the failure.

        """
        if budget_bounds is None:
            budget_bounds = {
                channel: (0, total_budget) for channel in self.parameters["channels"]
            }
            warnings.warn(
                "No budget bounds provided. Using default bounds (0, total_budget) for each channel.",
                stacklevel=2,
            )
        elif not isinstance(budget_bounds, dict):
            raise TypeError("`budget_bounds` should be a dictionary.")

        if custom_constraints is None:
            constraints = {"type": "eq", "fun": lambda x: np.sum(x) - total_budget}
            warnings.warn(
                "Using default equality constraint: The sum of all budgets should be equal to the total budget.",
                stacklevel=2,
            )
        elif not isinstance(custom_constraints, dict):
            raise TypeError("`custom_constraints` should be a dictionary.")
        else:
            constraints = custom_constraints

        num_channels = len(self.parameters["channels"])
        initial_guess = np.ones(num_channels) * total_budget / num_channels
        bounds = [
            (
                (budget_bounds[channel][0], budget_bounds[channel][1])
                if channel in budget_bounds
                else (0, total_budget)
            )
            for channel in self.parameters["channels"]
        ]

        if minimize_kwargs is None:
            minimize_kwargs = {
                "method": "SLSQP",
                "options": {"ftol": 1e-9, "maxiter": 1_000},
            }

        result = minimize(
            fun=self.objective,
            x0=initial_guess,
            bounds=bounds,
            constraints=constraints,
            **minimize_kwargs,
        )

        if result.success:
            optimal_budgets = {
                name: budget
                for name, budget in zip(
                    self.parameters["channels"], result.x, strict=False
                )
            }
            return optimal_budgets, -result.fun
        else:
            raise MinimizeException(f"Optimization failed: {result.message}")


class RiskAssessment:
    """A collection of static methods for assessing risk."""

    @staticmethod
    def tail_distance(samples: np.ndarray, confidence_level: float = 0.75) -> float:
        R"""Calculate the absolute distance between the mean and the quantiles.

        It is a simple and interpretable metric that can be used to assess the risk.

        The tail distance is calculated as:

            .. math::
                Tail\\ Distance = |Q_{(1 - \\alpha)} - \\mu| + |\\mu - Q_{\\alpha}|

        where:
            - :math:`\\mu` is the mean of the sample returns.
            - :math:`Q_{(1 - \\alpha)}` is the quantile at the specified confidence level.
            - :math:`Q_{\\alpha}` is the quantile at the specified confidence level.

        Parameters
        ----------
        samples : np.ndarray
            Array of sample returns or losses.
        confidence_level : float, optional
            Confidence level for the quantiles (default is 0.75).

        Returns
        -------
        float
            The tail distance metric.
        """
        mean = np.mean(samples)
        q1 = np.quantile(samples, confidence_level)
        q2 = np.quantile(samples, 1 - confidence_level)

        return abs(q1 - mean) + abs(mean - q2)

    @staticmethod
    def mean_tightness_score(
        samples: np.ndarray, alpha: float = 0.5, confidence_level: float = 0.75
    ) -> float:
        R"""
        Calculate the mean tightness score.

        The mean tightness score is a risk metric that balances the mean return and the tail variability.
        It is calculated as:

        .. math::
            Mean\ Tightness\ Score = \mu - \alpha \cdot Tail\ Distance

        where:
            - :math:`\mu` is the mean of the sample returns.
            - :math:`Tail\ Distance` is the tail distance metric.
            - :math:`\alpha` is the risk tolerance parameter.

        alpha (Risk Tolerance Parameter): This parameter controls the trade-off.
            - Higher :math:`\alpha` increases sensitivity to variability, making the metric value higher for spread dist
            - Lower :math:`\alpha` decreases sensitivity to variability, making the metric value lower for spread dist

        """
        mean = np.mean(samples)
        tail_metric = RiskAssessment.tail_distance(samples, confidence_level)
        return mean - alpha * tail_metric

    @staticmethod
    def calculate_roas_distribution_for_allocation(
        samples: np.ndarray, budget: float
    ) -> np.ndarray:
        """Calculate the ROAS distribution for a given total budget."""
        return samples / np.sum(budget)

    @staticmethod
    def value_at_risk(samples: np.ndarray, confidence_level: float = 0.95) -> float:
        R"""
        Calculate the Value at Risk (VaR) at a specified confidence level.

        VaR estimates the potential loss in value of an asset or portfolio over a defined period
        for a given confidence interval. It is a standard measure used in risk management to
        assess the risk of loss on a specific portfolio of financial assets.

        The Value at Risk (VaR) is calculated as:

            .. math::
                VaR = \mu - Q_{(1 - \alpha)}

        where:
            - :math:`\mu` is the mean of the sample returns.
            - :math:`Q_{(1 - \alpha)}` is the quantile at the specified confidence level.

        Parameters
        ----------
        samples : np.ndarray
            Array of sample returns or losses.
        confidence_level : float, optional
            Confidence level for VaR (default is 0.95).

        Returns
        -------
        float
            The VaR value at the specified confidence level.

        Raises
        ------
        ValueError
            If confidence_level is not between 0 and 1.

        References
        ----------
        - Jorion, P. (2006). Value at Risk: The New Benchmark for Managing Financial Risk.
        """
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1.")

        return np.percentile(samples, (1 - confidence_level) * 100)

    @staticmethod
    def conditional_value_at_risk(
        samples: np.ndarray, confidence_level: float = 0.95
    ) -> float:
        R"""
        Calculate the Conditional Value at Risk (CVaR) at a specified confidence level.

        CVaR, also known as Expected Shortfall, measures the average loss exceeding the VaR
        at a given confidence level, providing insight into the tail risk of the distribution.

        The Conditional Value at Risk (CVaR) is calculated as:

            .. math::
                CVaR = \mathbb{E}[X \mid X \leq VaR]

        where :math:`X` represents the loss distribution, and :math:`VaR` is the Value at Risk
        at the specified confidence level. CVaR provides a more comprehensive view of the risk
        associated with extreme losses beyond the VaR.

        Parameters
        ----------
        samples : np.ndarray
            Array of sample returns or losses.
        confidence_level : float, optional
            Confidence level for CVaR (default is 0.95).

        Returns
        -------
        float
            The CVaR value at the specified confidence level.

        Raises
        ------
        ValueError
            If confidence_level is not between 0 and 1.
        ValueError
            If no samples fall below the VaR threshold.

        References
        ----------
        - Rockafellar, R.T., & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk.
        """
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1.")
        VaR = np.percentile(samples, (1 - confidence_level) * 100)
        tail_losses = samples[samples <= VaR]
        if len(tail_losses) == 0:
            raise ValueError(
                "No samples fall below the VaR threshold; CVaR is undefined."
            )
        CVaR = tail_losses.mean()
        return CVaR

    @staticmethod
    def sharpe_ratio(samples: np.ndarray, risk_free_rate: float = 0.0) -> float:
        R"""
        Calculate the Sharpe Ratio.

        The Sharpe Ratio assesses the risk-adjusted return of an investment by comparing
        the excess return over the risk-free rate to the standard deviation of returns.

        The Sharpe Ratio is calculated as:

            .. math::
                Sharpe\ Ratio = \frac{\mathbb{E}[R - R_f]}{\sigma}

        where:
            - :math:`\mathbb{E}[R - R_f]` is the mean of excess returns.
            - :math:`\sigma` is the standard deviation of the excess returns.

        Parameters
        ----------
        samples : np.ndarray
            Array of sample returns.
        risk_free_rate : float, optional
            Risk-free rate of return (default is 0.0).

        Returns
        -------
        float
            The Sharpe Ratio.

        Raises
        ------
        ValueError
            If the standard deviation of excess returns is zero.

        References
        ----------
        - Sharpe, W.F. (1966). Mutual Fund Performance.
        """
        excess_returns = samples - risk_free_rate
        mean_excess_return = np.mean(excess_returns)
        std_excess_return = np.std(excess_returns, ddof=1)
        if std_excess_return == 0:
            raise ValueError(
                "Standard deviation of excess returns is zero; Sharpe Ratio is undefined."
            )
        sharpe_ratio = mean_excess_return / std_excess_return
        return sharpe_ratio

    @staticmethod
    def raroc(
        samples: np.ndarray, capital: float, risk_free_rate: float = 0.0
    ) -> float:
        R"""
        Calculate the Risk-Adjusted Return on Capital (RAROC).

        RAROC measures the efficiency of capital utilization by assessing the return
        generated above a risk-free benchmark, normalized by the capital at risk.
        This metric provides insight into the value created by taking on additional risk,
        relative to a safe investment.

        The Risk-Adjusted Return on Capital (RAROC) is calculated as:

            .. math::
                RAROC = \frac{\mathbb{E}[R] - R_f}{C}

        where:
            - :math:`\mathbb{E}[R]` is the expected return (mean of samples).
            - :math:`R_f` is the risk-free rate.
            - :math:`C` is the capital at risk.

        Parameters
        ----------
        samples : np.ndarray
            Array of sample returns representing the distribution of possible outcomes
            for an investment. Typically generated through simulations or historical data.
        capital : float
            The amount of capital at risk in the investment. RAROC normalizes the return by
            this value to show how efficiently the capital is utilized.
        risk_free_rate : float, optional
            The rate of return on a risk-free investment, such as a government bond, used as
            a baseline for measuring excess returns (default is 0.0).

        Returns
        -------
        float
            The RAROC value, calculated as the ratio of risk-adjusted return to capital.

        Raises
        ------
        ValueError
            If `capital` is less than or equal to zero, as RAROC requires positive capital.

        References
        ----------
        - Matten, C. (2000). Managing Bank Capital: Capital Allocation and Performance Measurement.
        """
        if capital <= 0:
            raise ValueError("Capital must be greater than zero.")

        expected_return = np.mean(samples)
        risk_adjusted_return = expected_return - risk_free_rate
        raroc_value = risk_adjusted_return / capital

        return raroc_value

    @staticmethod
    def adjusted_value_at_risk_score(
        samples: np.ndarray, confidence_level: float = 0.95, risk_aversion: float = 0.8
    ) -> float:
        R"""
        Calculate adjusted Value at Risk (AVaR) score.

        The adjusted Value at Risk (AVaR) score is a risk-adjusted metric that combines the
        mean and Value at Risk (VaR) based on a risk aversion parameter. It provides a single
        metric that accounts for both return and risk preferences.

        The score is calculated as:

            .. math::
                AVaR\ Score = (1 - \alpha) \cdot \mu + \alpha \cdot VaR

        where:
            - :math:`\mu` is the mean of the sample returns.
            - :math:`VaR` is the Value at Risk at the specified confidence level.
            - :math:`\alpha` is the risk aversion parameter.

        Parameters
        ----------
        samples : np.ndarray
            Observed data from the distribution.
        confidence_level : float
            Confidence level for VaR (e.g., 0.95 for 95% VaR).
        risk_aversion : float
            Risk aversion parameter (0 = low risk aversion, 1 = high risk aversion).

        Returns
        -------
        score : float
            Risk-adjusted score for the distribution.

        Raises
        ------
        ValueError
            If the risk aversion parameter is not between 0 and 1.
            If confidence_level is not between 0 and 1.
        """
        if not 0 <= risk_aversion <= 1:
            raise ValueError("Risk aversion parameter must be between 0 and 1.")
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1.")
        # Calculate the empirical VaR directly from the samples
        var = np.percentile(samples, (1 - confidence_level) * 100)

        mean = np.mean(samples)
        return (1 - risk_aversion) * mean + risk_aversion * var

    @staticmethod
    def portfolio_entropy(assets: np.ndarray) -> float:
        R"""
        Calculate the entropy of a portfolio's asset weights to assess diversification.

        Portfolio entropy, derived from Shannon entropy in information theory, quantifies
        the dispersion of asset weights within a portfolio. A higher entropy value indicates
        a more diversified portfolio, as investments are more evenly distributed across assets.
        Conversely, a lower entropy suggests concentration in fewer assets, implying higher risk.

        The entropy is calculated using the formula:

        .. math::
            E = -\sum_{i=1}^{n} w_i \cdot \log(w_i)

        where :math:`w_i` represents the weight of asset \( i \) in the portfolio.

        Parameters
        ----------
        assets : np.ndarray
            1D array representing the investment amounts in each asset.

        Returns
        -------
        float
            Portfolio entropy value.

        References
        ----------
        - Bera, A. K., & Park, S. Y. (2008). Optimal Portfolio Diversification using the Maximum Entropy Principle.
        - Pola, G. (2013). On entropy and portfolio diversification. *Journal of Asset Management*, 14(4), 228-238.
        """
        weights = assets / np.sum(assets)
        entropy = -np.sum(weights * np.log(weights))
        return entropy

    @staticmethod
    def diversification_ratio(samples: np.ndarray, assets: np.ndarray) -> float:
        R"""
        Calculate the Diversification Ratio of a portfolio to evaluate risk distribution.

        The Diversification Ratio measures the effectiveness of diversification by comparing
        the weighted average volatility of individual assets to the overall portfolio volatility.
        A higher ratio indicates better diversification, as it reflects lower correlations among
        assets, leading to reduced portfolio risk.

        The Diversification Ratio is calculated as:

        .. math::
            DR = \frac{\\sum_{i=1}^{n} w_i \\cdot \\sigma_i}{\\sigma_p}

        where:
            - :math:`w_i` is the weight of asset \\( i \\)
            - :math:`\\sigma_i` is the volatility (standard deviation) of asset \\( i \\)
            - :math:`\\sigma_p` is the volatility of the portfolio

        Parameters
        ----------
        samples : np.ndarray
            2D array where each column represents the returns of an asset.
        assets : np.ndarray
            1D array representing the investment amounts in each asset.

        Returns
        -------
        float
            Diversification Ratio.

        This ratio provides insight into how individual asset volatilities and their correlations
        contribute to the overall portfolio risk.

        References
        ----------
        - Choueifaty, Y., & Coignard, Y. (2008). Toward Maximum Diversification. *Journal of Portfolio Management*.
        - Meucci, A. (2009). Managing Diversification. *Risk*, 22(5), 74-79.
        """
        weights = assets / np.sum(assets)
        individual_volatilities = np.std(samples, axis=0, ddof=1)
        portfolio_volatility = np.sqrt(
            weights @ np.cov(samples, rowvar=False) @ weights.T
        )
        weighted_avg_volatility = np.sum(weights * individual_volatilities)
        diversification_ratio = weighted_avg_volatility / portfolio_volatility
        return diversification_ratio
