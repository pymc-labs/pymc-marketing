
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
"""Risk assessment module."""
import numpy as np
from typing import Callable

ObjectiveFunction = Callable[[np.ndarray, np.ndarray], float]

def default_assessment(samples: np.ndarray, budgets: np.ndarray):
    """Default assessment function."""
    return np.mean(samples)

def tail_distance(confidence_level: float = 0.75) -> ObjectiveFunction:
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
    confidence_level : float, optional
        Confidence level for the quantiles (default is 0.75).

    Returns
    -------
    ObjectiveFunction
        A function that calculates the tail distance metric given samples and budgets.
    """
    def _tail_distance(samples: np.ndarray, budgets: np.ndarray) -> float:
        mean = np.mean(samples)
        q1 = np.quantile(samples, confidence_level)
        q2 = np.quantile(samples, 1 - confidence_level)

        return abs(q1 - mean) + abs(mean - q2)

    return _tail_distance

def mean_tightness_score(
    alpha: float = 0.5, confidence_level: float = 0.75
) -> ObjectiveFunction:
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

    Parameters
    ----------
    alpha : float, optional
        Risk tolerance parameter (default is 0.5).
    confidence_level : float, optional
        Confidence level for the quantiles (default is 0.75).

    Returns
    -------
    ObjectiveFunction
        A function that calculates the mean tightness score given samples and budgets.
    """
    def _mean_tightness_score(samples: np.ndarray, budgets: np.ndarray) -> float:
        mean = np.mean(samples)
        tail_metric = tail_distance(confidence_level)
        return mean - alpha * tail_metric(samples, budgets)

    return _mean_tightness_score

def _calculate_roas_distribution_for_allocation(
    samples: np.ndarray, budgets: np.ndarray
) -> np.ndarray:
    """Calculate the ROAS distribution for a given total budget."""
    return samples / np.sum(budgets)


def value_at_risk(confidence_level: float = 0.95) -> ObjectiveFunction:
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
    confidence_level : float, optional
        Confidence level for VaR (default is 0.95).

    Returns
    -------
    ObjectiveFunction
        A function that calculates the VaR value at the specified confidence level given samples and budgets.

    Raises
    ------
    ValueError
        If confidence_level is not between 0 and 1.

    References
    ----------
    - Jorion, P. (2006). Value at Risk: The New Benchmark for Managing Financial Risk.
    """
    def _value_at_risk(samples: np.ndarray, budgets: np.ndarray) -> float:
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1.")
        return np.percentile(samples, (1 - confidence_level) * 100)

    return _value_at_risk


def conditional_value_at_risk(
    confidence_level: float = 0.95
) -> ObjectiveFunction:
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
    confidence_level : float, optional
        Confidence level for CVaR (default is 0.95).

    Returns
    -------
    ObjectiveFunction
        A function that calculates the CVaR value at the specified confidence level given samples and budgets.

    Raises
    ------
    ValueError
        If confidence_level is not between 0 and 1.
        If no samples fall below the VaR threshold.

    References
    ----------
    - Rockafellar, R.T., & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk.
    """
    def _conditional_value_at_risk(samples: np.ndarray, budgets: np.ndarray) -> float:
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

    return _conditional_value_at_risk


def sharpe_ratio(risk_free_rate: float = 0.0) -> ObjectiveFunction:
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
    risk_free_rate : float, optional
        Risk-free rate of return (default is 0.0).

    Returns
    -------
    ObjectiveFunction
        A function that calculates the Sharpe Ratio given samples and budgets.

    Raises
    ------
    ValueError
        If the standard deviation of excess returns is zero.

    References
    ----------
    - Sharpe, W.F. (1966). Mutual Fund Performance.
    """
    def _sharpe_ratio(samples: np.ndarray, budgets: np.ndarray) -> float:
        excess_returns = samples - risk_free_rate
        mean_excess_return = np.mean(excess_returns)
        std_excess_return = np.std(excess_returns, ddof=1)
        if std_excess_return == 0:
            raise ValueError(
                "Standard deviation of excess returns is zero; Sharpe Ratio is undefined."
            )
        sharpe_ratio = mean_excess_return / std_excess_return
        return sharpe_ratio

    return _sharpe_ratio


def raroc(
    risk_free_rate: float = 0.0
) -> ObjectiveFunction:
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
    risk_free_rate : float, optional
        The rate of return on a risk-free investment, such as a government bond, used as
        a baseline for measuring excess returns (default is 0.0).

    Returns
    -------
    ObjectiveFunction
        A function that calculates the RAROC value given samples and budgets.

    Raises
    ------
    ValueError
        If `capital` is less than or equal to zero, as RAROC requires positive capital.

    References
    ----------
    - Matten, C. (2000). Managing Bank Capital: Capital Allocation and Performance Measurement.
    """
    def _raroc(samples: np.ndarray, budgets: np.ndarray) -> float:
        capital = np.sum(budgets)
        if capital <= 0:
            raise ValueError("Capital must be greater than zero.")

        expected_return = np.mean(samples)
        risk_adjusted_return = expected_return - risk_free_rate
        raroc_value = risk_adjusted_return / capital
        return raroc_value

    return _raroc


def adjusted_value_at_risk_score(
    confidence_level: float = 0.95, risk_aversion: float = 0.8
) -> ObjectiveFunction:
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
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.95 for 95% VaR).
    risk_aversion : float, optional
        Risk aversion parameter (0 = low risk aversion, 1 = high risk aversion).

    Returns
    -------
    ObjectiveFunction
        A function that calculates the adjusted Value at Risk score given samples and budgets.

    Raises
    ------
    ValueError
        If the risk aversion parameter is not between 0 and 1.
        If confidence_level is not between 0 and 1.
    """
    def _adjusted_value_at_risk_score(samples: np.ndarray, budgets: np.ndarray) -> float:
        if not 0 <= risk_aversion <= 1:
            raise ValueError("Risk aversion parameter must be between 0 and 1.")
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1.")
        # Calculate the empirical VaR directly from the samples
        var = np.percentile(samples, (1 - confidence_level) * 100)
        mean = np.mean(samples)
        return (1 - risk_aversion) * mean + risk_aversion * var

    return _adjusted_value_at_risk_score


def portfolio_entropy(samples: np.ndarray, budgets: np.ndarray) -> float:
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
    weights = budgets / np.sum(budgets)
    entropy = -np.sum(weights * np.log(weights))
    return entropy


def diversification_ratio(samples: np.ndarray, budgets: np.ndarray) -> float:
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
    weights = budgets / np.sum(budgets)
    individual_volatilities = np.std(samples, axis=0, ddof=1)
    portfolio_volatility = np.sqrt(
        weights @ np.cov(samples, rowvar=False) @ weights.T
    )
    weighted_avg_volatility = np.sum(weights * individual_volatilities)
    diversification_ratio = weighted_avg_volatility / portfolio_volatility
    return diversification_ratio