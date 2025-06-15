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
r"""Bass diffusion model for product adoption.

Adapted from Wiki: https://en.wikipedia.org/wiki/Bass_diffusion_model

The Bass diffusion model, developed by Frank Bass in 1969, is a mathematical model that describes
the process of how new products get adopted in a population over time. It is widely used in
marketing, forecasting, and innovation studies to predict the adoption rates of new products
and technologies.

Mathematical Formulation
-----------------------
The model is based on a differential equation that describes the rate of adoption:

.. math::

    \frac{f(t)}{1-F(t)} = p + q F(t)

Where:

- :math:`F(t)` is the installed base fraction (cumulative proportion of adopters)
- :math:`f(t)` is the rate of change of the installed base fraction (:math:`f(t) = F'(t)`)
- :math:`p` is the coefficient of innovation or external influence
- :math:`q` is the coefficient of imitation or internal influence

The solution to this equation gives the adoption curve:

.. math::

    F(t) = \frac{1 - e^{-(p+q)t}}{1 + (\frac{q}{p})e^{-(p+q)t}}

The adoption rate at time t is given by:

.. math::

    f(t) = (p + q F(t))(1 - F(t))

Key Parameters
-------------
The model has three main parameters:

- :math:`m`: Market potential (total number of eventual adopters)
- :math:`p`: Coefficient of innovation (external influence) - typically 0.01-0.03
- :math:`q`: Coefficient of imitation (internal influence) - typically 0.3-0.5

Parameter Interpretation
-----------------------
- A higher :math:`p` value indicates stronger external influence (advertising, marketing)
- A higher :math:`q` value indicates stronger internal influence (word-of-mouth, social interactions)
- The ratio :math:`q/p` indicates the relative strength of internal vs. external influences
- The peak of adoption occurs at time :math:`t^* = \frac{\ln(q/p)}{p+q}`

Applications
-----------
The Bass model has been applied to forecast the adoption of various products and technologies:

- Consumer durables (TVs, refrigerators)
- Technology products (smartphones, software)
- Pharmaceutical products
- Entertainment products
- Services and subscriptions

This implementation provides a Bayesian version of the Bass model using PyMC, allowing for:
- Uncertainty quantification through prior distributions
- Hierarchical modeling for multiple products/markets
- Extension to incorporate additional factors

Examples
--------
Create a basic Bass model for multiple products:

.. plot::
    :context: close-figs

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pymc as pm

    from pymc_marketing.bass.model import create_bass_model
    from pymc_marketing.plot import plot_curve
    from pymc_marketing.prior import Prior

    # Create time points - 3 years of monthly data
    n_dates = 12 * 3
    dates = pd.date_range(start="2020-01-01", periods=n_dates, freq="MS")
    t = np.arange(n_dates)

    # Define coordinates for multiple products
    coords = {"T": t, "product": ["A", "B", "C"]}

    # Define priors
    priors = {
        "m": Prior("DiracDelta", c=10_000),  # Market potential
        "p": Prior("Beta", alpha=13.85, beta=692.43, dims="product"),  # Innovation coefficient
        "q": Prior("Beta", alpha=36.2, beta=54.4),  # Imitation coefficient
        "likelihood": Prior("Poisson", dims=("T", "product")),
    }

    # Create the Bass model
    model = create_bass_model(t, observed=None, priors=priors, coords=coords)

    # Sample from the prior predictive distribution
    with model:
        idata = pm.sample_prior_predictive()

    # Plot the adoption curves
    fig, axes = plt.subplots(1, 3, figsize=(10, 6))
    idata.prior["y"].pipe(plot_curve, "T", axes=axes)
    plt.suptitle("Bass Model Prior Predictive Adoption Curves")
    plt.tight_layout()
    plt.show()

"""

from typing import Any

import pymc as pm
import pytensor.tensor as pt
from pymc.model import Model

from pymc_marketing.prior import Censored, Prior, VariableFactory, create_dim_handler


def F(
    p: float | pt.TensorVariable,
    q: float | pt.TensorVariable,
    t: float | pt.TensorVariable,
) -> pt.TensorVariable:
    r"""Installed base fraction (cumulative adoption proportion).

    This function calculates the cumulative proportion of adopters at time t,
    representing the fraction of the potential market that has adopted the product.

    Parameters
    ----------
    p : float or TensorVariable
        Coefficient of innovation (external influence)
    q : float or TensorVariable
        Coefficient of imitation (internal influence)
    t : array-like or TensorVariable
        Time points

    Returns
    -------
    TensorVariable
        The cumulative proportion of adopters at each time point

    Notes
    -----
    This is the solution to the Bass differential equation:

    .. math::

        F(t) = \frac{1 - e^{-(p+q)t}}{1 + (\frac{q}{p})e^{-(p+q)t}}

    When :math:`t=0`, :math:`F(t)=0`, and as :math:`t` approaches infinity, :math:`F(t)` approaches 1.
    """
    return (1 - pt.exp(-(p + q) * t)) / (1 + (q / p) * pt.exp(-(p + q) * t))


def f(
    p: float | pt.TensorVariable,
    q: float | pt.TensorVariable,
    t: float | pt.TensorVariable,
) -> pt.TensorVariable:
    r"""Installed base fraction rate of change (adoption rate).

    This function calculates the rate of new adoptions at time t as a
    proportion of the potential market. It represents the probability density
    function of adoption time.

    Parameters
    ----------
    p : float or TensorVariable
        Coefficient of innovation (external influence)
    q : float or TensorVariable
        Coefficient of imitation (internal influence)
    t : array-like or TensorVariable
        Time points

    Returns
    -------
    TensorVariable
        The adoption rate at each time point as a fraction of potential market

    Notes
    -----
    This is the derivative of F(t) with respect to time:

    .. math::

        f(t) = \frac{(p+q)^2 \cdot e^{-(p+q)t}}{p \cdot (1+\frac{q}{p}e^{-(p+q)t})^2}

    Alternatively:

    .. math::

        f(t) = (p + q \cdot F(t)) \cdot (1 - F(t))

    The peak adoption rate occurs at time :math:`t^* = \frac{\ln(q/p)}{p+q}`
    """
    return (p * pt.square(p + q) * pt.exp(t * (p + q))) / pt.square(
        p * pt.exp(t * (p + q)) + q
    )


def create_bass_model(
    t: pt.TensorLike,
    observed: pt.TensorLike | None,
    priors: dict[str, Prior | Censored | VariableFactory],
    coords: dict[str, Any],
) -> Model:
    r"""Define a Bass diffusion model for product adoption forecasting.

    This function creates a Bayesian Bass diffusion model using PyMC to forecast
    product adoption over time. The Bass model captures both innovation (external
    influence like advertising) and imitation (internal influence like word-of-mouth)
    effects in the adoption process.

    The model includes the following components:

    - Market potential 'm': Total number of eventual adopters
    - Innovation coefficient 'p': Measures external influence
    - Imitation coefficient 'q': Measures internal influence
    - Adopters over time: Number of new adopters at each time point
    - Innovators: Adopters influenced by external factors
    - Imitators: Adopters influenced by previous adopters
    - Peak adoption time: When adoption rate reaches maximum

    Parameters
    ----------
    t : pt.TensorLike
        Time points for which the adoption is modeled.
    observed : pt.TensorLike | None
        Observed adoption data at each time point. If None, only
        prior predictive sampling is possible.
    priors : dict[str, Prior | Censored | VariableFactory]
        Dictionary containing priors for:
        - 'm': Market potential prior
        - 'p': Innovation coefficient prior
        - 'q': Imitation coefficient prior
        - 'likelihood': Observation likelihood model
    coords : dict[str, Any]
        Coordinate values for dimensions in the model, including
        'date' for the time dimension and any other dimensions
        included in the prior specifications.

    Returns
    -------
    Model
        A PyMC model object for the Bass diffusion model, containing
        the variables m, p, q, adopters, innovators, imitators, peak,
        and the likelihood y.

    Notes
    -----
    The returned model can be used for prior predictive checks, posterior
    sampling, and posterior predictive checks to forecast product adoption.

    The model implements the following mathematical relationships:

    .. math::

        \text{adopters}(t) &= m \cdot f(p, q, t) \\
        \text{innovators}(t) &= m \cdot p \cdot (1 - F(p, q, t)) \\
        \text{imitators}(t) &= m \cdot q \cdot F(p, q, t) \cdot (1 - F(p, q, t)) \\
        \text{peak} &= \frac{\ln(q) - \ln(p)}{p + q}
    """
    with pm.Model(coords=coords) as model:
        parameter_dims = (
            set(priors["p"].dims).union(priors["q"].dims).union(priors["m"].dims)
        )
        likelihood_dims = set(getattr(priors["likelihood"], "dims", ()) or ())

        combined_dims = (
            "T",
            *tuple(parameter_dims.union(likelihood_dims).difference(["T"])),
        )
        dim_handler = create_dim_handler(combined_dims)

        m = dim_handler(priors["m"].create_variable("m"), priors["m"].dims)
        p = dim_handler(priors["p"].create_variable("p"), priors["p"].dims)
        q = dim_handler(priors["q"].create_variable("q"), priors["q"].dims)

        time = dim_handler(t, "T")

        adopters = pm.Deterministic("adopters", m * f(p, q, time), dims=combined_dims)

        pm.Deterministic(
            "innovators",
            m * p * (1 - F(p, q, time)),
            dims=combined_dims,
        )
        pm.Deterministic(
            "imitators",
            m * q * F(p, q, time) * (1 - F(p, q, time)),
            dims=combined_dims,
        )

        peak = (pt.log(q) - pt.log(p)) / (p + q)
        peak_dims = tuple(parameter_dims) if parameter_dims else None
        pm.Deterministic("peak", peak, dims=peak_dims)

        priors["likelihood"].dims = combined_dims
        priors["likelihood"].create_likelihood_variable(  # type: ignore
            "y",
            mu=adopters,
            observed=observed,
        )

    return model
