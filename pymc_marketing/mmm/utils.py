from typing import Any, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import curve_fit, minimize_scalar


def generate_fourier_modes(
    periods: npt.NDArray[np.float_], n_order: int
) -> pd.DataFrame:
    """Generate Fourier modes.

    Parameters
    ----------
    periods : array-like of float
        Input array denoting the period range.
    n_order : int
        Maximum order of Fourier modes.

    Returns
    -------
    pd.DataFrame
        Fourier modes (sin and cos with different frequencies) as columns in a dataframe.

    References
    ----------
    See :ref:`examples:Air_passengers-Prophet_with_Bayesian_workflow` in PyMC examples collection.
    """
    if n_order < 1:
        raise ValueError("n_order must be greater than or equal to 1")
    return pd.DataFrame(
        {
            f"{func}_order_{order}": getattr(np, func)(2 * np.pi * periods * order)
            for order in range(1, n_order + 1)
            for func in ("sin", "cos")
        }
    )


def michaelis_menten(x, alpha, lam) -> float:
    """
    Calculate the Michaelis-Menten function value.

    Parameters
    ----------
    x : float
        The spent on a channel.
    alpha (Limit/Vmax) : float
        The maximum contribution a channel can make (also known as the plateau point).
    lam (k) : float
        The elbow on the function in `x` (Point where the curve change their direction)

    Returns
    -------
    float
        The value of the Michaelis-Menten function given the parameters.
    """

    return alpha * x / (lam + x)


def extense_sigmoid(x, alpha, lam) -> float:
    """
    Parameters
    ----------
    - alpha
        α (alpha): Represent the Asymptotic Maximum or Ceiling Value.
    - lam
        λ (lambda): affects how quickly the function approaches its upper and lower asymptotes. A higher value of
        lam makes the curve steeper, while a lower value makes it more gradual.
    """
    return (alpha - alpha * np.exp(-lam * x)) / (1 + np.exp(-lam * x))


def estimate_menten_parameters(
    channel: str,
    original_dataframe,
    contributions,
) -> List[float]:
    """
    Estimate the parameters for the michaelis-menten function.

    This function uses the scipy.optimize.curve_fit method to estimate the parameters
    of the extended sigmoid function. The parameters are estimated by minimizing the
    least squares error between the observed data and the values predicted by the
    extended sigmoid function.

    Parameters
    ----------
    x : array-like
        The input data for which the parameters are to be estimated.
    y : array-like
        The observed data for which the parameters are to be estimated.

    Returns
    -------
    List[float]
        The estimated parameters of the extended sigmoid function.
    """
    x = original_dataframe[channel].to_numpy()
    y = contributions.sel(channel=channel).to_numpy()

    # Initial guess for L and k
    initial_guess = [max(y), 0.001]
    # Curve fitting
    popt, pcov = curve_fit(michaelis_menten, x, y, p0=initial_guess)

    # Save the parameters
    return popt


def estimate_sigmoid_parameters(
    channel: str, original_dataframe, contributions, **kwargs
) -> List[float]:
    """
    Estimate the parameters for the extended sigmoid function.

    This function uses the scipy.optimize.curve_fit method to estimate the parameters
    of the extended sigmoid function. The parameters are estimated by minimizing the
    least squares error between the observed data and the values predicted by the
    extended sigmoid function.

    Parameters
    ----------
    x : array-like
        The input data for which the parameters are to be estimated.
    y : array-like
        The observed data for which the parameters are to be estimated.

    Returns
    -------
    List[float]
        The estimated parameters of the extended sigmoid function.
    """
    x = original_dataframe[channel].to_numpy()
    y = contributions.sel(channel=channel).to_numpy()

    alpha_initial_estimate = 3 * max(y)

    parameter_bounds_modified = ([0, 0], [alpha_initial_estimate, np.inf])
    popt, _ = curve_fit(
        extense_sigmoid,
        x,
        y,
        p0=[alpha_initial_estimate, 0.001],
        bounds=parameter_bounds_modified,
    )

    return popt


def compute_sigmoid_second_derivative(x, alpha, lam) -> float:
    """
    Compute the second derivative of the extended sigmoid function.

    The second derivative of a function gives us information about the curvature of the function.
    In the context of the sigmoid function, it helps us identify the inflection point, which is
    the point where the function changes from being concave up to concave down, or vice versa.

    Parameters
    ----------
    x : float
        The input value for which the second derivative is to be computed.
    alpha : float
        The asymptotic maximum or ceiling value of the sigmoid function.
    lam : float
        The parameter that affects how quickly the function approaches its upper and lower asymptotes.

    Returns
    -------
    float
        The second derivative of the sigmoid function at the input value.
    """
    # Compute the second derivative
    return (
        -alpha
        * lam**2
        * np.exp(-lam * x)
        * (1 - np.exp(-lam * x) - 2 * lam * x * np.exp(-lam * x))
        / (1 + np.exp(-lam * x)) ** 3
    )


def find_sigmoid_inflection_point(alpha, lam) -> Tuple[Any, float]:
    """
    Find the inflection point of the extended sigmoid function.

    The inflection point of a function is the point where the function changes its curvature,
    i.e., it changes from being concave up to concave down, or vice versa. For the sigmoid
    function, this is the point where the function has its maximum rate of growth.

    Parameters
    ----------
    alpha : float
        The asymptotic maximum or ceiling value of the sigmoid function.
    lam : float
        The parameter that affects how quickly the function approaches its upper and lower asymptotes.

    Returns
    -------
    tuple
        The x and y coordinates of the inflection point.
    """

    # Minimize the negative of the absolute value of the second derivative
    result = minimize_scalar(
        lambda x: -abs(compute_sigmoid_second_derivative(x, alpha, lam))
    )

    # Evaluate the original function at the inflection point
    x_inflection = result.x
    y_inflection = extense_sigmoid(x_inflection, alpha, lam)

    return x_inflection, y_inflection
