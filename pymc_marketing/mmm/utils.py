from typing import List

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import curve_fit


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


def michaelis_menten(x, L, k) -> float:
    """
    Calculate the Michaelis-Menten function value.

    Parameters
    ----------
    x : float
        The spent on a channel.
    L : float
        The maximum contribution a channel can make (also known as the plateau point).
    k : float
        The elbow on the function in `x` (Point where the curve change their direction)

    Returns
    -------
    float
        The value of the Michaelis-Menten function given the parameters.
    """

    return L * x / (k + x)


def estimate_menten_parameters(
    channel: str,
    original_dataframe,
    contributions,
) -> List[float]:

    x = original_dataframe[channel].to_numpy()
    y = contributions.quantile(q=0.5).sel(channel=channel).to_numpy()

    # Initial guess for L and k
    initial_guess = [max(y), 0.001]
    # Curve fitting
    popt, pcov = curve_fit(michaelis_menten, x, y, p0=initial_guess)

    # Save the parameters
    return popt
