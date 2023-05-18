import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial import distance


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

def find_elbow(x, y):
    """
    Finds the elbow point in a curve by measuring the distance between the points and a line connecting the first and last points.

    Args:
        x: array-like
            The x-coordinates of the points on the curve.
        y: array-like
            The y-coordinates of the points on the curve.

    Returns:
        index: int
            The index of the point representing the elbow in the curve.

    Notes:
        The function calculates the coefficients of the line connecting the first and last points using polynomial fitting.
        It then calculates the y-values of the line and measures the distances from the points of the curve to the line using the Euclidean distance.
        The index of the point with the maximum distance to the line is returned as the elbow point.
    """
    # Calculate the coefficients of the line connecting the first and last points
    line_coeffs = np.polyfit([x[0], x[-1]], [y[0], y[-1]], 1)

    # Calculate the y-values of the line
    line_y = np.poly1d(line_coeffs)(x)

    # Calculate the distances from the points of the curve to the line
    distances = distance.cdist(np.column_stack((x, y)), np.column_stack((x, line_y)), 'euclidean')

    # Return the index of the point with the maximum distance to the line
    return np.argmax(distances.diagonal())