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

def find_elbow(x: np.array, y: np.array) -> int:
    """
    Finds the elbow point in a curve by measuring the distance between the points and a line connecting the first and last points.

    Parameters
    ----------
    x : array-like
        The x-coordinates of the points on the curve.
    y : array-like
        The y-coordinates of the points on the curve.

    Returns
    -------
    index : int
        The index of the point representing the elbow in the curve.

    Notes
    -----
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

def calculate_curve(x: np.array, y: np.array) -> tuple[np.poly1d, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Calculate the quadratic curve, its derivative, roots, and y values for given x values.

    Parameters
    ----------
    x : array-like
        The x-coordinates of the points.
    y : array-like
        The y-coordinates of the points.

    Returns
    -------
    polynomial : numpy.poly1d
        The quadratic polynomial representing the curve.
    x_space_actual : numpy.ndarray
        The x-values for the actual curve.
    y_space_actual : numpy.ndarray
        The y-values for the actual curve.
    x_space_projected : numpy.ndarray
        The x-values for the projected curve, including the roots.
    y_space_projected : numpy.ndarray
        The y-values for the projected curve, including the roots.
    roots : list of float
        The real roots of the derivative of the curve.

    Notes
    -----
    This function fits a quadratic curve to the given points using the numpy.polyfit function.
    It calculates the derivative of the curve using the numpy.poly1d.deriv method.
    The function finds the real roots of the derivative using numpy.poly1d.r.
    It defines the x-values for the actual curve using numpy.linspace with the minimum and maximum x values.
    The x-values for the projected curve are defined using the minimum and maximum x values, including the roots.
    The y-values for both curves are calculated using the polynomial function.

    """

    # Fit a quadratic curve
    coefficients = np.polyfit(x, y, 2)
    polynomial = np.poly1d(coefficients)

    # Calculate derivative
    derivative = polynomial.deriv()

    # Find roots
    roots = derivative.r
    roots = [root.real for root in roots if root.imag == 0]

    # Define x spaces
    x_space_actual = np.linspace(x.min(), x.max(), 100)
    x_space_projected = np.linspace(min(x.min(), min(roots)), max(x.max(), max(roots)), 100)

    # Calculate y spaces
    y_space_actual = polynomial(x_space_actual)
    y_space_projected = polynomial(x_space_projected)

    return polynomial, x_space_actual, y_space_actual, x_space_projected, y_space_projected, roots
