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


def find_elbow(x: npt.NDArray[np.float_], y: npt.NDArray[np.float_]) -> int:
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
    distances = distance.cdist(
        np.column_stack((x, y)), np.column_stack((x, line_y)), "euclidean"
    )

    # Return the index of the point with the maximum distance to the line
    return int(np.argmax(distances.diagonal()))


class CurveCalculator:
    """
    A class used to calculate the quadratic curve, its derivative, roots, and y values for given x values.

    ...

    Attributes
    ----------
    coefficients : numpy.ndarray
        Coefficients of the quadratic curve.
    polynomial : numpy.poly1d
        The quadratic polynomial representing the curve.
    derivative : numpy.poly1d
        The derivative of the polynomial.
    roots : list of float
        The real roots of the derivative of the curve.
    x_space_actual : numpy.ndarray
        The x-values for the actual curve.
    y_space_actual : numpy.ndarray
        The y-values for the actual curve.
    x_space_projected : numpy.ndarray
        The x-values for the projected curve, including the roots.
    y_space_projected : numpy.ndarray
        The y-values for the projected curve, including the roots.
    """

    def __init__(self, x: npt.NDArray[np.float_], y: npt.NDArray[np.float_]):
        """Fit a quadratic curve, calculate its derivative and find its roots.

        Parameters
        ----------
        x : array-like
            The x-coordinates of the points.
        y : array-like
            The y-coordinates of the points.
        """
        # Fit a quadratic curve
        self.coefficients = np.polyfit(x, y, 2)
        self.polynomial = np.poly1d(self.coefficients)

        # Calculate derivative
        self.derivative = self.polynomial.deriv()

        # Find roots
        self.roots = self.derivative.r
        self.real_roots = [root.real for root in self.roots if root.imag == 0]

        # Define x spaces
        self.x_space_actual = np.linspace(x.min(), x.max(), 100)
        self.x_space_projected = np.linspace(
            min(x.min(), min(self.roots)), max(x.max(), max(self.roots)), 100
        )

        # Calculate y spaces
        self.y_space_actual = self.polynomial(self.x_space_actual)
        self.y_space_projected = self.polynomial(self.x_space_projected)
