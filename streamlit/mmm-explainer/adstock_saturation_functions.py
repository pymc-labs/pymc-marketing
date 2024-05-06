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
import numpy as np
from scipy.stats import weibull_min
from sklearn.preprocessing import MinMaxScaler

# ------------------- SATURATION CURVE FUNCTIONS ------------------------


# Hill function
def threshold_hill_saturation(x, alpha, gamma, threshold=None):
    """
    Compute the value of a Hill function with a threshold for activation.
    The threshold is added for visualisation purposes,
    it makes the graphs display a better S-shape.

    Parameters:
        x (float or array-like): Input variable(s).
        alpha (float): Controls the shape of the curve.
        gamma (float): Controls inflection point of saturation curve.
        threshold (float): Minimum amount of spend before response starts.

    Returns:
        float or array-like: Values of the modified Hill function for the given inputs.
    """
    if threshold:
        # Apply threshold condition
        y = np.where(x > threshold, (x**alpha) / ((x**alpha) + (gamma**alpha)), 0)
    else:
        y = (x**alpha) / ((x**alpha) + (gamma**alpha))
    return y


# Root function
def root_saturation(x, alpha):
    """
    Compute the value of a root function.
    The root function raises the input variable to a power specified by the alpha parameter.

    Parameters:
        x (float or array-like): Input variable(s).
        alpha (float): Exponent controlling the root function.

    Returns:
        float or array-like: Values of the root function for the given inputs.
    """
    return x**alpha


# Logistic function
def logistic_saturation(x, lam):
    """
    Compute the value of a logistic function for saturation.

    Parameters:
        x (float or array-like): Input variable(s).
        lam (float): Growth rate or steepness of the curve.

    Returns:
        float or array-like: Values of the modified logistic function for the given inputs.
    """
    return (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))


# Custom tanh saturation
def tanh_saturation(x, b=0.5, c=0.5):
    """
    Tanh saturation transformation.
    Credit to PyMC-Marketing: https://github.com/pymc-labs/pymc-marketing/blob/main/pymc_marketing/mmm/transformers.py

    Parameters:
        x (array-like): Input variable(s).
        b (float): Scales the output. Must be non-negative.
        c (float): Affects the steepness of the curve. Must be non-zero.

    Returns:
        array-like: Transformed values using the tanh saturation formula.
    """
    return b * np.tanh(x / (b * c))


# Michaelis-Menten saturation
def michaelis_menten_saturation(x, alpha, lam):
    """
    Evaluate the Michaelis-Menten function for given values of x, alpha, and lambda.

    Parameters:
    ----------
    x : float or np.ndarray
        The spending on a channel.
    alpha : float or np.ndarray
        The maximum contribution a channel can make (Limit/Vmax).
    lam : float or np.ndarray
        The point on the function in `x` where the curve changes direction (elbow/k).

    Returns:
    -------
    float or np.ndarray
        The value of the Michaelis-Menten function given the parameters.
    """
    return alpha * x / (lam + x)


# ------------------- ADSTOCK TRANSFORMATION FUNCTIONS ------------------------


def geometric_adstock_decay(impact, decay_factor, periods):
    """
    Calculate the geometric adstock effect.

    Parameters:
        impact (float): Initial advertising impact.
        decay_factor (float): Decay factor between 0 and 1.
        periods (int): Number of periods.

    Returns:
        list: List of adstock values for each period.
    """
    adstock_values = [impact]

    for _ in range(1, periods):
        impact *= decay_factor
        adstock_values.append(impact)

    return adstock_values


def delayed_geometric_decay(impact, decay_factor, theta, L):
    """
    Calculate the geometric adstock effect with a delayed peak and a specified maximum lag length.

    Parameters:
        impact (float): Peak advertising impact.
        decay_factor (float): Decay factor between 0 and 1, applied throughout.
        theta (int): Period at which peak impact occurs.
        L (int): Maximum lag length for adstock effect.

    Returns:
        np.array: Array of adstock values for each lag up to L.
    """
    adstock_values = np.zeros(L)

    # Calculate adstock values
    for lag in range(L):
        if lag < theta:
            # Before peak, apply decay to grow towards peak
            adstock_values[lag] = impact * (decay_factor ** abs(lag - theta))
        else:
            # After peak, apply decay normally
            adstock_values[lag] = impact * (decay_factor ** abs(lag - theta))

    return adstock_values


def weibull_adstock_decay(
    impact, shape, scale, periods, adstock_type="cdf", normalised=True
):
    """
    Calculate the Weibull PDF or CDF adstock decay for media mix modeling.

    Parameters:
        impact (float): Initial advertising impact.
        shape (float): Shape parameter of the Weibull distribution.
        scale (float): Scale parameter of the Weibull distribution.
        periods (int): Number of periods.
        adstock_type (str): Type of adstock ('cdf' or 'pdf').
        normalise (bool): If True, normalises decay values between 0 and 1,
                        otherwise leaves unnormalised.

    Returns:
        list: List of adstock-decayed values for each period.
    """
    # Create an array of time periods
    x_bin = np.arange(1, periods + 1)

    # Transform the scale parameter according to percentile of time period
    transformed_scale = round(np.quantile(x_bin, scale))

    # Handle the case when shape or scale is 0
    if shape == 0 or scale == 0:
        theta_vec_cum = np.zeros(periods)
    else:
        if adstock_type.lower() == "cdf":
            # Calculate the Weibull adstock decay using CDF
            theta_vec = np.concatenate(
                ([1], 1 - weibull_min.cdf(x_bin[:-1], shape, scale=transformed_scale))
            )
            theta_vec_cum = np.cumprod(theta_vec)
        elif adstock_type.lower() == "pdf":
            # Calculate the Weibull adstock decay using PDF
            theta_vec_cum = weibull_min.pdf(x_bin, shape, scale=transformed_scale)
            theta_vec_cum /= np.sum(theta_vec_cum)

    # Return adstock decay values, normalized or not
    if normalised:
        # Normalize the values between 0 and 1 using Min-Max scaling
        norm_theta_vec_cum = (
            MinMaxScaler().fit_transform(theta_vec_cum.reshape(-1, 1)).flatten()
        )
        # Scale by initial impact variable
        return norm_theta_vec_cum * impact
    else:
        # Scale by initial impact variable
        return theta_vec_cum * impact
