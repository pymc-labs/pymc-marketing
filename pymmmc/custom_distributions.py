import pymc as pm
import numpy as np
import aesara.tensor as at


# TODO: Turn this into a proper PyMC custom distribution
def truncated_geometric(name, data, θ):
    """Truncated geometric distribution"""
    pm.Potential(name, truncated_geometric_logp(θ, data))


def truncated_geometric_logp(theta, customers):
    """Calculate log probability of the truncated geometric distribution
    Parameters
    ----------
    theta : float
        `theta` is the churn rate
    customers : array of ints
        Vector of number of customers. Should be non-increasing.
    """
    churned_in_period_t = customers[:-1] - customers[1:]
    T = len(customers)
    time_periods = np.arange(start=1, stop=T)
    logp = 0
    logp += at.math.sum(
        churned_in_period_t * (at.log(theta) + ((time_periods - 1) * at.log(1 - theta)))
    )
    logp += customers[T - 1] * (T * at.log(1 - theta))
    return logp
