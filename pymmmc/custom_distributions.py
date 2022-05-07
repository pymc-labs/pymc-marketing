import aesara.tensor as at
import numpy as np
import pymc as pm

# TODO: Turn this into a proper PyMC custom distribution


def truncated_geometric(name, data, θ):
    """Truncated geometric distribution"""
    pm.Potential(name, truncated_geometric_logp(θ, data))


# TODO: Make this robust to data where there is a single observation. Ie the most recent cohort
def truncated_geometric_logp(theta, customers):
    """Calculate log probability of the truncated geometric distribution
    Parameters
    ----------
    theta : float or arrray
        `theta` is the churn rate
    customers : array of ints
        Vector of number of customers. Should be non-increasing.
    """
    # We can't learn anything about theta if we only have one observation (equal to the initial cohort size)
    assert len(customers) > 1

    churned_in_period_t = customers[:-1] - customers[1:]

    nT = customers[-1]
    T = len(customers)
    t_vec = np.arange(start=1, stop=len(customers))

    logp = 0

    # likelihood for all non-truncated time steps [1,... T-1]
    logp += at.math.sum(
        churned_in_period_t * (at.log(theta) + ((t_vec - 1) * at.log(1 - theta)))
    )

    # likelihood for final time step
    logp += nT * T * at.log(1 - theta)

    # # likelihood for all non-truncated time steps [1,... T-1]
    # logp += at.math.sum(
    #     churned_in_period_t * (at.log(theta * at.pow((1 - theta), t_vec - 1)))
    # )

    # # likelihood for final time step T
    # logp += nT * at.log(at.pow(1 - theta, T))

    return logp
