import pymc as pm
import numpy as np
import aesara.tensor as at


# TODO: Turn this into a proper PyMC custom distribution
def truncated_geometric(name, data, θ):
    """
    niave implementation...
    pm.Potential(
        name + "observed",
        churned_in_period_t * pm.math.log(θ * (1 - θ) ** (time_periods - 1)),
    )
    pm.Potential(name + "final", data[T - 1] * pm.math.log((1 - θ) ** T))
    But we will remove the exponents from inside the logs:
    a * ln(x*b^C) = a * (ln(x) + c * In(b))
    In(x^b) = b In(x)
    """
    pm.Potential(name, truncated_geometric_logp(θ, data))


def truncated_geometric_logp(theta, customers):
    churned_in_period_t = customers[:-1] - customers[1:]
    T = len(customers)
    time_periods = np.arange(start=1, stop=T)
    logp = 0
    logp += at.math.sum(
        churned_in_period_t * (at.log(theta) + ((time_periods - 1) * at.log(1 - theta)))
    )
    logp += customers[T - 1] * (T * at.log(1 - theta))
    return logp
