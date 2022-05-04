import pymc as pm
import numpy as np

# TODO: Is this even a vaguely correct name for this distribution?
# TODO: Turn this into a proper PyMC custom distribution
def truncated_geometric(name, data, θ):
    churned_in_period_t = data[:-1] - data[1:]
    T = len(data)
    time_periods = np.arange(start=1, stop=T)
    pm.Potential(
        name + "observed",
        churned_in_period_t * pm.math.log(θ * (1 - θ) ** (time_periods - 1)),
    )
    pm.Potential(name + "final", data[T - 1] * pm.math.log((1 - θ) ** T))
    return
