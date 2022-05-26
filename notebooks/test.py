import pymmmc

import aesara.tensor as at
import numpy as np
import pandas as pd

from aesara.tensor import TensorVariable

import pymc as pm

import matplotlib.pyplot as plt
import arviz as az


if __name__ == "__main__":
    rng = np.random.RandomState(seed=34)
    T = 10
    T0 = 0

    # individual-level model
    lam = 0.5; p = 0.2

    data = pymmmc.distributions.continuous_contractual.rng_fn(rng, lam, p, T, T0, size=[1000,])

    with pm.Model() as model:
        λ = pm.Gamma(name="λ", alpha=0.1, beta=0.1)
        π = pm.Beta(name="π", alpha=1, beta=1)
        
        cont_contractual = pymmmc.ContinuousContractual(
            name="continuous-contractual-clv",
            lam=λ,
            p=π,
            T=10,
            T0=0,
            observed=data,
        )
    
        trace = pm.sample(draws=10000, chains=1, tune=5000)
