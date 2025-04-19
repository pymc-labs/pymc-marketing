# How we compare

Given the popularity of the Media Mix Modelling (MMM) approach, there are many packages available to perform MMM. Here's a high-level overview of how PyMC-Marketing compares to some of the most popular packages.

|            | PyMC-Marketing      | Lightweight-MMM | Robyn                 | Orbit KTR | Meridian              |
|------------|---------------------|-----------------|-----------------------|-----------|---------------------|
| Language   | Python              | Python          | R                     | Python    | Python              |
| Approach | Bayesian            | Bayesian        |  Traditional ML    | Bayesian | Bayesian            |
| Foundation | PyMC                | NumPyro/JAX     |                       | STAN/Pyro | Tensor Flow Probability                |
| Company    | PyMC Labs           | Google          | Meta                  | Uber      | Google              |
| Open source| ✅                  | ✅              | ✅                    | ✅       | ✅                  |
| Model   | 🏗️ Build            | 🏗️ Build        |  🏗️ Build       | 🏗️ Build        | 🏗️ Build               |
| Budget optimizer | ✅ | ✅ |   ✅        |   ❌        |    ✅                  |
| Time-varying intercept | ✅ | ❌ | ❌ | ✅ | ✅ |
| Time-varying coefficients |  ✅ | ❌ | ❌ | ✅ | ❌  |
| Custom priors | ✅ | ✅ | ❌ | ❌ | ✅ |
| Lift-test calibration | ✅  | ❌ | ✅ | ❌ | ✅ |
| Out of sample predictions | ✅ | ✅ | ❌ | ✅ | ❌ |
| Unit-tested | ✅ | ✅ | ❌ | ✅ | ✅ |
