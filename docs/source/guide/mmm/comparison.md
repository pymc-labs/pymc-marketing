# How we compare

Given the popularity of the Media Mix Modelling (MMM) approach, there are many packages available to perform MMM. Here's a high-level overview of how PyMC-Marketing compares to some of the most popular packages.

|            | PyMC-Marketing      | Lightweight-MMM | Robyn                 | Orbit KTR | Recast              |
|------------|---------------------|-----------------|-----------------------|-----------|---------------------|
| Language   | Python              | Python          | R                     | Python    | R                   |
| Approach | Bayesian            | Bayesian        |  Traditional ML    | Bayesian | Bayesian    |
| Foundation | PyMC                | NumPyro/JAX     |                       | STAN/Pyro | STAN                |
| Company    | PyMC Labs           | Google          | Meta                  | Uber      | Recast              |
| Open source| ✅                  | ✅              | ✅                    | ✅       | ❌                 |
| Model   | 🏗️ Build            | 🏗️ Build        |  🏗️ Build       | 🏗️ Build        | 🛒 Buy               |
| Budget optimizer | ✅ | ✅ |   ✅        |   ❌        |    ✅                  |
| Time-varying parameters | coming soon | ❌ | ❌ | ✅ | ✅ |
| Custom priors | ✅ | ✅ | ❌ | ❌ | ✅ |
| Lift-test calibration | ✅  | ❌ | ✅ | ❌ | ✅ |
| Out of sample predictions | ✅ | ✅ | ❌ | ✅ | ✅ |
| Unit-tested | ✅ | ✅ | ❌ | ✅ | ? |
