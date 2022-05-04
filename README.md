# pymmmc

![Build](https://github.com/pymc-labs/pymmmc/workflows/ci/badge.svg)

## Bayesian MMMs in PyMC

...

## Bayesian CLVs in PyMC
[Customer Lifetime Value](https://en.wikipedia.org/wiki/Customer_lifetime_value) models is another important clas of models. There are many different types of CLV models and it can be helpful to conceptualise them as fitting in a 2-dimensional grid as below.

### Examples

|                | **Non-contractual** | **Contractual**                 |
|----------------|---------------------|---------------------------------|
| **Continuous** | Buying groceries    | Audible                         |
| **Discrete**   | Cinema ticket       | Monthly or yearly subscriptions |

To explain further:
- **Contractual:** In contractual settings a customer has a contract which continues to be active until it is explicitly cancelled. Therefore in contractual settings, customer churn events are observed.

- **Non-contractual:** In non-contractual settings, there is no ongoing contract that a customer has with a company. Instead, purchases can be ad hoc and churn events are unobserved.

- **Discrete:** Here, purchases are made at discrete points in time. This obviously depends upon the timescale that we are working on, but typically a relevant time period would be a month or year. However it could be more granualar than this - think of taking the 2nd of 4 inter-city train journeys offered per day.

- **Continuous:** In the continuous-time domain, purchases can be made at any point within a firms opening hours. For online ordering this could be any point within a 24 hour cycle, or purchases in physical stores could be made at any point during the trading day.

Below are links to notebooks we've written that outline CLV models by type

### Continuous non-contractual models
[links to notebook(s) here]

### Continuous contractual models
[links to notebook(s) here]

### Discrete non-conntractual models
[links to notebook(s) here]

### Discrete contractual models
[links to notebook(s) here]


---

## Local Development

1. Create conda environment. For example:

```shell
conda create -n pymmmc_env python=3.8
```

2. Activate environment.

```shell
conda activate pymmmc_env
```

3. Install `pymmmc` package:

```shell
make init
```

4. To run tests:

```shell
make test
```

5. To check code style:

```shell
make check_lint
```

6. Set [pre-commit hooks](https://pre-commit.com/) (Optional):

```shell
pre-commit install
```
