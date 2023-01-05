# PyMC-Marketing

![Build](https://github.com/pymc-labs/pymmmc/workflows/ci/badge.svg)

## Bayesian Media Mix Models (MMMs) in PyMC

In this package we provide an API for a Bayesian media mix model (MMM) specification following [Jin, Yuxue, et al. “Bayesian methods for media mix modeling with carryover and shape effects.” (2017).](https://research.google/pubs/pub46001/). Concretely, given a time series target variable $y_{t}$ (e.g. sales on conversions), media variiables $x_{m, t}$ (e.g. impressions, clicks or costs) and a set of control covariates $z_{c, t}$ (e.g. holidays, special events) we consider a linear model of the form

$$
y_{t} = \alpha + \sum_{m=1}^{M}\beta_{m}f(x_{m, t}) +  \sum_{c=1}^{C}\gamma_{c}z_{c, t} + \varepsilon_{t},
$$

where $\alpha$ is the intercept, $f$ is a media transformation function and $\varepsilon_{t}$ is the error therm which we assume is normally distributed. The function $f$ encodes the contribution of media on the target variable. Typically we consider two types of transformation: adstock (carry-over) and saturation effects.

### References:

- [Jin, Yuxue, et al. “Bayesian methods for media mix modeling with carryover and shape effects.” (2017).](https://research.google/pubs/pub46001/)
- PyMC Labs Blog:
  - [Bayesian Media Mix Modeling for Marketing Optimization](https://www.pymc-labs.io/blog-posts/bayesian-media-mix-modeling-for-marketing-optimization/)
  - [Improving the Speed and Accuracy of Bayesian Media Mix Models](https://www.pymc-labs.io/blog-posts/reducing-customer-acquisition-costs-how-we-helped-optimizing-hellofreshs-marketing-budget/)
- [Johns, Michael and Wang,  Zhenyu. "A Bayesian Approach to Media Mix Modeling"](https://www.youtube.com/watch?v=UznM_-_760Y)
- [Orduz, Juan. "Media Effect Estimation with PyMC: Adstock, Saturation & Diminishing Returns"](https://juanitorduz.github.io/pymc_mmm/)

---

## Bayesian CLVs in PyMC
[Customer Lifetime Value](https://en.wikipedia.org/wiki/Customer_lifetime_value) models is another important class of models. There are many different types of CLV models and it can be helpful to conceptualise them as fitting in a 2-dimensional grid as below. An excellent set of introduction slides to CLV's is provided in [Probability Models for Customer-Base Analysis](https://www.brucehardie.com/talks/ho_cba_tut_art_09.pdf) by Fader & Hardie (2009).

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
