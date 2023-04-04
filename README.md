# PyMC-Marketing

![Build](https://github.com/pymc-labs/pymc-marketing/workflows/ci/badge.svg)
[![codecov](https://codecov.io/gh/pymc-labs/pymc-marketing/branch/main/graph/badge.svg?token=OBV3BS5TYE)](https://codecov.io/gh/pymc-labs/pymc-marketing)
[![docs](https://readthedocs.org/projects/pymc-marketing/badge/?version=latest)](https://docs.readthedocs.io/en/latest/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI Version](https://img.shields.io/pypi/v/pymc-marketing.svg)](https://pypi.python.org/pypi/pymc-marketing)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Unlock the power of marketing analytics with PyMC-Marketing – the open source solution for smarter decision-making.** Media mix modeling and customer lifetime value modules allow businesses to make data-driven decisions about their marketing campaigns. Optimize your marketing strategy and unlock the full potential of your customer data.

---

## Installation

Start by setting up an environment (e.g. `marketing_env`) with PyMC. It may look something like the following:

```bash
mamba create -c conda-forge -n marketing_env python "pymc>=5"
mamba activate marketing_env
```

See the official [PyMC installation guide](https://www.pymc.io/projects/docs/en/latest/installation.html) if more detail is needed.

Assuming you have an environment set up then install PyMC-Marketing with the following command. This will give you the latest version of the library from PyPI.

```bash
pip install pymc-marketing
```

Alternatively you can install from GitHub directly:

```bash
pip install git+https://github.com/pymc-labs/pymc-marketing.git
```

## Bayesian Media Mix Models (MMMs) in PyMC

In this package we provide an API for a Bayesian media mix model (MMM) specification following [Jin, Yuxue, et al. “Bayesian methods for media mix modeling with carryover and shape effects.” (2017).](https://research.google/pubs/pub46001/) Concretely, given a time series target variable $y_{t}$ (e.g. sales on conversions), media variables $x_{m, t}$ (e.g. impressions, clicks or costs) and a set of control covariates $z_{c, t}$ (e.g. holidays, special events) we consider a linear model of the form

$$
y_{t} = \alpha + \sum_{m=1}^{M}\beta_{m}f(x_{m, t}) +  \sum_{c=1}^{C}\gamma_{c}z_{c, t} + \varepsilon_{t},
$$

where $\alpha$ is the intercept, $f$ is a media transformation function and $\varepsilon_{t}$ is the error therm which we assume is normally distributed. The function $f$ encodes the contribution of media on the target variable. Typically we consider two types of transformation: adstock (carry-over) and saturation effects.

[Here](https://pymc-marketing.readthedocs.io/en/stable/notebooks/mmm/mmm_example.html) you can find a simulated example:

1. First, we describe the data genaration process of a simulated dataset.
2. Next, we describe how to specify and fit a media mix model (as described above) using the `pymc-marketing` MMM's API.
3. Finally, we describe the model results: channel constribution and ROAS estimation. We also show how the model recovers the parameters from the data generation process step.

### References:

- [Jin, Yuxue, et al. “Bayesian methods for media mix modeling with carryover and shape effects.” (2017).](https://research.google/pubs/pub46001/)
- PyMC Labs Blog:
  - [Bayesian Media Mix Modeling for Marketing Optimization](https://www.pymc-labs.io/blog-posts/bayesian-media-mix-modeling-for-marketing-optimization/)
  - [Improving the Speed and Accuracy of Bayesian Media Mix Models](https://www.pymc-labs.io/blog-posts/reducing-customer-acquisition-costs-how-we-helped-optimizing-hellofreshs-marketing-budget/)
- [Johns, Michael and Wang,  Zhenyu. "A Bayesian Approach to Media Mix Modeling"](https://www.youtube.com/watch?v=UznM_-_760Y)
- [Orduz, Juan. "Media Effect Estimation with PyMC: Adstock, Saturation & Diminishing Returns"](https://juanitorduz.github.io/pymc_mmm/)

---

## Bayesian CLVs in PyMC
[Customer Lifetime Value](https://en.wikipedia.org/wiki/Customer_lifetime_value) (CLV) models are another important class of models. There are many different types of CLV models and it can be helpful to conceptualise them as fitting in a 2-dimensional grid as below. An excellent set of introduction slides to CLV's is provided in [Probability Models for Customer-Base Analysis](https://www.brucehardie.com/talks/ho_cba_tut_art_09.pdf) by Fader & Hardie (2009).

### Examples

|                | **Non-contractual** | **Contractual**                 |
|----------------|---------------------|---------------------------------|
| **Continuous** | Buying groceries    | Audible                         |
| **Discrete**   | Cinema ticket       | Monthly or yearly subscriptions |

To explain further:
- **Contractual:** In contractual settings, a customer has a contract which continues to be active until it is explicitly cancelled. Therefore, customer churn events are observed.

- **Non-contractual:** In non-contractual settings, there is no ongoing contract that a customer has with a company. Instead, purchases can be ad hoc and churn events are unobserved.

- **Discrete:** Here, purchases are made at discrete points in time. This obviously depends upon the timescale that we are working on, but typically a relevant time period would be a month or year. However it could be more granualar than this - think of taking the 2nd of 4 inter-city train journeys offered per day.

- **Continuous:** In the continuous-time domain, purchases can be made at any point within a firms opening hours. For online ordering, this could be any point within a 24 hour cycle, or purchases in physical stores could be made at any point during the trading day.

In the documentation, we provide some examples on how to use the CLV API. We use the data from the [`lifetimes`](https://github.com/CamDavidsonPilon/lifetimes) package to illustrate the models.

- [CLV Quickstart](https://pymc-marketing.readthedocs.io/en/stable/notebooks/clv/clv_quickstart.html)
- [BG/NBD model](https://pymc-marketing.readthedocs.io/en/stable/notebooks/clv/bg_nbd.html)
- [Gamma-Gamma model](https://pymc-marketing.readthedocs.io/en/stable/notebooks/clv/gamma_gamma.html)

---

## 📞 Schedule a Consultation
Unlock your potential with a free 30-minute strategy session with our PyMC experts. Discover how open source solutions and pymc-marketing can elevate your media-mix models and customer lifetime value analyses. Boost your career and organization by making smarter, data-driven decisions. Don't wait—[claim your complimentary session](https://calendly.com/benjamin-vincent/pymc-marketing) today and lead the way in marketing and data science innovation.

## Using PyMC-Marketing and how PyMC Labs can help you
PyMC-Marketing uses the [Apache 2.0 licence](LICENSE) which permits commercial use, amongst other things.

If you want to build upon the package, please feel free to fork the repo and submit a pull request. If in doubt, please open an issue.

For companies that want to use PyMC-Marketing in production, [PyMC Labs](https://www.pymc-labs.io) is available for consulting and training. We can help you build and deploy your models in production. We have experience with cutting edge Bayesian modelling techniques in general, and in particular with MMMs and CLVs. For example, see our video on [Bayesian Marketing Mix Models: State of the Art and their Future](https://www.youtube.com/watch?v=xVx91prC81g).
