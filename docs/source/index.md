---
og:description: "PyMC-Marketing is an open-source Python library for Bayesian Marketing Mix Modeling (MMM), Customer Lifetime Value (CLV), and media spend optimization. Built on PyMC by PyMC Labs, it provides adstock transformations, saturation curves, budget allocation, and CLV forecasting with full uncertainty quantification."
---

# PyMC-Marketing: Open Source Bayesian Marketing Mix Modeling & CLV in Python

```{eval-rst}
.. meta::
   :description: PyMC-Marketing is an open-source Python library for Bayesian Marketing Mix Modeling (MMM), Customer Lifetime Value (CLV), and media spend optimization. Built on PyMC by PyMC Labs, it provides adstock transformations, saturation curves, budget allocation, and CLV forecasting with full uncertainty quantification.
```

:::{image} _static/marketing-logo-dark.jpg
:align: center
:class: only-dark
:::

:::{image} _static/marketing-logo-light.jpg
:align: center
:class: only-light
:::

PyMC-Marketing is an open-source Python library for Bayesian marketing analytics, built and maintained by [PyMC Labs](https://www.pymc-labs.com). It provides production-ready implementations of Marketing Mix Modeling (MMM) and Customer Lifetime Value (CLV) models, enabling data scientists to measure media effectiveness, optimize marketing spend, and forecast customer value — all with full Bayesian uncertainty quantification. Whether you call it marketing mix modeling or media mix modeling, PyMC-Marketing is the most comprehensive open-source MMM solution available in Python.

:::{div} sd-text-center
[![Downloads](https://static.pepy.tech/badge/pymc-marketing)](https://pepy.tech/project/pymc-marketing)
[![Downloads](https://static.pepy.tech/badge/pymc-marketing/month)](https://pepy.tech/project/pymc-marketing)
[![GitHub stars](https://img.shields.io/github/stars/pymc-labs/pymc-marketing?style=flat&logo=github&label=Stars)](https://github.com/pymc-labs/pymc-marketing)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/pymc-labs/pymc-marketing/blob/main/LICENSE)
:::

:::{div} sd-text-center sd-text-muted
Powered by
:::

:::{image} _static/labs-logo-dark.png
:align: center
:target: https://www.pymc-labs.com
:scale: 10 %
:alt: PyMC Labs logo
:class: only-dark
:::

:::{image} _static/labs-logo-light.png
:align: center
:target: https://www.pymc-labs.com
:scale: 10 %
:alt: PyMC Labs logo
:class: only-light
:::

## Trusted in production

PyMC-Marketing is built and maintained by the **core developers of [PyMC](https://www.pymc.io)** — the Bayesian modeling library at the heart of the scientific Python stack. It isn't a research prototype: data science teams use it to plan and defend real marketing budgets, with every result validated against experiments and lift tests rather than back-tested in isolation.

:::{div} sd-text-center
**750,000+ downloads** · **1,100+ GitHub stars** · **Apache-2.0 licensed** — free, open, and auditable
:::

- **[Bolt](https://www.youtube.com/watch?v=djXoPq60bRM)** — the mobility super-app evolved its marketing measurement towards Bayesian MMM with PyMC-Marketing, validating spend decisions against experiments and quasi-experiments.
- **[HelloFresh](https://www.pymc-labs.com/blog-posts/reducing-customer-acquisition-costs-how-we-helped-optimizing-hellofreshs-marketing-budget/)** — runs Bayesian MMM in production for live budget planning. Putting the model into production cut inference time from **20 minutes to 2 minutes** and reduced prediction variance by **60%** — fast and stable enough to use inside ongoing planning cycles.

:::{div} sd-text-center
```{button-link} https://calendly.com/niall-oulton
:color: primary
:shadow:
📞 Take your marketing models to production — book a free 30-min session with the team behind PyMC-Marketing
```
:::

## Get started

```bash
pip install pymc-marketing
```

```python
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation, MMM

mmm = MMM(
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
    date_column="date_week",
    channel_columns=["x1", "x2"],
    control_columns=["event_1", "event_2", "t"],
    yearly_seasonality=2,
)
```

See how PyMC-Marketing compares to [Google Meridian, Meta Robyn, and other MMM frameworks](guide/mmm/comparison), or read the [benchmark comparison to Meridian](https://www.pymc-labs.com/blog-posts/pymc-marketing-vs-google-meridian).

## Quick links

:::::{grid} 1 1 2 3
:gutter: 2

::::{grid-item-card} Getting Started
:class-header: sd-text-center no-border
:class-title: sd-text-center
:class-footer: no-border

{material-outlined}`photo_library;5em`
^^^^^^^^^^^^^^^

Browse our getting started guide to quickly
get started with PyMC-Marketing.

+++

:::{button-ref} getting_started/index
:expand:
:color: secondary
:click-parent:
:ref-type: doc

To the getting started guide
:::
::::

::::{grid-item-card} Example notebooks
:class-header: sd-text-center no-border
:class-title: sd-text-center
:class-footer: no-border

{material-outlined}`menu_book;5em`
^^^^^^^^^^^^^^^

The example notebooks provide examples of using
the library in both real case scenarios
and synthetic data. They explain how to use
the library and showcase its features.

+++

:::{button-ref} gallery/gallery
:expand:
:color: secondary
:click-parent:
:ref-type: doc

To the example notebooks
:::
::::
::::{grid-item-card} API Reference
:class-header: sd-text-center no-border
:class-title: sd-text-center
:class-footer: no-border

{material-outlined}`data_object;5em`
^^^^^^^^^^^^^^^

The reference guide contains a detailed description of the functions,
modules, and objects included in the library. The reference describes how the
methods work and which parameters can be used. It assumes that you have an
understanding of the key concepts.

+++

:::{button-ref} api/index
:expand:
:color: secondary
:click-parent:
:ref-type: doc

To the reference guide
:::
::::
:::::

## Bayesian Marketing Mix Modeling (MMM) in PyMC

Leverage our Bayesian MMM API to tailor your marketing strategies effectively. Leveraging on top of the research article [Jin, Yuxue, et al. "Bayesian methods for media mix modeling with carryover and shape effects." (2017)](https://research.google/pubs/pub46001/),  and extending it by integrating the expertise from core PyMC developers, our API provides:

| Feature                                    | Benefit                                                                                                                                                                                                                                                                                                                                                                                 |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Custom Priors and Likelihoods              | Tailor your model to your specific business needs by including domain knowledge via prior distributions.                                                                                                                                                                                                                                                                                |
| Adstock Transformation                     | Optimize the carry-over effects in your marketing channels.                                                                                                                                                                                                                                                                                                                             |
| Saturation Effects                         | Understand the diminishing returns in media investments.                                                                                                                                                                                                                                                                                                                                |
| Customize adstock and saturation functions | You can select from a variety of adstock and saturation functions. You can even implement your own custom functions. See [documentation guide](https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_components.html).                                                                                                                                                              |
| Time-varying Intercept                     | Capture time-varying baseline contributions in your model (using modern and efficient Gaussian processes approximation methods). See [guide notebook](https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_time_varying_media_example.html).                                                                                                                                       |
| Time-varying Media Contribution            | Capture time-varying media efficiency in your model (using modern and efficient Gaussian processes approximation methods). See the [guide notebook](https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_tvp_example.html).                                                                                                                                                        |
| Visualization and Model Diagnostics        | Get a comprehensive view of your model's performance and insights.                                                                                                                                                                                                                                                                                                                      |
| Choose among many inference algorithms     | We provide the option to choose between various NUTS samplers (e.g. BlackJax, NumPyro and Nutpie). See the [example notebook](https://www.pymc-marketing.io/en/stable/notebooks/general/other_nuts_samplers.html) for more details.                                                                                                                                                     |
| GPU Support                                | PyMC's multiple backends allow for GPU acceleration.                                                                                                                                                                                                                                                                                                                                    |
| Out-of-sample Predictions                  | Forecast future marketing performance with credible intervals. Use this for simulations and scenario planning.                                                                                                                                                                                                                                                                          |
| Budget Optimization                        | Allocate your marketing spend efficiently across various channels for maximum ROI. See the [budget optimization example notebook](https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_budget_allocation_example.html)                                                                                                                                                             |
| Experiment Calibration                     | Fine-tune your model based on empirical experiments for a more unified view of marketing. See the [lift test integration explanation](https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_lift_test.html) for more details. [Here](https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_roas.html) you can find a *Case Study: Unobserved Confounders, ROAS and Lift Tests*. |

## Unlock Customer Lifetime Value (CLV) with PyMC

Understand and optimize your customer's value with our **CLV models**. Our API supports various types of CLV models, catering to both contractual and non-contractual settings, as well as continuous and discrete transaction modes:

- [CLV Quickstart](https://www.pymc-marketing.io/en/stable/notebooks/clv/clv_quickstart.html)
- [BG/NBD model](https://www.pymc-marketing.io/en/stable/notebooks/clv/bg_nbd.html)
- [Pareto/NBD model](https://www.pymc-marketing.io/en/stable/notebooks/clv/pareto_nbd.html)
- [Gamma-Gamma model](https://www.pymc-marketing.io/en/stable/notebooks/clv/gamma_gamma.html)
- [Shifted BG model](https://www.pymc-marketing.io/en/stable/notebooks/clv/sbg.html)
- [Modified BG/NBD model](https://www.pymc-marketing.io/en/stable/notebooks/clv/mbg_nbd.html)

Each of these models is tailored to different types of data and business scenarios:

|                | **Non-contractual**      | **Contractual**         |
| -------------- | ------------------------ | ----------------------- |
| **Continuous** | online purchases         | ad conversion time      |
| **Discrete**   | concerts & sports events | recurring subscriptions |

## Customer Choice Analysis

Analyze the impact of new product launches and understand customer choice behavior with our **Multivariate Interrupted Time Series (MVITS)** models. Our API supports analysis in both saturated and unsaturated markets to help you:

| Feature                     | Benefit                                                           |
| --------------------------- | ----------------------------------------------------------------- |
| Market Share Analysis       | Understand how new products affect existing product market shares |
| Causal Impact Assessment    | Measure the true causal effect of product launches on sales       |
| Saturated Market Analysis   | Model scenarios where total market size remains constant          |
| Unsaturated Market Analysis | Handle cases where new products grow the total market size        |
| Visualization Tools         | Plot market shares, causal impacts, and counterfactuals           |
| Bayesian Inference          | Get uncertainty estimates around all predictions                  |

See our example notebooks for [saturated markets](https://www.pymc-marketing.io/en/stable/notebooks/customer_choice/mv_its_saturated.html) and [unsaturated markets](https://www.pymc-marketing.io/en/stable/notebooks/customer_choice/mv_its_unsaturated.html) to learn more about customer choice modeling with PyMC-Marketing.

## Bass Diffusion Model

Forecast the adoption of new products with the **Bass Diffusion Model**. The model captures how innovators and imitators drive cumulative adoption over time, which is useful for product launch forecasts, demand planning, and growth analysis.

See the [Bass Diffusion Model example notebook](https://www.pymc-marketing.io/en/stable/notebooks/bass/bass_example.html) for a worked example.

---

## Resources

### Bolt's success story with PyMC-Marketing
**Checkout the video below to see how Bolt leverages PyMC-Marketing to assess the impact of their marketing efforts.**
<iframe width="800" height="450" src="https://www.youtube.com/embed/djXoPq60bRM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

### Time-varying parameters in MMMs in PyMC-Marketing
<iframe width="800" height="450" src="https://www.youtube.com/embed/2biNgpUpLik" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

### Customer Lifetime Value Modeling in Marine Industry
<iframe width="800" height="450" src="https://www.youtube.com/embed/u3oMWgStIZY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

For more videos, webinars and resources, check out the [PyMC Labs YouTube channel](https://www.youtube.com/@PyMCLabs).

---

### More [PyMC Labs](https://www.pymc-labs.com/) Blog Posts and Resources

#### Marketing Mix Models

- [Unobserved Confounders, ROAS and Lift Tests in Media Mix Models](https://www.pymc-labs.com/blog-posts/mmm_roas_lift/)
- [MMM Explainer App: Dynamic and Interactive Visualization of Key MMM Concepts](https://pymc-marketing-app.streamlit.app/)

#### Customer Lifetime Value

- [Hierarchical Customer Lifetime Value Models](https://www.pymc-labs.com/blog-posts/hierarchical_clv/)
- [Customer Lifetime Value in the non-contractual continuous case: The Bayesian Pareto NBD Model](https://www.pymc-labs.com/blog-posts/pareto-nbd/)
- [Cohort Revenue & Retention Analysis](https://www.pymc-labs.com/blog-posts/cohort-revenue-retention/)

### Case Studies

- [Building an in-house marketing analytics solution](https://www.pymc-labs.com/blog-posts/2023-07-18-niall-In-house-marketing/)
- [Bayesian Media Mix Models: Modelling changes in marketing effectiveness over time](https://www.pymc-labs.com/blog-posts/modelling-changes-marketing-effectiveness-over-time/)
- [Improving the Speed and Accuracy of Bayesian Media Mix Models](https://www.pymc-labs.com/blog-posts/reducing-customer-acquisition-costs-how-we-helped-optimizing-hellofreshs-marketing-budget/)
- [Bayesian Media Mix Modeling for Marketing Optimization](https://www.pymc-labs.com/blog-posts/bayesian-media-mix-modeling-for-marketing-optimization/)
- [Bayesian inference at scale: Running A/B tests with millions of observations](https://www.pymc-labs.com/blog-posts/bayesian-inference-at-scale-running-ab-tests-with-millions-of-observations/)

For more blogposts and resources, check out the [PyMC Labs Blog](https://www.pymc-labs.com/blog-posts/).

---

## Frequently Asked Questions

### What is PyMC-Marketing?

PyMC-Marketing is an open-source Python library for Bayesian marketing analytics. It includes production-ready implementations of Marketing Mix Modeling (MMM) for measuring media effectiveness and optimizing marketing spend, Customer Lifetime Value (CLV) models for forecasting customer value over time, Customer Choice models for analyzing product launch impacts, and a Bass Diffusion Model for forecasting new product adoption.

### How does PyMC-Marketing compare to Google Meridian or Meta Robyn?

PyMC-Marketing is the only open-source framework that combines MMM, CLV, Customer Choice, and Bass Diffusion models in a single library, with full Bayesian uncertainty quantification via [PyMC](https://www.pymc.io). Unlike Robyn (frequentist ridge regression) or Meridian (limited to MMM), PyMC-Marketing gives you posterior distributions over all parameters, custom priors for incorporating domain knowledge, and a unified API across all model types. See the [full comparison](guide/mmm/comparison) and the [benchmark comparison to Meridian](https://www.pymc-labs.com/blog-posts/pymc-marketing-vs-google-meridian).

### Is PyMC-Marketing free to use?

Yes. PyMC-Marketing is completely free and open source under the [Apache 2.0 license](https://github.com/pymc-labs/pymc-marketing/blob/main/LICENSE). You can install it with `pip install pymc-marketing`.

### What Python version does PyMC-Marketing require?

PyMC-Marketing supports Python 3.12 and above.

### Can PyMC-Marketing optimize my marketing budget?

Yes. PyMC-Marketing includes built-in [budget optimization](https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_budget_allocation_example.html) that allocates spend across channels to maximize ROI, accounting for diminishing returns via saturation curves and carry-over effects via adstock transformations.

---

## Need help with PyMC-Marketing?

Book a [free 30-minute strategy session](https://calendly.com/niall-oulton) with our experts or [send us a message](https://www.pymc-labs.com/contact).

:::{toctree}
:hidden:

getting_started/index
guide/index
gallery/gallery
guide/mmm/comparison
api/index
contributing/index
:::
