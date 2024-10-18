:::{image} _static/marketing-logo-dark.jpg
:align: center
:class: only-dark
:::

:::{image} _static/marketing-logo-light.jpg
:align: center
:class: only-light
:::

<h1 style="text-align: center;">Open Source Marketing Analytics Solution</h1>

:::{div} sd-text-center
[![Downloads](https://static.pepy.tech/badge/pymc-marketing)](https://pepy.tech/project/pymc-marketing)
[![Downloads](https://static.pepy.tech/badge/pymc-marketing/month)](https://pepy.tech/project/pymc-marketing)
[![Downloads](https://static.pepy.tech/badge/pymc-marketing/week)](https://pepy.tech/project/pymc-marketing)
:::

<h1 style="text-align: center;">Powered by</h1>

:::{image} _static/labs-logo-dark.png
:align: center
:target: https://www.pymc-labs.io
:scale: 20 %
:alt: PyMC Labs logo
:class: only-dark
:::

:::{image} _static/labs-logo-light.png
:align: center
:target: https://www.pymc-labs.io
:scale: 20 %
:alt: PyMC Labs logo
:class: only-light
:::

---

## 📞 Schedule a Free Strategy Consultation

Maximize your marketing ROI with a [free 30-minute strategy session](https://calendly.com/niall-oulton) with our PyMC-Marketing experts. Learn how Bayesian Marketing Mix Modeling and Customer Lifetime Value analytics can boost your organization by making smarter, data-driven decisions.

For businesses looking to integrate PyMC-Marketing into their operational framework, [PyMC Labs](https://www.pymc-labs.com) offers expert consulting and training. Our team is proficient in state-of-the-art Bayesian modeling techniques, with a focus on Marketing Mix Models (MMMs) and Customer Lifetime Value (CLV).

We provide the following professional services:

- **Custom Models**: We develop models that fit your organization's unique needs.
- **Coaching**: Regular, personalized coaching to ensure your team is well-equipped to confidently use PyMC-Marketing and related approaches.
- **SaaS Solutions**: Harness the power of our state-of-the-art software solutions to streamline your data-driven marketing initiatives.

### PyMC Labs Client Testimonials

<iframe width="800" height="450" src="https://www.youtube.com/embed/_CVEygFxFRA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>


## Quick links

:::::{grid} 1 1 2 2
:gutter: 2

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

:::{button-ref} notebooks/index
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

Leverage our Bayesian MMM API to tailor your marketing strategies effectively. Leveraging on top of the research article [Jin, Yuxue, et al. “Bayesian methods for media mix modeling with carryover and shape effects.” (2017)](https://research.google/pubs/pub46001/),  and extending it by integrating the expertise from core PyMC developers, our API provides:

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
| Out-of-sample Predictions                  | Forecast future marketing performance with credible intervals. Use this for simulations and scenario planning.                                                                                                                                                                                                                                                                          |
| Budget Optimization                        | Allocate your marketing spend efficiently across various channels for maximum ROI. See the [budget optimization example notebook](https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_budget_allocation_example.html)                                                                                                                                                             |
| Experiment Calibration                     | Fine-tune your model based on empirical experiments for a more unified view of marketing. See the [lift test integration explanation](https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_lift_test.html) for more details. [Here](https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_roas.html) you can find a *Case Study: Unobserved Confounders, ROAS and Lift Tests*. |

## Unlock Customer Lifetime Value (CLV) with PyMC

Understand and optimize your customer's value with our **CLV models**. Our API supports various types of CLV models, catering to both contractual and non-contractual settings, as well as continuous and discrete transaction modes.

Explore our detailed CLV examples using data from the [`lifetimes`](https://github.com/CamDavidsonPilon/lifetimes) package:

- [CLV Quickstart](https://pymc-marketing.readthedocs.io/en/stable/notebooks/clv/clv_quickstart.html)
- [BG/NBD model](https://pymc-marketing.readthedocs.io/en/stable/notebooks/clv/bg_nbd.html)
- [Pareto/NBD model](https://pymc-marketing.readthedocs.io/en/stable/notebooks/clv/pareto_nbd.html)
- [Gamma-Gamma model](https://pymc-marketing.readthedocs.io/en/stable/notebooks/clv/gamma_gamma.html)

Each of these models is tailored to different types of data and business scenarios:

|                | **Non-contractual**      | **Contractual**         |
| -------------- | ------------------------ | ----------------------- |
| **Continuous** | online purchases         | ad conversion time      |
| **Discrete**   | concerts & sports events | recurring subscriptions |

---

<h1 style="text-align: center;">Resources</h1>

### Bolt's success story with PyMC-Marketing
**Checkout the video below to see how Bolt leverages PyMC Marketing to assess the impact of their marketing efforts.**
<iframe width="800" height="450" src="https://www.youtube.com/embed/djXoPq60bRM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

### Time-varying parameters in MMMs in PyMC-Marketing
<iframe width="800" height="450" src="https://www.youtube.com/embed/2biNgpUpLik" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

### Customer Lifetime Value Modeling in Marine Industry
<iframe width="800" height="450" src="https://www.youtube.com/embed/u3oMWgStIZY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

For more videos, webinars and resources, check out the [PyMC Labs YouTube channel](https://www.youtube.com/@PyMC-Labs).

---

### More [PyMC Labs](https://www.pymc-labs.com/) Blog Posts and Resources

#### Marketing Mix Models

- [Unobserved Confounders, ROAS and Lift Tests in Media Mix Models](https://www.pymc-labs.com/blog-posts/mmm_roas_lift/)

#### Customer Lifetime Value

- [Hierarchical Customer Lifetime Value Models](https://www.pymc-labs.com/blog-posts/hierarchical_clv/)
- [Customer Lifetime Value in the non-contractual continuous case: The Bayesian Pareto NBD Model](https://www.pymc-labs.com/blog-posts/pareto-nbd/)
- [Cohort Revenue & Retention Analysis](https://www.pymc-labs.com/blog-posts/cohort-revenue-retention/)

### Case Studies

```bash
conda install -c conda-forge ipykernel
```

Start VS Code and ensure that the "Jupyter" extension is installed. Press Ctrl + Shift + P and type "Python: Select Interpreter". Ensure that `marketing_env` is selected. Press Ctrl + Shift + P and type "Create: New Jupyter Notebook".

### MMM Quickstart

```python
import pandas as pd

from pymc_marketing.mmm import (
    GeometricAdstock,
    LogisticSaturation,
    MMM,
)

data_url = "https://raw.githubusercontent.com/pymc-labs/pymc-marketing/main/data/mmm_example.csv"
data = pd.read_csv(data_url, parse_dates=["date_week"])

mmm = MMM(
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
    date_column="date_week",
    channel_columns=["x1", "x2"],
    control_columns=[
        "event_1",
        "event_2",
        "t",
    ],
    yearly_seasonality=2,
)
```

Once the model is fitted, we can further optimize our budget allocation as we are including diminishing returns and carry-over effects in our model.

Explore a hands-on [simulated example](https://pymc-marketing.readthedocs.io/en/stable/notebooks/mmm/mmm_example.html) for more insights into MMM with PyMC-Marketing.

Alternatively, dive into the concepts of adstock, saturation and Bayesian priors with our [MMM Explainer App](https://pymc-marketing-app.streamlit.app/).

### CLV Quickstart

We can choose from a variety of models, depending on the type of data and business nature. Let us look into a simple example with the Beta-Geo/NBD model for non-contractual continuous data.

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pymc_marketing import clv

data_url = "https://raw.githubusercontent.com/pymc-labs/pymc-marketing/main/data/clv_quickstart.csv"
data = pd.read_csv(data_url)
data["customer_id"] = data.index

beta_geo_model = clv.BetaGeoModel(data=data)

beta_geo_model.fit()
```

Once fitted, we can use the model to predict the number of future purchases for known customers, the probability that they are still alive, and get various visualizations plotted.

See the {ref}`howto` section for more on this.

## 📞 Schedule a Free Consultation for MMM & CLV Strategy

Maximize your marketing ROI with a [free 30-minute strategy session](https://calendly.com/niall-oulton) with our PyMC-Marketing experts. Learn how Bayesian Marketing Mix Modeling and Customer Lifetime Value analytics can boost your organization by making smarter, data-driven decisions.

For businesses looking to integrate PyMC-Marketing into their operational framework, [PyMC Labs](https://www.pymc-labs.com) offers expert consulting and training. Our team is proficient in state-of-the-art Bayesian modeling techniques, with a focus on Marketing Mix Models (MMMs) and Customer Lifetime Value (CLV). Explore these topics further by watching our video on [Bayesian Marketing Mix Models: State of the Art](https://www.youtube.com/watch?v=xVx91prC81g).

We provide the following professional services:

- **Custom Models**: We tailor niche marketing anayltics models to fit your organization's unique needs.
- **Build Within PyMC-Marketing**: Our team are experts leveraging the capabilities of PyMC-Marketing to create robust marketing models for precise insights.
- **SLA & Coaching**: Get guaranteed support levels and personalized coaching to ensure your team is well-equipped and confident in using our tools and approaches.
- **SaaS Solutions**: Harness the power of our state-of-the-art software solutions to streamline your data-driven marketing initiatives.

## Support

This repository is supported by [PyMC Labs](https://www.pymc-labs.io).

For companies that want to use PyMC-Marketing in production, [PyMC Labs](https://www.pymc-labs.io) is available for consulting and training. We can help you build and deploy your models in production. We have experience with cutting edge Bayesian modelling techniques which we have applied to a range of business domains including marketing analytics.

:::{image} _static/labs-logo-dark.png
:align: center
:target: https://www.pymc-labs.io
:scale: 20 %
:alt: PyMC Labs logo
:class: only-dark
:::

:::{image} _static/labs-logo-light.png
:align: center
:target: https://www.pymc-labs.io
:scale: 20 %
:alt: PyMC Labs logo
:class: only-light
:::

For more blogposts and resources, check out the [PyMC Labs Blog](https://www.pymc-labs.com/blog-posts/).

:::{toctree}
:hidden:
getting_started/index
contributing/index
guide/index
api/index
notebooks/index
:::
