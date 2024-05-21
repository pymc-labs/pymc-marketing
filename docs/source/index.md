# PyMC Marketing: Open Source Marketing Analytics Solution

**Unlock the power of marketing analytics with PyMC-Marketing – the python based open source solution for smarter decision-making.** Marketing mix modeling and customer lifetime value modules allow businesses to make data-driven decisions about their marketing campaigns. Optimize your marketing strategy and unlock the full potential of your customer data.

**Checkout the video below to see how Bolt leverages PyMC Marketing to assess the impact of their marketing efforts.**
<iframe width="560" height="315" src="https://www.youtube.com/embed/djXoPq60bRM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

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

## Installation

Install and activate an environment (e.g. `marketing_env`) with the `pymc-marketing` package from [conda-forge](https://conda-forge.org). It may look something like the following:

```bash
conda create -c conda-forge -n marketing_env pymc-marketing
conda activate marketing_env
```

### Installation for developers
If you are a developer of pymc-marketing, or want to start contributing, [refer to the contributing guide](https://github.com/pymc-labs/pymc-marketing/blob/main/CONTRIBUTING.md) to get started.

See the official [PyMC installation guide](https://www.pymc.io/projects/docs/en/latest/installation.html) if more detail is needed.

## Quickstart

Create a new Jupyter notebook with either JupyterLab or VS Code.

### JupyterLab Notebook

After installing the `pymc-marketing` package (see above), run the following with `marketing_env` activated:

```bash
conda install -c conda-forge jupyterlab
jupyter lab
```

### VS Code Notebook

After installing the `pymc-marketing` package (see above), run the following with `marketing_env` activated:

```bash
conda install -c conda-forge ipykernel
```

Start VS Code and ensure that the "Jupyter" extension is installed. Press Ctrl + Shift + P and type "Python: Select Interpreter". Ensure that `marketing_env` is selected. Press Ctrl + Shift + P and type "Create: New Jupyter Notebook".

### MMM Quickstart

```python
import pandas as pd
from pymc_marketing.mmm import DelayedSaturatedMMM


data_url = "https://raw.githubusercontent.com/pymc-labs/pymc-marketing/main/data/mmm_example.csv"
data = pd.read_csv(data_url, parse_dates=['date_week'])

mmm = DelayedSaturatedMMM(
    date_column="date_week",
    channel_columns=["x1", "x2"],
    control_columns=[
        "event_1",
        "event_2",
        "t",
    ],
    adstock_max_lag=8,
    yearly_seasonality=2,
)
```

Initiate fitting and get a visualization of some of the outputs with:

```python
X = data.drop('y',axis=1)
y = data['y']
mmm.fit(X,y)
mmm.plot_components_contributions();
```

See the Example notebooks section for examples of further types of plot you can get, as well as introspect the results of the fitting.

### CLV Quickstart

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pymc_marketing import clv


data_url = "https://raw.githubusercontent.com/pymc-labs/pymc-marketing/main/data/clv_quickstart.csv"
data = pd.read_csv(data_url)
data['customer_id'] = data.index

beta_geo_model = clv.BetaGeoModel(
    data = data
)

beta_geo_model.fit()
```

Once fitted, we can use the model to predict the number of future purchases for known customers, the probability that they are still alive, and get various visualizations plotted. See the Examples section for more on this.

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


:::{toctree}
:hidden:

guide/index
api/index
notebooks/index
:::
