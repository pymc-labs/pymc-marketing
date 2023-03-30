# PyMC Marketing

Bayesian MMMs and CLVs in PyMC.

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

You may have already set up an environment (e.g. `marketing_env`) you’d like to work in. It may look something like the following:

```
mamba create -c conda-forge -n marketing_env python=3.10 matplotlib jupyterlab ipykernel seaborn pandas pymc
mamba activate marketing_env
python -m ipykernel install --user --name marketing_env
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

## Quickstart

Once you've installed the library (see above), you can get started.

### MMM Quickstart
You could then get started in a Python session, or a jupyter lab notebook with:

```python
import pandas as pd
from pymc_marketing import mmm


data_url = "https://raw.githubusercontent.com/pymc-labs/pymc-marketing/main/datasets/mmm_example.csv"
data = pd.read_csv(data_url, parse_dates=['date_week'])

model = mmm.DelayedSaturatedMMM(
    data_df=data,
    target_column="y",
    date_column="date_week",
    channel_columns=["x1", "x2"],
    control_columns=[
        "event_1",
        "event_2",
        "t",
        "sin_order_1",
        "cos_order_1",
        "sin_order_2",
        "cos_order_2",
    ],
    adstock_max_lag=8,
)
```

Initiate fitting and get a visualization of some of the outputs with:

```python
model.fit()
model.plot_components_contributions()
```

See the Example notebooks section for examples of further types of plot you can get, as well as introspect the results of the fitting.

### CLV Quickstart

Again, in a Python session, or juputer lab notebook:

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pymc_marketing import clv


data_url = "https://raw.githubusercontent.com/pymc-labs/pymc-marketing/main/datasets/clv_quickstart.csv"
data = pd.read_csv(data_url)

beta_geo_model = clv.BetaGeoModel(
    customer_id=data.index,
    frequency=data["frequency"],
    recency=data["recency"],
    T=data["T"],
)

beta_geo_model.fit()
```
Once fitted, we can use the model to predict the number of future purchases for known customers, the probability that they are still alive, and get various visualizations plotted. See the Examples section for more on this.

## Support

This repository is supported by [PyMC Labs](https://www.pymc-labs.io).

:::{image} _static/logo-dark.png
:align: center
:target: https://www.pymc-labs.io
:scale: 20 %
:alt: PyMC Labs logo
:class: only-dark
:::

:::{image} _static/logo-light.png
:align: center
:target: https://www.pymc-labs.io
:scale: 20 %
:alt: PyMC Labs logo
:class: only-light
:::


:::{toctree}
:hidden:

api/index
notebooks/index
:::
