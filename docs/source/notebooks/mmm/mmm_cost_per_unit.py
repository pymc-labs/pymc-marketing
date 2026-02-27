"""Cost-per-Unit Demo: Train on Impressions, Report in Dollars.

A marimo notebook showcasing cost_per_unit for MMM channels measured in
non-monetary units (e.g., impressions).
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App()


@app.cell
def _():
    import warnings

    import arviz as az
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import xarray as xr

    from pymc_marketing.mmm.builders.yaml import build_mmm_from_yaml
    from pymc_marketing.paths import data_dir

    warnings.filterwarnings("ignore")
    az.style.use("arviz-darkgrid")
    plt.rcParams["figure.figsize"] = [12, 7]
    plt.rcParams["figure.dpi"] = 100
    return az, build_mmm_from_yaml, data_dir, mo, np, pd, plt, xr


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Introduction: Cost-per-Unit for Mixed-Unit Channels

    The causal chain for media impact is: **Spend → Impressions → Revenue**.

    - Impressions have a direct effect on revenue, so models trained on impressions
      are often less noisy and more accurate than models trained on spend.
    - But ROAS (Return on Ad Spend) and budget planning are done in **dollars**, not impressions.
    - **`cost_per_unit`** bridges this gap: train on impressions, report and plan in dollars.

    In this notebook:
    - **channel_1** (TV) is already in dollars — cost_per_unit = 1 (implicit via omission).
    - **channel_2** (Social Media) is in impressions — we set cost_per_unit ≈ $0.05/impression.
    """
    )
    return


@app.cell
def _(data_dir, pd):
    X_raw = pd.read_csv(data_dir / "processed" / "X.csv", parse_dates=["date"])
    y_raw = pd.read_csv(data_dir / "processed" / "y.csv")
    us_mask = X_raw["market"] == "US"
    x_train = X_raw.loc[us_mask].drop(columns=["market"]).reset_index(drop=True)
    y_train = y_raw.loc[us_mask].reset_index(drop=True)["y"]
    x_train.head()
    return x_train, y_train


@app.cell
def _(build_mmm_from_yaml, data_dir, x_train, y_train):
    mmm = build_mmm_from_yaml(
        X=x_train,
        y=y_train,
        config_path=data_dir / "config_files" / "example_with_original_scale_vars.yml",
        model_kwargs={
            "sampler_config": {
                "chains": 2,
                "tune": 500,
                "draws": 200,
                "random_seed": 42,
            }
        },
    )
    idata = mmm.fit(x_train, y_train)
    return idata, mmm


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Setting cost_per_unit

    Channel 2 is measured in **impressions**. We need to tell the model the cost per
    impression so it can convert to dollars for ROAS and budget planning.
    """
    )
    return


@app.cell
def _(mmm, np, pd):
    dates = mmm.data.dates
    rng = np.random.default_rng(42)
    cost_per_unit_df = pd.DataFrame(
        {
            "date": dates,
            "channel_2": np.abs(0.05 + rng.normal(0, 0.005, size=len(dates))),
        }
    )
    cost_per_unit_df.head(10)
    return cost_per_unit_df, rng


@app.cell
def _(cost_per_unit_df, mmm, mo):
    mmm.set_cost_per_unit(cost_per_unit_df)
    mo.md(
        r"""
    **cost_per_unit set!**

    - `channel_1` (TV): cost_per_unit = 1.0 (already in dollars, default)
    - `channel_2` (Social Media): cost_per_unit ≈ $0.05/impression

    `get_channel_data()` returns raw data (impressions for channel_2).
    `get_channel_spend()` returns data × cost_per_unit (dollars for both channels).
    """
    )
    return


@app.cell
def _(mmm, pd):
    raw = mmm.data.get_channel_data()
    spend = mmm.data.get_channel_spend()
    raw_df = (
        raw.isel(date=slice(0, 10)).to_dataframe().unstack("channel").add_suffix("_raw")
    )
    spend_df = (
        spend.isel(date=slice(0, 10))
        .to_dataframe()
        .unstack("channel")
        .add_suffix("_spend")
    )
    comparison = pd.concat([raw_df, spend_df], axis=1)
    comparison
    return comparison, raw, spend


if __name__ == "__main__":
    app.run()
