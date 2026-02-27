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
    from pymc_marketing.mmm.multidimensional import (
        MultiDimensionalBudgetOptimizerWrapper,
    )
    from pymc_marketing.paths import data_dir

    warnings.filterwarnings("ignore")
    az.style.use("arviz-darkgrid")
    plt.rcParams["figure.figsize"] = [12, 7]
    plt.rcParams["figure.dpi"] = 100
    return (
        az,
        build_mmm_from_yaml,
        data_dir,
        mo,
        np,
        pd,
        plt,
        xr,
        MultiDimensionalBudgetOptimizerWrapper,
    )


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


@app.cell
def _(mo):
    mo.md(
        r"""
    ## ROAS with cost_per_unit

    With cost_per_unit set, ROAS is now computed in consistent **$/$** terms for both
    channels. Below we show the interactive ROAS plot.
    """
    )
    return


@app.cell
def _(mmm):
    fig_roas = mmm.plot_interactive.roas()
    fig_roas
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Both channels now have ROAS in $/$ terms. Without cost_per_unit, channel_2's ROAS
    would be revenue-per-impression — meaningless for comparison with channel_1's
    revenue-per-dollar.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Static Plots with cost_per_unit

    The static plot suite supports cost_per_unit via the `apply_cost_per_unit` flag.
    When True, x-axes show **spend (dollars)** instead of raw channel data.
    """
    )
    return


@app.cell
def _(mmm, plt):
    fig_sat, axes_sat = mmm.plot.saturation_scatterplot(
        original_scale=True,
        apply_cost_per_unit=True,
    )
    plt.tight_layout()
    fig_sat
    return axes_sat, fig_sat


@app.cell
def _(mmm, np, plt):
    mmm.sensitivity.run_sweep(
        var_input="channel_data",
        sweep_values=np.linspace(0.5, 1.5, 11),
        var_names="channel_contribution",
        sweep_type="multiplicative",
        extend_idata=True,
    )
    sens_result = mmm.plot.sensitivity_analysis(
        apply_cost_per_unit=True,
        hue_dim="channel",
        x_sweep_axis="absolute",
    )
    fig_sens = sens_result.figure if hasattr(sens_result, "figure") else sens_result[0]
    plt.tight_layout()
    fig_sens
    return fig_sens, sens_result


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Budget Optimization with cost_per_unit

    For budget optimization, we provide a *future* cost_per_unit for the optimization
    window. This is independent of the historical cost_per_unit. The optimizer takes a
    dollar budget, distributes it over time, converts channel_2's portion to impressions
    using cost_per_unit, then optimizes. Output budgets are in dollars.
    """
    )
    return


@app.cell
def _(MultiDimensionalBudgetOptimizerWrapper, mmm, np, pd, rng):
    num_periods = 4
    future_dates = pd.date_range(
        start=mmm.data.dates[-1] + pd.Timedelta(days=1),
        periods=num_periods,
        freq="D",
    )
    start_date = future_dates[0]
    end_date = future_dates[-1]
    budget_wrapper = MultiDimensionalBudgetOptimizerWrapper(
        model=mmm,
        start_date=start_date,
        end_date=end_date,
    )
    future_cpu_df = pd.DataFrame(
        {
            "date": future_dates,
            "channel_2": np.abs(0.05 + rng.normal(0, 0.005, size=num_periods)),
        }
    )
    future_cpu_df
    return (
        budget_wrapper,
        end_date,
        future_cpu_df,
        future_dates,
        num_periods,
        start_date,
    )


@app.cell
def _(budget_wrapper, future_cpu_df):
    optimal_budgets, result = budget_wrapper.optimize_budget(
        budget=10_000,
        cost_per_unit=future_cpu_df,
    )
    optimal_budgets
    return optimal_budgets, result


@app.cell
def _(mo):
    mo.md(
        r"""
    `optimal_budgets` are in dollars. The optimizer internally converted channel_2's
    dollar allocation to impressions via cost_per_unit before feeding to the model's
    response function.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Sensitivity to Different cost_per_unit Rates

    The cost_per_unit rate directly affects budget allocation. Cheaper impressions
    ($0.03) make Social Media more attractive per dollar; expensive impressions
    ($0.10) shift budget toward TV.
    """
    )
    return


@app.cell
def _(budget_wrapper, future_dates, num_periods, pd):
    rates = [0.03, 0.05, 0.10]
    results_list = []
    for rate in rates:
        cpu_df = pd.DataFrame(
            {
                "date": future_dates,
                "channel_2": [rate] * num_periods,
            }
        )
        budgets, _ = budget_wrapper.optimize_budget(
            budget=10_000,
            cost_per_unit=cpu_df,
        )
        row = budgets.to_dataframe(name="budget").reset_index()
        row["cost_per_unit_channel_2"] = f"${rate:.2f}"
        results_list.append(row)

    comparison_df = pd.concat(results_list)
    comparison_df.pivot_table(
        index="cost_per_unit_channel_2",
        columns="channel",
        values="budget",
        aggfunc="sum",
    )
    return comparison_df, cpu_df, rate, results_list


@app.cell
def _(mo):
    mo.md(
        r"""
    As channel_2 gets cheaper, more budget flows there (more impressions per dollar =
    more value). As it gets expensive, budget shifts to channel_1 (TV).
    """
    )
    return


if __name__ == "__main__":
    app.run()
