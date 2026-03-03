"""Cost-per-Unit Demo: Train on Impressions, Report in Dollars.

A marimo notebook showcasing cost_per_unit for MMM channels measured in
non-monetary units (e.g., impressions).
"""

import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Cost-per-Unit: When Raw ROAS Tells the Wrong Story

    Imagine you are a marketing analyst at an e-commerce company. You run two
    advertising channels:

    | Channel | Data Unit | Typical Weekly Values |
    |---------|----------|-----------------------|
    | **Social** (Social Media) | Impressions | 10–100 |
    | **TV** | Dollars ($) | 10–50 |

    Your data scientist fits a Media Mix Model on this data. The model learns how
    each channel's *input* drives *revenue*. But there is a catch: **the two channels
    are measured in completely different units.** Social Media is in impressions,
    while TV is already in dollars.

    When you look at the fitted ROAS (Return on Ad Spend), TV appears to be
    **far more efficient** than Social Media. Should you shift your entire budget to
    TV?

    **Not so fast.** The comparison is unfair. Social Media's "ROAS" is actually
    *revenue per impression* — not *revenue per dollar*. Impressions are cheap
    (~$0.20 each), so the true dollar-for-dollar ROAS of Social Media is much
    higher than it first appears.

    This is exactly the problem that **`cost_per_unit`** solves. In this notebook
    we will:

    1. **Fit** an MMM on the raw data (impressions + dollars)
    2. **Inspect** ROAS and saturation curves *before* any unit conversion — and see
       how misleading they can be
    3. **Set `cost_per_unit`** for Social Media ($0.20/impression) and watch the
       story change dramatically
    4. **Run budget optimization** under different cost-per-impression assumptions
       ($0.10, $0.20, $0.30) to see how pricing affects allocation
    """)
    return


@app.cell
def _():
    import warnings

    import arviz as az
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

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
        MultiDimensionalBudgetOptimizerWrapper,
        build_mmm_from_yaml,
        data_dir,
        go,
        mo,
        np,
        pd,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 1: Load Data and Fit the Model

    We use a two-channel dataset filtered to the US market. The model is built
    from a YAML config with `GeometricAdstock` and `MichaelisMentenSaturation`.
    """)
    return


@app.cell
def _(data_dir, pd):
    X_raw = pd.read_csv(data_dir / "processed" / "X.csv", parse_dates=["date"])
    y_raw = pd.read_csv(data_dir / "processed" / "y.csv")
    us_mask = X_raw["market"] == "US"
    x_train = X_raw.loc[us_mask].drop(columns=["market"]).reset_index(drop=True)
    x_train = x_train.rename(columns={"channel_1": "Social", "channel_2": "TV"})
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
                "tune": 1000,
                "draws": 1000,
                "random_seed": 42,
            }
        },
    )
    _ = mmm.fit(x_train, y_train)
    return (mmm,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 2: The Misleading Picture — ROAS Without Unit Conversion

    Let's look at the fitted ROAS and saturation curves *as-is*, without any
    cost-per-unit adjustment. Remember:

    - **`channel_1` (Social Media)** is in **impressions**
    - **`channel_2` (TV)** is in **dollars**

    So the "ROAS" for Social Media is really *revenue per impression*, while TV's
    ROAS is a proper *revenue per dollar*. Comparing them side-by-side is like
    comparing apples and oranges.
    """)
    return


@app.cell
def _(mmm):
    fig_roas_before = mmm.plot_interactive.roas()
    fig_roas_before
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Notice how TV (channel_2) dominates.** Its ROAS is expressed in $/$ and
    looks much larger than Social Media's revenue-per-impression metric.

    A naive reading of this chart would suggest: *"TV is far more efficient —
    pour all the money there."*

    Let's look at the saturation curves next. The x-axis for each channel is in
    its *native unit*, so the scales are not comparable.
    """)
    return


@app.cell
def _(mmm):
    mmm.plot_interactive.saturation_curves()
    return


@app.cell
def _(mo):
    mo.md(r"""
    The saturation curves above look reasonable individually, but they are
    **not comparable** — Social Media's x-axis is *impressions* while TV's is
    *dollars*. We cannot visually judge which channel gives more bang for the buck.

    ---

    ## Step 3: Setting `cost_per_unit` — Leveling the Playing Field

    Now we tell the model what Social Media impressions actually *cost*. The
    causal chain for media is:

    > **Spend ($) → Impressions → Revenue ($)**

    By setting `cost_per_unit = $0.20` for Social Media, we bridge the gap from
    impressions back to dollars. The model can then express everything in
    consistent **$/\$** terms.

    - **`channel_1` (Social Media):** `cost_per_unit = 0.20` ($0.20 per impression)
    - **`channel_2` (TV):** `cost_per_unit = 1.0` (already in dollars — this is the
      default when omitted)
    """)
    return


@app.cell
def _(mmm, np, pd):
    dates = mmm.data.dates
    cost_per_unit_df = pd.DataFrame(
        {
            "date": dates,
            "Social": np.ones(len(dates)) * 0.2,
        }
    )
    cost_per_unit_df.head(10)
    mmm.set_cost_per_unit(cost_per_unit_df)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Verifying the Conversion

    `get_channel_data()` returns the **raw** values (impressions for Social Media,
    dollars for TV). `get_channel_spend()` multiplies by `cost_per_unit`, so now
    **both** columns are in dollars.
    """)
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
        .add_suffix("_spend($)")
    )
    comparison = pd.concat([raw_df, spend_df], axis=1)
    comparison
    return


@app.cell
def _(mo):
    mo.md(r"""
    Notice that `channel_1_spend($)` = `channel_1_raw` × 0.20 (impressions → dollars),
    while `channel_2_spend($)` = `channel_2_raw` × 1.0 (already dollars).

    ---

    ## Step 4: The True Picture — ROAS *After* Cost-per-Unit

    Now that both channels are in consistent dollar terms, let's revisit the ROAS
    chart.
    """)
    return


@app.cell
def _(mmm):
    fig_roas_after = mmm.plot_interactive.roas()
    fig_roas_after
    return


@app.cell
def _(mo):
    mo.md(r"""
    **The story has changed dramatically.** Social Media's ROAS has jumped —
    because dividing revenue by *actual dollar spend* (impressions × $0.20)
    instead of raw impression counts gives a much larger number. The two
    channels are now **much more comparable**.

    The takeaway: TV is *not* overwhelmingly better. Social Media delivers
    competitive returns when measured properly.

    Let's confirm this with the saturation curves, now plotted with a consistent
    **dollar (Spend)** x-axis.
    """)
    return


@app.cell
def _(mmm):
    mmm.plot_interactive.saturation_curves()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Now both saturation curves share the same unit on the x-axis — **dollars
    spent**. You can directly compare the marginal return of an extra dollar on
    Social Media vs. TV, which is exactly what you need for budget planning.

    ---

    ## Step 5: Budget Optimization with `cost_per_unit`

    For budget optimization, we provide a *future* `cost_per_unit` for the
    planning window. This is independent of the historical values — impression
    prices may change over time.

    The optimizer:
    1. Takes a **total dollar budget**
    2. Allocates dollars across channels and time periods
    3. Converts Social Media's dollar allocation to *impressions* using
       `cost_per_unit` before evaluating the response function
    4. Returns optimal budgets **in dollars**
    """)
    return


@app.cell
def _(MultiDimensionalBudgetOptimizerWrapper, mmm, pd):
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
    return budget_wrapper, future_dates, num_periods


@app.cell
def _(mo):
    mo.md(r"""
    ### Sensitivity Analysis: How Impression Price Affects Allocation

    The cost per impression is not fixed — it varies with market conditions,
    bidding strategy, and seasonality. Let's see how the optimal budget
    allocation shifts when Social Media impressions cost **$0.10**, **$0.20**,
    or **$0.30** each.

    - **Cheaper impressions ($0.10)** → more impressions per dollar → Social
      Media becomes more attractive
    - **Baseline ($0.20)** → our best estimate of current pricing
    - **Expensive impressions ($0.30)** → fewer impressions per dollar → budget
      shifts toward TV
    """)
    return


@app.cell
def _(budget_wrapper, future_dates, num_periods, pd):
    def compare_budgets_optimization_for_different_cpus(budget):
        rates = [0.10, 0.20, 0.30]
        results_list = []
        for rate in rates:
            cpu_df = pd.DataFrame(
                {
                    "date": future_dates,
                    "Social": [rate] * num_periods,
                }
            )
            budgets, _ = budget_wrapper.optimize_budget(
                budget=budget,
                cost_per_unit=cpu_df,
            )
            row = budgets.to_dataframe(name="budget").reset_index()
            row["cost_per_impression"] = f"${rate:.2f}"
            results_list.append(row)

        comparison_df = pd.concat(results_list)
        return comparison_df.pivot_table(
            index="cost_per_impression",
            columns="channel",
            values="budget",
            aggfunc="sum",
        )

    compare_budgets_optimization_for_different_cpus(10_000)
    return (compare_budgets_optimization_for_different_cpus,)


@app.cell
def _(compare_budgets_optimization_for_different_cpus):
    compare_budgets_optimization_for_different_cpus(10)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Reading the Results

    As the cost per impression **decreases** (from $0.30 → $0.10), each dollar
    buys more Social Media impressions, increasing that channel's efficiency.
    The optimizer responds by shifting more budget toward Social Media.

    Conversely, as impressions get **more expensive**, the optimizer reallocates
    toward TV where the cost structure hasn't changed.

    This is exactly the kind of scenario analysis that `cost_per_unit` enables:
    same model, same posterior — but different economic assumptions about media
    pricing lead to different optimal strategies.

    ---

    ### Optimization Landscape: Budget × Cost-per-Unit

    To see the full picture, we sweep over a grid of **total budgets** and
    **cost-per-impression** values and record what share of the budget the
    optimizer allocates to Social Media. The resulting surface reveals how
    these two levers jointly shape the optimal allocation.
    """)
    return


@app.cell
def _(budget_wrapper, future_dates, go, np, num_periods, pd):
    cpu_grid = np.round(np.linspace(0.05, 1.0, 10), 2)
    budget_grid = np.round(np.linspace(500, 20_000, 10)).astype(int)

    social_share = np.empty((len(budget_grid), len(cpu_grid)))

    for i, total_budget in enumerate(budget_grid):
        for j, cpu_rate in enumerate(cpu_grid):
            cpu_df = pd.DataFrame(
                {
                    "date": future_dates,
                    "Social": [float(cpu_rate)] * num_periods,
                }
            )
            budgets_opt, _ = budget_wrapper.optimize_budget(
                budget=float(total_budget),
                cost_per_unit=cpu_df,
            )
            budget_df = budgets_opt.to_dataframe(name="budget").reset_index()
            channel_totals = budget_df.groupby("channel")["budget"].sum()
            social_share[i, j] = channel_totals["Social"] / channel_totals.sum() * 100

    fig_surface = go.Figure(
        data=[
            go.Surface(
                x=cpu_grid,
                y=budget_grid,
                z=social_share,
                colorscale="RdYlBu_r",
                colorbar=dict(title="Social %"),
            )
        ]
    )
    fig_surface.update_layout(
        title="Optimization Landscape: Social Media Budget Share",
        scene=dict(
            xaxis_title="Cost per Impression ($)",
            yaxis_title="Total Budget ($)",
            zaxis_title="Social Share (%)",
        ),
        width=800,
        height=600,
    )
    fig_surface
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Key Takeaways

    1. **Train on the natural unit** (impressions, clicks, GRPs) for better model
       accuracy
    2. **Set `cost_per_unit`** to convert to dollars for fair ROAS comparisons
       and budget planning
    3. **Raw ROAS can be misleading** when channels use different units — always
       check whether you're comparing apples to apples
    4. **Budget optimization respects `cost_per_unit`** — the optimizer works in
       dollars internally and converts to channel-native units before evaluating
       the response function
    5. **Sensitivity analysis** with different rates lets you plan for changing
       media prices without refitting the model
    """)
    return


if __name__ == "__main__":
    app.run()
