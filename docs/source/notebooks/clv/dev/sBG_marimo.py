import marimo

__generated_with = "0.17.7"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Shifted Beta-Geometric Modeling with Cohorts
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this notebook we replicate the main results and figures from ["How to Project Customer Retention"](https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_jim_07.pdf) by Hardie & Fader (2007), which introduces the Shifted Beta-Geometric (sBG) model for customer behavior in a discrete contractual setting. It is ideal for business cases involving recurring subscriptions and has the following assumptions:
    * Customer cancellation probabilities are Beta-distributed with hyperparameters `alpha` and `beta`.
    * Retention rates change over time due to customer heterogeneity.
    * All customers in a given cohort began their contract in the same time period.

    The last assumption in particular is an ideal application for hierarchical Bayesian modeling.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutorial Notebook Outstanding Tasks
    - [ ] Rename sBG data
    - [ ] Merge Static Covariate PR
    - [ ] Fit regular/highend data to both a cohort and covariate model to replicate research results
    - [ ] Synthesize time cohort data with covariates to showcase conventional cohort EDA, additional predictive methods, and mean/polar plotting
    """)
    return


@app.cell
def _():
    import pytensor

    #set flag to hotfix open c++ errors
    pytensor.config.cxx = '/usr/bin/clang++'
    return


@app.cell
def _():
    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import xarray as xr

    from pymc_marketing import clv
    import pymc as pm
    from pymc_extras.prior import Prior

    # Plotting configuration
    az.style.use("arviz-darkgrid")
    plt.rcParams["figure.figsize"] = [12, 7]
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["figure.facecolor"] = "white"

    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo
    # magic command not supported in marimo; please file an issue to add support
    # %config InlineBackend.figure_format = "retina"
    return Prior, az, clv, np, pd, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load Data and EDA
    Adapt plots from sBG-Individual notebook here.
    """)
    return


@app.cell
def _(pd):
    # TODO: Need full dataset to T=13!
    dataset = pd.read_csv("https://raw.githubusercontent.com/pymc-labs/pymc-marketing/refs/heads/main/data/sbg_reg_hi_cohorts.csv")
    dataset.describe()
    return (dataset,)


@app.cell
def _(dataset):
    # Add a barplot of raw data here
    dataset[["recency", "cohort"]].value_counts().plot(kind='bar');
    # Add aggregation code to re-create data from the research paper in the next cell
    # move imported dataframe generation code for all datasets into scripts/generate_data/
    return


@app.cell
def _(pd):
    # Data from research paper
    df = pd.DataFrame(
        {
            "regular": [
                100.0,
                63.1,
                46.8,
                38.2,
                32.6,
                28.9,
                26.2,
                24.1,
                22.3,
                20.7,
                19.4,
                18.3,
                17.3,
            ],
            "highend": [
                100.0,
                86.9,
                74.3,
                65.3,
                59.3,
                55.1,
                51.7,
                49.1,
                46.8,
                44.5,
                42.7,
                40.9,
                39.4,
            ],
        }
    )
    df
    return (df,)


@app.cell
def _(dataset, pd):
    # Calculate retention percentages for each cohort and time period
    # Retention at time t = % of customers with recency >= t

    # Group by cohort to get total counts
    cohort_totals = dataset.groupby('cohort')['customer_id'].count()

    # Create a list to store results for each time period
    results = []

    # For each time period from 0 to T (8 in this case)
    for t_ in range(dataset['T'].max()):
        row_data = {'time': t_}

        for cohort in ['regular', 'highend']:
            cohort_data = dataset[dataset['cohort'] == cohort]
            total_customers = len(cohort_data)

            if t_ == 0:
                # At time 0, 100% retention
                retention_pct = 100.0
            else:
                # Count customers who survived at least to time t (recency >= t)
                survived = len(cohort_data[cohort_data['recency'] >= t_])
                retention_pct = (survived / total_customers) * 100

            row_data[cohort] = retention_pct

        results.append(row_data)

    # Convert to DataFrame
    df_prcnt = pd.DataFrame(results)
    #df_prcnt = df_prcnt.set_index('time')[['regular', 'highend']]

    df_prcnt
    return


@app.cell
def _(df, pd, t):
    # Assume a base population size (e.g., 1000 customers per cohort)
    base_population = 1000

    # Get T from the length of df (minus 1 since index starts at 0)
    T = len(df) - 1

    rows_ = []
    customer_id_ = 1

    for cohort_ in ['regular', 'highend']:
        # Get retention percentages for this cohort
        retention = df[cohort_].values

        # Calculate number of customers at each time point
        customers_at_t = (retention / 100) * base_population

        # Calculate churns at each time period
        for t_ in range(len(customers_at_t) - 1):
            # Number of customers who churned at time t
            churned = customers_at_t[t_] - customers_at_t[t + 1]
            churned_count = int(round(churned))

            # Create rows for customers who churned at time t+1
            # (they have recency = t+1)
            for _ in range(churned_count):
                rows_.append({
                    'customer_id': customer_id_,
                    'recency': t_ + 1,
                    'T': T,
                    'cohort': cohort_
                })
                customer_id_ += 1

        # Remaining customers at final time period
        # (they have recency = T)
        remaining = int(round(customers_at_t[-1]))
        for _ in range(remaining):
            rows_.append({
                'customer_id': customer_id_,
                'recency': T,
                'T': T,
                'cohort': cohort_
            })
            customer_id_ += 1

    # Convert to DataFrame
    dataset_reconstructed = pd.DataFrame(rows_)
    dataset_reconstructed
    return


@app.cell
def _():
    # Add code here to format CSV data into `df`. Could be worth adding as a utility function.
    return


@app.cell
def _():
    # Add code here to transform `df` back into CSV data. Will be added to scripts folder in notebook PR.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Fitting
    """)
    return


@app.cell
def _(Prior, clv, dataset):
    sbg = clv.ShiftedBetaGeoModel(
        data=dataset,
        model_config = {
            "alpha": Prior("HalfFlat",dims="cohort"),
            "beta": Prior("HalfFlat",dims="cohort")
        }
    )
    sbg.build_model()
    sbg.fit(fit_method='map')
    sbg.fit_summary()
    return (sbg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Differential Evolution Metropolis
    NUTS defaults to a compound sampler since this is a discrete sampling distribution. It still works great for a dataset of this size, but with large numbers of customers and cohorts, [`DEMetropolisZ`](https://www.pymc.io/projects/docs/en/v5.6.1/api/generated/pymc.DEMetropolisZ.html) may be more performant.
    """)
    return


@app.cell
def _(clv, dataset):
    sbg_1 = clv.ShiftedBetaGeoModel(data=dataset)
    sbg_1.build_model()
    sbg_1.fit(fit_method='demz', tune=3000, draws=3000)
    sbg_1.thin_fit_result(keep_every=2)
    sbg_1.fit_summary(var_names=['alpha', 'beta'])  #'demz' needs a lot of tunes/draws
    return (sbg_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Contrasting posterior inferences with the repo MLE estimates
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The sBG model has 2 population parameters of interest: `alpha` and `beta`.
    These parameters define the population distribution of the latent churn rate distribution `theta`.
    The larger the values of `alpha` and `beta`, the more homogenous the churn rates across different customers.

    The ratio of `alpha` to `beta` tells us the expected churn rates. If `alpha/beta == 0.1`, we expect the average customer to have a `0.1` probability of churning between each time period.

    The model fitting agrees with the Maximum Likelihood estimates described in the original paper.
    In addition, MCMC sampling, gives us useful information about the uncertainty of the fits.
    """)
    return


@app.cell
def _(az, sbg_1):
    az.plot_trace(sbg_1.idata, var_names=['alpha', 'beta'])
    return


@app.cell
def _(az, sbg_1):
    ref_val = {'highend': [0.668, 3.806], 'regular': [0.704, 1.182]}
    ref_val_map = {}
    for _cohort, (a_ref, b_ref) in ref_val.items():
        ref_val_map[f'alpha\n{_cohort}'] = a_ref
        ref_val_map[f'beta\n{_cohort}'] = b_ref
    az.plot_posterior(sbg_1.idata, var_names=['alpha', 'beta'], ref_val=ref_val_map)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Below plotting code cells were lifted directly from sBG-Individual notebook and require adaptation:**
    """)
    return


@app.cell
def _():
    # obs = df["highend"]/100

    # plt.plot(survive_pred,color="b", label="predicted")
    # plt.plot(obs,color="k", label="observed")
    # plt.ylabel("Survival Rate")
    # plt.legend()
    # plt.title("High-End Customers MAP")

    # plt.plot(reg_retention_cohort[:8],color="b", label="predicted")
    # plt.plot(retention_rate_regular_obs[:8],color="k", label="observed")
    # plt.ylabel("Retention Rate")
    # plt.legend()
    # plt.title("Regular Customers MCMC Estimated with Cohorts")
    return


@app.cell
def _():
    # az.plot_hdi(
    #     weeks_,
    #     hi_retention.mean("customer_id"),
    #     hdi_prob=0.95,
    #     color="C0",
    #     fill_kwargs={"label": "high end"},
    # )
    # az.plot_hdi(
    #     weeks_,
    #     lo_retention.mean("customer_id"),
    #     hdi_prob=0.95,
    #     color="C1",
    #     fill_kwargs={"label": "regular"},
    # )

    # plt.plot(weeks_, retention_rate_highend_obs, color="k", label="observed")
    # plt.plot(weeks_, retention_rate_regular_obs, color="k")

    # plt.axvline(7, ls="--", color="k")
    # plt.ylim([0.5, 1.05])
    # plt.ylabel("Retention Rate")
    # plt.legend()
    # plt.title("Figure 5");
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Predictive Methods
    Retention Rate
    """)
    return


@app.cell
def _(dataset, sbg_1):
    _pred_data = dataset.query('recency==T')
    pred_cohort_retention = sbg_1.expected_retention_rate(_pred_data, future_t=0).mean(('chain', 'draw'))
    pred_cohort_retention.to_dataframe(name='retention').reset_index()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Below cell is a WIP for plotting a time period trend for model evaluation for Probability Alive.
    """)
    return


@app.cell
def _(np, pd, sbg_1):
    max_T = 12
    cohort_names = np.array(['regular', 'highend'])
    cohorts_covar = np.array([0, 1])
    T_rng = np.arange(1, max_T + 1, 1)
    _pred_data = pd.DataFrame({'customer_id': np.arange(1, 1 + max_T * 2, 1), 'T': np.repeat(T_rng, len(cohort_names)), 'cohort': np.tile(cohort_names, max_T), 'covar_cohort': np.tile(cohorts_covar, max_T)})
    retention_array = _pred_data.query('T <=12').copy()
    sbg_1.expected_probability_alive(data=retention_array, future_t=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Figures 4 and 5 show the predicted average churn and retention trends for the two groups.
    We can see that predictions nicely match the observed data (black line), even when extrapolating into the time periods that were held-out when fitting the model.

    The plots also highlight an interesting implication from the model:
    the retention rates are expected to increase over time, as the more precarious customers gradually drop out. This is a direct consequence of modelling individual churn rates as being fixed over time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    These additional predictive methods are described in https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_contractual_mksc_10.pdf
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *Expected Residual Lifetimes*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Expected Retention Elasticity
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simulate Data from Follow-Up Paper
    Adapt below code to create time-period cohorts with highend & regular covariate customers mixed into each.
    """)
    return


@app.cell
def _(np, pd):
    cohort_counts = {2001: {2001: 10000, 2002: 8000, 2003: 6480, 2004: 5307, 2005: 4391}, 2002: {2002: 10000, 2003: 8000, 2004: 6480, 2005: 5307}, 2003: {2003: 10000, 2004: 8000, 2005: 6480}, 2004: {2004: 10000, 2005: 8000}, 2005: {2005: 10000}}
    case2 = {2003: {2003: 10000, 2004: 8000, 2005: 7600, 2006: 7383, 2007: 7235}, 2004: {2004: 10000, 2005: 8000, 2006: 7600, 2007: 7383}, 2005: {2005: 10000, 2006: 8000, 2007: 7600}, 2006: {2006: 10000, 2007: 8000}, 2007: {2007: 10000}}
    rows = []
    for _cohort, year_counts in cohort_counts.items():
        years_sorted = sorted(year_counts)
        ages = [y - _cohort for y in years_sorted]
        S = [year_counts[y] for y in years_sorted]
        last_t = ages[-1]
        for t in range(len(S) - 1):
            count = S[t] - S[t + 1]
            if count > 0:
                rows.append({'cohort': _cohort, 'recency': t, 'T': last_t, 'count': count})
        if S[-1] > 0:
            rows.append({'cohort': _cohort, 'recency': last_t, 'T': last_t, 'count': S[-1]})
    counts_df = pd.DataFrame(rows)
    t_churn_array = np.repeat(counts_df['recency'].to_numpy(), counts_df['count'].to_numpy())
    T_array = np.repeat(counts_df['T'].to_numpy(), counts_df['count'].to_numpy())
    cohort_array = np.repeat(counts_df['cohort'].to_numpy(), counts_df['count'].to_numpy())
    customer_id = np.arange(t_churn_array.size)
    case_1_2_df = pd.DataFrame({'customer_id': customer_id + 1, 'recency': t_churn_array + 1, 'T': T_array + 1, 'cohort': cohort_array}).astype({'customer_id': int, 'recency': int, 'T': int, 'cohort': str})
    return case_1_2_df, t


@app.cell
def _(Prior, case_1_2_df, clv):
    fit_case_1_2_df = case_1_2_df.query("T>1")

    sbg_case1_2 = clv.ShiftedBetaGeoModel(
        data = fit_case_1_2_df,
        model_config = {
            "alpha": Prior("HalfFlat", dims="cohort"),
            "beta": Prior("HalfFlat", dims="cohort"),
        }
    )
    sbg_case1_2.fit(method="map")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Latent Dropout Distribution for Customer Population
    Distribution mean and polarization index plotting will be added in future PR rather than visualizations of the latent cohort dropout distributions, but for illustratrive examples in tutorial notebook, latent distribution plotting code is provided below:
    """)
    return


@app.cell
def _(az, np, sbg, xr):
    # Extract alpha and beta from fit results
    alpha = sbg.fit_result["alpha"]
    beta = sbg.fit_result["beta"]

    # Generate 100 random samples from Beta distribution for each cohort
    rng = np.random.default_rng(42)
    n_samples = 100

    cohorts = alpha.coords['cohort'].values
    dropout_samples = np.array([
        rng.beta(
            alpha.sel(cohort=c).values.item(),  # Use .item() to get scalar
            beta.sel(cohort=c).values.item(),   # Use .item() to get scalar
            size=n_samples
        )
        for c in cohorts
    ]).T  # Transpose to get (samples, cohorts) shape

    # Create xarray DataArray with chain, draw, and cohort dimensions
    # Reshape to add chain dimension (1 chain, n_samples draws)
    dropout = xr.DataArray(
        dropout_samples[np.newaxis, :, :],  # Add chain dimension
        dims=("chain", "draw", "cohort"),
        coords={
            "chain": [0],
            "draw": np.arange(n_samples),
            "cohort": cohorts,
        },
        name="dropout",
    )

    # Convert to InferenceData
    dropout_idata = az.convert_to_inference_data(dropout)

    # Plot with arviz
    axes = az.plot_forest(
        dropout_idata,
        kind='ridgeplot',
        combined=True,
        colors='white',
        ridgeplot_overlap=1,
        ridgeplot_truncate=False,
        ridgeplot_quantiles=[.25,.5,.75],
        figsize=(7,7),
    )
    axes[0].set_title("Dropout Distributions by Cohort")
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext watermark
    # magic command not supported in marimo; please file an issue to add support
    # %watermark -n -u -v -iv -w -p pymc,pytensor
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
