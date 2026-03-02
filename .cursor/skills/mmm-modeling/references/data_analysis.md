# Data Analysis and Preparation for MMM

## Table of Contents
- [Data Format Requirements](#data-format-requirements)
- [Single-Geo Data Format](#single-geo-data-format)
- [Multidimensional Data Format](#multidimensional-data-format)
- [Channel Naming and Mapping](#channel-naming-and-mapping)
- [Exploratory Analysis](#exploratory-analysis)
- [Data Splitting Philosophy](#data-splitting-philosophy)
- [Control Variables](#control-variables)
- [Spend Share Computation](#spend-share-computation)

## Data Format Requirements

The MMM requires a pandas DataFrame with:

| Column Type | Required | Description |
|-------------|----------|-------------|
| **Date** | Yes | Datetime column (weekly or daily granularity) |
| **Channels** | Yes | One column per media channel (spend or impressions) |
| **Target** | Yes | Response variable (sales, conversions, etc.) |
| **Controls** | No | Holidays, events, macro indicators |
| **Geo/Region** | No | Required for multidimensional models |

## Single-Geo Data Format

Wide format with one row per time period:

```python
import pandas as pd

data_df = pd.read_csv("data.csv")

date_column = "date"
target_column = "sales"
channel_columns = ["tv", "radio", "social", "online_display"]
control_columns = [col for col in data_df.columns if "holiday_" in col]

data_df[date_column] = pd.to_datetime(data_df[date_column])

X = data_df.drop(columns=[target_column])
y = data_df[target_column]
```

## Multidimensional Data Format

Long format with one row per (date, geo) pair:

```python
data_df = pd.read_csv("data.csv", parse_dates=["date"])
# Columns: date, geo, x1, x2, event_1, event_2, y
# geo values: 'geo_a', 'geo_b', ...

X = data_df.drop(columns=["y"])
y = data_df["y"]
```

The `geo` column values become coordinates in the model. The MMM automatically detects the geo structure when `dims=("geo",)` is specified.

## Channel Naming and Mapping

Raw channel names are often cryptic. Map them to readable names early:

```python
channel_mapping = {
    "mdsp_dm": "Direct Mail",
    "mdsp_vidtr": "TV",
    "mdsp_so": "Social Media",
    "mdsp_on": "Online Display",
}
channel_columns = sorted(list(channel_mapping.values()))

data_df = data_df.rename(columns=channel_mapping)
```

## Exploratory Analysis

### Target Time Series

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots()
sns.lineplot(data=data_df, x=date_column, y=target_column, color="black", ax=ax)
ax.set(xlabel="date", ylabel=target_column)
ax.set_title("Target Over Time", fontsize=18, fontweight="bold")
```

### Total Media Spend by Channel

```python
fig, ax = plt.subplots()
(
    data_df.melt(value_vars=channel_columns, var_name="channel", value_name="spend")
    .groupby("channel")
    .agg({"spend": "sum"})
    .sort_values(by="spend")
    .plot.barh(ax=ax)
)
ax.set(xlabel="Spend", ylabel="Channel")
ax.set_title("Total Media Spend", fontsize=18, fontweight="bold")
```

### Media Spend Over Time

```python
fig, ax = plt.subplots()
data_df.set_index(date_column)[channel_columns].plot(ax=ax)
ax.legend(title="Channel", fontsize=12)
ax.set(xlabel="Date", ylabel="Spend")
ax.set_title("Media Spend Over Time", fontsize=18, fontweight="bold")
```

### Channel-Target Correlation

```python
n_channels = len(channel_columns)

fig, axes = plt.subplots(
    nrows=n_channels, ncols=1,
    figsize=(15, 3 * n_channels), sharex=True, layout="constrained",
)

for i, channel in enumerate(channel_columns):
    ax = axes[i]
    ax_twin = ax.twinx()
    sns.lineplot(data=data_df, x=date_column, y=channel, color=f"C{i}", ax=ax)
    sns.lineplot(data=data_df, x=date_column, y=target_column, color="black", ax=ax_twin)
    correlation = data_df[[channel, target_column]].corr().iloc[0, 1]
    ax_twin.grid(None)
    ax.set(title=f"{channel} (Correlation: {correlation:.2f})")
```

### Multidimensional EDA (Geo-Level)

```python
import seaborn as sns

g = sns.relplot(
    data=data_df, x="date", y="y",
    color="black", col="geo", col_wrap=1,
    kind="line", height=4, aspect=3,
)
g.figure.suptitle("Target by Geo", fontsize=16, fontweight="bold", y=1.03)
```

## Data Splitting Philosophy

**The final model is always fit on the full dataset.** Train/test splits are only for assessing model stability via time-slice cross-validation, never for the production model fit.

The reasoning: MMMs are typically data-scarce (1-3 years of weekly data = 50-150 observations). Holding out data for a final test set wastes precious information. Instead, use `TimeSliceCrossValidator` to systematically evaluate out-of-sample performance and parameter stability across expanding time windows, then fit the production model on all available data.

See [model_fit.md](model_fit.md) for the full time-slice cross-validation workflow.

## Control Variables

Control variables capture known effects that are not media-driven:

```python
# Holidays (binary indicators)
control_columns = [col for col in data_df.columns if "hldy_" in col or "event_" in col]

# Macro indicators (continuous)
# e.g., temperature, unemployment rate, competitor activity
```

Controls enter the model linearly with prior `gamma_control`:

```python
from pymc_extras.prior import Prior

model_config = {
    "gamma_control": Prior("Normal", mu=0, sigma=1, dims="control"),
}
```

## Spend Share Computation

Spend shares inform the prior on `saturation_beta` -- channels with higher spend get proportionally wider priors:

```python
import numpy as np

spend_shares = (
    data_df.melt(value_vars=channel_columns, var_name="channel", value_name="spend")
    .groupby("channel", as_index=False)
    .agg({"spend": "sum"})
    .sort_values(by="channel")
    .assign(spend_share=lambda x: x["spend"] / x["spend"].sum())["spend_share"]
    .to_numpy()
)

model_config = {
    "saturation_beta": Prior("HalfNormal", sigma=spend_shares, dims="channel"),
}
```

This ensures channels with higher historical spend are allowed larger contribution effects a priori.

---

## See Also

- [model_specification.md](model_specification.md) -- MMM constructor and prior configuration
- [model_fit.md](model_fit.md) -- Fitting workflow and time-slice cross-validation
