import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from pymc_extras.prior import Prior

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.multidimensional import MMM
from pymc_marketing.paths import data_dir

# global parameters

seed = 12345
geo_index = ("geo_a", "geo_b")
chain = 0


def get_dates(
    start_date="2022-06-06", end_date="2025-06-16"
) -> tuple[pd.DatetimeIndex, int]:
    """Utility to generate date range for the synthetic data.

    Parameters
    ----------
    start_date : str, optional
        Start date in YYYY-MM-DD format, by default "2022-06-06"
    end_date : str, optional
        End date in YYYY-MM-DD format, by default "2025-06-16"

    Returns
    -------
    pd.DatetimeIndex
        pandas DatetimeIndex with weekly dates
    int
        number of dates

    Raises
    ------
    ValueError
        If end_date is before start_date or dates are invalid
    """
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        if end <= start:
            raise ValueError("End date must be after start date")
        dates = pd.date_range(start=start, end=end, freq="W-MON")
        if len(dates) < 1:
            raise ValueError("Date range must include at least one week")
        return dates, len(dates)
    except ValueError as e:
        if "cannot convert" in str(e).lower():
            raise ValueError("Dates must be in YYYY-MM-DD format") from e
        raise


def generate_channel_one(n_dates: int, rng: np.random.Generator) -> np.ndarray:
    """Generate channel one data with a consistent 'always on' type channel with AR(1) structure.

    Parameters
    ----------
    n_dates : int
        Number of dates to generate data for
    rng : np.random.Generator
        Random number generator instance

    Returns
    -------
    np.ndarray
        Channel one data with shape (n_dates * 2,) representing two geos
    """
    # build a covariance matrix for the between geo-correlation
    variances = np.array([[1500, 0], [0, 1500]])
    corr = np.array([[1, 0.25], [0.25, 1]])
    cov = variances @ corr @ variances

    # convert it into an AR(1) process
    innovations = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n_dates)
    channel_one = np.zeros((n_dates, 2))
    channel_one[0] = innovations[0]

    phi = 0.8

    for t in range(1, n_dates):
        channel_one[t] = phi * channel_one[t - 1] + innovations[t]

    channel_one = channel_one + np.array([[5000, 6000]])

    # Replace negative values with 0
    channel_one = np.maximum(channel_one, 0)

    # add a few off periods to make it easier to
    # identify the media vs intercept effect
    channel_one[30:40, :] = 0
    channel_one[70:80, :] = 0
    channel_one[115:125, :] = 0
    channel_one[130:140, :] = 0
    channel_one[150:160, :] = 0
    channel_one[190:200, :] = 0

    channel_one = channel_one.flatten()

    return channel_one


def generate_channel_two(n_dates: int, rng: np.random.Generator) -> np.ndarray:
    """Generate channel two data with a bursty pattern and long off periods.

    Parameters
    ----------
    n_dates : int
        Number of dates to generate data for
    rng : np.random.Generator
        Random number generator instance

    Returns
    -------
    np.ndarray
        Channel two data with shape (n_dates * 2,) representing two geos
    """
    # build a covariance matrix for the between geo-correlation
    variances = np.array([[2000, 0], [0, 2000]])
    corr = np.array([[1, 0.25], [0.25, 1]])
    cov = variances @ corr @ variances

    channel_two = rng.multivariate_normal(mean=[6000, 7000], cov=cov, size=n_dates)

    # add several off periods to give it the on/off look
    channel_two[0:30, 0] = 0
    channel_two[60:90, 0] = 0
    channel_two[120:150, 0] = 0
    channel_two[5:35, 1] = 0
    channel_two[65:95, 1] = 0
    channel_two[125:155, 1] = 0

    channel_two = channel_two.flatten()

    # Replace negative values with 0
    channel_two = np.maximum(channel_two, 0)

    return channel_two


def generate_dataframe(
    dates: pd.DatetimeIndex, geo_index: tuple[str, ...], rng: np.random.Generator
) -> pd.DataFrame:
    """Generate a DataFrame with channels and events.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Array of dates
    geo_index : tuple[str, ...]
        Array of geo locations
    rng : np.random.Generator
        Random number generator instance

    Returns
    -------
    pd.DataFrame
        DataFrame containing channels, events and target variable with a multi-index
        of dates and geos
    """
    # Get channel data
    channel_one = generate_channel_one(len(dates), rng)
    channel_two = generate_channel_two(len(dates), rng)

    # Create multi-index DataFrame
    index = pd.MultiIndex.from_product([dates, geo_index], names=["date", "geo"])

    # Create DataFrame with channels
    df = pd.DataFrame({"x1": channel_one, "x2": channel_two}, index=index)

    df["event_1"] = 0
    df["event_2"] = 0
    # Random dates for event_1
    random_dates = rng.choice(dates, size=3, replace=False)
    for date in random_dates:
        df.loc[(date, slice(None)), "event_1"] = 1

    # Random dates for event_2
    random_dates = rng.choice(dates, size=3, replace=False)
    for date in random_dates:
        df.loc[(date, slice(None)), "event_2"] = 1

    # build default target variable. This will be rewritten over later
    # but we need it to generate target scales.
    df = df.reset_index()
    df["y"] = rng.normal(loc=5000, scale=500, size=len(df))

    return df


def build_mmm_model(df: pd.DataFrame) -> MMM:
    """Build and return a Marketing Mix Model object.
    We'll sample the prior from this model to generate the synthetic data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing channel, event and target data

    Returns
    -------
    MMM
        Marketing Mix Model object with model built and ready for inference
    """
    saturation = LogisticSaturation(
        priors={
            "beta": Prior(
                "Normal",
                mu=Prior("Normal", mu=-1.5, sigma=0.25, dims=("channel")),
                sigma=Prior("Exponential", scale=0.25, dims=("channel")),
                dims=("channel", "geo"),
                transform="exp",
                centered=False,
            ),
            "lam": Prior(
                "Gamma",
                mu=0.25,
                sigma=0.15,
                dims=("channel"),
            ),
        }
    )

    adstock = GeometricAdstock(
        priors={"alpha": Prior("Beta", alpha=2, beta=5, dims=("geo", "channel"))},
        l_max=8,
    )

    model_config = {
        "intercept": Prior("Normal", mu=0.5, sigma=0.2, dims="geo"),
        "gamma_control": Prior("Normal", mu=0, sigma=0.5, dims="control"),
        "gamma_fourier": Prior(
            "Normal",
            mu=0,
            sigma=Prior("HalfNormal", sigma=0.2),
            dims=("geo", "fourier_mode"),
            centered=False,
        ),
        "likelihood": Prior(
            "TruncatedNormal",
            lower=0,
            sigma=Prior("HalfNormal", sigma=0.5),
            dims=("date", "geo"),
        ),
    }

    mmm = MMM(
        date_column="date",
        target_column="y",
        channel_columns=["x1", "x2"],
        control_columns=["event_1", "event_2"],
        dims=("geo",),
        scaling={
            "channel": {"method": "max", "dims": ()},
            "target": {"method": "max", "dims": ()},
        },
        adstock=adstock,
        saturation=saturation,
        yearly_seasonality=2,
        model_config=model_config,
    )

    x_train = df.drop(columns=["y"])
    y_train = df["y"]
    mmm.build_model(X=x_train, y=y_train)

    mmm.add_original_scale_contribution_variable(
        var=[
            "channel_contribution",
            "control_contribution",
            "intercept_contribution",
            "yearly_seasonality_contribution",
            "y",
        ]
    )

    return mmm


def generate_synthetic_data(
    mmm: MMM, df: pd.DataFrame, draw_num: int, seed: int = seed
) -> tuple[pd.DataFrame, xr.DataArray]:
    """Generate synthetic data using the MMM's prior predictive distribution.

    Parameters
    ----------
    mmm : MMM
        The Marketing Mix Model instance
    df : pd.DataFrame
        Input dataframe containing features
    draw_num : int
        Which draw from the prior predictive to use
    seed : int, optional
        Random seed for sampling, by default global seed value

    Returns
    -------
    pd.DataFrame
        DataFrame with synthetic target variable
    xr.DataArray
        xarray of the true parameters used to generate the data

    Raises
    ------
    ValueError
        If draw_num is not available in the prior predictive samples
    """
    with mmm.model:
        prior = pm.sample_prior_predictive(random_seed=seed)

    try:
        prior_selection = prior.prior.sel(chain=chain, draw=draw_num)
    except KeyError:
        max_draw = prior.prior.sizes["draw"] - 1
        raise ValueError(
            f"Draw {draw_num} not available. Available draws are 0 to {max_draw}"
        )

    # extract a dictionary of the prior parameters
    true_params = prior_selection[
        ["saturation_beta", "saturation_lam", "adstock_alpha"]
    ]

    target = prior_selection.y_original_scale.to_dataframe()
    target = target.drop(columns=["chain", "draw"])

    df = df.set_index(["date", "geo"]).join(target).reset_index()
    df = df.drop(columns=["y"])
    df = df.rename(columns={"y_original_scale": "y"})

    return df, true_params


def main(
    output_dir: str = "data",
    draw_num: int = 8,
    start_date: str = "2022-06-06",
    end_date: str = "2025-06-16",
) -> None:
    """Generate synthetic data and save artifacts.

    Parameters
    ----------
    output_dir : str, optional
        Directory where output files will be saved, by default "data"
    draw_num : int, optional
        Which draw from the prior predictive to use, by default 8
    start_date : str, optional
        Start date in YYYY-MM-DD format, by default "2022-06-06"
    end_date : str, optional
        End date in YYYY-MM-DD format, by default "2025-06-16"
    """
    # Set up random number generator
    rng = np.random.default_rng(seed)

    # Generate dates
    dates, _ = get_dates(start_date, end_date)

    # Generate initial dataframe
    df = generate_dataframe(dates, geo_index, rng)

    # Build MMM model
    mmm = build_mmm_model(df)

    # Generate synthetic data and true parameters
    df_synthetic, true_params = generate_synthetic_data(mmm, df, draw_num, seed)

    # Save dataframe as CSV
    csv_path = data_dir / "mmm_multidimensional_example.csv"
    df_synthetic.to_csv(csv_path, index=False)
    print(f"Saved synthetic data to {csv_path}")

    # Save true parameters as netCDF
    netcdf_path = data_dir / "mmm_multidimensional_example_true_parameters.nc"
    true_params.to_netcdf(netcdf_path)
    print(f"Saved true parameters to {netcdf_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic MMM data and true parameters"
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to save output files (default: data)",
    )
    parser.add_argument(
        "--draw",
        type=int,
        default=8,
        help="Which draw from the prior predictive to use (default: 8)",
    )
    parser.add_argument(
        "--start-date",
        default="2022-06-06",
        help="Start date in YYYY-MM-DD format (default: 2022-06-06)",
    )
    parser.add_argument(
        "--end-date",
        default="2025-06-16",
        help="End date in YYYY-MM-DD format (default: 2025-06-16)",
    )

    args = parser.parse_args()
    main(args.output_dir, args.draw, args.start_date, args.end_date)
