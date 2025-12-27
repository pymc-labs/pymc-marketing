import numpy as np
import pandas as pd

# Summary data from original research paper for ShiftedBetaGeoModel
# From Table 1 in https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_jim_07.pdf
sbg_research_data = pd.DataFrame(
    {
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
    }
)


def generate_sbg_data(
    n_customers: int,
    n_time_periods: int,
    survival_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate individual-level customer churn data from aggregate percentage alive data.

    Parameters
    ----------
    n_customers : int
        Number of customers to simulate for each cohort
    n_time_periods : int
        Number of time periods to include from the survival data
    survival_data : pd.DataFrame, optional
        DataFrame with columns representing cohorts and values as percentage alive.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: recency, T, cohort
        Contains individual customer data for all cohorts in the survival data
    """

    def _individual_data_from_percentage_alive(percentage_alive, initial_customers):
        """Convert percentage alive data to individual churn times."""
        n_alive = np.asarray(percentage_alive / 100 * initial_customers, dtype=int)

        died_at = np.zeros((initial_customers,), dtype=int)
        counter = 0
        for t, diff in enumerate((n_alive[:-1] - n_alive[1:]), start=1):
            died_at[counter : counter + diff] = t
            counter += diff

        censoring_t = t + 1
        died_at[counter:] = censoring_t

        return died_at

    # Truncate data to requested number of time periods
    truncated_df = survival_data[:n_time_periods]

    # Generate individual churn data for each cohort
    datasets = []
    for cohort_name in truncated_df.columns:
        churn_data = _individual_data_from_percentage_alive(
            truncated_df[cohort_name], n_customers
        )

        dataset = pd.DataFrame(
            {
                "recency": churn_data,
                "T": n_time_periods,
                "cohort": cohort_name,
            }
        )
        datasets.append(dataset)

    # Combine all cohorts into a single dataset
    combined_dataset = pd.concat(datasets, ignore_index=True)

    # Create customer_id column from index
    combined_dataset["customer_id"] = combined_dataset.index + 1

    return combined_dataset[["customer_id", "recency", "T", "cohort"]]
