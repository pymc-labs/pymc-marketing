import pandas as pd

from pymc_marketing.clv.utils import clv_summary

df = pd.read_csv(
    "tests/clv/datasets/cdnow_transactions.csv",
)
summary_df = clv_summary(df, "id", "date", "spent", datetime_format="%Y%m%d")

summary_df.to_csv("tests/clv/datasets/cdnow_sample.csv", index=False)
