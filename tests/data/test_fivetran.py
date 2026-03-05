#   Copyright 2022 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import pandas as pd
import pytest

from pymc_marketing.data.fivetran import (
    process_fivetran_ad_reporting,
    process_fivetran_shopify_unique_orders,
)


@pytest.fixture
def example_account_report_df() -> pd.DataFrame:
    # Schema: https://fivetran.github.io/dbt_ad_reporting/#!/model/model.ad_reporting.ad_reporting__account_report#details
    data = [
        # 2024-01-01
        {
            "source_relation": "facebook_ads",
            "date_day": "2024-01-01",
            "platform": "facebook_ads",
            "account_id": "acc_fb_1",
            "account_name": "FB Account 1",
            "clicks": 100,
            "impressions": 1000,
            "spend": 10.0,
            "conversions": 2.0,
            "conversions_value": 50.0,
        },
        {
            "source_relation": "facebook_ads",
            "date_day": "2024-01-01",
            "platform": "facebook_ads",
            "account_id": "acc_fb_2",
            "account_name": "FB Account 2",
            "clicks": 50,
            "impressions": 500,
            "spend": 20.0,
            "conversions": 1.0,
            "conversions_value": 20.0,
        },
        {
            "source_relation": "google_ads",
            "date_day": "2024-01-01",
            "platform": "google_ads",
            "account_id": "acc_gg_1",
            "account_name": "GG Account 1",
            "clicks": 40,
            "impressions": 400,
            "spend": 5.0,
            "conversions": 0.5,
            "conversions_value": 10.0,
        },
        {
            "source_relation": "google_ads",
            "date_day": "2024-01-01",
            "platform": "google_ads",
            "account_id": "acc_gg_2",
            "account_name": "GG Account 2",
            "clicks": 60,
            "impressions": 600,
            "spend": 5.0,
            "conversions": 0.7,
            "conversions_value": 15.0,
        },
        # 2024-01-02 (no google rows)
        {
            "source_relation": "facebook_ads",
            "date_day": "2024-01-02",
            "platform": "facebook_ads",
            "account_id": "acc_fb_1",
            "account_name": "FB Account 1",
            "clicks": 30,
            "impressions": 300,
            "spend": 7.0,
            "conversions": 0.3,
            "conversions_value": 7.0,
        },
        {
            "source_relation": "facebook_ads",
            "date_day": "2024-01-02",
            "platform": "facebook_ads",
            "account_id": "acc_fb_2",
            "account_name": "FB Account 2",
            "clicks": 20,
            "impressions": 200,
            "spend": 3.0,
            "conversions": 0.2,
            "conversions_value": 5.0,
        },
    ]
    df = pd.DataFrame(data)
    # Convert string dates to proper datetime objects for multi-backend support
    df["date_day"] = pd.to_datetime(df["date_day"])
    return df


@pytest.fixture
def example_campaign_report_df() -> pd.DataFrame:
    # Schema: https://fivetran.github.io/dbt_ad_reporting/#!/model/model.ad_reporting.ad_reporting__campaign_report
    data = [
        # 2024-01-01
        {
            "source_relation": "facebook_ads",
            "date_day": "2024-01-01",
            "platform": "facebook_ads",
            "account_id": "acc_fb_1",
            "campaign_id": "camp_fb_1",
            "campaign_name": "Campaign FB 1",
            "clicks": 100,
            "impressions": 1000,
            "spend": 10.0,
            "conversions": 2.0,
            "conversions_value": 50.0,
        },
        {
            "source_relation": "facebook_ads",
            "date_day": "2024-01-01",
            "platform": "facebook_ads",
            "account_id": "acc_fb_1",
            "campaign_id": "camp_fb_2",
            "campaign_name": "Campaign FB 2",
            "clicks": 50,
            "impressions": 500,
            "spend": 20.0,
            "conversions": 1.0,
            "conversions_value": 20.0,
        },
        {
            "source_relation": "google_ads",
            "date_day": "2024-01-01",
            "platform": "google_ads",
            "account_id": "acc_gg_1",
            "campaign_id": "camp_gg_1",
            "campaign_name": "Campaign GG 1",
            "clicks": 40,
            "impressions": 400,
            "spend": 5.0,
            "conversions": 0.5,
            "conversions_value": 10.0,
        },
        {
            "source_relation": "google_ads",
            "date_day": "2024-01-01",
            "platform": "google_ads",
            "account_id": "acc_gg_1",
            "campaign_id": "camp_gg_2",
            "campaign_name": "Campaign GG 2",
            "clicks": 60,
            "impressions": 600,
            "spend": 5.0,
            "conversions": 0.7,
            "conversions_value": 15.0,
        },
        # 2024-01-02 (no google rows)
        {
            "source_relation": "facebook_ads",
            "date_day": "2024-01-02",
            "platform": "facebook_ads",
            "account_id": "acc_fb_1",
            "campaign_id": "camp_fb_3",
            "campaign_name": "Campaign FB 3",
            "clicks": 30,
            "impressions": 300,
            "spend": 7.0,
            "conversions": 0.3,
            "conversions_value": 7.0,
        },
        {
            "source_relation": "facebook_ads",
            "date_day": "2024-01-02",
            "platform": "facebook_ads",
            "account_id": "acc_fb_1",
            "campaign_id": "camp_fb_4",
            "campaign_name": "Campaign FB 4",
            "clicks": 20,
            "impressions": 200,
            "spend": 3.0,
            "conversions": 0.2,
            "conversions_value": 5.0,
        },
    ]
    df = pd.DataFrame(data)
    # Convert string dates to proper datetime objects for multi-backend support
    df["date_day"] = pd.to_datetime(df["date_day"])
    return df


@pytest.fixture
def example_ad_report_df() -> pd.DataFrame:
    # Minimal columns following Fivetran dbt_ad_reporting ad_report schema
    # https://fivetran.github.io/dbt_ad_reporting/#!/model/model.ad_reporting.ad_reporting__ad_report
    data = [
        {
            "source_relation": "facebook_ads",
            "date_day": "2024-01-01",
            "platform": "facebook_ads",
            "account_id": "acc_fb_1",
            "account_name": "FB Account 1",
            "campaign_id": "camp_fb_1",
            "campaign_name": "Campaign FB 1",
            "ad_group_id": "adgrp_fb_1",
            "ad_group_name": "Ad Group FB 1",
            "ad_id": "ad_fb_1",
            "ad_name": "Ad FB 1",
            "clicks": 100,
            "impressions": 1000,
            "spend": 10.0,
            "conversions": 2.0,
            "conversions_value": 50.0,
        },
        {
            "source_relation": "facebook_ads",
            "date_day": "2024-01-01",
            "platform": "facebook_ads",
            "account_id": "acc_fb_1",
            "account_name": "FB Account 1",
            "campaign_id": "camp_fb_2",
            "campaign_name": "Campaign FB 2",
            "ad_group_id": "adgrp_fb_2",
            "ad_group_name": "Ad Group FB 2",
            "ad_id": "ad_fb_2",
            "ad_name": "Ad FB 2",
            "clicks": 50,
            "impressions": 500,
            "spend": 20.0,
            "conversions": 1.0,
            "conversions_value": 20.0,
        },
        {
            "source_relation": "google_ads",
            "date_day": "2024-01-01",
            "platform": "google_ads",
            "account_id": "acc_gg_1",
            "account_name": "GG Account 1",
            "campaign_id": "camp_gg_1",
            "campaign_name": "Campaign GG 1",
            "ad_group_id": "adgrp_gg_1",
            "ad_group_name": "Ad Group GG 1",
            "ad_id": "ad_gg_1",
            "ad_name": "Ad GG 1",
            "clicks": 40,
            "impressions": 400,
            "spend": 5.0,
            "conversions": 0.5,
            "conversions_value": 10.0,
        },
        {
            "source_relation": "google_ads",
            "date_day": "2024-01-01",
            "platform": "google_ads",
            "account_id": "acc_gg_1",
            "account_name": "GG Account 1",
            "campaign_id": "camp_gg_2",
            "campaign_name": "Campaign GG 2",
            "ad_group_id": "adgrp_gg_2",
            "ad_group_name": "Ad Group GG 2",
            "ad_id": "ad_gg_2",
            "ad_name": "Ad GG 2",
            "clicks": 60,
            "impressions": 600,
            "spend": 5.0,
            "conversions": 0.7,
            "conversions_value": 15.0,
        },
        {
            "source_relation": "facebook_ads",
            "date_day": "2024-01-02",
            "platform": "facebook_ads",
            "account_id": "acc_fb_1",
            "account_name": "FB Account 1",
            "campaign_id": "camp_fb_3",
            "campaign_name": "Campaign FB 3",
            "ad_group_id": "adgrp_fb_3",
            "ad_group_name": "Ad Group FB 3",
            "ad_id": "ad_fb_3",
            "ad_name": "Ad FB 3",
            "clicks": 30,
            "impressions": 300,
            "spend": 7.0,
            "conversions": 0.3,
            "conversions_value": 7.0,
        },
        {
            "source_relation": "facebook_ads",
            "date_day": "2024-01-02",
            "platform": "facebook_ads",
            "account_id": "acc_fb_1",
            "account_name": "FB Account 1",
            "campaign_id": "camp_fb_4",
            "campaign_name": "Campaign FB 4",
            "ad_group_id": "adgrp_fb_4",
            "ad_group_name": "Ad Group FB 4",
            "ad_id": "ad_fb_4",
            "ad_name": "Ad FB 4",
            "clicks": 20,
            "impressions": 200,
            "spend": 3.0,
            "conversions": 0.2,
            "conversions_value": 5.0,
        },
        # No google_ads record on 2024-01-02 to test fill_value behavior
    ]
    df = pd.DataFrame(data)
    # Convert string dates to proper datetime objects for multi-backend support
    df["date_day"] = pd.to_datetime(df["date_day"])
    return df


# ==================== Multi-backend tests (function calls) ====================


@pytest.mark.parametrize(
    "value_columns, expected_columns, expected_values",
    [
        (
            "spend",
            ["facebook_ads_spend", "google_ads_spend"],
            [[30.0, 10.0], [10.0, 0.0]],
        ),
        (
            ["spend", "impressions"],
            [
                "facebook_ads_spend",
                "google_ads_spend",
                "facebook_ads_impressions",
                "google_ads_impressions",
            ],
            [[30.0, 10.0, 1500.0, 1000.0], [10.0, 0.0, 500.0, 0.0]],
        ),
    ],
)
def test_ad_report_schema_multibackend(
    example_ad_report_df,
    backend_converter,
    value_columns,
    expected_columns,
    expected_values,
):
    """Test process_fivetran_ad_reporting with ad_report schema across all backends."""
    # Convert to target backend
    df_backend = backend_converter.to_backend(example_ad_report_df)

    # Call function
    result = process_fivetran_ad_reporting(
        df_backend, value_columns=value_columns, rename_date_to="date"
    )

    # Convert result back to pandas for assertions
    result_pd = backend_converter.to_pandas(result)

    # Build expected DataFrame
    expected = (
        pd.DataFrame(
            data=expected_values,
            columns=expected_columns,
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )
        .reset_index()
        .rename(columns={"index": "date"})
    )

    # Ensure required columns exist and match expected values
    assert set(["date", *expected_columns]).issubset(set(result_pd.columns))

    result_subset = result_pd[["date", *expected_columns]]
    pd.testing.assert_frame_equal(result_subset, expected, check_dtype=False)


@pytest.mark.parametrize(
    "value_columns, expected_columns, expected_values",
    [
        (
            "spend",
            ["facebook_ads_spend", "google_ads_spend"],
            [[30.0, 10.0], [10.0, 0.0]],
        ),
        (
            ["spend", "impressions"],
            [
                "facebook_ads_spend",
                "google_ads_spend",
                "facebook_ads_impressions",
                "google_ads_impressions",
            ],
            [[30.0, 10.0, 1500.0, 1000.0], [10.0, 0.0, 500.0, 0.0]],
        ),
    ],
)
def test_account_report_schema_multibackend(
    example_account_report_df,
    backend_converter,
    value_columns,
    expected_columns,
    expected_values,
):
    """Test process_fivetran_ad_reporting with account_report schema across all backends."""
    # Convert to target backend
    df_backend = backend_converter.to_backend(example_account_report_df)

    # Call function
    result = process_fivetran_ad_reporting(
        df_backend, value_columns=value_columns, rename_date_to="date"
    )

    # Convert result back to pandas for assertions
    result_pd = backend_converter.to_pandas(result)

    # Build expected DataFrame
    expected = (
        pd.DataFrame(
            data=expected_values,
            columns=expected_columns,
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )
        .reset_index()
        .rename(columns={"index": "date"})
    )

    assert set(["date", *expected_columns]).issubset(set(result_pd.columns))
    result_subset = result_pd[["date", *expected_columns]]
    pd.testing.assert_frame_equal(result_subset, expected, check_dtype=False)


@pytest.mark.parametrize(
    "value_columns, expected_columns, expected_values",
    [
        (
            "spend",
            ["facebook_ads_spend", "google_ads_spend"],
            [[30.0, 10.0], [10.0, 0.0]],
        ),
        (
            ["spend", "impressions"],
            [
                "facebook_ads_spend",
                "google_ads_spend",
                "facebook_ads_impressions",
                "google_ads_impressions",
            ],
            [[30.0, 10.0, 1500.0, 1000.0], [10.0, 0.0, 500.0, 0.0]],
        ),
    ],
)
def test_campaign_report_schema_multibackend(
    example_campaign_report_df,
    backend_converter,
    value_columns,
    expected_columns,
    expected_values,
):
    """Test process_fivetran_ad_reporting with campaign_report schema across all backends."""
    # Convert to target backend
    df_backend = backend_converter.to_backend(example_campaign_report_df)

    # Call function
    result = process_fivetran_ad_reporting(
        df_backend, value_columns=value_columns, rename_date_to="date"
    )

    # Convert result back to pandas for assertions
    result_pd = backend_converter.to_pandas(result)

    # Build expected DataFrame
    expected = (
        pd.DataFrame(
            data=expected_values,
            columns=expected_columns,
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )
        .reset_index()
        .rename(columns={"index": "date"})
    )

    assert set(["date", *expected_columns]).issubset(set(result_pd.columns))
    result_subset = result_pd[["date", *expected_columns]]
    pd.testing.assert_frame_equal(result_subset, expected, check_dtype=False)


# ==================== Pandas-only accessor tests ====================


@pytest.mark.parametrize(
    "value_columns, expected_columns, expected_values",
    [
        (
            "spend",
            ["facebook_ads_spend", "google_ads_spend"],
            [[30.0, 10.0], [10.0, 0.0]],
        ),
        (
            ["spend", "impressions"],
            [
                "facebook_ads_spend",
                "google_ads_spend",
                "facebook_ads_impressions",
                "google_ads_impressions",
            ],
            [[30.0, 10.0, 1500.0, 1000.0], [10.0, 0.0, 500.0, 0.0]],
        ),
    ],
)
def test_ad_report_schema_accessor(
    example_ad_report_df,
    value_columns,
    expected_columns,
    expected_values,
):
    """Test .fivetran.process_ad_reporting accessor with ad_report schema (pandas only)."""
    result = example_ad_report_df.fivetran.process_ad_reporting(
        value_columns=value_columns, rename_date_to="date"
    )

    expected = (
        pd.DataFrame(
            data=expected_values,
            columns=expected_columns,
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )
        .reset_index()
        .rename(columns={"index": "date"})
    )

    # Ensure required columns exist and match expected values
    assert set(["date", *expected_columns]).issubset(set(result.columns))

    result_subset = result[["date", *expected_columns]]
    pd.testing.assert_frame_equal(result_subset, expected, check_dtype=False)


@pytest.mark.parametrize(
    "value_columns, expected_columns, expected_values",
    [
        (
            "spend",
            ["facebook_ads_spend", "google_ads_spend"],
            [[30.0, 10.0], [10.0, 0.0]],
        ),
        (
            ["spend", "impressions"],
            [
                "facebook_ads_spend",
                "google_ads_spend",
                "facebook_ads_impressions",
                "google_ads_impressions",
            ],
            [[30.0, 10.0, 1500.0, 1000.0], [10.0, 0.0, 500.0, 0.0]],
        ),
    ],
)
def test_account_report_schema_accessor(
    example_account_report_df,
    value_columns,
    expected_columns,
    expected_values,
):
    """Test .fivetran.process_ad_reporting accessor with account_report schema (pandas only)."""
    result = example_account_report_df.fivetran.process_ad_reporting(
        value_columns=value_columns, rename_date_to="date"
    )

    expected = (
        pd.DataFrame(
            data=expected_values,
            columns=expected_columns,
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )
        .reset_index()
        .rename(columns={"index": "date"})
    )

    assert set(["date", *expected_columns]).issubset(set(result.columns))
    result_subset = result[["date", *expected_columns]]
    pd.testing.assert_frame_equal(result_subset, expected, check_dtype=False)


@pytest.mark.parametrize(
    "value_columns, expected_columns, expected_values",
    [
        (
            "spend",
            ["facebook_ads_spend", "google_ads_spend"],
            [[30.0, 10.0], [10.0, 0.0]],
        ),
        (
            ["spend", "impressions"],
            [
                "facebook_ads_spend",
                "google_ads_spend",
                "facebook_ads_impressions",
                "google_ads_impressions",
            ],
            [[30.0, 10.0, 1500.0, 1000.0], [10.0, 0.0, 500.0, 0.0]],
        ),
    ],
)
def test_campaign_report_schema_accessor(
    example_campaign_report_df,
    value_columns,
    expected_columns,
    expected_values,
):
    """Test .fivetran.process_ad_reporting accessor with campaign_report schema (pandas only)."""
    result = example_campaign_report_df.fivetran.process_ad_reporting(
        value_columns=value_columns, rename_date_to="date"
    )

    expected = (
        pd.DataFrame(
            data=expected_values,
            columns=expected_columns,
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )
        .reset_index()
        .rename(columns={"index": "date"})
    )

    assert set(["date", *expected_columns]).issubset(set(result.columns))
    result_subset = result[["date", *expected_columns]]
    pd.testing.assert_frame_equal(result_subset, expected, check_dtype=False)


@pytest.mark.requires_polars
def test_polars_include_missing_dates_backend_alignment():
    """Test that polars backend is preserved with include_missing_dates=True.

    This is a regression test for the critical backend mismatch bug where
    creating date ranges with pandas.date_range() would break polars DataFrames.

    See: https://github.com/pymc-labs/pymc-marketing/pull/2224#discussion_r2741140644
    """
    from datetime import datetime

    import polars as pl

    # Create test data with a date gap: 2024-01-01, 2024-01-03 (missing 2024-01-02)
    df_pl = pl.DataFrame(
        {
            "date_day": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
                datetime(2024, 1, 3),  # Gap: missing 2024-01-02
                datetime(2024, 1, 3),
            ],
            "platform": ["facebook_ads", "google_ads", "facebook_ads", "google_ads"],
            "spend": [10.0, 20.0, 15.0, 25.0],
        }
    )

    # Process with include_missing_dates=True (should fill the gap with 2024-01-02)
    result = process_fivetran_ad_reporting(
        df_pl,
        value_columns="spend",
        include_missing_dates=True,
        freq="1d",
        fill_value=0.0,
        rename_date_to=None,
    )

    # CRITICAL: Verify result is still polars DataFrame (not pandas)
    assert isinstance(result, pl.DataFrame), (
        f"Expected polars DataFrame, got {type(result)}"
    )

    # Verify missing date was filled
    result_dates = sorted(result.select("date_day").to_series().to_list())
    expected_dates = [
        datetime(2024, 1, 1),
        datetime(2024, 1, 2),  # This was missing, should be filled
        datetime(2024, 1, 3),
    ]
    assert len(result_dates) == 3, f"Expected 3 rows, got {len(result_dates)}"
    assert result_dates == expected_dates, (
        f"Expected dates {expected_dates}, got {result_dates}"
    )

    # Verify fill_value was applied to missing date
    result_row_2024_01_02 = result.filter(pl.col("date_day") == datetime(2024, 1, 2))
    assert result_row_2024_01_02["facebook_ads_spend"][0] == 0.0
    assert result_row_2024_01_02["google_ads_spend"][0] == 0.0


# -------------------- Shopify orders unique orders --------------------


@pytest.fixture
def example_shopify_orders_df() -> pd.DataFrame:
    # Minimal columns from Shopify orders schema needed for the function
    # We include duplicates within the same day and invalid timestamps, plus a few
    # extra fields inspired by the Shopify orders schema CSV
    df = pd.DataFrame(
        [
            {
                "order_id": 7001,
                "user_id": 102,
                "orders_unique_key": "o1",
                "processed_timestamp": "2025-07-12 10:00:00",
                "updated_timestamp": "2025-07-12 12:00:00",
                "currency": "USD",
                "customer_id": 2001,
                "email": "alice@example.com",
                "financial_status": "paid",
                "fulfillment_status": "fulfilled",
                "shipping_address_country_code": "US",
                "is_deleted": False,
                "is_test_order": False,
                "line_item_count": 3,
                "new_vs_repeat": "new",
            },
            {
                "order_id": 7002,
                "user_id": 118,
                "orders_unique_key": "o2",
                "processed_timestamp": "2025-07-12 11:00:00",
                "updated_timestamp": "2025-07-12 12:30:00",
                "currency": "USD",
                "customer_id": 2001,
                "email": "alice@example.com",
                "financial_status": "paid",
                "fulfillment_status": "partial",
                "shipping_address_country_code": "US",
                "is_deleted": False,
                "is_test_order": False,
                "line_item_count": 1,
                "new_vs_repeat": "repeat",
            },
            {
                # duplicate same order same day
                "order_id": 7002,
                "user_id": 118,
                "orders_unique_key": "o2",
                "processed_timestamp": "2025-07-12 11:30:00",
                "updated_timestamp": "2025-07-12 12:31:00",
                "currency": "USD",
                "customer_id": 2001,
                "email": "alice@example.com",
                "financial_status": "paid",
                "fulfillment_status": "partial",
                "shipping_address_country_code": "US",
                "is_deleted": False,
                "is_test_order": False,
                "line_item_count": 1,
                "new_vs_repeat": "repeat",
            },
            {
                "order_id": 7004,
                "user_id": 117,
                "orders_unique_key": "o4",
                "processed_timestamp": "2025-07-12 23:59:59",
                "updated_timestamp": "2025-07-13 01:00:00",
                "currency": "USD",
                "customer_id": 2001,
                "email": "alice@example.com",
                "financial_status": "paid",
                "fulfillment_status": "fulfilled",
                "shipping_address_country_code": "US",
                "is_deleted": False,
                "is_test_order": False,
                "line_item_count": 5,
                "new_vs_repeat": "repeat",
            },
            {
                "order_id": 7017,
                "user_id": 120,
                "source_relation": "shopify.eu",
                "orders_unique_key": "o3",
                "processed_timestamp": "2025-07-13 09:00:00",
                "updated_timestamp": "2025-07-13 11:00:00",
                "currency": "EUR",
                "customer_id": 2002,
                "email": "bob@example.com",
                "financial_status": "refunded",
                "fulfillment_status": "unfulfilled",
                "shipping_address_country_code": "DE",
                "is_deleted": False,
                "is_test_order": False,
                "line_item_count": 6,
                "new_vs_repeat": "new",
            },
            {
                # invalid timestamp -> dropped
                "order_id": 7018,
                "user_id": 110,
                "source_relation": "shopify.eu",
                "orders_unique_key": "ox",
                "processed_timestamp": None,
                "updated_timestamp": "2025-07-14 12:05:00",
                "currency": "EUR",
                "customer_id": 2002,
                "email": "bob@example.com",
                "financial_status": "pending",
                "fulfillment_status": "unfulfilled",
                "shipping_address_country_code": "DE",
                "is_deleted": False,
                "is_test_order": False,
                "line_item_count": 3,
                "new_vs_repeat": "repeat",
            },
            {
                # invalid timestamp -> dropped
                "order_id": 7019,
                "user_id": 106,
                "source_relation": "shopify.eu",
                "orders_unique_key": "oy",
                "processed_timestamp": "invalid",
                "updated_timestamp": "2025-07-15 12:05:00",
                "currency": "EUR",
                "customer_id": 2002,
                "email": "bob@example.com",
                "financial_status": "pending",
                "fulfillment_status": "fulfilled",
                "shipping_address_country_code": "DE",
                "is_deleted": False,
                "is_test_order": False,
                "line_item_count": 4,
                "new_vs_repeat": "repeat",
            },
        ]
    )
    # Convert string timestamps to proper datetime objects for multi-backend support
    # Use errors="coerce" to handle None and "invalid" values (they become NaT)
    df["processed_timestamp"] = pd.to_datetime(
        df["processed_timestamp"], errors="coerce"
    )
    df["updated_timestamp"] = pd.to_datetime(df["updated_timestamp"], errors="coerce")
    return df


def test_shopify_orders_unique_orders_multibackend(
    example_shopify_orders_df: pd.DataFrame,
    backend_converter,
) -> None:
    """Test process_fivetran_shopify_unique_orders across all backends."""
    # Convert to target backend
    df_backend = backend_converter.to_backend(example_shopify_orders_df)

    # Call function
    result = process_fivetran_shopify_unique_orders(df_backend)

    # Convert result back to pandas for assertions
    result_pd = backend_converter.to_pandas(result)

    expected = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-07-12", "2025-07-13"]).normalize(),
            "orders": [3, 1],  # 2025-07-12: o1, o2, o4; 2025-07-13: o3
        }
    )

    pd.testing.assert_frame_equal(result_pd, expected, check_dtype=False)


def test_shopify_orders_unique_orders_accessor(
    example_shopify_orders_df: pd.DataFrame,
) -> None:
    """Test .fivetran.process_shopify_unique_orders accessor (pandas only)."""
    result = example_shopify_orders_df.fivetran.process_shopify_unique_orders()

    expected = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-07-12", "2025-07-13"]).normalize(),
            "orders": [3, 1],  # 2025-07-12: o1, o2, o4; 2025-07-13: o3
        }
    )

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)
