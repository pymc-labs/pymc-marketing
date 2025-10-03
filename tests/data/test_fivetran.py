#   Copyright 2022 - 2025 The PyMC Labs Developers
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
    return pd.DataFrame(data)


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
    return pd.DataFrame(data)


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
    return pd.DataFrame(data)


@pytest.mark.parametrize("accessor", [True, False])
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
def test_ad_report_schema(
    example_ad_report_df,
    accessor: bool,
    value_columns,
    expected_columns,
    expected_values,
):
    kwargs = dict(value_columns=value_columns, rename_date_to="date")
    if accessor:
        result = example_ad_report_df.fivetran.process_ad_reporting(**kwargs)
    else:
        result = process_fivetran_ad_reporting(example_ad_report_df, **kwargs)

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


@pytest.mark.parametrize("accessor", [True, False])
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
def test_account_report_schema(
    example_account_report_df,
    accessor: bool,
    value_columns,
    expected_columns,
    expected_values,
):
    kwargs = dict(value_columns=value_columns, rename_date_to="date")
    if accessor:
        result = example_account_report_df.fivetran.process_ad_reporting(**kwargs)
    else:
        result = process_fivetran_ad_reporting(example_account_report_df, **kwargs)

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


@pytest.mark.parametrize("accessor", [True, False])
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
def test_campaign_report_schema(
    example_campaign_report_df,
    accessor: bool,
    value_columns,
    expected_columns,
    expected_values,
):
    kwargs = dict(value_columns=value_columns, rename_date_to="date")
    if accessor:
        result = example_campaign_report_df.fivetran.process_ad_reporting(**kwargs)
    else:
        result = process_fivetran_ad_reporting(example_campaign_report_df, **kwargs)

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


# -------------------- Shopify orders unique orders --------------------


@pytest.fixture
def example_shopify_orders_df() -> pd.DataFrame:
    # Minimal columns from Shopify orders schema needed for the function
    # We include duplicates within the same day and invalid timestamps, plus a few
    # extra fields inspired by the Shopify orders schema CSV
    return pd.DataFrame(
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


@pytest.mark.parametrize("accessor", [True, False])
def test_shopify_orders_unique_orders(
    example_shopify_orders_df: pd.DataFrame,
    accessor: bool,
) -> None:
    if accessor:
        result = example_shopify_orders_df.fivetran.process_shopify_unique_orders()
    else:
        result = process_fivetran_shopify_unique_orders(example_shopify_orders_df)

    expected = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-07-12", "2025-07-13"]).normalize(),
            "orders": [3, 1],  # 2025-07-12: o1, o2, o4; 2025-07-13: o3
        }
    )

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)
