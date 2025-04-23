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
from datetime import datetime

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray
from pandas.testing import assert_frame_equal

from pymc_marketing.clv import GammaGammaModel, ParetoNBDModel
from pymc_marketing.clv.utils import (
    _expected_cumulative_transactions,
    _find_first_transactions,
    _rfm_quartile_labels,
    customer_lifetime_value,
    rfm_segments,
    rfm_summary,
    rfm_train_test_split,
    to_xarray,
)
from pymc_marketing.prior import Prior
from tests.conftest import set_model_fit


def test_to_xarray():
    customer_id = np.arange(10) + 100
    x = pd.Index(range(10), name="hello")
    y = pd.Series(range(10), name="y")
    z = np.arange(10)

    new_x = to_xarray(customer_id, x)
    assert isinstance(new_x, xarray.DataArray)
    assert new_x.dims == ("customer_id",)
    np.testing.assert_array_equal(new_x.coords["customer_id"], customer_id)
    np.testing.assert_array_equal(x, new_x.values)

    for old, new in zip((x, y, z), to_xarray(customer_id, x, y, z), strict=False):
        assert isinstance(new, xarray.DataArray)
        assert new.dims == ("customer_id",)
        np.testing.assert_array_equal(new.coords["customer_id"], customer_id)
        np.testing.assert_array_equal(old, new.values)

    new_y = to_xarray(customer_id, y, dim="test_dim")
    assert new_y.dims == ("test_dim",)
    np.testing.assert_array_equal(new_y.coords["test_dim"], customer_id)


@pytest.fixture(scope="module")
def fitted_gg(test_summary_data) -> GammaGammaModel:
    rng = np.random.default_rng(40)
    pd.Series({"p": 6.25, "q": 3.74, "v": 15.44})

    model_config = {
        # Params used in lifetimes test
        "p": Prior("DiracDelta", c=6.25),
        "q": Prior("DiracDelta", c=3.74),
        "v": Prior("DiracDelta", c=15.44),
    }
    model = GammaGammaModel(
        data=test_summary_data,
        model_config=model_config,
    )
    model.build_model()
    fake_fit = pm.sample_prior_predictive(
        draws=50, model=model.model, random_seed=rng
    ).prior
    set_model_fit(model, fake_fit)

    return model


# TODO: Consolidate this fixture into the tests requiring it?
@pytest.fixture()
def df_cum_transactions():
    cdnow_transactions = pd.read_csv("data/cdnow_transactions.csv")

    rfm_data = rfm_summary(
        cdnow_transactions,
        customer_id_col="id",
        datetime_col="date",
        datetime_format="%Y%m%d",
        time_unit="D",
        observation_period_end="19970930",
        time_scaler=7,
    )

    model_config = {
        "r": Prior("HalfFlat"),
        "alpha": Prior("HalfFlat"),
        "s": Prior("HalfFlat"),
        "beta": Prior("HalfFlat"),
    }

    pnbd = ParetoNBDModel(data=rfm_data, model_config=model_config)
    pnbd.fit()

    df_cum = _expected_cumulative_transactions(
        model=pnbd,
        transactions=cdnow_transactions,
        customer_id_col="id",
        datetime_col="date",
        t=25 * 7,
        datetime_format="%Y%m%d",
        time_unit="D",
        time_scaler=7,
    )
    return df_cum


class TestCustomerLifetimeValue:
    def test_missing_col(self, fitted_bg, test_summary_data):
        data_invalid = test_summary_data.drop(columns="future_spend")

        with pytest.raises(ValueError, match="Required column future_spend missing"):
            customer_lifetime_value(
                transaction_model=fitted_bg,
                data=data_invalid,
                future_t=10,
                discount_rate=0.1,
            )

    @pytest.mark.parametrize(
        "t, discount_rate, expected_change",
        [
            (1, 0, 1),
            (1, 1, 0.5),
            (2, 0, 2),
            (2, 1, 0.75),
        ],
    )
    def test_customer_lifetime_value_bg_with_known_values(
        self, fitted_bg, t, discount_rate, expected_change
    ):
        # Test borrowed from
        # https://github.com/CamDavidsonPilon/lifetimes/blob/aae339c5437ec31717309ba0ec394427e19753c4/tests/test_utils.py#L527

        # time=1, discount_rate=0 means the clv will be the same as the predicted
        # time=1, discount_rate=1 means the clv will halve over a period
        # time=2, discount_rate=0 means the clv will be twice the initial
        # time=2, discount_rate=1 means the clv will be twice the initial

        expected = np.array([0.016053, 0.021171, 0.030461, 0.031686, 0.001607])

        data = pd.DataFrame(
            {
                "customer_id": [0, 1, 2, 3, 4],
                "frequency": [0, 0, 6, 0, 2],
                "recency": [0, 0, 142, 0, 9],
                "T": [298, 224, 292, 147, 183],
                "future_spend": [1, 1, 1, 1, 1],
            }
        )

        clv = customer_lifetime_value(
            fitted_bg,
            data=data,
            future_t=t,
            discount_rate=discount_rate,
        ).mean(("chain", "draw"))

        np.testing.assert_allclose(clv, expected * expected_change, rtol=0.1)

    @pytest.mark.parametrize("transaction_model", ("fitted_bg", "fitted_pnbd"))
    def test_customer_lifetime_value_as_gg_method(
        self, request, test_summary_data, fitted_gg, transaction_model
    ):
        transaction_model = request.getfixturevalue(transaction_model)
        t = test_summary_data.head()

        ggf_clv = fitted_gg.expected_customer_lifetime_value(
            transaction_model=transaction_model,
            data=t,
        )

        # create future_spend column from fitted gg
        ggf_spend = fitted_gg.expected_customer_spend(data=t)
        t.loc[:, "future_spend"] = ggf_spend.mean(("chain", "draw")).copy()

        utils_clv = customer_lifetime_value(
            transaction_model=transaction_model,
            data=t,
        )
        np.testing.assert_equal(ggf_clv.values, utils_clv.values)

    @pytest.mark.parametrize("gg_map", (True, False))
    @pytest.mark.parametrize("transaction_model_map", (True, False))
    @pytest.mark.parametrize("transaction_model", ("fitted_bg", "fitted_pnbd"))
    def test_map_posterior_mix_fit_types(
        self,
        request,
        test_summary_data,
        fitted_gg,
        gg_map,
        transaction_model_map,
        transaction_model,
    ):
        """Test we can mix a model that was fit with MAP and one that was fit with sample."""
        transaction_model = request.getfixturevalue(transaction_model)
        t = test_summary_data.head()

        # Copy model with thinned chain/draw as would be obtained from MAP
        if transaction_model_map:
            transaction_model = transaction_model._build_with_idata(
                transaction_model.idata.sel(chain=slice(0, 1), draw=slice(0, 1))
            )

        if gg_map:
            fitted_gg = fitted_gg._build_with_idata(
                fitted_gg.idata.sel(chain=slice(0, 1), draw=slice(0, 1))
            )
            # create future_spend column from fitted gg
            ggf_spend = fitted_gg.expected_customer_spend(data=t)
            t.loc[:, "future_spend"] = ggf_spend.mean(("chain", "draw")).copy()

        res = customer_lifetime_value(
            transaction_model=transaction_model,
            data=t,
        )

        assert res.dims == ("chain", "draw", "customer_id")

    @pytest.mark.parametrize("transaction_model", ("fitted_bg", "fitted_pnbd"))
    def test_clv_after_thinning(
        self,
        request,
        test_summary_data,
        fitted_gg,
        transaction_model,
    ):
        transaction_model = request.getfixturevalue(transaction_model)
        t = test_summary_data.head()

        ggf_clv = fitted_gg.expected_customer_lifetime_value(
            transaction_model=transaction_model,
            data=t,
        )

        fitted_gg_thinned = fitted_gg.thin_fit_result(keep_every=10)
        transaction_model_thinned = transaction_model.thin_fit_result(keep_every=10)
        ggf_clv_thinned = fitted_gg_thinned.expected_customer_lifetime_value(
            transaction_model=transaction_model_thinned,
            data=t,
        )

        assert ggf_clv.shape == (1, 50, 5)
        assert ggf_clv_thinned.shape == (1, 5, 5)

        np.testing.assert_allclose(
            ggf_clv.isel(draw=slice(None, None, 10)).values,
            ggf_clv_thinned.values,
            rtol=1e-14,  # pandas dataframe arguments create tiny rounding errors
        )


class TestRFM:
    @pytest.fixture(scope="class")
    def transaction_data(self) -> pd.DataFrame:
        d = [
            [1, "2015-01-01", 1],
            [1, "2015-02-06", 2],
            [2, "2015-01-01", 2],
            [3, "2015-01-01", 3],
            [3, "2015-01-02", 1],
            [3, "2015-01-05", 5],
            [4, "2015-01-16", 6],
            [4, "2015-02-02", 3],
            [4, "2015-02-05", 3],
            [4, "2015-02-05", 6],
            [5, "2015-01-16", 3],
            [5, "2015-01-17", 1],
            [5, "2015-01-18", 8],
            [6, "2015-02-02", 5],
        ]
        return pd.DataFrame(d, columns=["identifier", "date", "monetary_value"])

    def test_find_first_transactions_test_period_end_none(self, transaction_data):
        max_date = transaction_data["date"].max()
        pd.testing.assert_frame_equal(
            left=_find_first_transactions(
                transactions=transaction_data,
                customer_id_col="identifier",
                datetime_col="date",
                observation_period_end=None,
            ),
            right=_find_first_transactions(
                transactions=transaction_data,
                customer_id_col="identifier",
                datetime_col="date",
                observation_period_end=max_date,
            ),
        )

    @pytest.mark.parametrize(
        argnames="today",
        argvalues=["2015-02-07", pd.Period("2015-02-07"), datetime(2015, 2, 7), None],
        ids=["string", "period", "datetime", "none"],
    )
    def test_find_first_transactions_returns_correct_results(
        self, transaction_data, today
    ):
        # Test adapted from
        # https://github.com/CamDavidsonPilon/lifetimes/blob/aae339c5437ec31717309ba0ec394427e19753c4/tests/test_utils.py#L137

        actual = _find_first_transactions(
            transaction_data,
            "identifier",
            "date",
            observation_period_end=today,
        )
        expected = pd.DataFrame(
            [
                [1, pd.Period("2015-01-01", "D"), True],
                [1, pd.Period("2015-02-06", "D"), False],
                [2, pd.Period("2015-01-01", "D"), True],
                [3, pd.Period("2015-01-01", "D"), True],
                [3, pd.Period("2015-01-02", "D"), False],
                [3, pd.Period("2015-01-05", "D"), False],
                [4, pd.Period("2015-01-16", "D"), True],
                [4, pd.Period("2015-02-02", "D"), False],
                [4, pd.Period("2015-02-05", "D"), False],
                [5, pd.Period("2015-01-16", "D"), True],
                [5, pd.Period("2015-01-17", "D"), False],
                [5, pd.Period("2015-01-18", "D"), False],
                [6, pd.Period("2015-02-02", "D"), True],
            ],
            columns=["identifier", "date", "first"],
            index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13],
        )  # row indices are skipped for time periods with multiple transactions
        assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        argnames="today",
        argvalues=["2015-02-07", pd.Period("2015-02-07"), datetime(2015, 2, 7), None],
        ids=["string", "period", "datetime", "none"],
    )
    def test_find_first_transactions_with_specific_non_daily_frequency(
        self, transaction_data, today
    ):
        # Test adapted from
        # https://github.com/CamDavidsonPilon/lifetimes/blob/aae339c5437ec31717309ba0ec394427e19753c4/tests/test_utils.py#L161

        actual = _find_first_transactions(
            transaction_data,
            "identifier",
            "date",
            observation_period_end=today,
            time_unit="W",
        )
        expected = pd.DataFrame(
            [
                [1, pd.Period("2014-12-29/2015-01-04", "W-SUN"), True],
                [1, pd.Period("2015-02-02/2015-02-08", "W-SUN"), False],
                [2, pd.Period("2014-12-29/2015-01-04", "W-SUN"), True],
                [3, pd.Period("2014-12-29/2015-01-04", "W-SUN"), True],
                [3, pd.Period("2015-01-05/2015-01-11", "W-SUN"), False],
                [4, pd.Period("2015-01-12/2015-01-18", "W-SUN"), True],
                [4, pd.Period("2015-02-02/2015-02-08", "W-SUN"), False],
                [5, pd.Period("2015-01-12/2015-01-18", "W-SUN"), True],
                [6, pd.Period("2015-02-02/2015-02-08", "W-SUN"), True],
            ],
            columns=["identifier", "date", "first"],
            index=actual.index,
        )  # we shouldn't really care about row ordering or indexing, but assert_frame_equals is strict about it
        assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        argnames="today",
        argvalues=["2015-02-07", pd.Period("2015-02-07"), datetime(2015, 2, 7), None],
        ids=["string", "period", "datetime", "none"],
    )
    def test_find_first_transactions_with_monetary_values(
        self, transaction_data, today
    ):
        # Test adapted from
        # https://github.com/CamDavidsonPilon/lifetimes/blob/aae339c5437ec31717309ba0ec394427e19753c4/tests/test_utils.py#L184

        actual = _find_first_transactions(
            transaction_data,
            "identifier",
            "date",
            "monetary_value",
            observation_period_end=today,
        )
        expected = pd.DataFrame(
            [
                [1, pd.Period("2015-01-01", "D"), 1, True],
                [1, pd.Period("2015-02-06", "D"), 2, False],
                [2, pd.Period("2015-01-01", "D"), 2, True],
                [3, pd.Period("2015-01-01", "D"), 3, True],
                [3, pd.Period("2015-01-02", "D"), 1, False],
                [3, pd.Period("2015-01-05", "D"), 5, False],
                [4, pd.Period("2015-01-16", "D"), 6, True],
                [4, pd.Period("2015-02-02", "D"), 3, False],
                [4, pd.Period("2015-02-05", "D"), 9, False],
                [5, pd.Period("2015-01-16", "D"), 3, True],
                [5, pd.Period("2015-01-17", "D"), 1, False],
                [5, pd.Period("2015-01-18", "D"), 8, False],
                [6, pd.Period("2015-02-02", "D"), 5, True],
            ],
            columns=["identifier", "date", "monetary_value", "first"],
        )
        assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        argnames="today",
        argvalues=["2015-02-07", pd.Period("2015-02-07"), datetime(2015, 2, 7), None],
        ids=["string", "period", "datetime", "none"],
    )
    def test_find_first_transactions_with_monetary_values_with_specific_non_daily_frequency(
        self, transaction_data, today
    ):
        # Test adapted from
        # https://github.com/CamDavidsonPilon/lifetimes/blob/aae339c5437ec31717309ba0ec394427e19753c4/tests/test_utils.py#L210

        actual = _find_first_transactions(
            transaction_data,
            "identifier",
            "date",
            "monetary_value",
            observation_period_end=today,
            time_unit="W",
        )
        expected = pd.DataFrame(
            [
                [1, pd.Period("2014-12-29/2015-01-04", "W-SUN"), 1, True],
                [1, pd.Period("2015-02-02/2015-02-08", "W-SUN"), 2, False],
                [2, pd.Period("2014-12-29/2015-01-04", "W-SUN"), 2, True],
                [3, pd.Period("2014-12-29/2015-01-04", "W-SUN"), 4, True],
                [3, pd.Period("2015-01-05/2015-01-11", "W-SUN"), 5, False],
                [4, pd.Period("2015-01-12/2015-01-18", "W-SUN"), 6, True],
                [4, pd.Period("2015-02-02/2015-02-08", "W-SUN"), 12, False],
                [5, pd.Period("2015-01-12/2015-01-18", "W-SUN"), 12, True],
                [6, pd.Period("2015-02-02/2015-02-08", "W-SUN"), 5, True],
            ],
            columns=["identifier", "date", "monetary_value", "first"],
        )
        assert_frame_equal(actual, expected)

    @pytest.mark.parametrize(
        argnames="today",
        argvalues=["2015-02-07", pd.Period("2015-02-07"), datetime(2015, 2, 7)],
        ids=["string", "period", "datetime"],
    )
    def test_rfm_summary_returns_correct_results(self, transaction_data, today):
        # Test adapted from
        # https://github.com/CamDavidsonPilon/lifetimes/blob/aae339c5437ec31717309ba0ec394427e19753c4/tests/test_utils.py#L239

        actual = rfm_summary(
            transaction_data, "identifier", "date", observation_period_end=today
        )
        expected = pd.DataFrame(
            [
                [1, 1.0, 36.0, 37.0],
                [2, 0.0, 0.0, 37.0],
                [3, 2.0, 4.0, 37.0],
                [4, 2.0, 20.0, 22.0],
                [5, 2.0, 2.0, 22.0],
                [6, 0.0, 0.0, 5.0],
            ],
            columns=["customer_id", "frequency", "recency", "T"],
        )
        assert_frame_equal(actual, expected)

    def test_rfm_summary_works_with_string_customer_ids(self):
        # Test adapted from
        # https://github.com/CamDavidsonPilon/lifetimes/blob/aae339c5437ec31717309ba0ec394427e19753c4/tests/test_utils.py#L250

        d = [
            ["X", "2015-02-01"],
            ["X", "2015-02-06"],
            ["Y", "2015-01-01"],
            ["Y", "2015-01-01"],
            ["Y", "2015-01-02"],
            ["Y", "2015-01-05"],
        ]
        df = pd.DataFrame(d, columns=["identifier", "date"])
        rfm_summary(df, "identifier", "date")

    def test_rfm_summary_works_with_int_customer_ids_and_doesnt_coerce_to_float(self):
        # Test adapted from
        # https://github.com/CamDavidsonPilon/lifetimes/blob/aae339c5437ec31717309ba0ec394427e19753c4/tests/test_utils.py#L263

        d = [
            [1, "2015-02-01"],
            [1, "2015-02-06"],
            [1, "2015-01-01"],
            [2, "2015-01-01"],
            [2, "2015-01-02"],
            [2, "2015-01-05"],
        ]
        df = pd.DataFrame(d, columns=["identifier", "date"])
        actual = rfm_summary(df, "identifier", "date")
        assert actual.index.dtype == "int64"

    def test_rfm_summary_with_specific_datetime_format(
        self,
        transaction_data,
    ):
        # Test adapted from
        # https://github.com/CamDavidsonPilon/lifetimes/blob/aae339c5437ec31717309ba0ec394427e19753c4/tests/test_utils.py#L279

        transaction_data["date"] = transaction_data["date"].map(
            lambda x: x.replace("-", "")
        )
        format = "%Y%m%d"
        today = "20150207"
        actual = rfm_summary(
            transaction_data,
            "identifier",
            "date",
            observation_period_end=today,
            datetime_format=format,
            sort_transactions=False,
        )
        expected = pd.DataFrame(
            [
                [1, 1.0, 36.0, 37.0],
                [2, 0.0, 0.0, 37.0],
                [3, 2.0, 4.0, 37.0],
                [4, 2.0, 20.0, 22.0],
                [5, 2.0, 2.0, 22.0],
                [6, 0.0, 0.0, 5.0],
            ],
            columns=["customer_id", "frequency", "recency", "T"],
        )
        assert_frame_equal(actual, expected)

    def test_rfm_summary_non_daily_frequency(
        self,
        transaction_data,
    ):
        # Test adapted from
        # https://github.com/CamDavidsonPilon/lifetimes/blob/aae339c5437ec31717309ba0ec394427e19753c4/tests/test_utils.py#L292

        today = "20150207"
        actual = rfm_summary(
            transaction_data,
            "identifier",
            "date",
            observation_period_end=today,
            time_unit="W",
        )
        expected = pd.DataFrame(
            [
                [1, 1.0, 5.0, 5.0],
                [2, 0.0, 0.0, 5.0],
                [3, 1.0, 1.0, 5.0],
                [4, 1.0, 3.0, 3.0],
                [5, 0.0, 0.0, 3.0],
                [6, 0.0, 0.0, 0.0],
            ],
            columns=["customer_id", "frequency", "recency", "T"],
        )
        assert_frame_equal(actual, expected)

    def test_rfm_summary_monetary_values_and_first_transactions(
        self,
        transaction_data,
    ):
        # Test adapted from
        # https://github.com/CamDavidsonPilon/lifetimes/blob/aae339c5437ec31717309ba0ec394427e19753c4/tests/test_utils.py#L311

        today = "20150207"
        actual = rfm_summary(
            transaction_data,
            "identifier",
            "date",
            monetary_value_col="monetary_value",
            observation_period_end=today,
        )
        expected = pd.DataFrame(
            [
                [1, 1.0, 36.0, 37.0, 2],
                [2, 0.0, 0.0, 37.0, 0],
                [3, 2.0, 4.0, 37.0, 3],
                [4, 2.0, 20.0, 22.0, 6],
                [5, 2.0, 2.0, 22.0, 4.5],
                [6, 0.0, 0.0, 5.0, 0],
            ],
            columns=["customer_id", "frequency", "recency", "T", "monetary_value"],
        )
        assert_frame_equal(actual, expected)

        actual_first_trans = rfm_summary(
            transaction_data,
            "identifier",
            "date",
            monetary_value_col="monetary_value",
            observation_period_end=today,
            include_first_transaction=True,
        )
        expected_first_trans = pd.DataFrame(
            [
                [1, 2.0, 1.0, 1.5],
                [2, 1.0, 37.0, 2],
                [3, 3.0, 33.0, 3],
                [4, 3.0, 2.0, 6],
                [5, 3.0, 20.0, 4],
                [6, 1.0, 5.0, 5],
            ],
            columns=["customer_id", "frequency", "recency", "monetary_value"],
        )
        assert_frame_equal(actual_first_trans, expected_first_trans)

    def test_rfm_summary_will_choose_the_correct_first_order_to_drop_in_monetary_transactions(
        self,
    ):
        # Test adapted from
        # https://github.com/CamDavidsonPilon/lifetimes/blob/aae339c5437ec31717309ba0ec394427e19753c4/tests/test_utils.py#L334

        cust = pd.Series([2, 2, 2])
        dates_ordered = pd.to_datetime(
            pd.Series(
                ["2014-03-14 00:00:00", "2014-04-09 00:00:00", "2014-05-21 00:00:00"]
            )
        )
        sales = pd.Series([10, 20, 25])
        transaction_data = pd.DataFrame(
            {"date": dates_ordered, "identifier": cust, "sales": sales}
        )
        summary_ordered_data = rfm_summary(
            transaction_data, "identifier", "date", "sales"
        )

        dates_unordered = pd.to_datetime(
            pd.Series(
                ["2014-04-09 00:00:00", "2014-03-14 00:00:00", "2014-05-21 00:00:00"]
            )
        )
        sales = pd.Series([20, 10, 25])
        transaction_data = pd.DataFrame(
            {"date": dates_unordered, "identifier": cust, "sales": sales}
        )
        summary_unordered_data = rfm_summary(
            transaction_data, "identifier", "date", "sales"
        )

        assert_frame_equal(summary_ordered_data, summary_unordered_data)
        assert summary_ordered_data["monetary_value"].loc[0] == 22.5

    def test_rfm_summary_statistics_identical_to_hardie_paper(
        self,
        cdnow_trans,
    ):
        # Test adapted from
        # https://github.com/CamDavidsonPilon/lifetimes/blob/aae339c5437ec31717309ba0ec394427e19753c4/tests/test_utils.py#L353

        # see http://brucehardie.com/papers/rfm_clv_2005-02-16.pdf
        # RFM and CLV: Using Iso-value Curves for Customer Base Analysis
        summary = rfm_summary(
            cdnow_trans,
            "id",
            "date",
            "spent",
            observation_period_end="19971001",
            datetime_format="%Y%m%d",
        )
        results = summary[summary["frequency"] > 0]["monetary_value"].describe()

        assert np.round(results.loc["mean"]) == 35
        assert np.round(results.loc["std"]) == 30
        assert np.round(results.loc["min"]) == 3
        assert np.round(results.loc["50%"]) == 27
        assert np.round(results.loc["max"]) == 300
        assert np.round(results.loc["count"]) == 946

    def test_rfm_summary_squashes_period_purchases_to_one_purchase(self):
        # Test adapted from
        # https://github.com/CamDavidsonPilon/lifetimes/blob/aae339c5437ec31717309ba0ec394427e19753c4/tests/test_utils.py#L472

        transactions = pd.DataFrame(
            [[1, "2015-01-01"], [1, "2015-01-01"]], columns=["identifier", "t"]
        )
        actual = rfm_summary(transactions, "identifier", "t", time_unit="W")
        assert actual.loc[0]["frequency"] == 1.0 - 1.0

    def test_rfm_train_test_split(self, transaction_data):
        # Test adapted from
        # https://github.com/CamDavidsonPilon/lifetimes/blob/aae339c5437ec31717309ba0ec394427e19753c4/tests/test_utils.py#L374

        train_end = "2015-02-01"
        actual = rfm_train_test_split(transaction_data, "identifier", "date", train_end)
        assert actual.loc[0]["test_frequency"] == 1
        assert actual.loc[1]["test_frequency"] == 0

        with pytest.raises(KeyError):
            actual.loc[6]

    @pytest.mark.parametrize("train_end", ("2014-02-07", "2015-02-08"))
    def test_rfm_train_test_split_throws_better_error_if_test_period_end_is_too_early(
        self,
        train_end,
        transaction_data,
    ):
        # Test adapted from
        # https://github.com/CamDavidsonPilon/lifetimes/blob/aae339c5437ec31717309ba0ec394427e19753c4/tests/test_utils.py#L387

        test_end = "2014-02-07"

        error_msg = """No data available. Check `test_transactions` and `train_period_end`
        and confirm values in `transactions` occur prior to those time periods."""

        with pytest.raises(ValueError, match=error_msg):
            rfm_train_test_split(
                transaction_data,
                "identifier",
                "date",
                train_end,
                test_period_end=test_end,
            )

    def test_rfm_train_test_split_works_with_specific_frequency(self, transaction_data):
        # Test adapted from
        # https://github.com/CamDavidsonPilon/lifetimes/blob/aae339c5437ec31717309ba0ec394427e19753c4/tests/test_utils.py#L412

        test_end = "2015-02-07"
        train_end = "2015-02-01"
        actual = rfm_train_test_split(
            transaction_data,
            "identifier",
            "date",
            train_end,
            test_period_end=test_end,
            time_unit="W",
        )
        expected_cols = [
            "customer_id",
            "frequency",
            "recency",
            "T",
            "test_frequency",
            "test_T",
        ]
        expected = pd.DataFrame(
            [
                [1, 0.0, 0.0, 4.0, 1, 1],
                [2, 0.0, 0.0, 4.0, 0, 1],
                [3, 1.0, 1.0, 4.0, 0, 1],
                [4, 0.0, 0.0, 2.0, 1, 1],
                [5, 0.0, 0.0, 2.0, 0, 1],
            ],
            columns=expected_cols,
        )
        assert_frame_equal(actual, expected, check_dtype=False)

    def test_rfm_train_test_split_gives_correct_date_boundaries(self, transaction_data):
        # Test adapted from
        # https://github.com/CamDavidsonPilon/lifetimes/blob/aae339c5437ec31717309ba0ec394427e19753c4/tests/test_utils.py#L432

        actual = rfm_train_test_split(
            transaction_data,
            "identifier",
            "date",
            train_period_end="2015-02-01",
            test_period_end="2015-02-04",
        )
        assert actual["test_frequency"].loc[1] == 0
        assert actual["test_frequency"].loc[3] == 1

    def test_rfm_train_test_split_with_monetary_value(self, transaction_data):
        # Test adapted from
        # https://github.com/CamDavidsonPilon/lifetimes/blob/aae339c5437ec31717309ba0ec394427e19753c4/tests/test_utils.py#L457

        test_end = "2015-02-07"
        train_end = "2015-02-01"
        actual = rfm_train_test_split(
            transaction_data,
            "identifier",
            "date",
            train_end,
            test_period_end=test_end,
            monetary_value_col="monetary_value",
        )
        assert (actual["monetary_value"] == [0, 0, 3, 0, 4.5]).all()
        assert (actual["test_monetary_value"] == [2, 0, 0, 6, 0]).all()

    @pytest.mark.parametrize("config", (None, "custom"))
    def test_rfm_segmentation_config(self, transaction_data, config):
        if config is not None:
            config = {
                "Test Segment": [
                    "111",
                    "222",
                    "333",
                    "444",
                ]
            }
            expected = ["Other", "Test Segment", "Other", "Other", "Other", "Other"]
        else:
            expected = [
                "Other",
                "Inactive Customer",
                "At Risk Customer",
                "Premium Customer",
                "Repeat Customer",
                "Top Spender",
            ]

        actual = rfm_segments(
            transaction_data,
            "identifier",
            "date",
            "monetary_value",
            segment_config=config,
        )

        assert (actual["segment"] == expected).all()

    def test_rfm_segmentation_warning(self):
        # this data will only return two bins for the frequency variable
        d = [
            [1, "2015-01-01", 1],
            [1, "2015-02-06", 2],
            [2, "2015-01-01", 2],
            [3, "2015-01-02", 1],
            [3, "2015-01-05", 3],
            [4, "2015-01-16", 4],
            [4, "2015-02-05", 5],
            [5, "2015-01-17", 1],
            [5, "2015-01-18", 2],
            [5, "2015-01-19", 2],
        ]
        repetitive_data = pd.DataFrame(d, columns=["id", "date", "monetary_value"])

        with pytest.warns(
            UserWarning,
            match="RFM score will not exceed 2 for f_quartile. Specify a custom segment_config",
        ):
            rfm_segments(
                repetitive_data,
                "id",
                "date",
                "monetary_value",
            )

    def test_rfm_quartile_labels(self):
        # assert recency labels are in reverse order
        recency = _rfm_quartile_labels("r_quartile", 5)
        assert recency == [4, 3, 2, 1]

        # assert max_quartile_range = 4 returns a range function for three labels
        frequency = _rfm_quartile_labels("f_quartile", 4)
        assert frequency == range(1, 4)


def test_expected_cumulative_transactions_dedups_inside_a_time_period(
    fitted_bg, cdnow_trans
):
    """
    Test adapted from lifetimes:
    https://github.com/CamDavidsonPilon/lifetimes/blob/master/tests/test_utils.py#L623
    """
    by_week = _expected_cumulative_transactions(
        fitted_bg, cdnow_trans, "date", "id", 10, time_unit="W"
    )
    by_day = _expected_cumulative_transactions(
        fitted_bg, cdnow_trans, "date", "id", 10, time_unit="D"
    )
    assert (by_week["actual"] >= by_day["actual"]).all()


def test_expected_cumulative_incremental_transactions_equals_r_btyd_walkthrough(
    cdnow_trans, fitted_pnbd
):
    """
    Validate expected cumulative & incremental transactions with BTYD walktrough

    https://cran.r-project.org/web/packages/BTYD/vignettes/BTYD-walkthrough.pdf

    cum.tracking[,20:25]
    # [,1] [,2] [,3] [,4] [,5] [,6]
    # actual 1359 1414 1484 1517 1573 1672
    # expected 1309 1385 1460 1533 1604 1674

    inc.tracking[,20:25]
    # [,1] [,2] [,3] [,4] [,5] [,6]
    # actual 73.00 55.00 70.00 33.00 56.00 99.00
    # expected 78.31 76.42 74.65 72.98 71.41 69.93

    Test adapted from lifetimes:
    https://github.com/CamDavidsonPilon/lifetimes/blob/master/tests/test_utils.py#L601
    """
    df_cum_trans = _expected_cumulative_transactions(
        model=fitted_pnbd,
        transactions=cdnow_trans,
        customer_id_col="id",
        datetime_col="date",
        t=25 * 7,
        datetime_format="%Y%m%d",
        time_unit="D",
        time_scaler=7,
    )

    actual_btyd = [1359, 1414, 1484, 1517, 1573, 1672]
    expected_btyd = [1309, 1385, 1460, 1533, 1604, 1674]

    actual = df_cum_trans["actual"].iloc[19:25].values
    predicted = df_cum_trans["predicted"].iloc[19:25].values.round()

    np.testing.assert_allclose(actual, actual_btyd)
    np.testing.assert_allclose(predicted, expected_btyd, rtol=1e-1)

    # get incremental from cumulative transactions
    df_inc_trans = df_cum_trans.apply(lambda x: x - x.shift(1))

    actual_btyd = [73.00, 55.00, 70.00, 33.00, 56.00, 99.00]
    expected_btyd = [78.31, 76.42, 74.65, 72.98, 71.41, 69.93]

    actual = df_inc_trans["actual"].iloc[19:25].values
    predicted = df_inc_trans["predicted"].iloc[19:25].values.round(2)

    np.testing.assert_allclose(actual, actual_btyd)
    np.testing.assert_allclose(predicted, expected_btyd, rtol=1e-2)


def test_expected_cumulative_transactions_date_index(fitted_bg, cdnow_trans):
    """
    Test set_index as date for cumulative transactions and bgf fitter.

    Get first 14 cdnow transactions dates and validate that date index,
    freq_multiplier = 1 working and compare with tested data for last 4 records.

    Test adapted from lifetimes:
    https://github.com/CamDavidsonPilon/lifetimes/blob/master/tests/test_utils.py#L648
    """
    df_cum = _expected_cumulative_transactions(
        fitted_bg,
        cdnow_trans,
        customer_id_col="id",
        datetime_col="date",
        t=14,
        datetime_format="%Y%m%d",
        time_unit="D",
        time_scaler=1,
        set_index_date=True,
    )

    dates = ["1997-01-11", "1997-01-12", "1997-01-13", "1997-01-14"]
    actual_trans = [11, 12, 15, 19]
    # these values differ from the lifetimes test because we are reusing the existing BG/NBD model fixture
    # rather than fitting a new model to a different subset of the data
    expected_trans = [76.27, 88.42, 101.53, 115.28]

    date_index = df_cum.iloc[-4:].index.to_timestamp().astype(str)
    actual = df_cum["actual"].iloc[-4:].values
    predicted = df_cum["predicted"].iloc[-4:].values.round(2)

    assert all(dates == date_index)
    np.testing.assert_allclose(actual, actual_trans)
    np.testing.assert_allclose(predicted, expected_trans, rtol=1e-2)
