import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray

from pymc_marketing.clv import BetaGeoModel, GammaGammaModel
from pymc_marketing.clv.utils import customer_lifetime_value, to_xarray


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

    for old, new in zip((x, y, z), to_xarray(customer_id, x, y, z)):
        assert isinstance(new, xarray.DataArray)
        assert new.dims == ("customer_id",)
        np.testing.assert_array_equal(new.coords["customer_id"], customer_id)
        np.testing.assert_array_equal(old, new.values)

    new_y = to_xarray(customer_id, y, dim="test_dim")
    new_y.dims == ("test_dim",)
    np.testing.assert_array_equal(new_y.coords["test_dim"], customer_id)


@pytest.fixture(scope="module")
def test_summary_data() -> pd.DataFrame:
    rng = np.random.default_rng(14)
    df = pd.read_csv("tests/clv/datasets/test_summary_data.csv", index_col=0)
    df["monetary_value"] = rng.lognormal(size=(len(df)))
    return df


@pytest.fixture(scope="module")
def fitted_bg(test_summary_data) -> BetaGeoModel:
    rng = np.random.default_rng(13)

    model = BetaGeoModel(
        customer_id=test_summary_data.index,
        frequency=test_summary_data["frequency"],
        recency=test_summary_data["recency"],
        T=test_summary_data["T"],
        # Narrow Gaussian centered at MLE params from lifetimes BetaGeoFitter
        a_prior=pm.DiracDelta.dist(1.85034151),
        alpha_prior=pm.DiracDelta.dist(1.86428187),
        b_prior=pm.DiracDelta.dist(3.18105431),
        r_prior=pm.DiracDelta.dist(0.16385072),
    )

    fake_fit = pm.sample_prior_predictive(
        samples=50, model=model.model, random_seed=rng
    )
    fake_fit.add_groups(dict(posterior=fake_fit.prior))
    model._fit_result = fake_fit

    return model


@pytest.fixture(scope="module")
def fitted_gg(test_summary_data) -> GammaGammaModel:
    rng = np.random.default_rng(40)

    pd.Series({"p": 6.25, "q": 3.74, "v": 15.44})

    model = GammaGammaModel(
        customer_id=test_summary_data.index,
        mean_transaction_value=test_summary_data["monetary_value"],
        frequency=test_summary_data["frequency"],
        # Params used in lifetimes test
        p_prior=pm.DiracDelta.dist(6.25),
        q_prior=pm.DiracDelta.dist(3.74),
        v_prior=pm.DiracDelta.dist(15.44),
    )

    fake_fit = pm.sample_prior_predictive(
        samples=50, model=model.model, random_seed=rng
    )
    fake_fit.add_groups(dict(posterior=fake_fit.prior))
    model._fit_result = fake_fit

    return model


def test_customer_lifetime_value_with_known_values(test_summary_data, fitted_bg):
    # Test borrowed from
    # https://github.com/CamDavidsonPilon/lifetimes/blob/aae339c5437ec31717309ba0ec394427e19753c4/tests/test_utils.py#L527

    t = test_summary_data.head()

    expected = np.array([0.016053, 0.021171, 0.030461, 0.031686, 0.001607])
    monetary_value = np.ones_like(expected)

    # discount_rate=0 means the clv will be the same as the predicted
    clv_d0 = customer_lifetime_value(
        fitted_bg,
        t.index,
        t["frequency"],
        t["recency"],
        t["T"],
        monetary_value=monetary_value,
        time=1,
        discount_rate=0.0,
    ).mean(("chain", "draw"))
    np.testing.assert_almost_equal(clv_d0, expected, decimal=5)

    # discount_rate=1 means the clv will halve over a period
    clv_d1 = (
        customer_lifetime_value(
            fitted_bg,
            t.index,
            t["frequency"],
            t["recency"],
            t["T"],
            monetary_value=pd.Series([1, 1, 1, 1, 1]),
            time=1,
            discount_rate=1.0,
        )
        .mean(("chain", "draw"))
        .values
    )
    np.testing.assert_almost_equal(clv_d1, expected / 2.0, decimal=5)

    # time=2, discount_rate=0 means the clv will be twice the initial
    clv_t2_d0 = (
        customer_lifetime_value(
            fitted_bg,
            t.index,
            t["frequency"],
            t["recency"],
            t["T"],
            monetary_value=pd.Series([1, 1, 1, 1, 1]),
            time=2,
            discount_rate=0,
        )
        .mean(("chain", "draw"))
        .values
    )
    np.testing.assert_allclose(clv_t2_d0, expected * 2.0, rtol=0.1)

    # time=2, discount_rate=1 means the clv will be twice the initial
    clv_t2_d1 = (
        customer_lifetime_value(
            fitted_bg,
            t.index,
            t["frequency"],
            t["recency"],
            t["T"],
            monetary_value=pd.Series([1, 1, 1, 1, 1]),
            time=2,
            discount_rate=1.0,
        )
        .mean(("chain", "draw"))
        .values
    )
    np.testing.assert_allclose(clv_t2_d1, expected / 2.0 + expected / 4.0, rtol=0.1)


def test_customer_lifetime_value_gg_with_bgf(test_summary_data, fitted_gg, fitted_bg):
    t = test_summary_data.head()

    ggf_clv = fitted_gg.expected_customer_lifetime_value(
        transaction_model=fitted_bg,
        customer_id=t.index,
        frequency=t["frequency"],
        recency=t["recency"],
        T=t["T"],
        mean_transaction_value=t["monetary_value"],
    )

    utils_clv = customer_lifetime_value(
        transaction_model=fitted_bg,
        customer_id=t.index,
        frequency=t["frequency"],
        recency=t["recency"],
        T=t["T"],
        monetary_value=fitted_gg.expected_customer_spend(
            t.index,
            mean_transaction_value=t["monetary_value"],
            frequency=t["frequency"],
        ),
    )
    np.testing.assert_equal(ggf_clv.values, utils_clv.values)
