import numpy as np
import pandas as pd
import pytest

from pymc_marketing.mmm.preprocessing import (
    FourierTransformer,
    create_mmm_transformer,
    create_target_transformer,
)

seed: int = sum(map(ord, "pymc_marketing"))
rng: np.random.Generator = np.random.default_rng(seed=seed)


@pytest.fixture
def toy_X() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date_col": pd.date_range("2020-01-01", periods=4, freq="D"),
            "x1": [1, 2, 3, 4],
            "x2": [5, 6, 7, 8],
            "control_1": [9, 10, 11, 12],
            "control_2": [13, 14, 15, 16],
        },
        index=[100, 101, 102, 103],
    )


@pytest.fixture
def toy_X_new() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date_col": pd.date_range("2020-01-05", periods=1, freq="D"),
            "x1": [1],
            "x2": [5],
            "control_1": [9],
            "control_2": [13],
        },
        index=[104],
    )


@pytest.mark.parametrize("n_order", [1, 2, 3])
def test_fourier_transformer(n_order) -> None:
    transformer = FourierTransformer(n_order=n_order)

    n_fit = 4
    dates = pd.date_range("2023-01-01", periods=n_fit, freq="D")
    x_fit = pd.Series(dates, index=dates)
    transformer.fit(x_fit)

    assert len(transformer.columns) == 2 * n_order

    x_transform = transformer.transform(x_fit)
    assert x_transform.shape == (n_fit, 2 * n_order)

    assert x_transform.max().max() <= 1
    assert x_transform.min().min() >= -1

    x_new = pd.Series(pd.date_range("2023-01-01", periods=1, freq="D"))
    x_transform_new = transformer.transform(x_new)
    assert x_transform_new.shape == (1, 2 * n_order)

    assert x_transform_new.max().max() <= 1
    assert x_transform_new.min().min() >= -1


@pytest.mark.parametrize(
    "channel_cols, date_col, yearly_fourier_order, control_cols",
    [
        (["x1", "x2"], "date_col", 2, ["control_1", "control_2"]),
        # No fourier order
        (["x1", "x2"], "date_col", None, ["control_1", "control_2"]),
        # No control cols
        (["x1", "x2"], "date_col", 2, None),
        # Only channel cols
        (["x1", "x2"], "date_col", None, None),
    ],
)
def test_default_mmm_preprocessing(
    toy_X, toy_X_new, channel_cols, date_col, yearly_fourier_order, control_cols
) -> None:
    transformer = create_mmm_transformer(
        channel_cols=channel_cols,
        date_col=date_col,
        yearly_fourier_order=yearly_fourier_order,
        control_cols=control_cols,
    )

    # Call during fit
    X_transform = transformer.fit_transform(toy_X)
    # Call for new data
    X_transform_new = transformer.transform(toy_X_new)

    pd.testing.assert_index_equal(X_transform.index, toy_X.index)
    pd.testing.assert_index_equal(X_transform_new.index, toy_X_new.index)

    named_transformers = transformer.named_transformers_
    expect_cols = channel_cols

    if control_cols is not None:
        expect_cols += control_cols

    if "fourier_mode" in named_transformers:
        expect_cols += named_transformers["fourier_mode"].columns.tolist()

    for frame in [X_transform, X_transform_new]:
        pd.testing.assert_index_equal(frame.columns, pd.Index(expect_cols))


def test_target_transformer() -> None:
    target_transformer = create_target_transformer()

    y = pd.Series([1, 2, 3, 4], index=[100, 101, 102, 103])
    with pytest.raises(ValueError):
        target_transformer.fit_transform(y)

    y_transformed = target_transformer.fit_transform(y.to_numpy().reshape(-1, 1))
    assert y_transformed.shape == (4, 1)
    np.testing.assert_almost_equal(
        y_transformed, np.array([1 / 4, 2 / 4, 3 / 4, 1.0]).reshape(-1, 1)
    )

    y_new = pd.Series([5], index=[104])
    y_new_transformed = target_transformer.transform(y_new.to_numpy().reshape(-1, 1))
    assert y_new_transformed.shape == (1, 1)
    np.testing.assert_almost_equal(y_new_transformed, np.array([5 / 4]).reshape(-1, 1))
