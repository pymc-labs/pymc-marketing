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
import warnings

import numpy as np
import pandas as pd
import pytest

from pymc_marketing.mmm.causal import TBF_FCI, TBFPC, CausalGraphModel

# Suppress specific dowhy warnings globally
warnings.filterwarnings("ignore", message="The graph defines .* variables")


@pytest.mark.filterwarnings("ignore:The graph defines .* variables")
@pytest.mark.parametrize(
    "dag, treatment, outcome, expected_adjustment_set",
    [
        (
            """
            digraph {
                X -> Y;
                Z -> X;
                Z -> Y;
            }
            """,
            ["X"],
            "Y",
            ["Z"],  # Z is needed to block backdoor paths
        ),
        (
            """
            digraph {
                X -> Y;
                Z1 -> X;
                Z1 -> Y;
                Z2 -> X;
                Z2 -> Y;
            }
            """,
            ["X"],
            "Y",
            ["Z1", "Z2"],  # Both Z1 and Z2 are needed
        ),
        (
            """
            digraph {
                X -> Y;
            }
            """,
            ["X"],
            "Y",
            [],  # No adjustment is needed
        ),
        (
            """
            digraph {
                X1 -> Y;
                X2 -> Y;
                Z -> X1;
                Z -> X2;
                Z -> Y;
            }
            """,
            ["X1", "X2"],
            "Y",
            ["Z"],  # Z is needed for both treatments
        ),
    ],
    ids=[
        "simple_backdoor_path",
        "multiple_confounders",
        "no_confounders",
        "multiple_treatments",
    ],
)
def test_get_unique_adjustment_nodes(dag, treatment, outcome, expected_adjustment_set):
    causal_model = CausalGraphModel.build_graphical_model(
        graph=dag, treatment=treatment, outcome=outcome
    )
    adjustment_set = causal_model.get_unique_adjustment_nodes()
    assert set(adjustment_set) == set(expected_adjustment_set), (
        f"Expected {expected_adjustment_set}, but got {adjustment_set}"
    )


@pytest.mark.parametrize(
    "dag, treatment, outcome, control_columns, channel_columns, expected_controls",
    [
        (
            """
            digraph {
                X -> Y;
                Z -> X;
                Z -> Y;
            }
            """,
            ["X"],
            "Y",
            ["Z"],  # Control columns provided
            ["X"],  # Channels
            ["Z"],  # Z remains
        ),
        (
            """
            digraph {
                X -> Y;
                Z -> X;
                Z -> Y;
            }
            """,
            ["X"],
            "Y",
            ["W"],  # Irrelevant control
            ["X"],
            [],  # W is removed
        ),
        (
            """
            digraph {
                X -> Y;
                Z -> X;
                Z -> Y;
            }
            """,
            ["X"],
            "Y",
            None,  # No controls
            ["X"],
            None,  # Return None unchanged
        ),
        (
            """
            digraph {
                X -> Y;
                Z -> X;
                Z -> Y;
                W -> X;
            }
            """,
            ["X"],
            "Y",
            ["Z", "W", "V"],  # Mixed controls
            ["X"],
            ["Z"],  # Only Z remains, as W and V are irrelevant for adjustment
        ),
    ],
    ids=[
        "relevant_control",
        "irrelevant_control",
        "no_controls",
        "mixed_controls",
    ],
)
def test_compute_adjustment_sets(
    dag, treatment, outcome, control_columns, channel_columns, expected_controls
):
    causal_model = CausalGraphModel.build_graphical_model(
        graph=dag, treatment=treatment, outcome=outcome
    )
    adjusted_controls = causal_model.compute_adjustment_sets(
        control_columns=control_columns, channel_columns=channel_columns
    )
    assert adjusted_controls == expected_controls, (
        f"Expected {expected_controls}, but got {adjusted_controls}"
    )


@pytest.fixture(scope="module")
def df_non_ts() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    n = 100
    A = rng.gamma(2, 1, n)
    eB = rng.gamma(2, 1, n)
    eC = rng.gamma(2, 1, n)
    eY = rng.gamma(2, 1, n)

    B = 0.8 * A + eB
    C = eC
    Y = 0.5 * A + 0.9 * B + 0.7 * C + eY

    return pd.DataFrame({"A": A, "B": B, "C": C, "Y": Y})


@pytest.mark.parametrize("target_edge_rule", ["any", "conservative", "fullS"])
@pytest.mark.parametrize("bf_thresh", [0.5, 1.0, 2.0])
@pytest.mark.parametrize(
    "forbidden_edges",
    [
        [],
        [("A", "C")],  # edge not involving target is allowed
        [("A", "Y")],  # forbid a potential target edge
    ],
)
def test_tbfpc_public_api_types(
    df_non_ts: pd.DataFrame,
    target_edge_rule: str,
    bf_thresh: float,
    forbidden_edges,
):
    model = TBFPC(
        target="Y",
        target_edge_rule=target_edge_rule,
        bf_thresh=bf_thresh,
        forbidden_edges=forbidden_edges,
    )
    out = model.fit(df_non_ts, drivers=["A", "B", "C"])  # returns self

    assert out is model
    # public API returns
    assert isinstance(model.summary(), str)
    assert isinstance(model.to_digraph(), str)

    directed = model.get_directed_edges()
    undirected = model.get_undirected_edges()
    assert isinstance(directed, list)
    assert isinstance(undirected, list)
    if directed:
        assert all(
            isinstance(e, tuple) and len(e) == 2 and all(isinstance(x, str) for x in e)
            for e in directed
        )
    if undirected:
        assert all(
            isinstance(e, tuple) and len(e) == 2 and all(isinstance(x, str) for x in e)
            for e in undirected
        )


def test_tbfpc_invalid_drivers_raises(df_non_ts: pd.DataFrame):
    model = TBFPC(target="Y", target_edge_rule="fullS")
    with pytest.raises(KeyError):
        model.fit(df_non_ts, drivers=["A", "B", "D"])  # "D" not in df


@pytest.mark.parametrize("edge_rule", ["random", "", None])
def test_tbfpc_invalid_edge_rule_raises(edge_rule):
    with pytest.raises(ValueError):
        TBFPC(target="Y", target_edge_rule=edge_rule)  # type: ignore[arg-type]


def test_tbfpc_emits_experimental_warning(df_non_ts: pd.DataFrame):
    with pytest.warns(UserWarning, match="experimental"):
        TBFPC(target="Y", target_edge_rule="fullS")


@pytest.mark.parametrize("bf", [0, -1.0])
def test_tbfpc_invalid_bf_thresh_raises(bf):
    with pytest.raises(ValueError):
        TBFPC(target="Y", bf_thresh=bf)  # type: ignore[arg-type]


def test_tbfpc_internal_key_and_sep():
    m = TBFPC(target="Y")
    # _key should sort endpoints
    assert m._key("B", "A") == ("A", "B")
    m._set_sep("A", "B", ["C"])
    assert m.sep_sets[("A", "B")] == {"C"}


def test_tbfpc_has_forbidden_blocks_edges(df_non_ts: pd.DataFrame):
    m = TBFPC(target="Y", forbidden_edges=[("A", "Y")])
    # if forbidden, CI returns True (treat as independent)
    assert m._has_forbidden("A", "Y") is True
    # Build minimal state to call _ci_independent
    m.fit(df_non_ts, drivers=["A", "B", "C"])  # initializes y_sh and bic_fn
    assert m._ci_independent(df_non_ts, "A", "Y", []) is True


@pytest.fixture(scope="module")
def df_ts() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    n = 300
    x1 = rng.uniform(low=0.0, high=1.0, size=n)
    X1_t = np.where(x1 > 0.9, x1, x1 / 2)

    x2 = rng.uniform(low=0.3, high=1.0, size=n)
    X2_t = np.where(x2 > 0.8, x2, x2 / 4)

    x3 = rng.uniform(low=0.0, high=1.0, size=n)
    X3_t = x3 + (X2_t * 0.2)

    Y_t = (
        (X1_t * 0.2)
        + (X2_t * 0.1)
        + (X3_t * 0.3)
        + rng.normal(loc=0.0, scale=0.05, size=n)
    )

    return pd.DataFrame({"X1": X1_t, "X2": X2_t, "X3": X3_t, "Y": Y_t})


@pytest.mark.parametrize("target_edge_rule", ["any", "conservative", "fullS"])
@pytest.mark.parametrize("bf_thresh", [0.5, 1.0, 2.0])
@pytest.mark.parametrize(
    "forbidden_edges",
    [
        [],
        [("X2", "Y"), ("X1", "X2")],
        [("X1", "Y")],
    ],
)
def test_tbf_fci_public_api_types(
    df_ts: pd.DataFrame,
    target_edge_rule: str,
    bf_thresh: float,
    forbidden_edges,
):
    model = TBF_FCI(
        target="Y",
        target_edge_rule=target_edge_rule,
        bf_thresh=bf_thresh,
        forbidden_edges=forbidden_edges,
        max_lag=1,
        allow_contemporaneous=True,
    )
    out = model.fit(df_ts, drivers=["X1", "X2", "X3"])  # returns self

    assert out is model
    # public API returns
    assert isinstance(model.summary(), str)
    assert isinstance(model.to_digraph(collapsed=False), str)
    assert isinstance(model.to_digraph(collapsed=True), str)

    directed = model.get_directed_edges()
    undirected = model.get_undirected_edges()
    assert isinstance(directed, list)
    assert isinstance(undirected, list)
    if directed:
        assert all(
            isinstance(e, tuple) and len(e) == 2 and all(isinstance(x, str) for x in e)
            for e in directed
        )
    if undirected:
        assert all(
            isinstance(e, tuple) and len(e) == 2 and all(isinstance(x, str) for x in e)
            for e in undirected
        )

    collapsed_directed, collapsed_undirected = model.collapsed_summary()
    assert isinstance(collapsed_directed, list)
    assert isinstance(collapsed_undirected, list)
    for e in collapsed_directed:
        assert isinstance(e, tuple) and len(e) == 3
        u, v, lag = e
        assert isinstance(u, str) and isinstance(v, str) and isinstance(lag, int)
    for e in collapsed_undirected:
        assert isinstance(e, tuple) and len(e) == 2
        u, v = e
        assert isinstance(u, str) and isinstance(v, str)


@pytest.mark.parametrize("edge_rule", ["random", "", None])
def test_tbf_fci_invalid_edge_rule_raises(edge_rule):
    with pytest.raises(ValueError):
        TBF_FCI(target="Y", target_edge_rule=edge_rule)  # type: ignore[arg-type]


@pytest.mark.parametrize("bf", [0, -1.0])
def test_tbf_fci_invalid_bf_thresh_raises(bf):
    with pytest.raises(ValueError):
        TBF_FCI(target="Y", bf_thresh=bf)  # type: ignore[arg-type]


@pytest.mark.parametrize("lag", [-1, 1.5])
def test_tbf_fci_invalid_max_lag_raises(lag):
    with pytest.raises(ValueError):
        TBF_FCI(target="Y", max_lag=lag)  # type: ignore[arg-type]


def test_tbf_fci_emits_experimental_warning(df_ts: pd.DataFrame):
    with pytest.warns(UserWarning, match="experimental"):
        TBF_FCI(target="Y", max_lag=1)


def test_tbf_fci_lag_naming_and_parsing():
    m = TBF_FCI(target="Y", max_lag=2)
    assert m._lag_name("X", 0) == "X[t]"
    assert m._lag_name("X", 2) == "X[t-2]"
    assert m._parse_lag("X[t]") == ("X", 0)
    assert m._parse_lag("X[t-2]") == ("X", 2)


@pytest.mark.parametrize(
    "forbidden_in,expected_contains",
    [
        ([("X1", "Y")], {("X1[t]", "Y[t]")}),
        ([("X1", "Y")], {("X1[t-1]", "Y[t]")}),
        ([("X2[t]", "Y[t]")], {("X2[t]", "Y[t]")}),
    ],
)
def test_tbf_fci_expand_edges(forbidden_in, expected_contains):
    m = TBF_FCI(target="Y", max_lag=1, forbidden_edges=forbidden_in)
    # All expected edges should be in expanded set
    assert expected_contains.issubset(m.forbidden_edges)


def test_tbf_fci_admissible_cond_set(df_ts: pd.DataFrame):
    m = TBF_FCI(target="Y", max_lag=1)
    all_vars = ["X1[t]", "X1[t-1]", "X2[t]", "X2[t-1]", "Y[t]"]
    # conditioning for (X1[t-1], Y[t]) can include same-time and earlier variables
    cand = m._admissible_cond_set(all_vars, "X1[t-1]", "Y[t]")
    # excludes the tested variables themselves (X1[t-1], Y[t])
    assert set(cand).issuperset({"X1[t]", "X2[t-1]", "X2[t]"})


def test_tbf_fci_invalid_drivers_raises(df_ts: pd.DataFrame):
    model = TBF_FCI(
        target="Y", target_edge_rule="fullS", max_lag=1, allow_contemporaneous=True
    )
    with pytest.raises(KeyError):
        model.fit(df_ts, drivers=["X1", "X2", "X9"])  # "X9" not in df
