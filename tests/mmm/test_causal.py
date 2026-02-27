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
import warnings

import graphviz
import networkx as nx
import numpy as np
import pandas as pd
import pymc as pm
import pytest
from pydantic import ValidationError
from pymc_extras.prior import Prior

from pymc_marketing.mmm.causal import (
    TBFPC,
    BuildModelFromDAG,
    CausalGraphModel,
)

# Suppress specific dowhy warnings globally
warnings.filterwarnings("ignore", message="The graph defines .* variables")


@pytest.fixture
def causal_df():
    rng = np.random.default_rng(123)
    N = 500
    Q = rng.normal(size=N)
    X = rng.normal(loc=0.14 * Q, scale=0.4, size=N)
    Y = rng.normal(loc=0.7 * X + 0.11 * Q, scale=0.24, size=N)
    P = rng.normal(loc=0.43 * X + 0.21 * Y, scale=0.22, size=N)

    dates = pd.date_range("2023-01-01", periods=N, freq="D")
    return pd.DataFrame({"date": dates, "Q": Q, "X": X, "Y": Y, "P": P})


def test_build_raises_when_coords_key_not_in_df(causal_df):
    dag = """
    digraph {
        Q -> X;
        X -> Y;
        Y -> P;
    }
    """
    # Inject an extra coordinate not present in the dataframe columns
    coords = {"date": causal_df["date"].to_numpy()}
    coords["ghost"] = np.arange(len(causal_df))

    with pytest.raises(
        KeyError, match="Coordinate key 'ghost' not found in DataFrame columns"
    ):
        BuildModelFromDAG(
            dag=dag,
            df=causal_df,
            target="Y",
            dims=("date",),
            coords=coords,
        )


def test_build_raises_when_df_missing_column_present_in_coords(causal_df):
    dag = """
    digraph {
        Q -> X;
        X -> Y;
        Y -> P;
    }
    """
    # Inject an extra coordinate not present in the dataframe columns
    coords = {"date": causal_df["date"].to_numpy()}
    df_missing_date = causal_df.drop(columns=["date"])  # Remove date from dataset

    with pytest.raises(
        KeyError, match="Coordinate key 'date' not found in DataFrame columns"
    ):
        BuildModelFromDAG(
            dag=dag,
            df=df_missing_date,
            target="Y",
            dims=("date",),
            coords=coords,
        )


def test_build_with_custom_priors_builds(causal_df):
    dag = """
    digraph {
        Q -> X;
        X -> Y;
        Y -> P;
    }
    """

    # Custom priors with matching dims expectation (likelihood has 'date', slope has no dims)
    custom_config = {
        "intercept": Prior("Normal", mu=0, sigma=0.5),
        "slope": Prior("Normal", mu=0, sigma=0.5),  # no dims implies ()
        "likelihood": Prior(
            "Normal", sigma=Prior("HalfNormal", sigma=0.5), dims=("date",)
        ),
    }

    coords = {"date": causal_df["date"].unique()}

    builder = BuildModelFromDAG(
        dag=dag,
        df=causal_df,
        target="Y",
        dims=("date",),
        coords=coords,
        model_config=custom_config,
    )

    model = builder.build()
    assert isinstance(model, pm.Model)


def test_warning_when_slope_dims_missing_vs_likelihood_dims(causal_df):
    dag = """
    digraph {
        Q -> X;
        X -> Y;
        Y -> P;
    }
    """

    causal_df["country"] = "Venezuela"

    custom_config = {
        "intercept": Prior("Normal", mu=0, sigma=1),
        "slope": Prior("Normal", mu=0, sigma=1),  # no dims
        "likelihood": Prior(
            "Normal", sigma=Prior("HalfNormal", sigma=1), dims=("date", "country")
        ),
    }

    coords = {
        "date": causal_df["date"].unique(),
        "country": causal_df["country"].unique(),
    }

    with pytest.warns(UserWarning, match="Slope prior dims"):
        builder = BuildModelFromDAG(
            dag=dag,
            df=causal_df,
            target="Y",
            dims=("date", "country"),
            coords=coords,
            model_config=custom_config,
        )
        model = builder.build()
        assert isinstance(model, pm.Model)


def test_no_warning_when_slope_dims_match_likelihood_dims(causal_df):
    dag = """
    digraph {
        Q -> X;
        X -> Y;
        Y -> P;
    }
    """

    causal_df["country"] = "Venezuela"

    custom_config = {
        "intercept": Prior("Normal", mu=0, sigma=1, dims=("country",)),
        "slope": Prior("Normal", mu=0, sigma=1, dims=("country",)),
        "likelihood": Prior(
            "Normal", sigma=Prior("HalfNormal", sigma=1), dims=("date", "country")
        ),
    }

    coords = {
        "date": causal_df["date"].unique(),
        "country": causal_df["country"].unique(),
    }

    builder = BuildModelFromDAG(
        dag=dag,
        df=causal_df,
        target="Y",
        dims=("date", "country"),
        coords=coords,
        model_config=custom_config,
    )
    model = builder.build()

    assert isinstance(model, pm.Model)


def test_error_when_likelihood_dims_differ_from_class_dims(causal_df):
    dag = """
    digraph {
        Q -> X;
        X -> Y;
        Y -> P;
    }
    """

    causal_df["country"] = "Venezuela"

    # Class dims only includes date, while likelihood dims include date and country -> should error
    custom_config = {
        "intercept": Prior("Normal", mu=0, sigma=1),
        "slope": Prior("Normal", mu=0, sigma=1),
        "likelihood": Prior(
            "Normal", sigma=Prior("HalfNormal", sigma=1), dims=("date", "country")
        ),
    }

    coords = {
        "date": causal_df["date"].unique(),
        "country": causal_df["country"].unique(),
    }

    with pytest.raises(
        ValueError, match=r"Likelihood Prior dims .* must match class dims .*"
    ):
        BuildModelFromDAG(
            dag=dag,
            df=causal_df,
            target="Y",
            dims=("date",),
            coords=coords,
            model_config=custom_config,
        )


def test_model_and_dag_graph_return_types(causal_df):
    dag = """
    digraph {
        Q -> X;
        X -> Y;
    }
    """

    coords = {"date": causal_df["date"].unique()}

    builder = BuildModelFromDAG(
        dag=dag,
        df=causal_df,
        target="Y",
        dims=("date",),
        coords=coords,
    )
    model = builder.build()
    assert isinstance(model, pm.Model)

    mg = builder.model_graph()
    dg = builder.dag_graph()
    assert isinstance(mg, graphviz.Digraph)
    assert isinstance(dg, nx.DiGraph)


def test_default_model_config_contents_and_types(causal_df):
    dag = """
    digraph {
        Q -> X;
        X -> Y;
    }
    """

    coords = {"date": causal_df["date"].unique()}

    builder = BuildModelFromDAG(
        dag=dag,
        df=causal_df,
        target="Y",
        dims=("date",),
        coords=coords,
    )

    cfg = builder.model_config
    assert set(cfg.keys()) >= {"intercept", "slope", "likelihood"}
    assert isinstance(cfg["intercept"], Prior)
    assert isinstance(cfg["slope"], Prior)
    assert isinstance(cfg["likelihood"], Prior)

    # Check default dims
    like_dims = cfg["likelihood"].dims
    if isinstance(like_dims, str):
        like_dims = (like_dims,)
    elif isinstance(like_dims, list):
        like_dims = tuple(like_dims)
    assert like_dims == ("date",)

    slope_dims = cfg["slope"].dims
    if slope_dims is None:
        slope_dims = tuple()
    elif isinstance(slope_dims, str):
        slope_dims = (slope_dims,)
    elif isinstance(slope_dims, list):
        slope_dims = tuple(slope_dims)
    intercept_dims = cfg["intercept"].dims
    if intercept_dims is None:
        intercept_dims = tuple()
    elif isinstance(intercept_dims, str):
        intercept_dims = (intercept_dims,)
    elif isinstance(intercept_dims, list):
        intercept_dims = tuple(intercept_dims)
    # Expect dims without 'date' -> empty tuple
    assert slope_dims == tuple()
    assert intercept_dims == slope_dims


def test_parse_dag_parses_dot_and_simple_formats():
    # DOT format
    dag_dot = """
    digraph {
        A -> B;
        B -> C;
    }
    """
    g_dot = BuildModelFromDAG._parse_dag(dag_dot)
    assert isinstance(g_dot, nx.DiGraph)
    assert set(g_dot.edges()) == {("A", "B"), ("B", "C")}

    # Simple A->B tokens format
    dag_simple = "A->B, B->C"
    g_simple = BuildModelFromDAG._parse_dag(dag_simple)
    assert isinstance(g_simple, nx.DiGraph)
    assert set(g_simple.edges()) == {("A", "B"), ("B", "C")}

    # Cycle should raise
    with pytest.raises(ValueError, match="not a DAG"):
        BuildModelFromDAG._parse_dag("A->B, B->A")


def test_init_raises_when_target_not_in_dag(causal_df):
    dag = """
    digraph {
        A -> B;
    }
    """

    coords = {"date": causal_df["date"].unique()}

    with pytest.raises(ValueError, match=r"Target 'Z' not in DAG nodes"):
        BuildModelFromDAG(
            dag=dag,
            df=causal_df.rename(columns={"Q": "A", "X": "B"}),
            target="Z",
            dims=("date",),
            coords=coords,
        )


def test_parse_dag_malformed_dot_raises():
    malformed = "digraph { A -> B;"  # missing closing brace
    with pytest.raises(ValueError, match="Malformed DOT digraph: missing braces"):
        BuildModelFromDAG._parse_dag(malformed)


def test_parse_dag_handles_comments_and_standalone_nodes():
    dag = """
    digraph {
        // comment line
        A;
        A -> B; // edge comment
        C; # standalone node with hash comment
        B -> C;
    }
    """
    g = BuildModelFromDAG._parse_dag(dag)
    assert set(g.edges()) == {("A", "B"), ("B", "C")}
    assert set(g.nodes()) >= {"A", "B", "C"}


def test_parse_dag_invalid_simple_token_raises():
    with pytest.raises(ValueError, match="Invalid edge token"):
        BuildModelFromDAG._parse_dag("A-B, C->D")


def test_validate_coords_raises_when_coords_none(causal_df):
    dag = """
    digraph {
        Q -> X;
    }
    """
    # Pydantic validate_call intercepts before our internal check
    with pytest.raises(ValidationError):
        BuildModelFromDAG(
            dag=dag,
            df=causal_df,
            target="X",
            dims=("date",),
            coords=None,
        )


def test_validate_coords_raises_when_dim_missing_in_coords(causal_df):
    dag = """
    digraph {
        Q -> X;
    }
    """
    causal_df["country"] = "Venezuela"
    coords = {"date": causal_df["date"].unique()}
    with pytest.raises(
        ValueError, match=r"Missing coordinate values for dim 'country'"
    ):
        BuildModelFromDAG(
            dag=dag,
            df=causal_df,
            target="X",
            dims=("date", "country"),
            coords=coords,
        )


def test_validate_coords_raises_when_prior_dims_not_in_coords(causal_df):
    dag = """
    digraph {
        Q -> X;
    }
    """
    coords = {"date": causal_df["date"].unique()}
    custom_config = {
        prior_name: Prior("Normal", mu=0, sigma=1, dims=("country",))
        for prior_name in ("intercept", "slope")
    }
    with pytest.raises(
        ValueError,
        match=r"Dim 'country' declared in Prior '(?:intercept|slope)' must be present in coords",
    ):
        BuildModelFromDAG(
            dag=dag,
            df=causal_df,
            target="X",
            dims=("date",),
            coords=coords,
            model_config=custom_config,
        )


def test_no_warning_when_dims_given_as_str_and_list(causal_df):
    dag = """
    digraph {
        Q -> X;
        X -> Y;
    }
    """
    causal_df["country"] = "Venezuela"
    custom_config = {
        "intercept": Prior("Normal", mu=0, sigma=1, dims="country"),
        "slope": Prior("Normal", mu=0, sigma=1, dims="country"),
        "likelihood": Prior(
            "Normal", sigma=Prior("HalfNormal", sigma=1), dims=["date", "country"]
        ),
    }
    coords = {
        "date": causal_df["date"].unique(),
        "country": causal_df["country"].unique(),
    }
    builder = BuildModelFromDAG(
        dag=dag,
        df=causal_df,
        target="Y",
        dims=("date", "country"),
        coords=coords,
        model_config=custom_config,
    )
    model = builder.build()
    assert isinstance(model, pm.Model)


def test_likelihood_dims_none_init_ok(causal_df):
    dag = """
    digraph {
        Q -> X;
    }
    """
    coords = {"date": causal_df["date"].unique()}
    custom_config = {
        "intercept": Prior("Normal", mu=0, sigma=1),
        "slope": Prior("Normal", mu=0, sigma=1),
        "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=1), dims=None),
    }
    builder = BuildModelFromDAG(
        dag=dag,
        df=causal_df,
        target="X",
        dims=("date",),
        coords=coords,
        model_config=custom_config,
    )
    assert isinstance(builder, BuildModelFromDAG)


def test_validate_coords_required_raises_valueerror_when_none(causal_df):
    """Test that directly calling _validate_coords_required_are_consistent raises ValueError when coords is None."""
    dag = """
    digraph {
        Q -> X;
    }
    """

    # Create builder without going through pydantic validation
    builder = object.__new__(BuildModelFromDAG)
    builder.dag = dag
    builder.df = causal_df
    builder.target = "X"
    builder.dims = ("date",)
    builder.coords = None  # Explicitly set to None
    builder.graph = BuildModelFromDAG._parse_dag(dag)
    builder.nodes = list(nx.topological_sort(builder.graph))
    builder.model_config = {
        "intercept": Prior("Normal", mu=0, sigma=1),
        "slope": Prior("Normal", mu=0, sigma=1),
        "likelihood": Prior(
            "Normal", sigma=Prior("HalfNormal", sigma=1), dims=("date",)
        ),
    }

    # This should raise the specific ValueError
    with pytest.raises(ValueError, match=r"'coords' is required and cannot be None\."):
        builder._validate_coords_required_are_consistent()


def test_error_when_likelihood_in_model_config_is_none(causal_df):
    dag = """
    digraph {
        Q -> X;
    }
    """
    coords = {"date": causal_df["date"].unique()}
    with pytest.raises(
        TypeError, match=r"model_config\['likelihood'\] must be a Prior"
    ):
        BuildModelFromDAG(
            dag=dag,
            df=causal_df,
            target="X",
            dims=("date",),
            coords=coords,
            model_config={
                "intercept": Prior("Normal", mu=0, sigma=1),
                "likelihood": None,
                "slope": Prior("Normal", mu=0, sigma=1),
            },
        )


def test_build_raises_when_missing_column_from_df(causal_df):
    dag = """
    digraph {
        A -> B;
    }
    """
    # Create df missing column 'B'
    df = causal_df.rename(columns={"Q": "A"})[["date", "A"]]
    coords = {"date": df["date"].unique()}
    builder = BuildModelFromDAG(
        dag=dag,
        df=df,
        target="B",
        dims=("date",),
        coords=coords,
    )
    with pytest.raises(KeyError, match="Column 'B' not found in df"):
        builder.build()


def test_model_graph_raises_when_called_before_build(causal_df):
    dag = """
    digraph {
        Q -> X;
    }
    """
    coords = {"date": causal_df["date"].unique()}
    builder = BuildModelFromDAG(
        dag=dag,
        df=causal_df,
        target="X",
        dims=("date",),
        coords=coords,
    )
    with pytest.raises(RuntimeError, match=r"Call build\(\) first"):
        builder.model_graph()


def test_default_model_config_slope_dims_excludes_date_multi_dim(causal_df):
    dag = """
    digraph {
        Q -> X;
        X -> Y;
    }
    """
    causal_df["country"] = "Venezuela"
    coords = {
        "date": causal_df["date"].unique(),
        "country": causal_df["country"].unique(),
    }
    builder = BuildModelFromDAG(
        dag=dag,
        df=causal_df,
        target="Y",
        dims=("date", "country"),
        coords=coords,
    )
    slope_dims = builder.model_config["slope"].dims
    if isinstance(slope_dims, str):
        slope_dims = (slope_dims,)
    elif isinstance(slope_dims, list):
        slope_dims = tuple(slope_dims)
    elif slope_dims is None:
        slope_dims = tuple()
    assert slope_dims == ("country",)


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
