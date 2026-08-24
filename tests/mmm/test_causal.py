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
"""Tests for :mod:`pymc_marketing.mmm.causal`.

The causal algorithms themselves live in (and are tested by) pathmc. What is
covered here is this package's side of the contract: ``CausalGraphModel``
turning a DAG into the adjustment set the MMM consumes, and the lazy wiring that
keeps ``pathmc`` an optional dependency.
"""

import importlib
import sys

import pathmc
import pytest

from pymc_marketing.mmm import causal as causal_module
from pymc_marketing.mmm.causal import CausalGraphModel


@pytest.fixture
def causal_without_pathmc(monkeypatch):
    """Reload the causal module with ``import pathmc`` failing.

    ``None`` in ``sys.modules`` makes ``import pathmc`` raise ``ImportError``,
    which is what an environment without the ``dag`` extra looks like.
    """
    monkeypatch.setitem(sys.modules, "pathmc", None)
    module = importlib.reload(causal_module)
    yield module
    monkeypatch.undo()
    importlib.reload(causal_module)


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
        (
            # X <- Z1 -> C <- Z2 -> Y is a backdoor path already blocked by the
            # collider C: adjusting for any of its nodes would open it instead.
            """
            digraph {
                Z1 -> X;
                Z1 -> C;
                Z2 -> C;
                Z2 -> Y;
                X -> Y;
            }
            """,
            ["X"],
            "Y",
            [],
        ),
    ],
    ids=[
        "simple_backdoor_path",
        "multiple_confounders",
        "no_confounders",
        "multiple_treatments",
        "collider_blocked_backdoor_path",
    ],
)
def test_get_unique_adjustment_nodes(dag, treatment, outcome, expected_adjustment_set):
    causal_model = CausalGraphModel.build_graphical_model(
        graph=dag, treatment=treatment, outcome=outcome
    )
    adjustment_set = causal_model.get_unique_adjustment_nodes()
    assert adjustment_set == expected_adjustment_set, (
        f"Expected {expected_adjustment_set}, but got {adjustment_set}"
    )


def test_build_graphical_model_is_pathmc_backed():
    causal_model = CausalGraphModel.build_graphical_model(
        graph="digraph { Z -> X; Z -> Y; X -> Y; }", treatment=["X"], outcome="Y"
    )
    assert isinstance(causal_model.path_model, pathmc.PathModel)
    assert causal_model.path_model.adjustment_sets("X", "Y") == [{"Z"}]


def test_build_graphical_model_accepts_commented_dot():
    """A ``graphviz.Digraph(comment=...)`` source keeps working as a DAG string."""
    causal_model = CausalGraphModel.build_graphical_model(
        graph="// True Causal DAG\ndigraph {\n Z -> X;\n Z -> Y;\n X -> Y;\n}\n",
        treatment=["X"],
        outcome="Y",
    )
    assert causal_model.get_unique_adjustment_nodes() == ["Z"]


def test_treatment_absent_from_dag_raises():
    causal_model = CausalGraphModel.build_graphical_model(
        graph="digraph { Z -> X; Z -> Y; X -> Y; }",
        treatment=["not_a_node"],
        outcome="Y",
    )
    with pytest.raises(ValueError, match="Treatment 'not_a_node' not in DAG"):
        causal_model.get_unique_adjustment_nodes()


def test_unidentifiable_treatment_warns():
    """A latent confounder leaves no adjustment set, which must not pass silently."""
    spec = pathmc.dag_to_spec("digraph { W -> U; U -> X; U -> Y; X -> Y; }")
    causal_model = CausalGraphModel(
        pathmc.model(spec, latent=["U"]), treatment=["X"], outcome="Y"
    )

    with pytest.warns(UserWarning, match="not identifiable by backdoor adjustment"):
        adjustment_set = causal_model.get_unique_adjustment_nodes()

    assert adjustment_set == []


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
            [
                "Z2",
                "Z1",
            ],  # Order of the retained controls must not depend on input order
            ["X"],
            ["Z1", "Z2"],
        ),
    ],
    ids=[
        "relevant_control",
        "irrelevant_control",
        "no_controls",
        "mixed_controls",
        "sorted_controls",
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


# ---------------------------------------------------------------------------
# pathmc wiring
# ---------------------------------------------------------------------------


def test_discovery_tools_reexport_from_pathmc():
    """TBFPC / BuildModelFromDAG / TestResult resolve to pathmc's objects."""
    assert causal_module.TBFPC is pathmc.TBFPC
    assert causal_module.BuildModelFromDAG is pathmc.BuildModelFromDAG
    assert causal_module.TestResult is pathmc.TestResult


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="does_not_exist"):
        _ = causal_module.does_not_exist


def test_module_imports_without_pathmc(causal_without_pathmc):
    """Importing the module (and with it the core MMM) must not need pathmc."""
    assert causal_without_pathmc.CausalGraphModel is not None


@pytest.mark.parametrize("name", ["TBFPC", "BuildModelFromDAG", "TestResult"])
def test_reexports_error_actionably_without_pathmc(causal_without_pathmc, name):
    with pytest.raises(ImportError, match=r"pip install pymc-marketing\[dag\]"):
        getattr(causal_without_pathmc, name)


def test_build_graphical_model_errors_actionably_without_pathmc(causal_without_pathmc):
    with pytest.raises(ImportError, match=r"pip install pymc-marketing\[dag\]"):
        causal_without_pathmc.CausalGraphModel.build_graphical_model(
            graph="digraph { X -> Y; }", treatment=["X"], outcome="Y"
        )
