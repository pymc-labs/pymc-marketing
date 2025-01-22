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

import pytest

from pymc_marketing.mmm.causal import CausalGraphModel

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
