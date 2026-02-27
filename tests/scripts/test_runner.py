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
"""Tests for the notebook runner script."""

import sys
from pathlib import Path

# Add scripts to path for import
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent / "scripts" / "run_notebooks")
)

from runner import (
    BLACKLIST,
    clear_cell_outputs,
    filter_blacklist,
    run_parameters,
)


class TestFilterBlacklist:
    """Tests for the filter_blacklist function."""

    def test_filters_blacklisted_notebook(self):
        """Test that blacklisted notebooks are filtered out."""
        notebooks = [Path("docs/source/notebooks/mmm/mmm_chronos.ipynb")]
        result = filter_blacklist(notebooks)
        assert result == []

    def test_keeps_non_blacklisted_notebook(self):
        """Test that non-blacklisted notebooks are kept."""
        notebooks = [Path("docs/source/notebooks/mmm/mmm_example.ipynb")]
        result = filter_blacklist(notebooks)
        assert result == notebooks

    def test_mixed_list(self):
        """Test filtering a list with both blacklisted and non-blacklisted notebooks."""
        notebooks = [
            Path("docs/source/notebooks/mmm/mmm_chronos.ipynb"),
            Path("docs/source/notebooks/mmm/mmm_example.ipynb"),
        ]
        result = filter_blacklist(notebooks)
        assert result == [Path("docs/source/notebooks/mmm/mmm_example.ipynb")]

    def test_empty_list(self):
        """Test filtering an empty list."""
        result = filter_blacklist([])
        assert result == []

    def test_blacklist_contains_chronos(self):
        """Test that the BLACKLIST contains the expected notebook."""
        assert "docs/source/notebooks/mmm/mmm_chronos.ipynb" in BLACKLIST


class TestClearCellOutputs:
    """Tests for the clear_cell_outputs function."""

    def test_clears_code_cell_outputs(self):
        """Test that code cell outputs are cleared."""
        cells = [
            {"cell_type": "code", "outputs": ["some output"], "execution_count": 1}
        ]
        clear_cell_outputs(cells)
        assert cells[0]["outputs"] == []
        assert cells[0]["execution_count"] is None

    def test_ignores_markdown_cells(self):
        """Test that markdown cells are not affected."""
        cells = [{"cell_type": "markdown", "source": "# Title"}]
        clear_cell_outputs(cells)
        assert "outputs" not in cells[0]

    def test_clears_multiple_code_cells(self):
        """Test clearing outputs from multiple code cells."""
        cells = [
            {"cell_type": "code", "outputs": ["output1"], "execution_count": 1},
            {"cell_type": "markdown", "source": "# Title"},
            {"cell_type": "code", "outputs": ["output2"], "execution_count": 2},
        ]
        clear_cell_outputs(cells)
        assert cells[0]["outputs"] == []
        assert cells[0]["execution_count"] is None
        assert "outputs" not in cells[1]
        assert cells[2]["outputs"] == []
        assert cells[2]["execution_count"] is None


class TestRunParameters:
    """Tests for the run_parameters function."""

    def test_creates_run_params(self):
        """Test that run_parameters creates correct RunParams for each notebook."""
        notebooks = [Path("notebook1.ipynb"), Path("notebook2.ipynb")]
        result = run_parameters(notebooks)
        assert len(result) == 2
        assert result[0]["notebook_path"] == Path("notebook1.ipynb")
        assert result[0]["mock"] is True
        assert result[1]["notebook_path"] == Path("notebook2.ipynb")
        assert result[1]["mock"] is True

    def test_empty_list(self):
        """Test run_parameters with an empty list."""
        result = run_parameters([])
        assert result == []
