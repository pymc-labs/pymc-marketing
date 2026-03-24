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
import importlib
from os import fspath
from pathlib import Path
from unittest.mock import patch

import pytest
from pyprojroot import here

from pymc_marketing import paths


def test_root_path() -> None:
    """Test that root path is correctly set using pyprojroot."""
    expected_root = here()
    assert paths.root == expected_root
    assert paths.root.exists()


def test_data_dir() -> None:
    """Test that data directory path is correctly defined when inside the repo."""
    expected_data_dir = here() / "data"
    assert paths.data_dir == expected_data_dir
    assert isinstance(paths.data_dir, Path)


def test_paths_are_absolute() -> None:
    """Test that root and data_dir are absolute paths when inside the repo."""
    assert paths.root is not None
    assert paths.root.is_absolute()
    assert isinstance(paths.data_dir, Path)
    assert paths.data_dir.is_absolute()


@pytest.fixture()
def no_project_root():
    with patch("pyprojroot.here", side_effect=RuntimeError("Project root not found.")):
        importlib.reload(paths)
        yield
    importlib.reload(paths)


def test_data_dir_falls_back_to_url_when_no_project_root(no_project_root) -> None:
    """Test that data_dir falls back to a URL when pyprojroot cannot find the project root."""
    assert paths.root is None
    assert isinstance(paths.data_dir, paths.URLPath)
    assert "main" in paths.data_dir.url


@pytest.fixture(scope="module")
def data_url() -> paths.URLPath:
    return paths.create_data_url("main")


def test_create_data_url(data_url) -> None:
    assert (
        data_url.url
        == "https://raw.githubusercontent.com/pymc-labs/pymc-marketing/main/data"
    )


def test_url_path_fspath(data_url) -> None:
    """Test the __fspath__ method of URLPath."""
    assert fspath(data_url) == data_url.url


def test_url_path_truediv(data_url) -> None:
    """Test the __truediv__ method of URLPath."""
    new_path = data_url / "new_file.csv"
    expected_url = "https://raw.githubusercontent.com/pymc-labs/pymc-marketing/main/data/new_file.csv"
    assert new_path.url == expected_url
    assert isinstance(new_path, paths.URLPath)
