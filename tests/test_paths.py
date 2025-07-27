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
from os import fspath
from pathlib import Path

import pytest
from pyprojroot import here

from pymc_marketing import paths


def test_root_path() -> None:
    """Test that root path is correctly set using pyprojroot."""
    expected_root = here()
    assert paths.root == expected_root
    assert paths.root.exists()


def test_data_dir() -> None:
    """Test that data directory path is correctly defined."""
    expected_data_dir = here() / "data"
    assert paths.data_dir == expected_data_dir
    # Note: We don't assert exists() here as the data dir might not exist in CI
    assert isinstance(paths.data_dir, Path)


def test_paths_are_absolute() -> None:
    """Test that all defined paths are absolute."""
    assert paths.root.is_absolute()
    assert paths.data_dir.is_absolute()


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
