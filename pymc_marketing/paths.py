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
"""Paths for the project.

The `data_dir` will be the local directory where the data files are stored in forked
repositories. If the directory does not exist, it will create a URLPath object pointing
to the data directory on the main branch of the pymc-labs/pymc-marketing repository.

"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike

from pyprojroot import here

root = here()
data_dir = root / "data"


URL = "https://raw.githubusercontent.com/pymc-labs/pymc-marketing/{branch}/data"


@dataclass
class URLPath(PathLike):
    """A class representing a URL path which can be used like a file path.

    Parameters
    ----------
    url : str
        The URL to the data directory or file.

    """

    url: str

    def __fspath__(self) -> str:
        """Return the URL as a string when the object is used as a file path."""
        return self.url

    def __truediv__(self, other: str) -> URLPath:
        """Combine the URL with another path component."""
        return URLPath(f"{self.url}/{other}")


def create_data_url(branch: str) -> URLPath:
    """Create a URLPath object for the data directory on a specific branch.

    Parameters
    ----------
    branch : str
        The branch name to create the URL for.

    Returns
    -------
    URLPath
        An object representing the URL path to the data directory on the specified branch.

    Examples
    --------
    Read MMM data from the main branch:

    .. code-block:: python

        import pandas as pd

        from pymc_marketing.paths import create_data_url

        data_dir = create_data_url("main")
        file = data_dir / "mmm_example.csv"
        df = pd.read_csv(file)

    """
    return URLPath(URL.format(branch=branch))


if not data_dir.exists():
    data_dir = create_data_url("main")
