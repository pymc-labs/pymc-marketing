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
"""Run the doctest examples embedded in docstrings.

Docstring examples silently rot when the code around them changes, since
nothing normally executes them. This test runs the ones that are
self-contained (no fitted model or external data required) so a broken
example fails CI instead of confusing the next reader.

Only modules with such self-contained examples are listed here. Most
docstrings in this codebase illustrate usage against a fitted ``MMM``/
``NestedLogit`` model or an active ``pymc.Model`` context and are not
meant to run standalone; those are intentionally left out (or marked
``# doctest: +SKIP`` at the source) rather than forced into this list.
"""

import doctest

import pytest

from pymc_marketing.customer_choice import nested_logit
from pymc_marketing.mmm import tvp

DOCTEST_MODULES = [nested_logit, tvp]


@pytest.mark.parametrize(
    "module", DOCTEST_MODULES, ids=[module.__name__ for module in DOCTEST_MODULES]
)
def test_docstring_examples(module) -> None:
    results = doctest.testmod(module, optionflags=doctest.ELLIPSIS)
    assert results.failed == 0, (
        f"{results.failed} doctest example(s) failed in {module.__name__}"
    )
