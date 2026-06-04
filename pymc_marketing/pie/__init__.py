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
"""Predicted Incrementality by Experimentation (PIE).

The recommended entry point is :class:`PIEModel`, which wraps the model in a
:class:`~pymc_marketing.model_builder.RegressionModelBuilder` interface with
standard ``.fit()``, ``.save()``, and ``.load()`` methods.

Examples
--------
Fit on a corpus of past RCTs, then predict incrementality for new campaigns:

.. code-block:: python

    import pandas as pd
    from pymc_marketing.pie import PIEModel

    X = pd.DataFrame({...})
    y = pd.Series([...])

    model = PIEModel(
        pre_determined_features=["objective", "vertical", "budget"],
        post_determined_features=["exposure_rate"],
    )
    model.fit(X, y, random_seed=42)
    predictions = model.predict(X_new)
"""

from pymc_marketing.pie.model import PIEModel

__all__ = ["PIEModel"]
