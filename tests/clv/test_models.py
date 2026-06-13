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
import pandas as pd
import pytest
from pymc_extras.prior import Prior

from pymc_marketing.clv.models.basic import CLVModel
from pymc_marketing.clv.models.beta_geo import BetaGeoModel
from pymc_marketing.clv.models.beta_geo_beta_binom import BetaGeoBetaBinomModel
from pymc_marketing.clv.models.gamma_gamma import GammaGammaModel
from pymc_marketing.clv.models.modified_beta_geo import ModifiedBetaGeoModel
from pymc_marketing.clv.models.pareto_nbd import ParetoNBDModel
from pymc_marketing.clv.models.shifted_beta_geo import (
    ShiftedBetaGeoModel,
    ShiftedBetaGeoModelIndividual,
)


class CLVModelTest(CLVModel):
    _model_type = "CLVModelTest"

    @property
    def default_model_config(self):
        return {"x": Prior("Normal", mu=0, sigma=1)}

    def build_model(self):
        pass


@pytest.mark.parametrize(
    "data, expected_error, match",
    [
        (
            pd.DataFrame({"recency": [1], "T": [2]}),
            ValueError,
            r"The following required columns are missing from the input data: \['frequency'\]",
        ),
        (
            pd.DataFrame({"frequency": [-1], "recency": [1], "T": [2]}),
            ValueError,
            "Column frequency has negative values",
        ),
        (
            pd.DataFrame({"frequency": [1], "recency": [-1], "T": [2]}),
            ValueError,
            "Column recency has negative values",
        ),
        (
            pd.DataFrame({"frequency": [1], "recency": [1], "T": [-1]}),
            ValueError,
            "Column T has negative values",
        ),
        (
            pd.DataFrame({"frequency": [0], "recency": [1], "T": [2]}),
            ValueError,
            "recency cannot be greater than 0 if frequency is 0",
        ),
        (
            pd.DataFrame({"frequency": [1], "recency": [3], "T": [2]}),
            ValueError,
            "recency cannot be greater than T",
        ),
        (
            pd.DataFrame({"frequency": [1.5], "recency": [1], "T": [2]}),
            ValueError,
            "frequency column must contain only integer values",
        ),
    ],
)
def test_base_clv_model_validate_data_errors(data, expected_error, match):
    model = CLVModelTest.__new__(CLVModelTest)
    model.model_config = {}
    with pytest.raises(expected_error, match=match):
        model._validate_data(data)


@pytest.mark.parametrize(
    "data, expected_warning, match",
    [
        (
            pd.DataFrame({"frequency": [0], "recency": [0], "T": [0]}),
            UserWarning,
            "T=0 is mathematically valid but practically useless.",
        ),
    ],
)
def test_base_clv_model_validate_data_warnings(data, expected_warning, match):
    model = CLVModelTest.__new__(CLVModelTest)
    model.model_config = {}
    with pytest.warns(expected_warning, match=match):
        model._validate_data(data)


@pytest.mark.parametrize(
    "model_cls",
    [BetaGeoModel, ParetoNBDModel, ModifiedBetaGeoModel],
)
@pytest.mark.parametrize(
    "data, expected_error, match",
    [
        (
            pd.DataFrame({"recency": [1], "T": [2]}),
            ValueError,
            r"The following required columns are missing from the input data: \['frequency'\]",
        ),
        (
            pd.DataFrame(
                {
                    "customer_id": [1, 1],
                    "frequency": [1, 2],
                    "recency": [1, 1],
                    "T": [2, 2],
                }
            ),
            ValueError,
            "Column customer_id has duplicate entries",
        ),
        (
            pd.DataFrame(
                {
                    "customer_id": [1, 2],
                    "frequency": [-1, 2],
                    "recency": [1, 1],
                    "T": [2, 2],
                }
            ),
            ValueError,
            "Column frequency has negative values",
        ),
    ],
)
def test_rfm_models_validate_data_errors(model_cls, data, expected_error, match):
    model = model_cls.__new__(model_cls)
    model.model_config = {"purchase_covariate_cols": [], "dropout_covariate_cols": []}
    with pytest.raises(expected_error, match=match):
        model._validate_data(data)


@pytest.mark.parametrize(
    "data, expected_error, match",
    [
        (
            pd.DataFrame(
                {
                    "customer_id": [1, 2],
                    "frequency": [1, 2],
                    "recency": [1, 1],
                    "T": [2, 3],
                }
            ),
            ValueError,
            "Column T has non-homogeneous entries",
        ),
        (
            pd.DataFrame(
                {
                    "customer_id": [1],
                    "frequency": [1],
                    "recency": [3],
                    "T": [2],
                }
            ),
            ValueError,
            "recency cannot be greater than T",
        ),
    ],
)
def test_beta_geo_beta_binom_validate_data_errors(data, expected_error, match):
    model = BetaGeoBetaBinomModel.__new__(BetaGeoBetaBinomModel)
    model.model_config = {}
    with pytest.raises(expected_error, match=match):
        model._validate_data(data)


@pytest.mark.parametrize(
    "data, expected_error, match",
    [
        (
            pd.DataFrame(
                {
                    "customer_id": [1, 2],
                    "recency": [1, 1],
                    "T": [2, 2],
                }
            ),
            ValueError,
            r"The following required columns are missing from the input data: \['cohort'\]",
        ),
        (
            pd.DataFrame(
                {
                    "customer_id": [1, 2],
                    "recency": [0, 1],
                    "T": [2, 2],
                    "cohort": ["A", "A"],
                }
            ),
            ValueError,
            "Column recency must be at least 1",
        ),
        (
            pd.DataFrame(
                {
                    "customer_id": [1, 2],
                    "recency": [1, 3],
                    "T": [2, 2],
                    "cohort": ["A", "A"],
                }
            ),
            ValueError,
            "recency cannot be greater than T",
        ),
        (
            pd.DataFrame(
                {
                    "customer_id": [1, 2],
                    "recency": [1, 1],
                    "T": [1, 1],
                    "cohort": ["A", "A"],
                }
            ),
            ValueError,
            "Column T must be at least 2",
        ),
    ],
)
def test_shifted_beta_geo_validate_data_errors(data, expected_error, match):
    model = ShiftedBetaGeoModel.__new__(ShiftedBetaGeoModel)
    model.model_config = {"dropout_covariate_cols": []}
    with pytest.raises(expected_error, match=match):
        model._validate_data(data)


@pytest.mark.parametrize(
    "data, expected_warning, match",
    [
        (
            pd.DataFrame(
                {
                    "customer_id": [1],
                    "frequency": [1],
                    "monetary_value": [0],
                }
            ),
            UserWarning,
            "Non-positive monetary values are practically problematic.",
        ),
        (
            pd.DataFrame(
                {
                    "customer_id": [1],
                    "frequency": [1],
                    "monetary_value": [-1],
                }
            ),
            UserWarning,
            "Non-positive monetary values are practically problematic.",
        ),
    ],
)
def test_gamma_gamma_model_validate_data_warnings(data, expected_warning, match):
    model = GammaGammaModel.__new__(GammaGammaModel)
    model.model_config = {}
    with pytest.warns(expected_warning, match=match):
        model._validate_data(data)


@pytest.mark.parametrize(
    "data, expected_error, match",
    [
        (
            pd.DataFrame(
                {
                    "customer_id": [1],
                    "frequency": [-1],
                    "monetary_value": [10.0],
                }
            ),
            ValueError,
            "Column frequency has negative values",
        ),
        (
            pd.DataFrame(
                {
                    "customer_id": [1],
                    "frequency": [1.5],
                    "monetary_value": [10.0],
                }
            ),
            ValueError,
            "frequency column must contain only integer values",
        ),
    ],
)
def test_gamma_gamma_model_validate_data_errors(data, expected_error, match):
    model = GammaGammaModel.__new__(GammaGammaModel)
    model.model_config = {}
    with pytest.raises(expected_error, match=match):
        model._validate_data(data)


@pytest.mark.parametrize(
    "data, expected_error, match",
    [
        (
            pd.DataFrame({"customer_id": [1], "t_churn": [11], "T": [10]}),
            ValueError,
            "t_churn must respect 0 < t_churn <= T",
        ),
        (
            pd.DataFrame({"customer_id": [1], "t_churn": [0], "T": [10]}),
            ValueError,
            "t_churn must respect 0 < t_churn <= T",
        ),
    ],
)
def test_shifted_beta_geo_individual_validate_data_errors(data, expected_error, match):
    model = ShiftedBetaGeoModelIndividual.__new__(ShiftedBetaGeoModelIndividual)
    model.model_config = {}
    with pytest.raises(expected_error, match=match):
        model._validate_data(data)
