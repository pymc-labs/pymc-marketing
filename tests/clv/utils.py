from typing import Union

from arviz import InferenceData
from xarray import Dataset

from pymc_marketing.clv.models import CLVModel


def set_model_fit(model: CLVModel, fit: Union[InferenceData, Dataset]):
    if isinstance(fit, InferenceData):
        assert "posterior" in fit.groups()
    else:
        fit = InferenceData(posterior=fit)
    if model.model is None:
        model.build_model()
    model.idata = fit
    model.idata.add_groups(fit_data=model.data.to_xarray())
    model.set_idata_attrs(fit)
