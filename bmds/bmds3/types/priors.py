from pathlib import Path
from typing import Dict

import pandas as pd

from ...constants import Dtype
from ..constants import ContinuousModel, DichotomousModel, ModelPriors, PriorClass

# lazy mapping; saves copy as requested
_model_priors: Dict[str, ModelPriors] = {}


def _load_model_priors():
    # lazy load model priors from CSV file
    def set_param_type(df):
        return df.assign(variance_param=df.name.map({"rho": True, "alpha": True})).fillna(False)

    def build_priors(df):
        priors = {}
        for (data_class, model_id, prior_class), params in df:
            key = f"{data_class}-{model_id}-{prior_class}"
            gof_priors = params[params.variance_param == False]  # noqa: E712
            var_priors = params[params.variance_param == True]  # noqa: E712
            priors[key] = ModelPriors(
                prior_class=prior_class,
                priors=gof_priors.to_dict("records"),
                variance_priors=var_priors.to_dict("records") if var_priors.shape[0] > 0 else None,
            )
        return priors

    filename = Path(__file__).parent / "priors.csv"
    priors = (
        pd.read_csv(str(filename))
        .pipe(set_param_type)
        .groupby(["data_class", "model_id", "prior_class"])
        .pipe(build_priors)
    )
    _model_priors.update(priors)


def get_dichotomous_prior(model: DichotomousModel, prior_class: PriorClass) -> ModelPriors:
    if len(_model_priors) == 0:
        _load_model_priors()
    key = f"{Dtype.DICHOTOMOUS}-{model.id}-{prior_class}"
    return _model_priors[key].copy(deep=True)


def get_continuous_prior(model: ContinuousModel, prior_class: PriorClass) -> ModelPriors:
    if len(_model_priors) == 0:
        _load_model_priors()
    key = f"{Dtype.CONTINUOUS}-{model.id}-{prior_class}"
    return _model_priors[key].copy(deep=True)
