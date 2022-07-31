from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from ...constants import Dtype
from ...utils import pretty_table
from ..constants import ContinuousModel, DichotomousModel, DistType, PriorClass, PriorType


class Prior(BaseModel):
    name: str
    type: PriorType
    initial_value: float
    stdev: float
    min_value: float
    max_value: float

    def numeric_list(self) -> List[float]:
        return list(self.dict(exclude={"name"}).values())


class ModelPriors(BaseModel):
    prior_class: PriorClass  # if this is a predefined model class
    priors: List[Prior]  # priors for main model
    variance_priors: Optional[List[Prior]]  # priors for variance model (continuous-only)

    def __str__(self):
        headers = "name|type|initial|stdev|min|max".split("|")
        rows = [
            (p.name, p.type.name, p.initial_value, p.stdev, p.min_value, p.max_value)
            for p in chain(self.priors, self.variance_priors or ())
        ]
        return pretty_table(rows, headers)

    def get_prior(self, name: str) -> Prior:
        """Search all priors and return the match by name.

        Args:
            name (str): prior name

        Raises:
            ValueError: if no value is found
        """
        for p in self.priors:
            if p.name == name:
                return p
        if self.variance_priors:
            for p in self.variance_priors:
                if p.name == name:
                    return p
        raise ValueError(f"No parameter named {name}")

    def priors_list(
        self, degree: Optional[int] = None, dist_type: Optional[DistType] = None
    ) -> list[list]:
        priors = []
        for prior in self.priors:
            priors.append(prior.numeric_list())

        # for multistage/polynomial, this assumes that the 3rd
        # prior parameter is betaN ensure that this is always the case

        # remove degreeN; 1st order multistage/polynomial
        if degree and degree == 1:
            priors.pop(2)

        # copy degreeN; > 2rd order poly
        if degree and degree > 2:
            for i in range(2, degree):
                priors.append(priors[2])

        # add constant variance parameter
        if dist_type and dist_type in {DistType.normal, DistType.log_normal}:
            priors.append(self.variance_priors[0].numeric_list())

        # add non-constant variance parameter
        if dist_type and dist_type is DistType.normal_ncv:
            for prior in self.variance_priors:
                priors.append(prior.numeric_list())

        return priors

    def to_c(
        self, degree: Optional[int] = None, dist_type: Optional[DistType] = None
    ) -> np.ndarray:
        priors = self.priors_list(degree, dist_type)
        return np.array(priors, dtype=np.float64).flatten("F")


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
