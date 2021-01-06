import logging
from copy import copy, deepcopy
from typing import Dict, List, Optional, Tuple

import pandas as pd
from simple_settings import settings

from .. import constants
from ..datasets import DatasetType
from .models.base import BmdModel, BmdModelAveraging
from .models import continuous as c3
from .models import dichotomous as d3
from .models import ma
from .types import sessions as schema

logger = logging.getLogger(__name__)


class BmdsSession:
    """A BmdsSession is bmd modeling session for a single dataset.

    The session contains the dataset, model configuration and results, and model recommendations
    and potentially model averaging results too. BmdsSessions are a primary data type that
    should be able to be serialized and deserialized.
    """

    version_str: str
    version_pretty: str
    version_tuple: Tuple[int, ...]
    model_options: Dict[str, Dict]

    def __init__(self, dataset: DatasetType):
        self.dataset = dataset
        self.models: List[BmdModel] = []
        self.model_average: Optional[BmdModelAveraging] = None

    def add_default_models(self, global_settings=None):
        for name in self.model_options[self.dataset.dtype].keys():
            model_settings = deepcopy(global_settings) if global_settings is not None else None

            # TODO - change this; use `degree` in settings
            if name in constants.VARIABLE_POLYNOMIAL:
                min_poly_order = 1 if name == constants.M_MultistageCancer else 2
                max_poly_order = min(
                    self.dataset.num_dose_groups - 1, settings.MAXIMUM_POLYNOMIAL_ORDER + 1
                )
                for i in range(min_poly_order, max_poly_order):
                    poly_model_settings = (
                        deepcopy(model_settings) if model_settings is not None else {}
                    )
                    poly_model_settings["degree"] = i
                    self.add_model(name, settings=poly_model_settings)
            else:
                self.add_model(name, settings=model_settings)

    def add_model(self, name, settings=None):
        Model = self.model_options[self.dataset.dtype][name]
        instance = Model(dataset=self.dataset, settings=settings)
        self.models.append(instance)

    def add_model_averaging(self):
        """
        Must be added average other models are added since a shallow copy is taken, and the
        execution of model averaging assumes all other models were executed.
        """
        instance = ma.BmdModelAveragingDichotomous(dataset=self.dataset, models=copy((self.models)))
        self.model_average = instance

    def execute(self):
        # execute individual models
        for model in self.models:
            model.execute_job()

        # execute model average
        if self.model_average is not None:
            self.model_average.execute_job()

    def execute_and_recommend(self, drop_doses=False):
        raise NotImplementedError("TODO")

    # serializing
    # -----------
    def serialize(self) -> schema.SessionSchemaBase:
        raise NotImplementedError("implement!")

    def deserialize(self) -> "BmdsSession":
        raise NotImplementedError("implement!")

    # reporting
    # ---------
    def to_dict(self):
        return self.serialize().dict()

    def to_df(self, filename) -> pd.DataFrame:
        raise NotImplementedError("TODO - implement!")

    def to_docx(
        self,
        filename=None,
        title=None,
        input_dataset=True,
        summary_table=True,
        recommendation_details=True,
        recommended_model=True,
        all_models=False,
    ):
        raise NotImplementedError("TODO - implement!")


class Bmds330(BmdsSession):
    version_str = constants.BMDS330
    version_pretty = "BMDS v3.3.0"
    version_tuple = (3, 3, 0)
    model_options = {
        constants.DICHOTOMOUS: {
            constants.M_Logistic: d3.Logistic,
            constants.M_LogLogistic: d3.LogLogistic,
            constants.M_Probit: d3.Probit,
            constants.M_LogProbit: d3.LogProbit,
            constants.M_QuantalLinear: d3.QuantalLinear,
            constants.M_Multistage: d3.Multistage,
            constants.M_Gamma: d3.Gamma,
            constants.M_Weibull: d3.Weibull,
            constants.M_DichotomousHill: d3.DichotomousHill,
        },
        constants.DICHOTOMOUS_CANCER: {
            # constants.M_MultistageCancer: d3.Multistage
        },
        constants.CONTINUOUS: {
            # constants.M_Linear: c3.Linear, = Polynomial degree=1
            # constants.M_Polynomial: c3.Polynomial,
            constants.M_Power: c3.Power,
            constants.M_Hill: c3.Hill,
            constants.M_ExponentialM2: c3.ExponentialM2,
            constants.M_ExponentialM3: c3.ExponentialM3,
            constants.M_ExponentialM4: c3.ExponentialM4,
            constants.M_ExponentialM5: c3.ExponentialM5,
        },
        constants.CONTINUOUS_INDIVIDUAL: {
            # constants.M_Linear: c3.Linear,
            # constants.M_Polynomial: c3.Polynomial,
            # constants.M_Power: c3.Power,
            # constants.M_Hill: c3.Hill,
            # constants.M_ExponentialM2: c3.ExponentialM2,
            # constants.M_ExponentialM3: c3.ExponentialM3,
            # constants.M_ExponentialM4: c3.ExponentialM4,
            # constants.M_ExponentialM5: c3.ExponentialM5,
        },
    }

    def serialize(self) -> "Bmds330Schema":
        schema = Bmds330Schema(
            version=dict(
                string=self.version_str, pretty=self.version_pretty, numeric=self.version_tuple,
            ),
            dataset=self.dataset.serialize(),
            models=[model.serialize() for model in self.models],
        )
        if self.model_average is not None:
            schema.model_average = self.model_average.serialize(self)

        return schema


class Bmds330Schema(schema.SessionSchemaBase):
    def deserialize(self) -> Bmds330:
        session = Bmds330(dataset=self.dataset.deserialize())
        session.models = [model.deserialize(session.dataset) for model in self.models]
        if self.model_average:
            session.model_average = self.model_average.deserialize(session)
        return session
