import logging
from copy import deepcopy
from typing import Dict, Tuple

import pandas as pd
from simple_settings import settings

from .. import __version__, constants
from ..datasets import DatasetType
from .models import continuous as c3
from .models import dichotomous as d3
from .models import ma

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
        self.models = []
        self.dataset = dataset

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

    def add_model(self, name, settings=None, id=None):
        Model = self.model_options[self.dataset.dtype][name]
        instance = Model(dataset=self.dataset, settings=settings, id=id)
        self.models.append(instance)

    def add_model_averaging(self):
        """
        Must be added average other models are added since a shallow copy is taken, and the
        execution of model averaging assumes all other models were executed.
        """
        instance = ma.DichotomousMA(dataset=self.dataset, models=list(range(len(self.models))))
        self.models.append(instance)

    def execute(self):
        for model in self.models:
            if isinstance(model, ma.BaseModelAveraging):
                model.execute_job(self)
            else:
                model.execute_job()

    def execute_and_recommend(self, drop_doses=False):
        raise NotImplementedError("TODO")

    # serializing
    # -----------
    def to_dict(self):
        return dict(
            bmds_version=self.version_str,
            bmds_python_version=__version__,
            dataset=self.dataset.dict(),
            models=[model.dict(i) for i, model in enumerate(self.models)],
            recommended_model_index=getattr(self, "recommended_model_index", None),
        )

    # reporting
    # ---------
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


class BMDS_v330(BmdsSession):
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
