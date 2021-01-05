import logging
from copy import copy, deepcopy
from typing import Dict

from simple_settings import settings

from .. import constants
from ..bmds2.sessions import BMDS
from .models import continuous as c3
from .models import dichotomous as d3
from .models import ma

logger = logging.getLogger(__name__)


class Bmds3Version(BMDS):
    """
    A bmds3 session.

    TODO - update API to make it so you can use the same session
    TODO - refactor! and make sure it works with bmds2
    """

    def add_default_models(self, global_settings=None):
        for name in self.model_options[self.dtype].keys():
            model_settings = deepcopy(global_settings) if global_settings is not None else None
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

    def add_model_averaging(self):
        """
        Must be added average other models are added since a shallow copy is taken, and the
        execution of model averaging assumes all other models were executed.
        """
        instance = ma.DichotomousMA(dataset=self.dataset, models=copy(self.models))
        self.models.append(instance)

    def _can_execute_locally(self) -> bool:
        return True

    @classmethod
    def from_dict(cls, data: Dict) -> "Bmds3Version":
        raise NotImplementedError("TODO - implement!")


class BMDS_v330(Bmds3Version):
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
