import logging
from copy import deepcopy

from simple_settings import settings

from .. import constants
from ..bmds2.sessions import BMDS
from .models import dichotomous as d3

logger = logging.getLogger(__name__)


class Bmds3Version(BMDS):
    """
    A bmds3 session.

    TODO - update API to make it so you can use the same session
    TODO - refactor! and make sure it works with bmds2
    """

    def _can_execute_locally(self) -> bool:
        return True


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
            # constants.M_Linear: c3.Linear,
            # constants.M_Polynomial: c3.Polynomial,
            # constants.M_Power: c3.Power,
            # constants.M_Hill: c3.Hill,
            # constants.M_ExponentialM2: c3.ExponentialM2,
            # constants.M_ExponentialM3: c3.ExponentialM3,
            # constants.M_ExponentialM4: c3.ExponentialM4,
            # constants.M_ExponentialM5: c3.ExponentialM5,
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
