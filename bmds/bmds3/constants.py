import ctypes
from enum import Enum, IntEnum
from typing import List, Tuple

from pydantic import BaseModel


BMDS_BLANK_VALUE = -9999
NUM_PRIOR_COLS = 5
CDF_TABLE_SIZE = 99
MY_MAX_PARMS = 16
NUM_LIKELIHOODS_OF_INTEREST = 5
NUM_TESTS_OF_INTEREST = 4


class VarType_t(Enum):
    eVarTypeNone = 0, 0, "No variance model"
    eConstant = 1, 1, "Constant variance"
    eModeled = 2, 2, "Modeled variance"

    def __new__(cls, id: int, num_params: int, verbose: str):
        # https://stackoverflow.com/a/12680149/906385
        entry = object.__new__(cls)
        entry.id = entry._value_ = id  # set the value, and the extra attribute
        entry.num_params = num_params
        entry.verbose = verbose
        return entry


class DichotomousModel(Enum):
    d_hill = 1, "Hill", ("a", "b")
    d_gamma = 2, "Gamma", ("a", "b")
    d_logistic = 3, "Logistic", ("a", "b")
    d_loglogistic = 4, "LogLogistic", ("a", "b")
    d_logprobit = 5, "LogProbit", ("a", "b")
    d_multistage = 6, "Multistage", ("a", "b")
    d_probit = 7, "Probit", ("a", "b")
    d_qlinear = 8, "QuantalLinear", ("a", "b")
    d_weibull = 9, "Weibull", ("a", "b")

    def __new__(cls, id: int, verbose: str, params: Tuple[str, ...]):
        # https://stackoverflow.com/a/12680149/906385
        entry = object.__new__(cls)
        entry.id = entry._value_ = id  # set the value, and the extra attribute
        entry.verbose = verbose
        entry.params = params
        return entry

    def pretty_name(self, model) -> str:
        # TODO - move to model object
        if self.id == 6:
            return f"Multistage-{model.settings.degree}"
        return self.verbose


class ContinuousModel(Enum):
    eExp2 = 2, "Exponential 2"
    eExp3 = 3, "Exponential 3"
    eExp4 = 4, "Exponential 4"
    eExp5 = 5, "Exponential 5"
    eHill = 6, "Hill"
    ePoly = 7, "Linear/Polynomial"
    ePow = 8, "Power"
    gain_loss_model = 10, "Gain loss model"
    polynomial = 666, "Linear/Polynomial"

    def __new__(cls, id: int, verbose: str):
        # https://stackoverflow.com/a/12680149/906385
        entry = object.__new__(cls)
        entry.id = entry._value_ = id  # set the value, and the extra attribute
        entry.verbose = verbose
        return entry

    def pretty_name(self, model) -> str:
        # TODO - move to model object
        if self.id in {7, 666}:
            return "Linear" if model.settings.degree == 1 else f"Polynomial-{model.settings.degree}"
        return self.verbose


class PriorType(IntEnum):
    eNone = 0
    eNormal = 1
    eLognormal = 2


class Prior(BaseModel):
    type: PriorType
    initial_value: float
    stdev: float
    min_value: float
    max_value: float
