from enum import Enum, IntEnum
from typing import Tuple

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


class DichotomousModelData(BaseModel):
    id: int
    verbose: str
    params: Tuple[str, ...]


class DichotomousModel(Enum):
    d_hill = DichotomousModelData(id=1, verbose="Hill", params=("a", "b", "c", "d"))
    d_gamma = DichotomousModelData(id=2, verbose="Gamma", params=("a", "b", "c"))
    d_logistic = DichotomousModelData(id=3, verbose="Logistic", params=("a", "b"))
    d_loglogistic = DichotomousModelData(id=4, verbose="LogLogistic", params=("a", "b", "c"))
    d_logprobit = DichotomousModelData(id=5, verbose="LogProbit", params=("a", "b", "c"))
    d_multistage = DichotomousModelData(id=6, verbose="Multistage", params=("a", "b"))
    d_probit = DichotomousModelData(id=7, verbose="Probit", params=("a", "b"))
    d_qlinear = DichotomousModelData(id=8, verbose="QuantalLinear", params=("a", "b"))
    d_weibull = DichotomousModelData(id=9, verbose="Weibull", params=("a", "b", "c"))

    def __new__(cls, data: DichotomousModelData):
        # https://stackoverflow.com/a/12680149/906385
        entry = object.__new__(cls)
        entry.id = entry._value_ = data.id  # set the value, and the extra attribute
        entry.data = data
        return entry

    @property
    def num_params(self):
        return len(self.data.params)

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
    ePow = 8, "Power"
    ePolynomial = 666, "Linear/Polynomial"

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
