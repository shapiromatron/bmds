from enum import Enum, IntEnum
from typing import Tuple

from pydantic import BaseModel

BMDS_BLANK_VALUE = -9999
CDF_TABLE_SIZE = 99
MY_MAX_PARMS = 16
NUM_LIKELIHOODS_OF_INTEREST = 5
NUM_PRIOR_COLS = 5
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


class DichotomousModel(BaseModel):
    id: int
    verbose: str
    params: Tuple[str, ...]

    @property
    def num_params(self):
        return len(self.params)


class DichotomousModelChoices(Enum):
    d_hill = DichotomousModel(id=1, verbose="Hill", params=("a", "b", "c", "d"))
    d_gamma = DichotomousModel(id=2, verbose="Gamma", params=("a", "b", "c"))
    d_logistic = DichotomousModel(id=3, verbose="Logistic", params=("a", "b"))
    d_loglogistic = DichotomousModel(id=4, verbose="LogLogistic", params=("a", "b", "c"))
    d_logprobit = DichotomousModel(id=5, verbose="LogProbit", params=("a", "b", "c"))
    d_multistage = DichotomousModel(id=6, verbose="Multistage", params=("a", "b"))
    d_probit = DichotomousModel(id=7, verbose="Probit", params=("a", "b"))
    d_qlinear = DichotomousModel(id=8, verbose="QuantalLinear", params=("a", "b"))
    d_weibull = DichotomousModel(id=9, verbose="Weibull", params=("a", "b", "c"))


class ContinuousModel(BaseModel):
    id: int
    verbose: str
    params: Tuple[str, ...]

    @property
    def num_params(self):
        return len(self.params)


class ContinuousModelChoices(Enum):
    c_power = ContinuousModel(id=8, verbose="Power", params=("a", "b", "c","d"))
    c_hill = ContinuousModel(id=6, verbose="Hill", params=("a", "b", "c","d","e"))
    c_polynomial = ContinuousModel(id=666, verbose="Polynomial", params=("a", ))
    c_exp_m2 = ContinuousModel(id=2, verbose="ExponentialM3", params=("a", "b", "c","d","e"))
    c_exp_m3 = ContinuousModel(id=3, verbose="ExponentialM3", params=("a", "b", "c","d","e"))
    c_exp_m4 = ContinuousModel(id=4, verbose="ExponentialM3", params=("a", "b", "c","d","e"))
    c_exp_m5 = ContinuousModel(id=5, verbose="ExponentialM3", params=("a", "b", "c","d","e"))

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
