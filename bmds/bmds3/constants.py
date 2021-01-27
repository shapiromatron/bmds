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


class BmdModelSchema(BaseModel):
    id: int
    verbose: str
    model_form_str: str


class DichotomousModel(BmdModelSchema):
    params: Tuple[str, ...]

    @property
    def num_params(self):
        return len(self.params)


class DichotomousModelIds(IntEnum):
    d_hill = 1
    d_gamma = 2
    d_logistic = 3
    d_loglogistic = 4
    d_logprobit = 5
    d_multistage = 6
    d_probit = 7
    d_qlinear = 8
    d_weibull = 9


class DichotomousModelChoices(Enum):
    d_hill = DichotomousModel(
        id=DichotomousModelIds.d_hill.value,
        verbose="Hill",
        params=("g", "n", "a", "b"),
        model_form_str="P[dose] = g + (v - v * g) / (1 + exp(-a - b * Log(dose)))",
    )
    d_gamma = DichotomousModel(
        id=DichotomousModelIds.d_gamma.value,
        verbose="Gamma",
        params=("g", "a", "b"),
        model_form_str="P[dose]= g + (1 - g) * CumGamma(b * dose, a)",
    )
    d_logistic = DichotomousModel(
        id=DichotomousModelIds.d_logistic.value,
        verbose="Logistic",
        params=("a", "b"),
        model_form_str="P[dose] = 1 / [1 + exp(-a - b * dose)]",
    )
    d_loglogistic = DichotomousModel(
        id=DichotomousModelIds.d_loglogistic.value,
        verbose="LogLogistic",
        params=("g", "a", "b"),
        model_form_str="P[dose] = g + (1 - g)/(1 + exp(-a - b * Log(dose)))",
    )
    d_logprobit = DichotomousModel(
        id=DichotomousModelIds.d_logprobit.value,
        verbose="LogProbit",
        params=("g", "a", "b"),
        model_form_str="P[dose] = g + (1 - g) * CumNorm(a + b * Log(Dose))",
    )
    d_multistage = DichotomousModel(
        id=DichotomousModelIds.d_multistage.value,
        verbose="Multistage",
        params=("g", "x1", "x2"),
        model_form_str="P[dose] = g + (1 - g) * (1 - exp(-b1 * dose^1 - b2 * dose^2 - ...))",
    )
    d_probit = DichotomousModel(
        id=DichotomousModelIds.d_probit.value,
        verbose="Probit",
        params=("a", "b"),
        model_form_str="P[dose] = CumNorm(a + b * Dose)",
    )
    d_qlinear = DichotomousModel(
        id=DichotomousModelIds.d_qlinear.value,
        verbose="Quantal Linear",
        params=("g", "a"),
        model_form_str="P[dose] = g + (1 - g) * (1 - exp(-b * dose)",
    )
    d_weibull = DichotomousModel(
        id=DichotomousModelIds.d_weibull.value,
        verbose="Weibull",
        params=("g", "a", "b"),
        model_form_str="P[dose] = g + (1 - g) * (1 - exp(-b * dose^a))",
    )


class ContinuousModel(BmdModelSchema):
    params: Tuple[str, ...]

    @property
    def num_params(self):
        return len(self.params)


class ContinuousModelIds(IntEnum):
    c_exp_m3 = 3
    c_exp_m5 = 5
    c_hill = 6
    c_power = 8
    c_polynomial = 666


class ContinuousModelChoices(Enum):
    c_power = ContinuousModel(
        id=ContinuousModelIds.c_power.value,
        verbose="Power",
        params=("g", "v", "n", "alpha"),
        model_form_str="P[dose] = g + v * dose ^ n",
    )
    c_hill = ContinuousModel(
        id=ContinuousModelIds.c_hill.value,
        verbose="Hill",
        params=("g", "v", "k", "n", "alpha"),
        model_form_str="P[dose] = g + v * dose ^ n / (k ^ n + dose ^ n)",
    )
    c_polynomial = ContinuousModel(
        id=ContinuousModelIds.c_polynomial.value,
        verbose="Polynomial",
        params=("g", "b1", "b2"),
        model_form_str="P[dose] = g + b1*dose + b2*dose^2 + b3*dose^3...",
    )
    c_exp_m3 = ContinuousModel(
        id=ContinuousModelIds.c_exp_m3.value,
        verbose="ExponentialM3",
        params=("a", "b", "c", "d", "alpha"),
        model_form_str="P[dose] = a * exp(±1 * (b * dose) ^ d)",
    )
    c_exp_m5 = ContinuousModel(
        id=ContinuousModelIds.c_exp_m5.value,
        verbose="ExponentialM5",
        params=("a", "b", "c", "d", "alpha"),
        model_form_str="P[dose] = a * (c - (c - 1) * exp(-(b * dose) ^ d)",
    )


class PriorType(IntEnum):
    eNone = 0
    eNormal = 1
    eLognormal = 2


class PriorClass(IntEnum):
    frequentist_unrestricted = 0
    frequentist_restricted = 1
    bayesian = 2


class Prior(BaseModel):
    type: PriorType
    initial_value: float
    stdev: float
    min_value: float
    max_value: float

    @classmethod
    def parse_args(cls, *args) -> "Prior":
        return cls(**{key: arg for key, arg in zip(cls.__fields__.keys(), args)})
