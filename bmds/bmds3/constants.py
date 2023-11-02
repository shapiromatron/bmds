from enum import Enum, IntEnum

from pydantic import BaseModel, Field

BMDS_BLANK_VALUE = -9999
N_BMD_DIST = 100
NUM_PRIOR_COLS = 5


class BmdModelSchema(BaseModel):
    id: int
    verbose: str
    bmds_model_form_str: str = Field(alias="model_form_str")


class DichotomousModel(BmdModelSchema):
    params: tuple[str, ...]

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
        params=("g", "v", "a", "b"),
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
        params=("g", "b"),
        model_form_str="P[dose] = g + (1 - g) * (1 - exp(-b * dose)",
    )
    d_weibull = DichotomousModel(
        id=DichotomousModelIds.d_weibull.value,
        verbose="Weibull",
        params=("g", "a", "b"),
        model_form_str="P[dose] = g + (1 - g) * (1 - exp(-b * dose^a))",
    )


class ContinuousModel(BmdModelSchema):
    params: tuple[str, ...]
    variance_params: tuple[str, ...]


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
        params=("g", "v", "n"),
        variance_params=("rho", "alpha"),
        model_form_str="P[dose] = g + v * dose ^ n",
    )
    c_hill = ContinuousModel(
        id=ContinuousModelIds.c_hill.value,
        verbose="Hill",
        params=("g", "v", "k", "n"),
        variance_params=("rho", "alpha"),
        model_form_str="P[dose] = g + v * dose ^ n / (k ^ n + dose ^ n)",
    )
    c_polynomial = ContinuousModel(
        id=ContinuousModelIds.c_polynomial.value,
        verbose="Polynomial",
        params=("g", "b1", "b2"),
        variance_params=("rho", "alpha"),
        model_form_str="P[dose] = g + b1*dose + b2*dose^2 + b3*dose^3...",
    )
    c_exp_m3 = ContinuousModel(
        id=ContinuousModelIds.c_exp_m3.value,
        verbose="ExponentialM3",
        params=("a", "b", "c", "d"),
        variance_params=("rho", "log-alpha"),
        model_form_str="P[dose] = a * exp(±1 * (b * dose) ^ d)",
    )
    c_exp_m5 = ContinuousModel(
        id=ContinuousModelIds.c_exp_m5.value,
        verbose="ExponentialM5",
        params=("a", "b", "c", "d"),
        variance_params=("rho", "log-alpha"),
        model_form_str="P[dose] = a * (c - (c - 1) * exp(-(b * dose) ^ d)",
    )


class DistType(IntEnum):
    normal = 1  # f(i) = a * x(i)
    normal_ncv = 2  # f(i) = a * x(i) ^ p
    log_normal = 3

    @property
    def distribution_type(self) -> str:
        return _dt_name[self]

    @property
    def variance_model(self) -> str:
        return _dt_variance_model[self]


_dt_name = {
    DistType.normal: "Normal",
    DistType.normal_ncv: "Normal",
    DistType.log_normal: "Lognormal",
}
_dt_variance_model = {
    DistType.normal: "Constant variance",
    DistType.normal_ncv: "Nonconstant variance",
    DistType.log_normal: "Constant variance",
}


class PriorType(IntEnum):
    Uniform = 0
    Normal = 1
    Lognormal = 2


class PriorClass(IntEnum):
    frequentist_unrestricted = 0
    frequentist_restricted = 1
    bayesian = 2

    @property
    def name(self) -> str:
        return _pc_name[self]

    @property
    def restriction(self) -> str:
        return _pc_restriction[self]

    @property
    def is_bayesian(self) -> bool:
        return self == self.bayesian


_pc_name = {
    PriorClass.frequentist_unrestricted: "Frequentist unrestricted",
    PriorClass.frequentist_restricted: "Frequentist restricted",
    PriorClass.bayesian: "Bayesian",
}
_pc_restriction = {
    PriorClass.frequentist_unrestricted: "Unrestricted",
    PriorClass.frequentist_restricted: "Restricted",
}
