from enum import Enum, IntEnum
from itertools import chain
from typing import List, Optional, Tuple

import numpy as np
from pydantic import BaseModel

from ..utils import pretty_table

BMDS_BLANK_VALUE = -9999
CDF_TABLE_SIZE = 99
MY_MAX_PARMS = 16
N_BMD_DIST = 200
NUM_LIKELIHOODS_OF_INTEREST = 5
NUM_PRIOR_COLS = 5
NUM_TESTS_OF_INTEREST = 4


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
    variance_params: Tuple[str, ...]


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
        variance_params=("rho", "alpha"),
        model_form_str="P[dose] = a * exp(±1 * (b * dose) ^ d)",
    )
    c_exp_m5 = ContinuousModel(
        id=ContinuousModelIds.c_exp_m5.value,
        verbose="ExponentialM5",
        params=("a", "b", "c", "d"),
        variance_params=("rho", "alpha"),
        model_form_str="P[dose] = a * (c - (c - 1) * exp(-(b * dose) ^ d)",
    )


class DistType(IntEnum):
    normal = 1
    normal_ncv = 2
    log_normal = 3


class PriorType(IntEnum):
    Uniform = 0
    Normal = 1
    Lognormal = 2


class Prior(BaseModel):
    name: str
    type: PriorType
    initial_value: float
    stdev: float
    min_value: float
    max_value: float

    def tbl_str_hdr(self) -> str:
        return "| param | type       |    initial |      stdev |        min |        max |"

    def tbl_str(self) -> str:
        return f"| {self.name:5} | {self.type.name:10} | {self.initial_value:10.3g} | {self.stdev:10.3g} | {self.min_value:10.3g} | {self.max_value:10.3g} |"

    def numeric_list(self) -> List[float]:
        return list(self.dict(exclude={"name"}).values())


class PriorClass(IntEnum):
    frequentist_unrestricted = 0
    frequentist_restricted = 1
    bayesian = 2
    custom = 3

    @property
    def name(self):
        return _pc_name_mapping[self]


_pc_name_mapping = {
    PriorClass.frequentist_unrestricted: "Frequentist unrestricted",
    PriorClass.frequentist_restricted: "Frequentist restricted",
    PriorClass.bayesian: "Bayesian",
}


class ModelPriors(BaseModel):
    prior_class: PriorClass  # if this is a predefined model class
    priors: List[Prior]  # priors for main model
    variance_priors: Optional[List[Prior]]  # priors for variance model (continuous-only)

    def __str__(self):
        ps = [self.priors[0].tbl_str_hdr()]
        ps.extend([p.tbl_str() for p in self.priors])
        p = "\n".join(ps)
        if self.variance_priors is not None:
            vps = "\n".join([p.tbl_str() for p in self.variance_priors])
            p += f"""\n{vps}"""
        p += "\n"
        return p

    def tbl(self) -> str:
        headers = "name|type|initial_value|stdev|min_value|max_value".split("|")
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

    def to_c(
        self, degree: Optional[int] = None, dist_type: Optional[DistType] = None
    ) -> np.ndarray:

        priors = []
        for prior in self.priors:
            priors.append(prior.numeric_list())

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

        return np.array(priors, dtype=np.float64).flatten("F")
