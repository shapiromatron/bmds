import ctypes
from enum import IntEnum
from textwrap import dedent
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, confloat, conint

from bmds.bmds3.constants import DichotomousModelChoices

from ...datasets import DichotomousDataset
from .. import constants
from .common import BMDS_BLANK_VALUE, list_t_c


class DichotomousRiskType(IntEnum):
    eExtraRisk = 1
    eAddedRisk = 2


class DichotomousModelSettings(BaseModel):
    bmr: confloat(gt=0) = 0.1
    alpha: confloat(gt=0, lt=1) = 0.05
    bmr_type: DichotomousRiskType = DichotomousRiskType.eExtraRisk
    degree: conint(ge=0, le=8) = 0  # multistage only
    samples: conint(ge=10, le=1000) = 100
    burnin: conint(ge=5, le=1000) = 20


class DichotomousAnalysisStruct(ctypes.Structure):

    _fields_ = [
        ("model", ctypes.c_int),  # Model Type as listed in DichModel
        ("n", ctypes.c_int),  # total number of observations obs/n
        ("Y", ctypes.POINTER(ctypes.c_double)),  # observed +
        ("doses", ctypes.POINTER(ctypes.c_double)),
        ("n_group", ctypes.POINTER(ctypes.c_double)),  # size of the group
        ("prior", ctypes.POINTER(ctypes.c_double)),  # a column order matrix parms X prior_cols
        ("BMD_type", ctypes.c_int),  # 1 = extra ; added otherwise
        ("BMR", ctypes.c_double),
        ("alpha", ctypes.c_double),  # alpha of the analysis
        ("degree", ctypes.c_int),  # degree of polynomial used only multistage
        ("samples", ctypes.c_int),  # number of MCMC samples
        ("burnin", ctypes.c_int),  # size of burnin
        ("parms", ctypes.c_int),  # number of parameters in the model
        ("prior_cols", ctypes.c_int),  # columns in the prior
    ]

    def __str__(self) -> str:
        return dedent(
            f"""
            model: {self.model}
            n: {self.n}
            Y: {self.Y[:self.n]}
            doses: {self.doses[:self.n]}
            n_group: {self.n_group[:self.n]}
            prior: {self.prior[:self.parms*self.prior_cols]}
            BMD_type: {self.BMD_type}
            BMR: {self.BMR}
            alpha: {self.alpha}
            degree: {self.degree}
            samples: {self.samples}
            burnin: {self.burnin}
            parms: {self.parms}
            prior_cols: {self.prior_cols}
            """
        )


class DichotomousAnalysis(BaseModel):
    """
    Purpose - Contains all of the information for a dichotomous analysis.
    It is used do describe a single model analysis, in which all of the
    informationis used, or a MA analysis, in which all the information
    save prior, degree, parms and prior_cols are used.
    """

    model: constants.DichotomousModel
    dataset: DichotomousDataset
    priors: List[constants.Prior]
    BMD_type: int
    BMR: float
    alpha: float
    degree: int
    samples: int
    burnin: int

    class Config:
        arbitrary_types_allowed = True

    @property
    def num_params(self) -> int:
        return (
            self.degree + 1
            if self.model == DichotomousModelChoices.d_multistage.value
            else self.model.num_params
        )

    def _priors_to_list(self) -> List[float]:
        """
        allocate memory for all parameters and convert to columnwise matrix
        """
        if len(self.priors) == self.num_params:
            # most cases
            arr = np.array([list(prior.dict().values()) for prior in self.priors])
        elif len(self.priors) < self.num_params:
            # special case for multistage; apply all priors; copy last one
            data: List[List[float]] = []
            for prior in self.priors[:-1]:
                data.append(list(prior.dict().values()))
            for _ in range(len(self.priors) - 1, self.num_params):
                data.append(list(self.priors[-1].dict().values()))
            arr = np.array(data)
        else:
            raise ValueError("Unknown state")
        return arr.T.flatten().tolist()

    def to_c(self) -> DichotomousAnalysisStruct:
        priors = self._priors_to_list()
        return DichotomousAnalysisStruct(
            model=ctypes.c_int(self.model.id),
            n=ctypes.c_int(self.dataset.num_dose_groups),
            Y=list_t_c(self.dataset.incidences, ctypes.c_double),
            doses=list_t_c(self.dataset.doses, ctypes.c_double),
            n_group=list_t_c(self.dataset.ns, ctypes.c_double),
            prior=list_t_c(priors, ctypes.c_double),
            BMD_type=ctypes.c_int(self.BMD_type),
            BMR=ctypes.c_double(self.BMR),
            alpha=ctypes.c_double(self.alpha),
            degree=ctypes.c_int(self.degree),
            samples=ctypes.c_int(self.samples),
            burnin=ctypes.c_int(self.burnin),
            parms=ctypes.c_int(self.num_params),
            prior_cols=ctypes.c_int(constants.NUM_PRIOR_COLS),
        )


class DichotomousModelResultStruct(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_int),  # dichotomous model specification
        ("nparms", ctypes.c_int),  # number of parameters in the model
        ("parms", ctypes.POINTER(ctypes.c_double)),  # parameter estimate
        ("cov", ctypes.POINTER(ctypes.c_double)),  # covariance estimate
        ("max", ctypes.c_double),  # value of the likelihood/posterior at the maximum
        ("dist_numE", ctypes.c_int),  # number of entries in rows for the bmd_dist
        ("model_df", ctypes.c_double),  # Used model degrees of freedom
        ("total_df", ctypes.c_double),  # Total degrees of freedom
        ("bmd_dist", ctypes.POINTER(ctypes.c_double),),  # bmd distribution (dist_numE x 2) matrix
    ]


class DichotomousModelResult(BaseModel):
    """
    Single model fit.
    """

    model: constants.DichotomousModel
    num_params: int
    dist_numE: int
    params: Optional[List[float]]
    cov: Optional[np.ndarray]
    max: Optional[float]
    model_df: Optional[float]
    total_df: Optional[float]
    bmd_dist: Optional[np.ndarray]

    class Config:
        arbitrary_types_allowed = True

    def to_c(self) -> DichotomousModelResultStruct:
        parms = np.zeros(self.num_params, dtype=np.float64)
        self.cov = np.zeros(self.num_params ** 2, dtype=np.float64)
        self.bmd_dist = np.zeros(self.dist_numE * 2, dtype=np.float64)
        return DichotomousModelResultStruct(
            model=ctypes.c_int(self.model.id),
            nparms=ctypes.c_int(self.num_params),
            parms=parms.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cov=self.cov.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            dist_numE=ctypes.c_int(self.dist_numE),
            bmd_dist=self.bmd_dist.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

    def from_c(self, struct: DichotomousModelResultStruct):
        self.params = struct.parms[: self.num_params]
        self.cov = self.cov.reshape(self.num_params, self.num_params)
        self.max = struct.max
        self.model_df = struct.model_df
        self.total_df = struct.total_df

        # reshape; get rid of 0 and inf; must be JSON serializable
        arr = self.bmd_dist.reshape(2, self.dist_numE).T
        arr = arr[np.isfinite(arr[:, 0])]
        arr = arr[arr[:, 0] > 0]
        self.bmd_dist = arr

    def bmd_plot(self):
        df = pd.DataFrame(data=self.bmd_dist, columns="bmd quantile".split())
        df = df.query("bmd>0 & bmd < inf")
        df.plot.scatter("bmd", "quantile", xlabel="Dose", ylabel="Propotion")

    def dict(self, **kw) -> Dict:
        kw.update(exclude={"cov", "bmd_dist"})
        d = super().dict(**kw)
        d["cov"] = self.cov.tolist()
        d["bmd_dist"] = self.bmd_dist.T.tolist()
        return d


class DichotomousPgofDataStruct(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_int),  # total number of observations obs/n
        ("Y", ctypes.POINTER(ctypes.c_double)),  # observed +
        ("doses", ctypes.POINTER(ctypes.c_double)),
        ("n_group", ctypes.POINTER(ctypes.c_double)),  # size of the group
        ("model_df", ctypes.c_double),
        ("model", ctypes.c_int),  # Model Type as listed in DichModel
        ("parms", ctypes.c_int),  # number of parameters in the model
        ("est_parms", ctypes.POINTER(ctypes.c_double)),  # parameter estimate
    ]

    @classmethod
    def from_fit(
        cls, fit_input: DichotomousAnalysisStruct, fit_output: DichotomousModelResultStruct
    ):
        return cls(
            n=fit_input.n,
            Y=fit_input.Y,
            doses=fit_input.doses,
            n_group=fit_input.n_group,
            model_df=fit_output.model_df,
            model=fit_input.model,
            parms=fit_output.nparms,
            est_parms=fit_output.parms,
        )


class DichotomousPgofResultStruct(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_int),  # total number of observations obs/n
        ("expected", ctypes.POINTER(ctypes.c_double)),
        ("residual", ctypes.POINTER(ctypes.c_double)),
        ("test_statistic", ctypes.c_double),
        ("p_value", ctypes.c_double),
        ("df", ctypes.c_double),
    ]

    @classmethod
    def from_dataset(cls, dataset: DichotomousDataset):
        n = dataset.num_dose_groups
        return cls(
            n=n,
            expected=list_t_c([0.0 for _ in range(n)], ctypes.c_double),
            residual=list_t_c([0.0 for _ in range(n)], ctypes.c_double),
        )


class DichotomousBmdsResultsStruct(ctypes.Structure):
    _fields_ = [
        ("bmd", ctypes.c_double),
        ("bmdl", ctypes.c_double),
        ("bmdu", ctypes.c_double),
        ("aic", ctypes.c_double),
        ("bounded", ctypes.POINTER(ctypes.c_bool)),
    ]

    @classmethod
    def from_results(cls, results: DichotomousModelResult) -> "DichotomousBmdsResultsStruct":
        return cls(
            bmd=BMDS_BLANK_VALUE,
            bmdl=BMDS_BLANK_VALUE,
            bmdu=BMDS_BLANK_VALUE,
            aic=BMDS_BLANK_VALUE,
            bounded=list_t_c([False for _ in range(results.num_params)], ctypes.c_bool),
        )


class DichotomousPgofResult(BaseModel):
    expected: List[float]
    residual: List[float]
    test_statistic: float
    p_value: float
    df: float

    @classmethod
    def from_c(cls, struct: DichotomousPgofResultStruct):
        return cls(
            expected=struct.expected[: struct.n],
            residual=struct.residual[: struct.n],
            test_statistic=struct.test_statistic,
            p_value=struct.p_value,
            df=struct.df,
        )


class DichotomousResult(BaseModel):
    model_class: str
    model_name: str
    bmdl: float
    bmd: float
    bmdu: float
    aic: float
    bounded: List[bool]
    fit: DichotomousModelResult
    gof: DichotomousPgofResult


class DichotomousMAAnalysisStruct(ctypes.Structure):
    _fields_ = [
        ("nmodels", ctypes.c_int),  # number of models for the model average
        (
            "priors",
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ),  # list of pointers to prior arrays, priors[i] is the prior array for the ith model ect
        ("nparms", ctypes.POINTER(ctypes.c_int)),  # parameters in each model
        ("actual_parms", ctypes.POINTER(ctypes.c_int)),  # actual number of parameters in the model
        (
            "prior_cols",
            ctypes.POINTER(ctypes.c_int),
        ),  # columns in the prior if there are more in the future, presently there are only 5
        ("models", ctypes.POINTER(ctypes.c_int)),  # list of models
        ("modelPriors", ctypes.POINTER(ctypes.c_double)),  # prior probability on the model
    ]


class DichotomousMAAnalysis(BaseModel):
    models: List[constants.DichotomousModel]
    priors: List[List[constants.Prior]]
    model_priors: List[int]

    class Config:
        arbitrary_types_allowed = True

    def _priors_to_list(self, priors) -> List[float]:
        """
        allocate memory for all parameters and convert to columnwise matrix
        """
        arr = np.array([list(prior.dict().values()) for prior in priors])

        return arr.T.flatten().tolist()

    def to_c(self) -> DichotomousMAAnalysisStruct:
        _priors = [self._priors_to_list(model_priors) for model_priors in self.priors]
        _prior_arrays = [list_t_c(model_priors, ctypes.c_double) for model_priors in _priors]
        import pdb; pdb.set_trace()
        _prior_pointers = [ctypes.cast(prior_array, ctypes.POINTER(ctypes.c_double)) for prior_array in _prior_arrays]
        priors = list_t_c(_prior_pointers,ctypes.POINTER(ctypes.c_double))
        return DichotomousMAAnalysisStruct(
            nmodels=ctypes.c_int(len(self.models)),
            priors=priors,
            actual_parms=list_t_c([model.num_params for model in self.models], ctypes.c_int),
            priorCols=list_t_c([constants.NUM_PRIOR_COLS] * len(self.models), ctypes.c_int),
            models=list_t_c([model.id for model in self.models], ctypes.c_int),
            modelPriors=list_t_c(self.model_priors, ctypes.c_double),
        )


class DichotomousMAResultStruct(ctypes.Structure):
    _fields_ = [
        ("nmodels", ctypes.c_int),  # number of models for each
        (
            "models",
            ctypes.POINTER(ctypes.POINTER(DichotomousModelResultStruct)),
        ),  # individual model fits for each model average
        ("dist_numE", ctypes.c_int),  # number of entries in rows for the bmd_dist
        ("post_probs", ctypes.POINTER(ctypes.c_double)),  # posterior probabilities
        ("bmd_dist", ctypes.POINTER(ctypes.c_double)),  # bmd ma distribution (dist_numE x 2) matrix
    ]


class DichotomousMAResult(BaseModel):
    results: List[DichotomousModelResultStruct]
    num_models: int
    dist_numE: int
    post_probs: List[int]
    bmd_dist: List[int]

    class Config:
        arbitrary_types_allowed = True

    def to_c(self) -> DichotomousMAAnalysisStruct:
        _results = [ctypes.pointer(struct) for struct in self.results]
        return DichotomousMAResultStruct(
            nmodels=ctypes.c_int(self.num_models),
            models=list_t_c(_results, ctypes.POINTER(DichotomousModelResultStruct)),
            dist_numE=ctypes.c_int(self.dist_numE),
            post_probs=list_t_c(self.post_probs, ctypes.c_double),
            bmd_dist=list_t_c(self.bmd_dist, ctypes.c_double),
        )
