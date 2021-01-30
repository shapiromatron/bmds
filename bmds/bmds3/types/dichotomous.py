import ctypes
from enum import IntEnum
from textwrap import dedent
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, confloat, conint

from bmds.bmds3.constants import DichotomousModelChoices, ModelPriors

from ...datasets import DichotomousDataset
from .. import constants
from .common import NumpyFloatArray, list_t_c


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
    priors: Optional[ModelPriors]  # if None; default used


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
    information is used, or a MA analysis, in which all the information
    save prior, degree, parms and prior_cols are used.
    """

    model: constants.DichotomousModel
    dataset: DichotomousDataset
    priors: ModelPriors
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

    def _priors_array(self) -> np.ndarray:
        if self.model.id is constants.DichotomousModelIds.d_multistage:
            return self.priors.to_c(degree=self.degree)
        else:
            return self.priors.to_c()

    def to_c(self) -> DichotomousAnalysisStruct:
        priors = self._priors_array()
        priors_pointer = np.ctypeslib.as_ctypes(priors)
        return DichotomousAnalysisStruct(
            model=ctypes.c_int(self.model.id),
            n=ctypes.c_int(self.dataset.num_dose_groups),
            Y=list_t_c(self.dataset.incidences, ctypes.c_double),
            doses=list_t_c(self.dataset.doses, ctypes.c_double),
            n_group=list_t_c(self.dataset.ns, ctypes.c_double),
            prior=priors_pointer,
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # reference same memory in struct and numpy
        # https://stackoverflow.com/a/23330369/906385
        self.np_parms = np.zeros(kwargs["nparms"], dtype=np.float64)
        self.parms = np.ctypeslib.as_ctypes(self.np_parms)
        self.np_cov = np.zeros(kwargs["nparms"] ** 2, dtype=np.float64)
        self.cov = np.ctypeslib.as_ctypes(self.np_cov)
        self.np_bmd_dist = np.zeros(kwargs["dist_numE"] * 2, dtype=np.float64)
        self.bmd_dist = np.ctypeslib.as_ctypes(self.np_bmd_dist)

    def __str__(self) -> str:
        return dedent(
            f"""
            model: {self.model}
            nparms: {self.nparms}
            parms: {self.parms[:self.nparms]}
            cov: {self.cov[:self.nparms**2]}
            max: {self.max}
            dist_numE: {self.dist_numE}
            model_df: {self.model_df}
            total_df: {self.total_df}
            bmd_dist: {self.bmd_dist[:self.dist_numE*2]}
            """
        )


class DichotomousModelResult(BaseModel):
    """
    Single model fit.
    """

    num_params: int
    dist_numE: int
    params: Optional[List[float]]
    cov: Optional[NumpyFloatArray]
    max: Optional[float]
    model_df: Optional[float]
    total_df: Optional[float]
    bmd_dist: Optional[NumpyFloatArray]

    class Config:
        arbitrary_types_allowed = True

    def to_c(self, model_id: int) -> DichotomousModelResultStruct:
        return DichotomousModelResultStruct(
            model=model_id, nparms=self.num_params, dist_numE=self.dist_numE
        )

    def from_c(self, struct: DichotomousModelResultStruct, model):
        self.params = model.transform_params(struct)
        self.cov = struct.np_cov.reshape(self.num_params, self.num_params)
        self.max = struct.max
        self.model_df = struct.model_df
        self.total_df = struct.total_df

        # reshape; get rid of 0 and inf; must be JSON serializable
        arr = struct.np_bmd_dist.reshape(2, self.dist_numE)
        arr = arr[:, np.isfinite(arr[0, :])]
        arr = arr[:, arr[0, :] > 0]
        self.bmd_dist = arr

    def bmd_plot(self):
        df = pd.DataFrame(data=self.bmd_dist, columns="bmd quantile".split())
        df = df.query("bmd>0 & bmd < inf")
        df.plot.scatter("bmd", "quantile", xlabel="Dose", ylabel="Propotion")

    def dict(self, **kw) -> Dict:
        kw.update(exclude={"cov", "bmd_dist"})
        d = super().dict(**kw)
        d["cov"] = self.cov.tolist()
        d["bmd_dist"] = self.bmd_dist.tolist()
        return d


class DichotomousPgofResultStruct(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_int),  # total number of observations obs/n
        ("expected", ctypes.POINTER(ctypes.c_double)),
        ("residual", ctypes.POINTER(ctypes.c_double)),
        ("test_statistic", ctypes.c_double),
        ("p_value", ctypes.c_double),
        ("df", ctypes.c_double),
    ]

    def __str__(self) -> str:
        return dedent(
            f"""
            n: {self.n}
            expected: {self.expected[:self.n]}
            residual: {self.residual[:self.n]}
            test_statistic: {self.test_statistic}
            p_value: {self.p_value}
            df: {self.df}
            """
        )

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

    def __str__(self) -> str:
        return dedent(
            f"""
            bmd: {self.bmd}
            bmdl: {self.bmdl}
            bmdu: {self.bmdu}
            aic: {self.aic}
            bounded: <not shown>
            """
        )

    @classmethod
    def from_results(cls, results: DichotomousModelResult) -> "DichotomousBmdsResultsStruct":
        return cls(
            bmd=constants.BMDS_BLANK_VALUE,
            bmdl=constants.BMDS_BLANK_VALUE,
            bmdu=constants.BMDS_BLANK_VALUE,
            aic=constants.BMDS_BLANK_VALUE,
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
    bmdl: float
    bmd: float
    bmdu: float
    aic: float
    roi: float
    bounded: List[bool]
    fit: DichotomousModelResult
    gof: DichotomousPgofResult
    dr_x: List[float]
    dr_y: List[float]
    bmdl_y: float
    bmd_y: float
    bmdu_y: float
