import ctypes
from enum import IntEnum
from textwrap import dedent
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel

from bmds.bmds3.constants import ContinuousModelChoices
from bmds.datasets.continuous import ContinuousDataset

from .. import constants
from .common import NumpyFloatArray, list_t_c
from .priors import ModelPriors


class ContinuousRiskType(IntEnum):
    eAbsoluteDev = 1
    eStandardDev = 2
    eRelativeDev = 3
    ePointEstimate = 4
    eExtra = 5  # Not used
    eHybrid_Extra = 6
    eHybrid_Added = 7


class ContinuousModelSettings(BaseModel):
    suff_stat: bool = True
    bmr_type: ContinuousRiskType = ContinuousRiskType.eStandardDev
    is_increasing: Optional[bool]  # if None; autodetect used
    bmr: float = 1.0
    tail_prob: float = 0.01
    disttype: constants.DistType = constants.DistType.normal
    alpha: float = 0.05
    samples: int = 0
    degree: int = 0  # polynomial only
    burnin: int = 20
    priors: Optional[ModelPriors]  # if None; default used


class ContinuousAnalysisStruct(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_int),
        ("n", ctypes.c_int),
        ("suff_stat", ctypes.c_bool),  # true if the data are in sufficient statistics format
        ("Y", ctypes.POINTER(ctypes.c_double)),  # observed data means or actual data
        ("doses", ctypes.POINTER(ctypes.c_double)),
        (
            "sd",
            ctypes.POINTER(ctypes.c_double),
        ),  # SD of the group if suff_stat = true, null otherwise
        (
            "n_group",
            ctypes.POINTER(ctypes.c_double),
        ),  # N for each group if suff_stat = true, null otherwise
        (
            "prior",
            ctypes.POINTER(ctypes.c_double),
        ),  # a column order matrix px5 where p is the number of parameters
        ("BMD_type", ctypes.c_int),  # type of BMD
        ("isIncreasing", ctypes.c_bool),  # if the BMD is defined increasing or decreasing
        ("BMR", ctypes.c_double),  # benchmark response related to the BMD type
        ("tail_prob", ctypes.c_double),  # tail probability
        ("disttype", ctypes.c_int),  # distribution type defined in the enum distribution
        ("alpha", ctypes.c_double),  # specified alpha
        ("samples", ctypes.c_int),  # number of MCMC samples
        ("degree", ctypes.c_int),
        ("burnin", ctypes.c_int),
        ("parms", ctypes.c_int),  # number of parameters
        ("prior_cols", ctypes.c_int),
    ]

    def __str__(self) -> str:
        return dedent(
            f"""
            model: {self.model}
            n: {self.n}
            suff_stat: {self.suff_stat}
            Y: {self.Y[:self.n]}
            doses: {self.doses[:self.n]}
            sd: {self.sd[:self.n]}
            n_group: {self.n_group[:self.n]}
            prior: {self.prior[:self.parms*self.prior_cols]}
            BMD_type: {self.BMD_type}
            isIncreasing: {self.isIncreasing}
            BMR: {self.BMR}
            tail_prob: {self.tail_prob}
            disttype: {self.disttype}
            alpha: {self.alpha}
            samples: {self.samples}
            degree: {self.degree}
            burnin: {self.burnin}
            parms: {self.parms}
            prior_cols: {self.prior_cols}
            """
        )


class ContinuousAnalysis(BaseModel):
    model: constants.ContinuousModel
    dataset: ContinuousDataset
    priors: ModelPriors
    suff_stat: bool
    BMD_type: int
    is_increasing: bool
    BMR: float
    tail_prob: float
    disttype: int
    alpha: float
    samples: int
    burnin: int
    degree: int

    class Config:
        arbitrary_types_allowed = True

    @property
    def num_params(self) -> int:
        return (
            self.degree + 2
            if self.model == ContinuousModelChoices.c_polynomial.value
            else self.model.num_params
        )

    def _priors_array(self) -> np.ndarray:
        if self.model.id is constants.ContinuousModelIds.c_polynomial:
            return self.priors.to_c(degree=self.degree, dist_type=self.disttype)
        else:
            return self.priors.to_c(dist_type=self.disttype)

    def to_c(self) -> ContinuousAnalysisStruct:
        priors = self._priors_array()
        priors_pointer = np.ctypeslib.as_ctypes(priors)
        return ContinuousAnalysisStruct(
            BMD_type=ctypes.c_int(self.BMD_type),
            BMR=ctypes.c_double(self.BMR),
            Y=list_t_c(self.dataset.means, ctypes.c_double),
            alpha=ctypes.c_double(self.alpha),
            burnin=ctypes.c_int(self.burnin),
            degree=ctypes.c_int(self.degree),
            disttype=ctypes.c_int(self.disttype),
            doses=list_t_c(self.dataset.doses, ctypes.c_double),
            isIncreasing=ctypes.c_bool(self.is_increasing),
            model=ctypes.c_int(self.model.id),
            n=ctypes.c_int(self.dataset.num_dose_groups),
            n_group=list_t_c(self.dataset.ns, ctypes.c_double),
            parms=ctypes.c_int(self.num_params),
            prior=priors_pointer,
            prior_cols=ctypes.c_int(constants.NUM_PRIOR_COLS),
            samples=ctypes.c_int(self.samples),
            sd=list_t_c(self.dataset.stdevs, ctypes.c_double),
            suff_stat=ctypes.c_bool(self.suff_stat),
            tail_prob=ctypes.c_double(self.tail_prob),
        )


class ContinuousModelResultStruct(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_int),  # continuous model specification
        ("dist", ctypes.c_int),  # distribution type
        ("nparms", ctypes.c_int),  # number of parameters in the model
        ("parms", ctypes.POINTER(ctypes.c_double)),  # parameter estimate
        ("cov", ctypes.POINTER(ctypes.c_double)),  # covariance estimate
        ("max", ctypes.c_double),  # value of the likelihood/posterior at the maximum
        ("dist_numE", ctypes.c_int),  # number of entries in rows for the bmd_dist
        ("model_df", ctypes.c_double),
        ("total_df", ctypes.c_double),
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


class ContinuousModelResult(BaseModel):

    model: constants.ContinuousModel
    dist: Optional[int]
    num_params: int
    params: Optional[List[float]]
    cov: Optional[NumpyFloatArray]
    max: Optional[float]
    model_df: Optional[float]
    total_df: Optional[float]
    dist_numE: int
    bmd_dist: Optional[NumpyFloatArray]

    class Config:
        arbitrary_types_allowed = True

    def to_c(self):
        return ContinuousModelResultStruct(
            model=self.model.id, nparms=self.num_params, dist_numE=self.dist_numE
        )

    def from_c(self, struct: ContinuousModelResultStruct):
        self.params = struct.np_parms.tolist()
        self.cov = struct.np_cov.reshape(self.num_params, self.num_params)
        self.max = struct.max
        self.model_df = struct.model_df
        self.total_df = struct.total_df
        arr = struct.np_bmd_dist.reshape(2, self.dist_numE)
        arr = arr[:, np.isfinite(arr[0, :])]
        arr = arr[:, arr[0, :] > 0]
        self.bmd_dist = arr

    def dict(self, **kw) -> Dict:
        kw.update(exclude={"cov", "bmd_dist"})
        d = super().dict(**kw)
        d["cov"] = self.cov.tolist()
        d["bmd_dist"] = self.bmd_dist.tolist()
        return d


class ContinuousBmdsResultsStruct(ctypes.Structure):
    _fields_ = [
        ("bmd", ctypes.c_double),
        ("bmdl", ctypes.c_double),
        ("bmdu", ctypes.c_double),
        ("aic", ctypes.c_double),
        ("bounded", ctypes.POINTER(ctypes.c_bool)),
    ]

    @classmethod
    def from_results(cls, results: ContinuousModelResult) -> "ContinuousBmdsResultsStruct":
        return cls(
            bmd=constants.BMDS_BLANK_VALUE,
            bmdl=constants.BMDS_BLANK_VALUE,
            bmdu=constants.BMDS_BLANK_VALUE,
            aic=constants.BMDS_BLANK_VALUE,
            bounded=list_t_c([False for _ in range(results.num_params)], ctypes.c_bool),
        )


class ContinuousResult(BaseModel):
    bmdl: float
    bmd: float
    bmdu: float
    aic: float
    roi: float
    bounded: List[bool]
    fit: ContinuousModelResult
    dr_x: List[float]
    dr_y: List[float]
    bmdl_y: float
    bmd_y: float
    bmdu_y: float
