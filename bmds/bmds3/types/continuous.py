from bmds.datasets.continuous import ContinuousDataset
import ctypes
from enum import IntEnum
from textwrap import dedent
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

from .. import constants
from .common import list_t_c


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
    bmr_type: ContinuousRiskType = ContinuousRiskType.eAbsoluteDev
    isIncreasing: bool = True
    bmr: float = 1.0
    tail_prob: float = 1.0
    disttype: int = 1
    alpha: float = 1.0
    samples: int = 1
    burnin: int = 1


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
        ("burnin", ctypes.c_int),  # burn in
        ("parms", ctypes.c_int),  # number of parameters
        ("prior_cols", ctypes.c_int),
    ]

    def __str__(self) -> str:
        return dedent(
            f"""
            model: {self.model}
            BMR: {self.BMR}
            """
        )

class ContinuousAnalysis(BaseModel):
    model: constants.ContinuousModel
    dataset: ContinuousDataset
    priors: List[constants.Prior]
    suff_stat: bool
    BMD_type: int
    isIncreasing: bool
    BMR: float
    tail_prob: float
    disttype: int
    alpha: float
    samples: int

    class Config:
        arbitrary_types_allowed = True

    @property
    def num_params(self) -> int:
        return self.model.num_params

    def _priors_to_list(self) -> List[float]:
        """
        allocate memory for all parameters and convert to columnwise matrix
        """
        if len(self.priors) == self.model.num_params:
            # most cases
            arr = np.array([list(prior.dict().values()) for prior in self.priors])
        elif len(self.priors) < self.model.num_params:
            # special case for multistage; apply all priors; copy last one
            data: List[List[float]] = []
            for prior in self.priors[:-1]:
                data.append(list(prior.dict().values()))
            for _ in range(len(self.priors) - 1, self.model.num_params):
                data.append(list(self.priors[-1].dict().values()))
            arr = np.array(data)
        else:
            raise ValueError("Unknown state")
        return arr.T.flatten().tolist()

    def to_c(self) -> ContinuousAnalysisStruct:
        priors = self._priors_to_list()
        return ContinuousAnalysisStruct(
            BMD_type=ctypes.c_int(self.BMD_type),
            BMR=ctypes.c_double(self.BMR),
            Y=list_t_c(self.dataset.means, ctypes.c_double),
            alpha=ctypes.c_double(self.alpha),
            disttype=ctypes.c_int(self.disttype),
            doses=list_t_c(self.dataset.doses, ctypes.c_double),
            isIncreasing=ctypes.c_bool(self.isIncreasing),
            model=ctypes.c_int(self.model.id),
            n=ctypes.c_int(self.dataset.num_dose_groups),
            n_group=list_t_c(self.dataset.ns, ctypes.c_double),
            parms=ctypes.c_int(self.model.num_params),
            prior=list_t_c(priors, ctypes.c_double),
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
        (
            "bmd_dist",
            ctypes.POINTER(ctypes.c_double),
        ),  # bmd distribution (dist_numE x 2) matrix
    ]

class ContinuousModelResult(BaseModel):

    model: constants.ContinuousModel
    dist: Optional[int] = 1
    num_params: int
    params: Optional[List[float]]
    cov: Optional[np.ndarray]
    max: Optional[float] = 1.0
    dist_numE: int
    bmd_dist: Optional[np.ndarray]

    class Config:
        arbitrary_types_allowed = True


    def to_c(self):
        parms = np.zeros(self.num_params, dtype=np.float64)
        self.cov = np.zeros(self.num_params ** 2, dtype=np.float64)
        self.bmd_dist = np.zeros(self.dist_numE * 2, dtype=np.float64)
        return ContinuousModelResultStruct(
            model=ctypes.c_int(self.model.id),
            dist=ctypes.c_int(self.dist),
            nparms=ctypes.c_int(self.num_params),
            parms=parms.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cov=self.cov.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            max=ctypes.c_double(self.max),
            dist_numE=ctypes.c_int(self.dist_numE),
            bmd_dist=self.bmd_dist.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

    def from_c(self, struct: ContinuousModelResultStruct):
        self.params = struct.parms[: self.num_params]
        self.cov = self.cov.reshape(self.num_params, self.num_params)
        self.max = struct.max
