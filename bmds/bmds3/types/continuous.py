import ctypes
from enum import IntEnum
from typing import List

from pydantic import BaseModel

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
    pass


class ContinuousAnalysis(BaseModel):
    model: int
    n: int
    suff_stat: bool
    Y: List[float]
    doses: List[float]
    sd: List[float]
    n_group: List[float]
    prior: List[float]
    BMD_type: int
    isIncreasing: bool
    BMR: float
    tail_prob: float
    disttype: int
    alpha: float
    samples: int
    burnin: int
    parms: int
    prior_cols: int

    class Struct(ctypes.Structure):
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

    def to_c(self):
        return self.Struct(
            model=ctypes.c_int(self.model),
            n=ctypes.c_int(self.n),
            suff_stat=ctypes.c_bool(self.suff_stat),
            Y=list_t_c(self.Y, ctypes.c_double),
            doses=list_t_c(self.doses, ctypes.c_double),
            sd=list_t_c(self.sd, ctypes.c_double),
            n_group=list_t_c(self.n_group, ctypes.c_double),
            prior=list_t_c(self.prior, ctypes.c_double),
            BMD_type=ctypes.c_int(self.BMD_type),
            isIncreasing=ctypes.c_bool(self.isIncreasing),
            BMR=ctypes.c_double(self.BMR),
            tail_prob=ctypes.c_double(self.tail_prob),
            disttype=ctypes.c_int(self.disttype),
            alpha=ctypes.c_double(self.alpha),
            samples=ctypes.c_int(self.samples),
            burnin=ctypes.c_int(self.burnin),
            parms=ctypes.c_int(self.parms),
            prior_cols=ctypes.c_int(self.prior_cols),
        )

    @classmethod
    def from_c(cls, struct):
        return cls(
            model=struct.model.value,
            n=struct.n.value,
            suff_stat=struct.suff_stat.value,
            Y=struct.Y[: struct.n.value],
            doses=struct.doses[: struct.n.value],
            sd=struct.sd[: struct.n.value],
            n_group=struct.n_group[: struct.n.value],
            prior=struct.prior[: struct.parms.value * struct.prior_cols.value],
            BMD_type=struct.BMD_type.value,
            isIncreasing=struct.isIncreasing.value,
            BMR=struct.BMR.value,
            tail_prob=struct.tail_prob.value,
            disttype=struct.disttype.value,
            alpha=struct.alpha.value,
            samples=struct.samples.value,
            burnin=struct.burnin.value,
            parms=struct.parms.value,
            prior_cols=struct.prior_cols.value,
        )


class ContinuousModelResult(BaseModel):

    model: int
    dist: int
    nparms: int
    parms: List[float]
    cov: List[float]
    max: float
    dist_numE: int
    bmd_dist: List[float]

    class Struct(ctypes.Structure):
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

    def to_c(self):
        return self.Struct(
            model=ctypes.c_int(self.model),
            dist=ctypes.c_int(self.dist),
            nparms=ctypes.c_int(self.nparms),
            parms=list_t_c(self.parms, ctypes.c_double),
            cov=list_t_c(self.cov, ctypes.c_double),
            max=ctypes.c_double(self.max),
            dist_numE=ctypes.c_int(self.dist_numE),
            bmd_dist=list_t_c(self.bmd_dist, ctypes.c_double),
        )

    @classmethod
    def from_c(cls, struct):
        return cls(
            model=struct.model.value,
            dist=struct.dist.value,
            nparms=struct.nparms.value,
            parms=struct.parms[: struct.nparms.value],
            cov=struct.cov[: struct.nparms.value],
            max=struct.max.value,
            dist_numE=struct.dist_numE.value,
            bmd_dist=struct.bmd_dist[: struct.dist_numE ** 2],
        )
