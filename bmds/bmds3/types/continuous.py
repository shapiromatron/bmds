import ctypes
from enum import IntEnum
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel

from bmds.bmds3.constants import ContinuousModelChoices
from bmds.datasets.continuous import ContinuousDatasets

from ...constants import Dtype
from .. import constants
from .common import NumpyFloatArray, list_t_c
from .priors import ModelPriors
from .structs import (
    ContinuousAnalysisStruct,
    ContinuousBmdsResultsStruct,
    ContinuousModelResultStruct,
    ContinuousStructs,
)


class ContinuousRiskType(IntEnum):
    eAbsoluteDev = 1
    eStandardDev = 2
    eRelativeDev = 3
    ePointEstimate = 4
    eExtra = 5  # Not used
    eHybrid_Extra = 6
    eHybrid_Added = 7


class ContinuousModelSettings(BaseModel):
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


class ContinuousAnalysis(BaseModel):
    model: constants.ContinuousModel
    dataset: ContinuousDatasets
    priors: ModelPriors
    BMD_type: ContinuousRiskType
    is_increasing: bool
    BMR: float
    tail_prob: float
    disttype: constants.DistType
    alpha: float
    samples: int
    burnin: int
    degree: int

    class Config:
        arbitrary_types_allowed = True

    @property
    def num_params(self) -> int:
        if self.model == ContinuousModelChoices.c_polynomial.value:
            params = self.degree + 1
        else:
            params = len(self.model.params)

        if self.disttype is constants.DistType.normal_ncv:
            params += 2
        else:
            params += 1

        return params

    def _priors_array(self) -> np.ndarray:
        if self.model.id == constants.ContinuousModelIds.c_polynomial:
            return self.priors.to_c(degree=self.degree, dist_type=self.disttype)
        else:
            return self.priors.to_c(dist_type=self.disttype)

    def to_c(self) -> ContinuousStructs:
        priors = self._priors_array()
        priors_pointer = np.ctypeslib.as_ctypes(priors)
        nparms = self.num_params

        struct = ContinuousAnalysisStruct(
            BMD_type=ctypes.c_int(self.BMD_type),
            BMR=ctypes.c_double(self.BMR),
            alpha=ctypes.c_double(self.alpha),
            burnin=ctypes.c_int(self.burnin),
            degree=ctypes.c_int(self.degree),
            disttype=ctypes.c_int(self.disttype),
            isIncreasing=ctypes.c_bool(self.is_increasing),
            model=ctypes.c_int(self.model.id),
            parms=ctypes.c_int(nparms),
            prior=priors_pointer,
            prior_cols=ctypes.c_int(constants.NUM_PRIOR_COLS),
            samples=ctypes.c_int(self.samples),
            tail_prob=ctypes.c_double(self.tail_prob),
        )

        if self.dataset.dtype == Dtype.CONTINUOUS:
            struct.suff_stat = ctypes.c_bool(True)
            struct.Y = list_t_c(self.dataset.means, ctypes.c_double)
            struct.doses = list_t_c(self.dataset.doses, ctypes.c_double)
            struct.n = ctypes.c_int(self.dataset.num_dose_groups)
            struct.n_group = list_t_c(self.dataset.ns, ctypes.c_double)
            struct.sd = list_t_c(self.dataset.stdevs, ctypes.c_double)
        elif self.dataset.dtype == Dtype.CONTINUOUS_INDIVIDUAL:
            struct.suff_stat = ctypes.c_bool(False)
            struct.Y = list_t_c(self.dataset.responses, ctypes.c_double)
            struct.doses = list_t_c(self.dataset.individual_doses, ctypes.c_double)
            struct.n = ctypes.c_int(len(self.dataset.individual_doses))
            struct.n_group = list_t_c([], ctypes.c_double)
            struct.sd = list_t_c([], ctypes.c_double)
        else:
            raise ValueError(f"Invalid dtype: {self.dataset.dtype}")

        return ContinuousStructs(
            analysis=struct,
            result=ContinuousModelResultStruct(nparms=nparms, dist_numE=constants.N_BMD_DIST),
            summary=ContinuousBmdsResultsStruct(nparms=nparms),
        )


class ContinuousModelResult(BaseModel):

    dist: int
    params: List[float]
    cov: NumpyFloatArray
    max: float
    model_df: float
    total_df: float
    bmd_dist: NumpyFloatArray

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_c(cls, struct: ContinuousModelResultStruct) -> "ContinuousModelResult":

        arr = struct.np_bmd_dist.reshape(2, struct.dist_numE)
        arr = arr[:, np.isfinite(arr[0, :])]
        arr = arr[:, arr[0, :] > 0]

        return ContinuousModelResult(
            dist=struct.dist,
            params=struct.np_parms.tolist(),
            cov=struct.np_cov[: struct.nparms ** 2].reshape(struct.nparms, struct.nparms),
            max=struct.max,
            model_df=struct.model_df,
            total_df=struct.total_df,
            bmd_dist=arr,
        )

    def dict(self, **kw) -> Dict:
        kw.update(exclude={"cov", "bmd_dist"})
        d = super().dict(**kw)
        d["cov"] = self.cov.tolist()
        d["bmd_dist"] = self.bmd_dist.tolist()
        return d


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
