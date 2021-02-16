import ctypes
from enum import IntEnum
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel, confloat, conint

from bmds.bmds3.constants import DichotomousModelChoices, ModelPriors

from ...datasets import DichotomousDataset
from .. import constants
from .common import NumpyFloatArray, list_t_c, residual_of_interest
from .structs import (
    DichotomousAnalysisStruct,
    DichotomousAodStruct,
    DichotomousBmdsResultsStruct,
    DichotomousModelResultStruct,
    DichotomousPgofResultStruct,
    DichotomousStructs,
)


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
        if self.model.id == constants.DichotomousModelIds.d_multistage:
            return self.priors.to_c(degree=self.degree)
        else:
            return self.priors.to_c()

    def to_c(self) -> DichotomousStructs:
        priors = self._priors_array()
        priors_pointer = np.ctypeslib.as_ctypes(priors)
        return DichotomousStructs(
            analysis=DichotomousAnalysisStruct(
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
            ),
            result=DichotomousModelResultStruct(
                model=self.model.id, nparms=self.num_params, dist_numE=constants.N_BMD_DIST
            ),
            gof=DichotomousPgofResultStruct(n=self.dataset.num_dose_groups),
            summary=DichotomousBmdsResultsStruct(num_params=self.num_params),
            aod=DichotomousAodStruct(),
        )


class DichotomousModelResult(BaseModel):
    max: float  # likelihood
    model_df: float
    total_df: float
    bmd_dist: NumpyFloatArray

    @classmethod
    def from_c(cls, struct: DichotomousModelResultStruct) -> "DichotomousModelResult":
        # reshape; get rid of 0 and inf; must be JSON serializable
        arr = struct.np_bmd_dist.reshape(2, struct.dist_numE)
        arr = arr[:, np.isfinite(arr[0, :])]
        arr = arr[:, arr[0, :] > 0]

        return DichotomousModelResult(
            max=struct.max, model_df=struct.model_df, total_df=struct.total_df, bmd_dist=arr,
        )

    def dict(self, **kw) -> Dict:
        kw.update(exclude={"cov", "bmd_dist"})
        d = super().dict(**kw)
        d["bmd_dist"] = self.bmd_dist.tolist()
        return d


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


class DichotomousParameters(BaseModel):
    names: List[str]
    values: List[float]
    bounded: List[bool]
    cov: NumpyFloatArray

    @classmethod
    def from_model(cls, model) -> "DichotomousParameters":
        results = model.structs.result
        return cls(
            names=model.get_param_names(),
            values=model.transform_params(results),
            bounded=model.structs.summary.np_bounded.tolist(),
            cov=results.np_cov.reshape(results.nparms, results.nparms),
        )

    def dict(self, **kw) -> Dict:
        kw.update(exclude={"cov"})
        d = super().dict(**kw)
        d["cov"] = self.cov.tolist()
        return d


class DichotomousPlotting(BaseModel):
    dr_x: NumpyFloatArray
    dr_y: NumpyFloatArray
    bmdl_y: float
    bmd_y: float
    bmdu_y: float

    @classmethod
    def from_model(cls, model, params) -> "DichotomousPlotting":
        structs = model.structs
        dr_x = model.dataset.dose_linspace
        critical_xs = np.array([structs.summary.bmdl, structs.summary.bmd, structs.summary.bmdu])
        dr_y = model.dr_curve(dr_x, params)
        critical_ys = model.dr_curve(critical_xs, params)
        return cls(
            dr_x=dr_x,
            dr_y=dr_y,
            bmdl_y=critical_ys[0] if structs.summary.bmdl > 0 else constants.BMDS_BLANK_VALUE,
            bmd_y=critical_ys[1] if structs.summary.bmd > 0 else constants.BMDS_BLANK_VALUE,
            bmdu_y=critical_ys[2] if structs.summary.bmdu > 0 else constants.BMDS_BLANK_VALUE,
        )

    def dict(self, **kw) -> Dict:
        kw.update(exclude={"dr_x", "dr_y"})
        d = super().dict(**kw)
        d["dr_x"] = self.dr_x.tolist()
        d["dr_y"] = self.dr_y.tolist()
        return d


class DichotomousResult(BaseModel):
    bmdl: float
    bmd: float
    bmdu: float
    aic: float
    roi: float
    fit: DichotomousModelResult
    gof: DichotomousPgofResult
    parameters: DichotomousParameters
    plotting: DichotomousPlotting

    @classmethod
    def from_model(cls, model) -> "DichotomousResult":
        structs = model.structs
        fit = DichotomousModelResult.from_c(structs.result)
        gof = DichotomousPgofResult.from_c(structs.gof)
        parameters = DichotomousParameters.from_model(model)
        plotting = DichotomousPlotting.from_model(model, parameters.values)
        return cls(
            bmdl=structs.summary.bmdl,
            bmd=structs.summary.bmd,
            bmdu=structs.summary.bmdu,
            aic=structs.summary.aic,
            roi=residual_of_interest(structs.summary.bmd, model.dataset.doses, gof.residual),
            fit=fit,
            gof=gof,
            parameters=parameters,
            plotting=plotting,
        )

