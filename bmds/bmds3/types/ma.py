import ctypes
from typing import List

import numpy as np
from pydantic import BaseModel

from ..models.dichotomous import BmdModelDichotomous
from .common import list_t_c
from .structs import DichotomousAnalysisStruct, DichotomousModelResultStruct


class DichotomousMAAnalysisStruct(ctypes.Structure):
    _fields_ = [
        ("nmodels", ctypes.c_int),
        ("priors", ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),),
        ("nparms", ctypes.POINTER(ctypes.c_int)),
        ("actual_parms", ctypes.POINTER(ctypes.c_int)),
        ("prior_cols", ctypes.POINTER(ctypes.c_int),),
        ("models", ctypes.POINTER(ctypes.c_int)),
        ("modelPriors", ctypes.POINTER(ctypes.c_double)),
    ]

    @classmethod
    def from_python(cls, models: List[DichotomousAnalysisStruct]):

        # list of floats
        priors = [
            list_t_c(model.prior[: model.parms * model.prior_cols], ctypes.c_double,)
            for model in models
        ]

        # pointer of pointers
        priors2 = list_t_c(
            [ctypes.cast(el, ctypes.POINTER(ctypes.c_double)) for el in priors],
            ctypes.POINTER(ctypes.c_double),
        )

        return cls(
            nmodels=ctypes.c_int(len(models)),
            priors=priors2,
            nparms=list_t_c([model.parms for model in models], ctypes.c_int),
            actual_parms=list_t_c([model.parms for model in models], ctypes.c_int),
            prior_cols=list_t_c([model.prior_cols for model in models], ctypes.c_int),
            models=list_t_c([model.model for model in models], ctypes.c_int),
            modelPriors=list_t_c([1 / len(models)] * len(models), ctypes.c_double),
        )


class DichotomousMAResultStruct(ctypes.Structure):
    _fields_ = [
        ("nmodels", ctypes.c_int),
        ("models", ctypes.POINTER(ctypes.POINTER(DichotomousModelResultStruct))),
        ("dist_numE", ctypes.c_int),
        ("post_probs", ctypes.POINTER(ctypes.c_double)),
        ("bmd_dist", ctypes.POINTER(ctypes.c_double)),
    ]

    @classmethod
    def from_python(cls, models: List[DichotomousModelResultStruct]):
        _results = [ctypes.pointer(model) for model in models]
        nmodels = len(models)
        dist_numE = 200
        return DichotomousMAResultStruct(
            nmodels=nmodels,
            models=list_t_c(_results, ctypes.POINTER(DichotomousModelResultStruct)),
            dist_numE=ctypes.c_int(dist_numE),
            post_probs=(ctypes.c_double * nmodels)(),
            bmd_dist=(ctypes.c_double * (dist_numE * 2))(),
        )


class ModelAverageResult(BaseModel):
    pass


class DichotomousModelAverageResult(ModelAverageResult):
    """
    Model average fit
    """

    bmd: float
    bmdl: float
    bmdu: float
    bmd_quantile: List[float]
    bmd_value: List[float]
    priors: List[float]
    posteriors: List[float]
    dr_x: List[float]
    dr_y: List[float]

    @classmethod
    def from_execution(
        cls,
        inputs: DichotomousMAAnalysisStruct,
        outputs: DichotomousMAResultStruct,
        models: List[BmdModelDichotomous],
    ):
        arr = np.array(outputs.bmd_dist[: outputs.dist_numE * 2]).reshape(2, outputs.dist_numE).T
        arr = arr[np.isfinite(arr[:, 0])]
        arr = arr[arr[:, 0] > 0]

        priors = inputs.modelPriors[: inputs.nmodels]
        posteriors = np.array(outputs.post_probs[: outputs.nmodels])
        dr_x = models[0].results.dr_x

        values = np.array([m.results.dr_y for m in models])
        dr_y = values.T.dot(posteriors)

        values = np.array([[m.results.bmdl, m.results.bmd, m.results.bmdu] for m in models])
        bmds = values.T.dot(posteriors)

        return cls(
            bmdl=bmds[0],
            bmd=bmds[1],
            bmdu=bmds[2],
            bmd_quantile=arr.T[0, :].tolist(),
            bmd_value=arr.T[1, :].tolist(),
            priors=priors,
            posteriors=posteriors.tolist(),
            dr_x=dr_x,
            dr_y=dr_y.tolist(),
        )
