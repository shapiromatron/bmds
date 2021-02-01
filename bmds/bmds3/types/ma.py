from typing import List

import numpy as np
from pydantic import BaseModel

from ..models.dichotomous import BmdModelDichotomous
from .structs import DichotomousMAAnalysisStruct, DichotomousMAResultStruct


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
