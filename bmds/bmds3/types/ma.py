from typing import Dict, List

import numpy as np
from pydantic import BaseModel

from ..models.dichotomous import BmdModelDichotomous
from .continuous import NumpyFloatArray
from .dichotomous import DichotomousResult
from .structs import DichotomousMAAnalysisStruct, DichotomousMAResultStruct, MAResultsStruct


class ModelAverageResult(BaseModel):
    pass


class DichotomousModelAverageResult(ModelAverageResult):
    """
    Model average fit
    """

    bmd: float
    bmdl: float
    bmdu: float
    bmd_quantile: NumpyFloatArray
    bmd_value: NumpyFloatArray
    priors: NumpyFloatArray
    posteriors: NumpyFloatArray
    dr_x: NumpyFloatArray
    dr_y: NumpyFloatArray

    @classmethod
    def from_execution(
        cls,
        inputs: DichotomousMAAnalysisStruct,
        outputs: DichotomousMAResultStruct,
        model_results: List[DichotomousResult],
        ma_result_struct: MAResultsStruct,
    ):
        arr = np.array(outputs.bmd_dist[: outputs.dist_numE * 2]).reshape(2, outputs.dist_numE).T
        arr = arr[np.isfinite(arr[:, 0])]
        arr = arr[arr[:, 0] > 0]

        priors = inputs.modelPriors[: inputs.nmodels]
        posteriors = np.array(outputs.post_probs[: outputs.nmodels])
        values = np.array([result.plotting.dr_y for result in model_results])

        dr_x = model_results[0].plotting.dr_x
        dr_y = values.T.dot(posteriors)

        return cls(
            bmdl=ma_result_struct.bmdl_ma,
            bmd=ma_result_struct.bmd_ma,
            bmdu=ma_result_struct.bmdu_ma,
            bmd_quantile=arr.T[0, :],
            bmd_value=arr.T[1, :],
            priors=priors,
            posteriors=posteriors,
            dr_x=dr_x,
            dr_y=dr_y,
        )

    def dict(self, **kw) -> Dict:
        d = super().dict(**kw)
        return NumpyFloatArray.listify(d)
