from typing import Dict

import numpy as np
from pydantic import BaseModel

from .continuous import NumpyFloatArray
from .structs import DichotomousMAStructs


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
    def from_structs(cls, structs: DichotomousMAStructs, model_results):
        # only keep positive finite values
        arr = structs.dich_result.np_bmd_dist.reshape(2, structs.dich_result.dist_numE).T
        arr = arr[np.isfinite(arr[:, 0])]
        arr = arr[arr[:, 0] > 0]

        # calculate dr_y for model averaging
        priors = structs.analysis.np_modelPriors
        posteriors = structs.dich_result.np_post_probs
        values = np.array([result.plotting.dr_y for result in model_results])
        dr_x = model_results[0].plotting.dr_x
        dr_y = values.T.dot(posteriors)

        return cls(
            bmdl=structs.result.bmdl_ma,
            bmd=structs.result.bmd_ma,
            bmdu=structs.result.bmdu_ma,
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
