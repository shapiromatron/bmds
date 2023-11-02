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

    bmdl: float
    bmd: float
    bmdu: float
    bmdl_y: float
    bmd_y: float
    bmdu_y: float
    bmd_dist: NumpyFloatArray
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
        bmds = [structs.result.bmdl_ma, structs.result.bmd_ma, structs.result.bmdu_ma]
        bmds_ys = np.interp(bmds, dr_x, dr_y)
        return cls(
            bmdl=bmds[0],
            bmd=bmds[1],
            bmdu=bmds[2],
            bmdl_y=bmds_ys[0],
            bmd_y=bmds_ys[1],
            bmdu_y=bmds_ys[2],
            bmd_dist=arr.T,
            priors=priors,
            posteriors=posteriors,
            dr_x=dr_x,
            dr_y=dr_y,
        )

    def update_record(self, d: dict) -> None:
        """Update data record for a tabular-friendly export"""
        d.update(
            bmdl=self.bmdl,
            bmd=self.bmd,
            bmdu=self.bmdu,
        )

    def update_record_weights(self, d: dict, index: int) -> None:
        """Update data record for a tabular-friendly export"""
        d.update(
            model_prior=self.priors[index],
            model_posterior=self.posteriors[index],
        )
