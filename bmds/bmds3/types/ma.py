from textwrap import dedent
from typing import Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from bmds import bmdscore

from ..models.dichotomous import BmdModelDichotomous
from .continuous import NumpyFloatArray


class ModelAverageResult(BaseModel):
    pass


class DichotomousModelAverage:
    def __init__(self, dataset, models: list[BmdModelDichotomous], model_weights: npt.NDArray):
        self.average = bmdscore.python_dichotomousMA_analysis()
        self.average.nmodels = len(models)
        self.average.nparms = [model.structs.result.nparms for model in models]
        self.average.actual_parms = [
            model.structs.result.nparms for model in models
        ]  # TODO - different than prior code?
        # self.average.actual_parms = [model.structs.result.parms for model in models]
        self.average.prior_cols = [model.structs.analysis.prior_cols for model in models]
        self.average.models = [model.structs.analysis.model for model in models]
        self.average.priors = [model.structs.analysis.prior for model in models]
        self.average.modelPriors = model_weights

        first = models[0].structs.analysis
        self.analysis = bmdscore.python_dichotomous_analysis()
        self.analysis.BMD_type = first.BMD_type
        self.analysis.BMR = first.BMR
        self.analysis.alpha = first.alpha
        self.analysis.Y = dataset.incidences
        self.analysis.n_group = dataset.ns
        self.analysis.doses = dataset.doses
        self.analysis.n = dataset.num_dose_groups

        self.averageRes = bmdscore.python_dichotomousMA_result()
        self.averageRes.nmodels = len(models)
        self.averageRes.dist_numE = 200
        self.averageRes.models = [model.structs.result for model in models]

        self.bmdsRes = bmdscore.BMDSMA_results()
        self.bmdsRes.BMD = np.full(self.average.nmodels, -9999)
        self.bmdsRes.BMDL = np.full(self.average.nmodels, -9999)
        self.bmdsRes.BMDU = np.full(self.average.nmodels, -9999)
        self.bmdsRes.ebUpper = np.full(self.analysis.n, -9999)
        self.bmdsRes.ebLower = np.full(self.analysis.n, -9999)

    def execute(self) -> "DichotomousModelAverageResult":
        bmdscore.pythonBMDSDichoMA(self.average, self.analysis, self.averageRes, self.bmdsRes)

    def __str__(self):
        return dedent("TODO")


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
    def from_cpp(cls, analysis: DichotomousModelAverage, model_results) -> Self:
        # only keep positive finite values
        arr = np.array(analysis.averageRes.bmd_dist).reshape(2, analysis.averageRes.dist_numE).T
        arr = arr[np.isfinite(arr[:, 0])]
        arr = arr[arr[:, 0] > 0]

        # calculate dr_y for model averaging
        priors = np.array(analysis.average.modelPriors)
        posteriors = np.array(analysis.averageRes.post_probs)
        values = np.array([result.plotting.dr_y for result in model_results])
        dr_x = model_results[0].plotting.dr_x
        dr_y = values.T.dot(posteriors)
        bmds = [analysis.bmdsRes.BMDL_MA, analysis.bmdsRes.BMD_MA, analysis.bmdsRes.BMDU_MA]
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

    def dict(self, **kw) -> dict:
        d = super().dict(**kw)
        return NumpyFloatArray.listify(d)

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
