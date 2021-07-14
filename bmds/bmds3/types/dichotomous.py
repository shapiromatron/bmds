import ctypes
from enum import IntEnum
from typing import Dict, List, Union

import numpy as np
from pydantic import BaseModel, confloat, conint

from bmds.bmds3.constants import DichotomousModelChoices, ModelPriors

from ...constants import BOOL_ICON
from ...datasets import DichotomousDataset
from ...utils import multi_lstrip
from .. import constants
from .common import NumpyFloatArray, list_t_c, pretty_table, residual_of_interest
from .priors import PriorClass
from .structs import (
    BmdsResultsStruct,
    DichotomousAnalysisStruct,
    DichotomousAodStruct,
    DichotomousModelResultStruct,
    DichotomousPgofResultStruct,
    DichotomousStructs,
)


class DichotomousRiskType(IntEnum):
    ExtraRisk = 1
    AddedRisk = 2


class DichotomousModelSettings(BaseModel):
    bmr: confloat(gt=0) = 0.1
    alpha: confloat(gt=0, lt=1) = 0.05
    bmr_type: DichotomousRiskType = DichotomousRiskType.ExtraRisk
    degree: conint(ge=0, le=8) = 0  # multistage only
    samples: conint(ge=10, le=1000) = 100
    burnin: conint(ge=5, le=1000) = 20
    priors: Union[None, PriorClass, ModelPriors]  # if None; default used

    def text(self) -> str:
        return multi_lstrip(
            f"""\
        BMR Type: {self.bmr_type.name}
        BMR: {self.bmr}
        Alpha: {self.alpha}
        Degree: {self.degree}
        Samples: {self.samples}
        Burn-in: {self.burnin}

        Priors:
        {self.priors}"""
        )


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
            summary=BmdsResultsStruct(num_params=self.num_params),
            aod=DichotomousAodStruct(),
        )


class DichotomousModelResult(BaseModel):
    loglikelihood: float
    aic: float
    bic_equiv: float
    chisq: float
    model_df: float
    total_df: float
    bmd_dist: NumpyFloatArray

    @classmethod
    def from_model(cls, model) -> "DichotomousModelResult":
        result = model.structs.result
        summary = model.structs.summary
        # reshape; get rid of 0 and inf; must be JSON serializable
        arr = result.np_bmd_dist.reshape(2, result.dist_numE)
        arr = arr[:, np.isfinite(arr[0, :])]
        arr = arr[:, arr[0, :] > 0]

        return DichotomousModelResult(
            loglikelihood=result.max,
            aic=summary.aic,
            bic_equiv=summary.BIC_equiv,
            chisq=summary.chisq,
            model_df=result.model_df,
            total_df=result.total_df,
            bmd_dist=arr,
        )

    def dict(self, **kw) -> Dict:
        d = super().dict(**kw)
        return NumpyFloatArray.listify(d)


class DichotomousPgofResult(BaseModel):
    expected: List[float]
    residual: List[float]
    eb_lower: List[float]
    eb_upper: List[float]
    test_statistic: float
    p_value: float
    roi: float
    df: float

    @classmethod
    def from_model(cls, model):
        gof = model.structs.gof
        roi = residual_of_interest(model.structs.summary.bmd, model.dataset.doses, gof.residual)
        return cls(
            expected=gof.np_expected.tolist(),
            residual=gof.np_residual.tolist(),
            eb_lower=gof.np_ebLower.tolist(),
            eb_upper=gof.np_ebUpper.tolist(),
            test_statistic=gof.test_statistic,
            p_value=gof.p_value,
            roi=roi,
            df=gof.df,
        )

    def tbl(self, dataset: DichotomousDataset) -> str:
        headers = "Dose|EstProb|Expected|Observed|Size|ScaledRes".split("|")
        data = []
        for dg in range(dataset.num_dose_groups):
            data.append(
                [
                    dataset.doses[dg],
                    self.expected[dg] / dataset.ns[dg],
                    self.expected[dg],
                    dataset.incidences[dg],
                    dataset.ns[dg],
                    self.residual[dg],
                ]
            )
        return pretty_table(data, headers)


class DichotomousParameters(BaseModel):
    names: List[str]
    values: NumpyFloatArray
    se: NumpyFloatArray
    lower_ci: NumpyFloatArray
    upper_ci: NumpyFloatArray
    bounded: NumpyFloatArray
    cov: NumpyFloatArray

    @classmethod
    def from_model(cls, model) -> "DichotomousParameters":
        result = model.structs.result
        summary = model.structs.summary
        return cls(
            names=model.get_param_names(),
            values=result.np_parms,
            bounded=summary.np_bounded,
            se=summary.np_stdErr,
            lower_ci=summary.np_lowerConf,
            upper_ci=summary.np_upperConf,
            cov=result.np_cov.reshape(result.nparms, result.nparms),
        )

    def dict(self, **kw) -> Dict:
        d = super().dict(**kw)
        return NumpyFloatArray.listify(d)

    def tbl(self) -> str:
        headers = "parm|estimate|bounded".split("|")
        data = []
        for name, value, bounded in zip(self.names, self.values, self.bounded):
            data.append([name, value, BOOL_ICON[bounded]])
        return pretty_table(data, headers)


class DichotomousAnalysisOfDeviance(BaseModel):
    names: List[str]
    ll: List[float]
    params: List[int]
    deviance: List[float]
    df: List[int]
    p_value: List[float]

    @classmethod
    def from_model(cls, model) -> "DichotomousAnalysisOfDeviance":
        aod = model.structs.aod
        return cls(
            names=["Full model", "Fitted model", "Reduced model"],
            ll=[aod.fullLL, aod.fittedLL, aod.redLL],
            params=[aod.nFull, aod.nFit, aod.nRed],
            deviance=[constants.BMDS_BLANK_VALUE, aod.devFit, aod.devRed],
            df=[constants.BMDS_BLANK_VALUE, aod.dfFit, aod.dfRed],
            p_value=[constants.BMDS_BLANK_VALUE, aod.pvFit, aod.pvRed],
        )

    def tbl(self) -> str:
        headers = "Model|LL|#parms|deviance|test DF|pval".split("|")
        data = []
        for i in range(len(self.names)):
            # manually format columns b/c tabulate won't format if first row text is str
            data.append(
                [
                    self.names[i],
                    self.ll[i],
                    self.params[i],
                    f"{self.deviance[i]:g}" if i > 0 else "-",
                    f"{self.df[i]:g}" if i > 0 else "-",
                    f"{self.p_value[i]:g}" if i > 0 else "N/A",
                ]
            )
        return pretty_table(data, headers)


class DichotomousPlotting(BaseModel):
    dr_x: NumpyFloatArray
    dr_y: NumpyFloatArray
    bmdl_y: float
    bmd_y: float
    bmdu_y: float

    @classmethod
    def from_model(cls, model, params) -> "DichotomousPlotting":
        structs = model.structs
        critical_xs = np.array([structs.summary.bmdl, structs.summary.bmd, structs.summary.bmdu])
        dr_x = model.dataset.dose_linspace
        bad_params = np.isclose(params, constants.BMDS_BLANK_VALUE).any()
        dr_y = dr_x * 0 if bad_params else model.dr_curve(dr_x, params)
        critical_ys = critical_xs * 0 if bad_params else model.dr_curve(critical_xs, params)
        return cls(
            dr_x=dr_x,
            dr_y=dr_y,
            bmdl_y=critical_ys[0] if structs.summary.bmdl > 0 else constants.BMDS_BLANK_VALUE,
            bmd_y=critical_ys[1] if structs.summary.bmd > 0 else constants.BMDS_BLANK_VALUE,
            bmdu_y=critical_ys[2] if structs.summary.bmdu > 0 else constants.BMDS_BLANK_VALUE,
        )

    def dict(self, **kw) -> Dict:
        d = super().dict(**kw)
        return NumpyFloatArray.listify(d)


class DichotomousResult(BaseModel):
    bmdl: float
    bmd: float
    bmdu: float
    has_completed: bool
    fit: DichotomousModelResult
    gof: DichotomousPgofResult
    parameters: DichotomousParameters
    deviance: DichotomousAnalysisOfDeviance
    plotting: DichotomousPlotting

    @classmethod
    def from_model(cls, model) -> "DichotomousResult":
        summary = model.structs.summary
        fit = DichotomousModelResult.from_model(model)
        gof = DichotomousPgofResult.from_model(model)
        parameters = DichotomousParameters.from_model(model)
        deviance = DichotomousAnalysisOfDeviance.from_model(model)
        plotting = DichotomousPlotting.from_model(model, parameters.values)
        return cls(
            bmdl=summary.bmdl,
            bmd=summary.bmd,
            bmdu=summary.bmdu,
            has_completed=summary.validResult,
            fit=fit,
            gof=gof,
            parameters=parameters,
            deviance=deviance,
            plotting=plotting,
        )

    def text(self, dataset: DichotomousDataset) -> str:
        return multi_lstrip(
            f"""
        Summary:
        {self.tbl()}

        Goodness of fit:
        {self.gof.tbl(dataset)}

        Parameters:
        {self.parameters.tbl()}

        Deviances:
        {self.deviance.tbl()}
        """
        )

    def tbl(self) -> str:
        data = [
            ["BMDL", self.bmdl],
            ["BMD", self.bmd],
            ["BMDU", self.bmdu],
            ["AIC", self.fit.aic],
            ["LL", self.fit.loglikelihood],
            ["model_df", self.fit.model_df],
            ["p-value", self.gof.p_value],
            ["DOF", self.gof.df],
            ["Chi²", self.fit.chisq],
        ]
        return pretty_table(data, "")
