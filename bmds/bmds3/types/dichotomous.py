import ctypes
from enum import IntEnum
from typing import Annotated, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ...constants import BOOL_ICON
from ...datasets import DichotomousDataset
from ...utils import multi_lstrip, pretty_table
from .. import constants
from .common import NumpyFloatArray, NumpyIntArray, clean_array, residual_of_interest
from .priors import ModelPriors, PriorClass, PriorType
from .structs import (
    BmdsResultsStruct,
    DichotomousAnalysisStruct,
    DichotomousAodStruct,
    DichotomousModelResultStruct,
    DichotomousPgofResultStruct,
    DichotomousStructs,
)


class DichotomousRiskType(IntEnum):
    AddedRisk = 0
    ExtraRisk = 1


_bmr_text_map = {
    DichotomousRiskType.ExtraRisk: "{:.0%} Extra Risk",
    DichotomousRiskType.AddedRisk: "{:.0%} Added Risk",
}


class DichotomousModelSettings(BaseModel):
    bmr: Annotated[float, Field(gt=0)] = 0.1
    alpha: Annotated[float, Field(gt=0, lt=1)] = 0.05
    bmr_type: DichotomousRiskType = DichotomousRiskType.ExtraRisk
    degree: Annotated[int, Field(ge=0, le=8)] = 0  # multistage only
    samples: Annotated[int, Field(ge=10, le=1000)] = 100
    burnin: Annotated[int, Field(ge=5, le=1000)] = 20
    priors: PriorClass | ModelPriors | None = None  # if None; default used

    @property
    def bmr_text(self) -> str:
        return _bmr_text_map[self.bmr_type].format(self.bmr)

    @property
    def confidence_level(self) -> float:
        return 1 - self.alpha

    def tbl(self, show_degree: bool = True) -> str:
        data = [
            ["BMR", self.bmr_text],
            ["Confidence Level", self.confidence_level],
            ["Modeling approach", self.priors.prior_class.name],
        ]

        if show_degree:
            data.append(["Degree", self.degree])

        if self.priors.is_bayesian:
            data.extend((["Samples", self.samples], ["Burn-in", self.burnin]))

        return pretty_table(data, "")

    def docx_table_data(self) -> list:
        return [
            ["Setting", "Value"],
            ["BMR", self.bmr_text],
            ["Confidence Level", self.confidence_level],
            ["Maximum Multistage Degree", self.degree],
        ]

    def update_record(self, d: dict) -> None:
        """Update data record for a tabular-friendly export"""
        d.update(
            bmr=self.bmr_text,
            confidence_level=self.confidence_level,
            degree=self.degree,
            model_class=self.priors.prior_class.name,
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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def num_params(self) -> int:
        return (
            self.degree + 1
            if self.model == constants.DichotomousModelChoices.d_multistage.value
            else self.model.num_params
        )

    def _priors_array(self) -> np.ndarray:
        degree = (
            self.degree if self.model.id == constants.DichotomousModelIds.d_multistage else None
        )
        return self.priors.to_c(degree=degree)

    def to_c(self) -> DichotomousStructs:
        return DichotomousStructs(
            analysis=DichotomousAnalysisStruct(
                model=ctypes.c_int(self.model.id),
                n=ctypes.c_int(self.dataset.num_dose_groups),
                Y=self.dataset.incidences,
                doses=self.dataset.doses,
                n_group=self.dataset.ns,
                prior=self._priors_array(),
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
    bmds_model_df: float = Field(alias="model_df")
    total_df: float
    bmd_dist: NumpyFloatArray

    @classmethod
    def from_model(cls, model) -> Self:
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


class DichotomousPgofResult(BaseModel):
    expected: list[float]
    residual: list[float]
    eb_lower: list[float]
    eb_upper: list[float]
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
        headers = "Dose|Size|Observed|Expected|Est Prob|Scaled Residual".split("|")
        data = []
        for dg in range(dataset.num_dose_groups):
            data.append(
                [
                    dataset.doses[dg],
                    dataset.ns[dg],
                    dataset.incidences[dg],
                    self.expected[dg],
                    self.expected[dg] / dataset.ns[dg],
                    self.residual[dg],
                ]
            )
        return pretty_table(data, headers)


class DichotomousParameters(BaseModel):
    names: list[str]
    values: NumpyFloatArray
    se: NumpyFloatArray
    lower_ci: NumpyFloatArray
    upper_ci: NumpyFloatArray
    bounded: NumpyFloatArray
    cov: NumpyFloatArray
    prior_type: NumpyIntArray
    prior_initial_value: NumpyFloatArray
    prior_stdev: NumpyFloatArray
    prior_min_value: NumpyFloatArray
    prior_max_value: NumpyFloatArray

    @classmethod
    def get_priors(cls, model) -> np.ndarray:
        priors_list = model.get_priors_list()
        return np.array(priors_list, dtype=np.float64).T

    @classmethod
    def from_model(cls, model) -> Self:
        result = model.structs.result
        summary = model.structs.summary
        param_names = model.get_param_names()
        priors = cls.get_priors(model)
        return cls(
            names=param_names,
            values=result.np_parms,
            bounded=summary.np_bounded,
            se=summary.np_stdErr,
            lower_ci=summary.np_lowerConf,
            upper_ci=summary.np_upperConf,
            cov=result.np_cov.reshape(result.nparms, result.nparms),
            prior_type=priors[0],
            prior_initial_value=priors[1],
            prior_stdev=priors[2],
            prior_min_value=priors[3],
            prior_max_value=priors[4],
        )

    def tbl(self) -> str:
        headers = "Variable|Estimate|Bounded|Std Error|Lower CI|Upper CI".split("|")
        data = []
        for name, value, bounded, se, lower_ci, upper_ci in zip(
            self.names,
            self.values,
            self.bounded,
            self.se,
            self.lower_ci,
            self.upper_ci,
            strict=True,
        ):
            data.append(
                (
                    name,
                    value,
                    BOOL_ICON[bounded],
                    "NA" if bounded else f"{se:g}",
                    "NA" if bounded else f"{lower_ci:g}",
                    "NA" if bounded else f"{upper_ci:g}",
                )
            )
        return pretty_table(data, headers)

    def rows(self, extras: dict) -> list[dict]:
        rows = []
        for i in range(len(self.names)):
            rows.append(
                {
                    **extras,
                    **dict(
                        name=self.names[i],
                        value=self.values[i],
                        se=self.se[i],
                        lower_ci=self.lower_ci[i],
                        upper_ci=self.upper_ci[i],
                        bounded=bool(self.bounded[i]),
                        initial_distribution=PriorType(self.prior_type[i]).name,
                        initial_value=self.prior_initial_value[i],
                        initial_stdev=self.prior_stdev[i],
                        initial_min_value=self.prior_min_value[i],
                        initial_max_value=self.prior_max_value[i],
                    ),
                }
            )
        return rows


class DichotomousAnalysisOfDeviance(BaseModel):
    names: list[str]
    ll: list[float]
    params: list[int]
    deviance: list[float]
    df: list[int]
    p_value: list[float]

    @classmethod
    def from_model(cls, model) -> Self:
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
        headers = "Model|Log Likelihood|# Params|Deviance|Test DOF|P-Value".split("|")
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
                    f"{self.p_value[i]:g}" if i > 0 else "-",
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
    def from_model(cls, model, params) -> Self:
        summary = model.structs.summary
        xs = np.array([summary.bmdl, summary.bmd, summary.bmdu])
        dr_x = model.dataset.dose_linspace
        dr_y = clean_array(model.dr_curve(dr_x, params))
        critical_ys = clean_array(model.dr_curve(xs, params))
        critical_ys[critical_ys <= 0] = constants.BMDS_BLANK_VALUE
        return cls(
            dr_x=dr_x,
            dr_y=dr_y,
            bmdl_y=critical_ys[0],
            bmd_y=critical_ys[1],
            bmdu_y=critical_ys[2],
        )


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
    def from_model(cls, model) -> Self:
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

    def text(self, dataset: DichotomousDataset, settings: DichotomousModelSettings) -> str:
        return multi_lstrip(
            f"""
        Summary:
        {self.tbl()}

        Model Parameters:
        {self.parameters.tbl()}

        Goodness of Fit:
        {self.gof.tbl(dataset)}

        Analysis of Deviance:
        {self.deviance.tbl()}
        """
        )

    def tbl(self) -> str:
        data = [
            ["BMD", self.bmd],
            ["BMDL", self.bmdl],
            ["BMDU", self.bmdu],
            ["AIC", self.fit.aic],
            ["Log Likelihood", self.fit.loglikelihood],
            ["P-Value", self.gof.p_value],
            ["Overall DOF", self.gof.df],
            ["Chi²", self.fit.chisq],
        ]
        return pretty_table(data, "")

    def update_record(self, d: dict) -> None:
        """Update data record for a tabular-friendly export"""
        d.update(
            bmdl=self.bmdl,
            bmd=self.bmd,
            bmdu=self.bmdu,
            aic=self.fit.aic,
            loglikelihood=self.fit.loglikelihood,
            p_value=self.gof.p_value,
            overall_dof=self.gof.df,
            bic_equiv=self.fit.bic_equiv,
            chi_squared=self.fit.chisq,
            residual_of_interest=self.gof.roi,
            residual_at_lowest_dose=self.gof.residual[0],
        )
