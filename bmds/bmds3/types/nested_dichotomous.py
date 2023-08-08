from enum import IntEnum
from textwrap import dedent
from typing import NamedTuple, Self

import numpy as np
from pydantic import BaseModel, Field

from ... import bmdscore, constants
from .common import NumpyFloatArray, clean_array


class RiskType(IntEnum):
    AddedRisk = 0
    ExtraRisk = 1


class LitterSpecificCovariate(IntEnum):
    ControlGroupMean = 0
    OverallMean = 1


class IntralitterCorrelation(IntEnum):
    Zero = 0
    Estimate = 1


class Background(IntEnum):
    Zero = 0
    Estimate = 1


_bmr_text_map = {
    RiskType.ExtraRisk: "{:.0%} extra risk",
    RiskType.AddedRisk: "{:.0%} added risk",
}


class NestedDichotomousModelSettings(BaseModel):
    bmr_type: RiskType = RiskType.ExtraRisk
    bmr: float = Field(default=0.1, gt=0)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    litter_specific_covariate: LitterSpecificCovariate = LitterSpecificCovariate.ControlGroupMean
    intralitter_correlation: IntralitterCorrelation = IntralitterCorrelation.Estimate
    background: Background = Background.Estimate
    restricted: bool = True
    bootstrap_iterations: int = Field(default=1000, gt=10, lt=10000)
    bootstrap_seed: int = 0

    def bmr_text(self) -> str:
        return _bmr_text_map[self.bmr_type].format(self.bmr)

    @property
    def confidence_level(self) -> float:
        return 1 - self.alpha


class NestedDichotomousAnalysis(NamedTuple):
    analysis: bmdscore.python_nested_analysis
    result: bmdscore.python_nested_result

    @classmethod
    def blank(cls):
        return cls(bmdscore.python_nested_analysis(), bmdscore.python_nested_result())

    def execute(self):
        bmdscore.pythonBMDSNested(self.analysis, self.result)

    def __str__(self):
        return dedent(
            f"""
            Analysis:
            {self.analysis}

            Result:
            {self.result}
            """
        )


class BmdResult(BaseModel):
    aic: float
    bic_equiv: float
    bmd: float
    bmdl: float
    bmdu: float
    bounded: list[bool]
    chi_squared: float
    lower_ci: list[float]
    std_err: list[float]
    upper_ci: list[float]

    @classmethod
    def from_model(cls, data: bmdscore.BMDS_results) -> Self:
        return cls(
            aic=data.AIC,
            bic_equiv=data.BIC_equiv,
            bmd=data.BMD,
            bmdl=data.BMDL,
            bmdu=data.BMDU,
            bounded=data.bounded,
            chi_squared=data.chisq,
            lower_ci=data.lowerConf,
            std_err=data.stdErr,
            upper_ci=data.upperConf,
        )


class BootstrapResult(BaseModel):
    n_runs: int
    p_value: list[float]
    p50: list[float]
    p90: list[float]
    p95: list[float]
    p99: list[float]

    @classmethod
    def from_model(cls, data: bmdscore.nestedBootstrap) -> Self:
        return cls(
            n_runs=data.numRuns,
            p_value=data.pVal,
            p50=data.perc50,
            p90=data.perc90,
            p95=data.perc95,
            p99=data.perc99,
        )


class ReducedResult(BaseModel):
    dose: list[float]
    lowerConf: list[float]
    numRows: int
    propAffect: list[float]
    upperConf: list[float]

    @classmethod
    def from_model(cls, data: bmdscore.nestedReducedData) -> Self:
        return cls(
            dose=data.dose,
            lowerConf=data.lowerConf,
            numRows=data.numRows,
            propAffect=data.propAffect,
            upperConf=data.upperConf,
        )


class LitterResult(BaseModel):
    lsc: list[float]
    sr: list[float]
    dose: list[float]
    estimated_probabilities: list[float]
    expected: list[float]
    litter_size: list[float]
    nrows: int
    observed: list[int]

    @classmethod
    def from_model(cls, data: bmdscore.nestedLitterData) -> Self:
        return cls(
            lsc=data.LSC,
            sr=data.SR,
            dose=data.dose,
            estimated_probabilities=data.estProb,
            expected=data.expected,
            litter_size=data.litterSize,
            nrows=data.numRows,
            observed=data.observed,
        )


class Plotting(BaseModel):
    dr_x: NumpyFloatArray
    dr_y: NumpyFloatArray
    bmdl_y: float
    bmd_y: float
    bmdu_y: float

    @classmethod
    def from_model(cls, model, params) -> Self:
        summary = model.structs.result.bmdsRes
        xs = np.array([summary.BMDL, summary.BMD, summary.BMDU])
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

    def dict(self, **kw) -> dict:
        d = super().dict(**kw)
        return NumpyFloatArray.listify(d)


class NestedDichotomousResult(BaseModel):
    ll: float
    scaled_residuals: list[float]
    bmd: BmdResult
    bootstrap: BootstrapResult
    combined_pvalue: float
    cov: list[float]
    df: float
    fixed_lsc: float
    litter: LitterResult
    max: float
    # model: nested_model  # TODO remove?
    nparms: int
    obs_chi_sq: float
    parms: list[float]
    reduced: ReducedResult
    plotting: Plotting
    has_completed: bool = False

    @classmethod
    def from_model(cls, model) -> Self:
        result: bmdscore.python_nested_result = model.structs.result
        return cls(
            ll=result.LL,
            scaled_residuals=result.SRs,
            bmd=BmdResult.from_model(result.bmdsRes),
            bootstrap=BootstrapResult.from_model(result.boot),
            combined_pvalue=result.combPVal,
            cov=result.cov,
            df=result.df,
            fixed_lsc=result.fixedLSC,
            litter=LitterResult.from_model(result.litter),
            max=result.max,
            nparms=result.nparms,
            obs_chi_sq=result.obsChiSq,
            parms=result.parms,
            reduced=ReducedResult.from_model(result.reduced),
            plotting=Plotting.from_model(model, result.parms),
            has_completed=result.bmdsRes.validResult,
        )
