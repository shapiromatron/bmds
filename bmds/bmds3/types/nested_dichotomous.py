from enum import IntEnum
from textwrap import dedent
from typing import NamedTuple, Self

from pydantic import BaseModel, confloat, conint

from bmds import bmdscore

from ...datasets import NestedDichotomousDataset
from ...utils import multi_lstrip, pretty_table


class NestedDichotomousRiskType(IntEnum):
    AddedRisk = 0
    ExtraRisk = 1


class NestedDichotomousLSCType(IntEnum):
    OverallMean = 0
    ControlGroupMean = 1


class NestedDichotomousBackgroundType(IntEnum):
    Zero = 0
    Estimated = 1


_bmr_text_map = {
    NestedDichotomousRiskType.ExtraRisk: "{:.0%} extra risk",
    NestedDichotomousRiskType.AddedRisk: "{:.0%} added risk",
}


class NestedDichotomousModelSettings(BaseModel):
    bmr: confloat(gt=0) = 0.1
    alpha: confloat(gt=0, lt=1) = 0.05
    bmr_type: NestedDichotomousRiskType = NestedDichotomousRiskType.ExtraRisk
    litter_specific_covariate: NestedDichotomousLSCType = NestedDichotomousLSCType.ControlGroupMean
    background: NestedDichotomousBackgroundType = NestedDichotomousBackgroundType.Estimated
    bootstrap_iterations: conint(gt=0) = 1
    bootstrap_seed: conint(gt=0) = 0

    def bmr_text(self) -> str:
        return _bmr_text_map[self.bmr_type].format(self.bmr)

    @property
    def confidence_level(self) -> float:
        return 1 - self.alpha

    def tbl(self, show_degree: bool = True) -> str:
        data = [
            ["BMR", self.bmr_text],
            ["Confidence Level", self.confidence_level],
        ]
        return pretty_table(data, "")


class NestedDichotomousAnalysis(BaseModel):
    """
    Contains all of the information for a nested dichotomous analysis.
    """

    def to_cpp(self):
        analysis = bmdscore.python_nested_analysis()
        analysis.model = bmdscore.nested_model.nlogistic
        analysis.restricted = True
        analysis.doses = []  # dont need to still be np array right?
        analysis.litterSize = []
        analysis.incidence = []
        analysis.lsc = []
        analysis.LSC_type = 1
        analysis.ILC_type = 1
        analysis.BMD_type = 1
        analysis.background = 1
        analysis.BMR = 0.1
        analysis.alpha = 0.05
        analysis.iterations = 1000
        analysis.seed = -9999
        nested_result = bmdscore.python_nested_result()
        return NestedDichotomousAnalysisCPPStructs(analysis, nested_result)


class NestedDichotomousAnalysisCPPStructs(NamedTuple):
    analysis: bmdscore.python_nested_analysis
    result: bmdscore.python_nested_result

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
    valid_result: bool

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
            valid_result=data.validResult,
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


class NestedDichotomousResult(BaseModel):
    ll: float
    srs: list[float]  # TODO rename?
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

    @classmethod
    def from_model(cls, model) -> Self:
        result: bmdscore.python_nested_result = model.structs.result
        return cls(
            ll=result.LL,
            srs=result.SRs,
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
        )
