from enum import IntEnum
from textwrap import dedent
from typing import NamedTuple, Self

import numpy as np
from pydantic import BaseModel, Field

from ... import bmdscore, constants
from ...datasets import NestedDichotomousDataset
from ...utils import camel_to_title, multi_lstrip, pretty_table
from .common import NumpyFloatArray, clean_array, residual_of_interest


class RiskType(IntEnum):
    AddedRisk = 0
    ExtraRisk = 1


class LitterSpecificCovariate(IntEnum):
    ControlGroupMean = 0
    OverallMean = 1

    @property
    def text(self) -> str:
        return "lsc+" if self.OverallMean else "lsc-"


class IntralitterCorrelation(IntEnum):
    Zero = 0
    Estimate = 1

    @property
    def text(self) -> str:
        return "ilc+" if self.Estimate else "ilc-"


class Background(IntEnum):
    Zero = 0
    Estimate = 1


_bmr_text_map = {
    RiskType.ExtraRisk: "{:.0%} Extra Risk",
    RiskType.AddedRisk: "{:.0%} Added Risk",
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

    @property
    def bmr_text(self) -> str:
        return _bmr_text_map[self.bmr_type].format(self.bmr)

    @property
    def confidence_level(self) -> float:
        return 1 - self.alpha

    @property
    def restriction_text(self) -> str:
        return "Restricted" if self.restricted else "Unrestricted"

    def _tbl_rows(self) -> list:
        return [
            ["BMR", self.bmr_text],
            ["Confidence Level", self.confidence_level],
            ["Litter Specific Covariate", camel_to_title(self.litter_specific_covariate.name)],
            ["Intralitter Correlation", self.intralitter_correlation.name],
            ["Background", self.background.name],
            ["Model Restriction", self.restriction_text],
            ["Bootstrap Iterations", self.bootstrap_iterations],
            ["Bootstrap Key", self.bootstrap_seed],
        ]

    def tbl(self, degree_required: bool = False) -> str:
        return pretty_table(self._tbl_rows(), "")

    def docx_table_data(self) -> list:
        rows = self._tbl_rows()
        rows.insert(0, ["Setting", "Value"])
        return rows

    def update_record(self, d: dict) -> None:
        """Update data record for a tabular-friendly export"""
        d.update(
            bmr=self.bmr_text,
        )


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


class BootstrapRuns(BaseModel):
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

    def tbl(self) -> str:
        col1 = "1 2 3 Combined".split()
        data = list(zip(col1, self.p_value, self.p50, self.p90, self.p95, self.p99, strict=True))
        return pretty_table(data, headers="Run P-Value 50th 90th 95th 99th".split())


class ReducedResult(BaseModel):
    dose: list[float]
    prop_affected: list[float]
    lower_ci: list[float]
    upper_ci: list[float]

    @classmethod
    def from_model(cls, data: bmdscore.nestedReducedData) -> Self:
        return cls(
            dose=data.dose,
            prop_affected=data.propAffect,
            lower_ci=data.lowerConf,
            upper_ci=data.upperConf,
        )


class LitterResult(BaseModel):
    lsc: list[float]
    scaled_residuals: list[float]
    dose: list[float]
    estimated_probabilities: list[float]
    expected: list[float]
    litter_size: list[float]
    observed: list[int]
    roi: float

    @classmethod
    def from_model(cls, data: bmdscore.nestedLitterData, bmd: float) -> Self:
        return cls(
            lsc=data.LSC,
            scaled_residuals=data.SR,
            dose=data.dose,
            estimated_probabilities=data.estProb,
            expected=data.expected,
            litter_size=data.litterSize,
            observed=data.observed,
            roi=residual_of_interest(bmd, data.dose, data.SR),
        )

    def tbl(self) -> str:
        headers = (
            "Dose|Lit. Spec. Cov.|Est. Prob.|Litter Size|Expected|Observed|Scaled Residual".split(
                "|"
            )
        )
        data = list(
            zip(
                self.dose,
                self.lsc,
                self.estimated_probabilities,
                self.expected,
                self.observed,
                self.litter_size,
                self.observed,
                strict=True,
            )
        )
        return pretty_table(data, headers)


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
    summary: BmdResult
    bootstrap: BootstrapRuns
    combined_pvalue: float
    cov: list[float]
    dof: float
    fixed_lsc: float
    litter: LitterResult
    max: float
    obs_chi_sq: float
    parameter_names: list[str]
    parameters: list[float]
    reduced: ReducedResult
    plotting: Plotting
    has_completed: bool = False

    @classmethod
    def from_model(cls, model) -> Self:
        result: bmdscore.python_nested_result = model.structs.result
        return cls(
            ll=result.LL,
            scaled_residuals=result.SRs,
            summary=BmdResult.from_model(result.bmdsRes),
            bootstrap=BootstrapRuns.from_model(result.boot),
            combined_pvalue=result.combPVal,
            cov=result.cov,
            dof=result.df,
            fixed_lsc=result.fixedLSC,
            litter=LitterResult.from_model(result.litter, result.bmdsRes.BMD),
            max=result.max,
            obs_chi_sq=result.obsChiSq,
            parameter_names=model.get_param_names(),
            parameters=result.parms,
            reduced=ReducedResult.from_model(result.reduced),
            plotting=Plotting.from_model(model, result.parms),
            has_completed=result.bmdsRes.validResult,
        )

    def text(
        self, dataset: NestedDichotomousDataset, settings: NestedDichotomousModelSettings
    ) -> str:
        return multi_lstrip(
            f"""
        Summary:
        {self.tbl()}

        Model Parameters:
        {self.parameters_tbl()}

        Bootstrap Runs:
        {self.bootstrap.tbl()}

        Scaled Residuals:
        {self.scaled_residuals_tbl()}

        Litter Data:
        {self.litter.tbl()}
        """
        )

    def tbl(self) -> str:
        data = [
            ["BMD", self.summary.bmd],
            ["BMDL", self.summary.bmdl],
            ["BMDU", self.summary.bmdu],
            ["AIC", self.summary.aic],
            ["P-value", self.combined_pvalue],
            ["D.O.F", self.dof],
            ["Chi²", self.summary.chi_squared],
            ["Log-likelihood", self.ll],
        ]
        return pretty_table(data, "")

    def scaled_residuals_tbl(self) -> str:
        col1 = [
            "Minimum scaled residual for dose group nearest BMD",
            "Minimum ABS(scaled residual) for dose group nearest BMD",
            "Average scaled residual for dose group nearest BMD",
            "Average ABS(scaled residual) for dose group nearest BMD",
            "Maximum scaled residual for dose group nearest BMD",
            "Maximum ABS(scaled residual) for dose group nearest BMD",
        ]
        data = list(zip(col1, self.scaled_residuals, strict=True))
        return pretty_table(data, "")

    def parameters_tbl(self) -> str:
        data = list(zip(self.parameter_names, self.parameters, strict=True))
        return pretty_table(data, "")

    def update_record(self, d: dict) -> None:
        """Update data record for a tabular-friendly export"""
        d.update(
            bmd=self.summary.bmd,
            bmdl=self.summary.bmdl,
            bmdu=self.summary.bmdu,
        )

    def get_parameter(self, parameter: str) -> float:
        """Get parameter value by name"""
        match parameter:
            case "bmd":
                return self.summary.bmd
            case "bmdl":
                return self.summary.bmdl
            case "bmdu":
                return self.summary.bmdu
            case "aic":
                return self.summary.aic
            case "dof":
                return self.dof
            case "pvalue":
                return self.combined_pvalue
            case "roi":
                return self.litter.roi
            case "roi_control":
                return self.litter.scaled_residuals[0]  # TODO - ?
            case _:
                raise ValueError(f"Unknown parameter: {parameter}")
