from enum import IntEnum
from textwrap import dedent
from typing import NamedTuple, Self

from pydantic import BaseModel, confloat, conint
from ...utils import multi_lstrip, pretty_table

from bmds import bmdscore
from ...datasets import NestedDichotomousDataset


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


class NestedDichotomousAnalysis(BaseModel):
    """
    Purpose - Contains all of the information for a nested dichotomous analysis.
    """
    # model :
    # restricted :
    # doses :
    # litterSize :
    # incidence :
    # lsc :
    # LSC_type :
    # ILC_type :
    # BMD_type :
    # background :
    # BMR :
    # alpha :
    # iterations :
    # seed :

    def to_cpp(Self):
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
        nested_result.bmdsRes = bmdscore.BMDS_results()
        nested_result.boot = bmdscore.nestedBootstrap()
        nested_result.litter = bmdscore.nestedLitterData()
        nested_result.reduced = bmdscore.nestedReducedData()

        return NestedDichotomousAnalysisCPPStructs(analysis, nested_result)

class NestedDichotomousAnalysisCPPStructs(NamedTuple):
    analysis: bmdscore.python_nested_analysis
    nested_result: bmdscore.python_nested_result

    def execute(self):
        bmdscore.pythonBMDSNested(self.analysis, self.nested_result)

    def __str__(self):
        return dedent(
            f"""
            Analysis:
            {self.analysis}

            Result:
            {self.nested_result}
            """
        )
class NestedDichotomousResult(BaseModel):
    # model :
    # nparms :
    # parms :
    # cov :
    # max :
    # df :
    # fixedLSC :
    # LL :
    # obsChiSq :
    # combPVal :
    # SRs :
    # bmdsRes :
    # litter :
    # boot :
    # reduced :

    @classmethod
    def from_model(cls, model) -> Self:
        result = model.structs.result
        summary = result.bmdsRes
        # fit = NestedDichotomousModelResult.from_model(model)
        # gof = NestedDichotomousPgofResult.from_model(model)
        # parameters = NestedDichotomousParameters.from_model(model)
        # deviance = NestedDichotomousAnalysisOfDeviance.from_model(model)
        # plotting = NestedDichotomousPlotting.from_model(model, parameters.values)
        return cls(
            # bmdl=summary.BMDL,
            bmd=summary.BMD,
            # bmdu=summary.BMDU,
            # has_completed=summary.validResult,
            # fit=fit,
            # gof=gof,
            # parameters=parameters,
            # deviance=deviance,
            # plotting=plotting,
        )

    def text(self, dataset: NestedDichotomousDataset, settings: NestedDichotomousModelSettings) -> str:
        return multi_lstrip(
            f"""
        Summary:
        {self.tbl()}
        """
        )

    def tbl(self) -> str:
        data = [
            ["BMD", self.bmd],
            # ["BMDL", self.bmdl],
            # ["BMDU", self.bmdu],
            # ["AIC", self.fit.aic],
            # ["Log Likelihood", self.fit.loglikelihood],
            # ["P-Value", self.gof.p_value],
            # ["Overall DOF", self.gof.df],
            # ["Chi²", self.fit.chisq],
        ]
        return pretty_table(data, "")

    # def update_record(self, d: dict) -> None:
    #     """Update data record for a tabular-friendly export"""
    #     d.update(
    #         bmdl=self.bmdl,
    #         bmd=self.bmd,
    #         bmdu=self.bmdu,
    #         aic=self.fit.aic,
    #         loglikelihood=self.fit.loglikelihood,
    #         p_value=self.gof.p_value,
    #         overall_dof=self.gof.df,
    #         bic_equiv=self.fit.bic_equiv,
    #         chi_squared=self.fit.chisq,
    #         residual_of_interest=self.gof.roi,
    #         residual_at_lowest_dose=self.gof.residual[0],
    #     )
