from enum import IntEnum

from pydantic import BaseModel, confloat, conint

from bmds import bmdscore


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

    def to_cpp(Self):
        analysis = bmdscore.python_nested_analysis()
        analysis.model = bmdscore.nested_model.nlogistic
        analysis.restricted = True
        analysis.doses = [] # dont need to still be np array right?
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

        bmdscore.pythonBMDSNested(analysis, nested_result)
