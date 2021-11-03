from enum import IntEnum

from pydantic import BaseModel, confloat, conint


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
