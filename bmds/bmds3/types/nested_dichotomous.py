import ctypes
from enum import IntEnum
from typing import Dict, List, Union

import numpy as np
from pydantic import BaseModel, confloat, conint

class NestedDichotomousRiskType(IntEnum):
    AddedRisk = 0
    ExtraRisk = 1



class NestedDichotomousLSCType(IntEnum):
    OverallMean = 0
    ControlGroupMean = 1

class NestedDichotomousBackgroundType(IntEnum):
    Zero=0
    Estimated=1

class NestedDichotomousModelSettings(BaseModel):
    bmr: confloat(gt=0) = 0.1
    alpha: confloat(gt=0, lt=1) = 0.05
    bmr_type: NestedDichotomousRiskType = NestedDichotomousRiskType.ExtraRisk
    litter_specific_covariate: NestedDichotomousLSCType = NestedDichotomousLSCType.ControlGroupMean
    background : NestedDichotomousBackgroundType=NestedDichotomousBackgroundType.Estimated
    bootstrap_iterations : conint(gt=0) = 1
    bootstrap_seed : conint(gt=0) =0



    def bmr_text(self) -> str:
        return _bmr_text_map[self.bmr_type].format(self.bmr)

    def text(self) -> str:
        return multi_lstrip(
            f"""\
        BMR Type: {self.bmr_type.name}
        BMR: {self.bmr}
        Alpha: {self.alpha}
        litter_specific_covariate: {self.litter_specific_covariate}
        background: {self.background}
        bootstrap_iterations: {self.bootstrap_iterations}
        bootstrap_seed: {self.bootstrap_seed}
        """
        )

    def update_record(self, d: dict) -> None:
        """Update data record for a tabular-friendly export"""
        d.update(
            bmr=self.bmr,
            bmr_type=self.bmr_type.name,
            alpha=self.alpha,
            litter_specific_covariate=self.degree,
            background=self.priors.background,
            bootstrap_iterations=self.bootstrap_iterations,
            bootstrap_seed = self.bootstrap_seed
        )
