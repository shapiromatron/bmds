import ctypes
from typing import List

from .. import types33
from ..types33 import DichotomousModelSettings
from ..constants import DichotomousModel, Prior
from .base import BaseModel, BmdsLibraryManager, InputModelSettings


class Dichotomous(BaseModel):
    model_id: DichotomousModel

    @property
    def num_params(self) -> int:
        return len(self.model_id.params)

    def get_model_settings(self, settings: InputModelSettings) -> DichotomousModelSettings:
        if settings is None:
            settings = DichotomousModelSettings()
        if isinstance(settings, DichotomousModelSettings):
            return settings
        return DichotomousModelSettings.parse_obj(settings)

    def execute(self) -> types33.DichotomousModelResult:
        # setup inputs
        inputs = types33.DichotomousAnalysis(
            model=self.model_id,
            dataset=self.dataset,
            priors=self.default_frequentist_priors,
            BMD_type=self.settings.bmr_type,
            BMR=self.settings.bmr,
            alpha=self.settings.alpha,
            degree=self.settings.degree,
            samples=self.settings.samples,
            burnin=self.settings.burnin,
        )

        # setup outputs
        results = types33.DichotomousModelResult(model=self.model_id, dist_numE=200)
        results_struct = results.to_c()

        dll = BmdsLibraryManager.get_dll(bmds_version="BMDS330", base_name="libDRBMD")

        dll.estimate_sm_laplace_dicho(
            ctypes.pointer(inputs.to_c()), ctypes.pointer(results_struct), True
        )
        results.from_c()

        return results

    # required
    default_frequentist_priors: List[Prior]


class Logistic(Dichotomous):
    model_id = DichotomousModel.d_logistic

    default_frequentist_priors = [
        Prior(type=0, initial_value=-2, stdev=1, min_value=-18, max_value=18),
        Prior(type=0, initial_value=0.1, stdev=1, min_value=1, max_value=10),
    ]
