import ctypes
from typing import List

from ..constants import ContinuousModel, ContinuousModelChoices, Prior
from ..types.continuous import (
    ContinuousAnalysis,
    ContinuousModelResult,
    ContinuousModelSettings,
)
from .base import BaseModel, BmdsLibraryManager, InputModelSettings


class Continuous(BaseModel):
    # required settings
    model: ContinuousModel

    def get_model_settings(self, settings: InputModelSettings) -> ContinuousModelSettings:
        if settings is None:
            model = ContinuousModelSettings()
        elif isinstance(settings, ContinuousModelSettings):
            model = settings
        else:
            model = ContinuousModelSettings.parse_obj(settings)

        return model

    def execute(self, debug=False):
        # setup inputs
        priors = self.default_frequentist_priors()
        inputs = ContinuousAnalysis(
            model=self.model,
            dataset=self.dataset,
            priors=priors,
            BMD_type=self.settings.bmr_type,
            BMR=self.settings.bmr,
            alpha=self.settings.alpha,
            suff_stat=self.settings.suff_stat,
            isIncreasing=self.settings.isIncreasing,
            tail_prob=self.settings.tail_prob,
            disttype=self.settings.disttype,
            samples=self.settings.samples,
            burnin=self.settings.burnin,
        )
        import pdb; pdb.set_trace()
        # setup outputs
        fit_results = ContinuousModelResult(
            model=self.model, dist_numE=200, num_params=inputs.num_params
        )
        fit_results_struct = fit_results.to_c()

        dll = BmdsLibraryManager.get_dll(bmds_version="BMDS330", base_name="libDRBMD")

        inputs_struct = inputs.to_c()
        if debug:
            print(inputs_struct)

        dll.estimate_sm_laplace_cont(
            ctypes.pointer(inputs_struct), ctypes.pointer(fit_results_struct)
        )
        fit_results.from_c(fit_results_struct)

        return fit_results

    def default_frequentist_priors(self) -> List[Prior]:
        ...

    def get_default_model_degree(self) -> int:
        return self.model.num_params - 1

    def model_class(self) -> str:
        return self.model.verbose

    def model_name(self) -> str:
        return self.model_class()


class Power(Continuous):
    model = ContinuousModelChoices.c_power.value

    def default_frequentist_priors(self) -> List[Prior]:
        return [
            Prior(type=0, initial_value=-2.0, stdev=1, min_value=-18, max_value=18),
            Prior(type=0, initial_value=0.5, stdev=1, min_value=0, max_value=100),
        ]
