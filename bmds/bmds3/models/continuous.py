import ctypes
from typing import List

from ..constants import ContinuousModel, ContinuousModelChoices, Prior
from ..types.continuous import (
    ContinuousAnalysis,
    ContinuousModelResult,
    ContinuousModelSettings,
    ContinuousBmdsResultsStruct,
    ContinuousResult
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

        if model.degree == 0:
            model.degree = self.get_default_model_degree()

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
            degree=self.settings.degree
        )
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

        bmds_results_struct = ContinuousBmdsResultsStruct.from_results(fit_results)

        dll.collect_cont_bmd_values(ctypes.pointer(inputs_struct),
            ctypes.pointer(fit_results_struct),
            ctypes.pointer(bmds_results_struct),)

        result = ContinuousResult(
            model_class=self.model_class(),
            model_name=self.model_name(),
            bmdl=bmds_results_struct.bmdl,
            bmd=bmds_results_struct.bmd,
            bmdu=bmds_results_struct.bmdu,
            aic=bmds_results_struct.aic,
            bounded=[bmds_results_struct.bounded[i] for i in range(fit_results.num_params)],
            fit=fit_results,
        )
        return result

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
            Prior(type=0, initial_value=0, stdev=1, min_value=1e-8, max_value=1e8),
            Prior(type=0, initial_value=0, stdev=1, min_value=-1e8, max_value=1e8),
            Prior(type=0, initial_value=0, stdev=1, min_value=1e-8, max_value=100),
            Prior(type=0, initial_value=0, stdev=1, min_value=-1000, max_value=100),
        ]

class Hill(Continuous):
    model = ContinuousModelChoices.c_hill.value

    def default_frequentist_priors(self) -> List[Prior]:
        return [
            Prior(type=0, initial_value=0, stdev=1, min_value=1e-8, max_value=1e8),
            Prior(type=0, initial_value=0, stdev=1, min_value=-1e8, max_value=1e8),
            Prior(type=0, initial_value=0, stdev=1, min_value=1e-8, max_value=100),
            Prior(type=0, initial_value=0, stdev=1, min_value=-1000, max_value=100),
            Prior(type=0, initial_value=0, stdev=1, min_value=-1000, max_value=100),
        ]

class Polynomial(Continuous):
    model = ContinuousModelChoices.c_polynomial.value

    def default_frequentist_priors(self) -> List[Prior]:
        return [
            Prior(type=0, initial_value=0, stdev=1, min_value=1e-8, max_value=1e8),
        ]

class ExponentialM2(Continuous):
    model = ContinuousModelChoices.c_exp_m2.value

    def default_frequentist_priors(self) -> List[Prior]:
        return [
            Prior(type=0, initial_value=0, stdev=1, min_value=1e-8, max_value=1e8),
            Prior(type=0, initial_value=0, stdev=1, min_value=-1e8, max_value=1e8),
            Prior(type=0, initial_value=0, stdev=1, min_value=1e-8, max_value=100),
            Prior(type=0, initial_value=0, stdev=1, min_value=-1000, max_value=100),
            Prior(type=0, initial_value=0, stdev=1, min_value=-1000, max_value=100),
        ]

class ExponentialM3(Continuous):
    model = ContinuousModelChoices.c_exp_m3.value

    def default_frequentist_priors(self) -> List[Prior]:
        return [
            Prior(type=0, initial_value=0, stdev=1, min_value=1e-8, max_value=1e8),
            Prior(type=0, initial_value=0, stdev=1, min_value=-1e8, max_value=1e8),
            Prior(type=0, initial_value=0, stdev=1, min_value=1e-8, max_value=100),
            Prior(type=0, initial_value=0, stdev=1, min_value=-1000, max_value=100),
            Prior(type=0, initial_value=0, stdev=1, min_value=-1000, max_value=100),
        ]

class ExponentialM4(Continuous):
    model = ContinuousModelChoices.c_exp_m4.value

    def default_frequentist_priors(self) -> List[Prior]:
        return [
            Prior(type=0, initial_value=0, stdev=1, min_value=1e-8, max_value=1e8),
            Prior(type=0, initial_value=0, stdev=1, min_value=-1e8, max_value=1e8),
            Prior(type=0, initial_value=0, stdev=1, min_value=1e-8, max_value=100),
            Prior(type=0, initial_value=0, stdev=1, min_value=-1000, max_value=100),
            Prior(type=0, initial_value=0, stdev=1, min_value=-1000, max_value=100),
        ]

class ExponentialM5(Continuous):
    model = ContinuousModelChoices.c_exp_m5.value

    def default_frequentist_priors(self) -> List[Prior]:
        return [
            Prior(type=0, initial_value=0, stdev=1, min_value=1e-8, max_value=1e8),
            Prior(type=0, initial_value=0, stdev=1, min_value=-1e8, max_value=1e8),
            Prior(type=0, initial_value=0, stdev=1, min_value=1e-8, max_value=100),
            Prior(type=0, initial_value=0, stdev=1, min_value=-1000, max_value=100),
            Prior(type=0, initial_value=0, stdev=1, min_value=-1000, max_value=100),
        ]
