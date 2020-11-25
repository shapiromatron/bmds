import ctypes
from typing import List

from ..constants import DichotomousModel, DichotomousModelChoices, Prior
from ..types.dichotomous import (
    DichotomousAnalysis,
    DichotomousModelResult,
    DichotomousModelSettings,
    DichotomousPgofDataStruct,
    DichotomousPgofResult,
    DichotomousPgofResultStruct,
    DichotomousBmdsResultsStruct,
    DichotomousResult,
)
from .base import BaseModel, BmdsLibraryManager, InputModelSettings


class Dichotomous(BaseModel):
    # required settings
    model: DichotomousModel

    def get_model_settings(self, settings: InputModelSettings) -> DichotomousModelSettings:
        if settings is None:
            model = DichotomousModelSettings()
        elif isinstance(settings, DichotomousModelSettings):
            model = settings
        else:
            model = DichotomousModelSettings.parse_obj(settings)

        if model.degree == 0:
            model.degree = self.get_default_model_degree()

        return model

    def execute(self, debug=False) -> DichotomousResult:
        # setup inputs
        priors = self.default_frequentist_priors()
        inputs = DichotomousAnalysis(
            model=self.model,
            dataset=self.dataset,
            priors=priors,
            BMD_type=self.settings.bmr_type,
            BMR=self.settings.bmr,
            alpha=self.settings.alpha,
            degree=self.settings.degree,
            samples=self.settings.samples,
            burnin=self.settings.burnin,
        )

        # setup outputs
        fit_results = DichotomousModelResult(
            model=self.model, dist_numE=200, num_params=inputs.num_params
        )
        fit_results_struct = fit_results.to_c()

        dll = BmdsLibraryManager.get_dll(bmds_version="BMDS330", base_name="libDRBMD")

        inputs_struct = inputs.to_c()
        if debug:
            print(inputs_struct)

        dll.estimate_sm_laplace_dicho(
            ctypes.pointer(inputs_struct), ctypes.pointer(fit_results_struct), True
        )
        fit_results.from_c(fit_results_struct)

        # gof results call
        gof_data_struct = DichotomousPgofDataStruct.from_fit(inputs_struct, fit_results_struct)
        gof_results_struct = DichotomousPgofResultStruct.from_dataset(self.dataset)
        dll.compute_dichotomous_pearson_GOF(
            ctypes.pointer(gof_data_struct), ctypes.pointer(gof_results_struct)
        )
        gof_results = DichotomousPgofResult.from_c(gof_results_struct)

        bmds_results_struct = DichotomousBmdsResultsStruct.from_results(fit_results)
        dll.collect_dicho_bmd_values(
            ctypes.pointer(inputs_struct),
            ctypes.pointer(fit_results_struct),
            ctypes.pointer(bmds_results_struct),
        )
        result = DichotomousResult(
            model_class=self.model_class(),
            model_name=self.model_name(),
            bmdl=bmds_results_struct.bmdl,
            bmd=bmds_results_struct.bmd,
            bmdu=bmds_results_struct.bmdu,
            aic=bmds_results_struct.aic,
            bounded=[bmds_results_struct.bounded[i] for i in range(fit_results.num_params)],
            fit=fit_results,
            gof=gof_results,
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


class Logistic(Dichotomous):
    model = DichotomousModelChoices.d_logistic.value

    def default_frequentist_priors(self) -> List[Prior]:
        return [
            Prior(type=0, initial_value=-2, stdev=1, min_value=-18, max_value=18),
            Prior(type=0, initial_value=0.1, stdev=1, min_value=1, max_value=10),
        ]


class LogLogistic(Dichotomous):
    model = DichotomousModelChoices.d_loglogistic.value

    def default_frequentist_priors(self) -> List[Prior]:
        return [
            Prior(type=0, initial_value=-2.0, stdev=1, min_value=-18, max_value=18),
            Prior(type=0, initial_value=-2.0, stdev=1, min_value=-18, max_value=18),
            Prior(type=0, initial_value=1.0, stdev=1, min_value=1e-4, max_value=18),
        ]


class Probit(Dichotomous):
    model = DichotomousModelChoices.d_probit.value

    def default_frequentist_priors(self) -> List[Prior]:
        return [
            Prior(type=0, initial_value=-2, stdev=1, min_value=-18, max_value=18),
            Prior(type=0, initial_value=0.1, stdev=1, min_value=0, max_value=18),
        ]


class LogProbit(Dichotomous):
    model = DichotomousModelChoices.d_logprobit.value

    def default_frequentist_priors(self) -> List[Prior]:
        return [
            Prior(type=0, initial_value=-2.0, stdev=1, min_value=-18, max_value=18),
            Prior(type=0, initial_value=-3.0, stdev=1, min_value=-18, max_value=18),
            Prior(type=0, initial_value=1.0, stdev=1, min_value=1e-4, max_value=18),
        ]


class Gamma(Dichotomous):
    model = DichotomousModelChoices.d_gamma.value

    def default_frequentist_priors(self) -> List[Prior]:
        return [
            Prior(type=0, initial_value=-2.0, stdev=1, min_value=-18, max_value=18),
            Prior(type=0, initial_value=1.0, stdev=1, min_value=0.2, max_value=18),
            Prior(type=0, initial_value=0.1, stdev=1, min_value=0, max_value=100),
        ]


class QuantalLinear(Dichotomous):
    model = DichotomousModelChoices.d_qlinear.value

    def default_frequentist_priors(self) -> List[Prior]:
        return [
            Prior(type=0, initial_value=-2.0, stdev=1, min_value=-18, max_value=18),
            Prior(type=0, initial_value=0.5, stdev=1, min_value=0, max_value=100),
        ]


class Weibull(Dichotomous):
    model = DichotomousModelChoices.d_weibull.value

    def default_frequentist_priors(self) -> List[Prior]:
        return [
            Prior(type=0, initial_value=-2.0, stdev=1, min_value=-18, max_value=18),
            Prior(type=0, initial_value=0.5, stdev=1, min_value=1e-6, max_value=18),
            Prior(type=0, initial_value=1.0, stdev=1, min_value=1e-6, max_value=100),
        ]


class DichotomousHill(Dichotomous):
    model = DichotomousModelChoices.d_hill.value

    def default_frequentist_priors(self) -> List[Prior]:
        return [
            Prior(type=0, initial_value=0, stdev=1, min_value=-18, max_value=18),
            Prior(type=0, initial_value=0, stdev=1, min_value=-18, max_value=18),
            Prior(type=0, initial_value=0, stdev=1, min_value=-18, max_value=18),
            Prior(type=0, initial_value=0, stdev=1, min_value=-1e-8, max_value=18),
        ]


class Multistage(Dichotomous):
    model = DichotomousModelChoices.d_multistage.value

    def default_frequentist_priors(self) -> List[Prior]:
        # underlying dll code is duplicated based on the degree
        return [
            Prior(type=0, initial_value=0, stdev=0, min_value=-18, max_value=18),
            Prior(type=0, initial_value=0, stdev=0, min_value=-18, max_value=100),
            Prior(type=0, initial_value=0, stdev=0, min_value=-18, max_value=1e4),
        ]

    def get_default_model_degree(self) -> int:
        return self.dataset.num_dose_groups - 1

    def get_model_settings(self, settings: InputModelSettings) -> DichotomousModelSettings:
        model = super().get_model_settings(settings)

        if model.degree < 2:
            raise ValueError(f"Multistage must be ≥ 2; got {model.degree}")

        return model

    def model_name(self) -> str:
        return f"Multistage {self.settings.degree}°"
