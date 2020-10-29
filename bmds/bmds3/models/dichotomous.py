import ctypes
from typing import List

from .. import types33
from ..constants import DichotomousModel, DichotomousModelChoices, Prior
from ..types33 import DichotomousModelSettings
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

    def execute(self, debug=False) -> types33.DichotomousModelResult:
        # setup inputs
        priors = self.default_frequentist_priors()
        inputs = types33.DichotomousAnalysis(
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
        results = types33.DichotomousModelResult(
            model=self.model, dist_numE=200, num_params=inputs.num_params
        )
        results_struct = results.to_c()

        dll = BmdsLibraryManager.get_dll(bmds_version="BMDS330", base_name="libDRBMD")

        inputs_struct = inputs.to_c()
        if debug:
            print(inputs_struct)

        dll.estimate_sm_laplace_dicho(
            ctypes.pointer(inputs_struct), ctypes.pointer(results_struct), True
        )
        results.from_c()

        return results

    def default_frequentist_priors(self) -> List[Prior]:
        pass

    def get_default_model_degree(self) -> int:
        return self.model.num_params - 1


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
