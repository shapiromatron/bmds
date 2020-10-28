import ctypes
from typing import List

from .. import types33
from ..types33 import DichotomousModelSettings
from ..constants import DichotomousModel, Prior
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
            model.set_default_degree(self.model, self.dataset)

        return model

    def execute(self) -> types33.DichotomousModelResult:
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
        results = types33.DichotomousModelResult(model=self.model, dist_numE=200)
        results_struct = results.to_c()

        dll = BmdsLibraryManager.get_dll(bmds_version="BMDS330", base_name="libDRBMD")

        dll.estimate_sm_laplace_dicho(
            ctypes.pointer(inputs.to_c()), ctypes.pointer(results_struct), True
        )
        results.from_c()

        return results

    def default_frequentist_priors(self) -> List[Prior]:
        pass


class Logistic(Dichotomous):
    model = DichotomousModel.d_logistic
    default_frequentist_priors = [
        Prior(type=0, initial_value=-2, stdev=1, min_value=-18, max_value=18),
        Prior(type=0, initial_value=0.1, stdev=1, min_value=1, max_value=10),
    ]


class LogLogistic(Dichotomous):
    model = DichotomousModel.d_loglogistic
    default_frequentist_priors = [
        Prior(type=0, initial_value=-2.0, stdev=1, min_value=-18, max_value=18),
        Prior(type=0, initial_value=-2.0, stdev=1, min_value=-18, max_value=18),
        Prior(type=0, initial_value=1.0, stdev=1, min_value=1e-4, max_value=18),
    ]


class Probit(Dichotomous):
    model = DichotomousModel.d_probit
    default_frequentist_priors = [
        Prior(type=0, initial_value=-2, stdev=1, min_value=-18, max_value=18),
        Prior(type=0, initial_value=0.1, stdev=1, min_value=0, max_value=18),
    ]


class LogProbit(Dichotomous):
    model = DichotomousModel.d_logprobit
    default_frequentist_priors = [
        Prior(type=0, initial_value=-2.0, stdev=1, min_value=-18, max_value=18),
        Prior(type=0, initial_value=-3.0, stdev=1, min_value=-18, max_value=18),
        Prior(type=0, initial_value=1.0, stdev=1, min_value=1e-4, max_value=18),
    ]


class Gamma(Dichotomous):
    model = DichotomousModel.d_gamma
    default_frequentist_priors = [
        Prior(type=0, initial_value=-2.0, stdev=1, min_value=-18, max_value=18),
        Prior(type=0, initial_value=1.0, stdev=1, min_value=0.2, max_value=18),
        Prior(type=0, initial_value=0.1, stdev=1, min_value=0, max_value=100),
    ]


class QuantalLinear(Dichotomous):
    model = DichotomousModel.d_qlinear
    default_frequentist_priors = [
        Prior(type=0, initial_value=-2.0, stdev=1, min_value=-18, max_value=18),
        Prior(type=0, initial_value=0.5, stdev=1, min_value=0, max_value=100),
    ]


class Weibull(Dichotomous):
    model = DichotomousModel.d_weibull
    default_frequentist_priors = [
        Prior(type=0, initial_value=-2.0, stdev=1, min_value=-18, max_value=18),
        Prior(type=0, initial_value=0.5, stdev=1, min_value=1e-6, max_value=18),
        Prior(type=0, initial_value=1.0, stdev=1, min_value=1e-6, max_value=100),
    ]


class DichotomousHill(Dichotomous):
    model = DichotomousModel.d_hill
    default_frequentist_priors = [
        Prior(type=0, initial_value=-2.0, stdev=1, min_value=-18, max_value=18),
        Prior(type=0, initial_value=0.0, stdev=1, min_value=-18, max_value=18),
        Prior(type=0, initial_value=0.0, stdev=1, min_value=-18, max_value=18),
        Prior(type=0, initial_value=1.0, stdev=1, min_value=-1e-8, max_value=18),
    ]


class Multistage(Dichotomous):
    model = DichotomousModel.d_multistage


#     @property
#     def param_names(self) -> Tuple[str, ...]:
#         params: List[str] = ["g"]
#         params.extend(self.settings.degree_param_names())
#         return tuple(params)

#     @property
#     def num_params(self) -> int:
#         return self.settings.degree + 1

#     def get_dll_default_frequentist_priors(self) -> List[Prior]:
#         """
#         Returns the default list of frequentist priors for a model.
#         """

#         priors = [
#             Prior(type=0, initial_value=-17, stdev=0, min_value=-18, max_value=18),
#             Prior(type=0, initial_value=0.1, stdev=0, min_value=0, max_value=100),
#         ]
#         if self.settings.degree > 2:
#             priors.extend(
#                 [
#                     Prior(type=0, initial_value=0.1, stdev=0, min_value=0, max_value=1e4)
#                     for degree in range(2, self.settings.degree)
#                 ]
#             )
#         return priors
