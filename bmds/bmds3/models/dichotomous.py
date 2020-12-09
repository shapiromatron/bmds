import ctypes
from typing import Dict, List

import numpy as np
from scipy.stats import gamma, norm

from ..constants import DichotomousModel, DichotomousModelChoices, Prior, PriorClass
from ..types.dichotomous import (
    DichotomousAnalysis,
    DichotomousBmdsResultsStruct,
    DichotomousModelResult,
    DichotomousModelSettings,
    DichotomousPgofDataStruct,
    DichotomousPgofResult,
    DichotomousPgofResultStruct,
    DichotomousResult,
)
from .base import BaseModel, BmdsLibraryManager, InputModelSettings
from .priors import DichotomousPriorLookup


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
        priors = self.get_priors()
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

        bmds_results_struct = DichotomousBmdsResultsStruct.from_results(fit_results.num_params)
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
            dr_x=self.dataset.dose_linspace.tolist(),
            dr_y=self.dr_curve(fit_results.params).tolist(),
        )
        return result

    def get_priors(
        self, prior_class: PriorClass = PriorClass.frequentist_unrestricted
    ) -> List[Prior]:
        return DichotomousPriorLookup[(self.model.id, prior_class.value)]

    def get_default_model_degree(self) -> int:
        return self.model.num_params - 1

    def model_class(self) -> str:
        return self.model.verbose

    def model_name(self) -> str:
        return self.model_class()

    def dr_curve(self, params) -> Dict:
        raise NotImplementedError()


class Logistic(Dichotomous):
    model = DichotomousModelChoices.d_logistic.value

    def dr_curve(self, params) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-params[0] - params[1] * self.dataset.dose_linspace))


class LogLogistic(Dichotomous):
    model = DichotomousModelChoices.d_loglogistic.value

    def dr_curve(self, params) -> np.ndarray:
        return params[0] + (1.0 - params[0]) / (
            1.0 + np.exp(-params[1] - params[2] * np.log(self.dataset.dose_linspace))
        )


class Probit(Dichotomous):
    model = DichotomousModelChoices.d_probit.value

    def dr_curve(self, params) -> np.ndarray:
        return norm.cdf(params[0] + params[1] * self.dataset.dose_linspace)


class LogProbit(Dichotomous):
    model = DichotomousModelChoices.d_logprobit.value

    def dr_curve(self, params) -> np.ndarray:
        return params[0] + (1 - params[0]) * norm.cdf(
            params[1] + params[2] * np.log(self.dataset.dose_linspace)
        )


class Gamma(Dichotomous):
    model = DichotomousModelChoices.d_gamma.value

    def dr_curve(self, params) -> np.ndarray:
        return params[0] + (1 - params[1]) * gamma.cdf(
            self.dataset.dose_linspace * params[1], params[2]
        )


class QuantalLinear(Dichotomous):
    model = DichotomousModelChoices.d_qlinear.value

    def dr_curve(self, params) -> np.ndarray:
        return params[0] + (1 - params[0]) * (1 - np.exp(-params[1] * self.dataset.dose_linspace))


class Weibull(Dichotomous):
    model = DichotomousModelChoices.d_weibull.value

    def dr_curve(self, params) -> np.ndarray:
        return params[0] + (1 - params[0]) * (
            1 - np.exp(-1 * params[2] * self.dataset.dose_linspace ** params[1])
        )


class DichotomousHill(Dichotomous):
    model = DichotomousModelChoices.d_hill.value

    def dr_curve(self, params) -> np.ndarray:
        return params[0] + (params[1] - params[1] * params[0]) / (
            1 + np.exp(-params[2] - params[3] * np.log(self.dataset.dose_linspace))
        )


class Multistage(Dichotomous):
    model = DichotomousModelChoices.d_multistage.value

    def get_default_model_degree(self) -> int:
        return self.dataset.num_dose_groups - 1

    def get_model_settings(self, settings: InputModelSettings) -> DichotomousModelSettings:
        model = super().get_model_settings(settings)

        if model.degree < 2:
            raise ValueError(f"Multistage must be ≥ 2; got {model.degree}")

        return model

    def model_name(self) -> str:
        return f"Multistage {self.settings.degree}°"

    def dr_curve(self, params) -> np.ndarray:
        # TODO - handle higher degrees
        xs = self.dataset.dose_linspace
        return params[0] + (1 - params[0]) * (
            1 - np.exp((-params[1] * xs) - (1 - params[2] * xs) ** 2)
        )
