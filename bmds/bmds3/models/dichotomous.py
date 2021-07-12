import ctypes
from typing import List, Optional

import numpy as np
from scipy.stats import gamma, norm

from ...datasets import DichotomousDataset
from ..constants import (
    DichotomousModel,
    DichotomousModelChoices,
    DichotomousModelIds,
    ModelPriors,
    PriorClass,
)
from ..types.dichotomous import DichotomousAnalysis, DichotomousModelSettings, DichotomousResult
from ..types.priors import get_dichotomous_prior
from .base import BmdModel, BmdModelSchema, BmdsLibraryManager, InputModelSettings


class BmdModelDichotomous(BmdModel):
    bmd_model_class: DichotomousModel

    def get_model_settings(
        self, dataset: DichotomousDataset, settings: InputModelSettings
    ) -> DichotomousModelSettings:
        if settings is None:
            model_settings = DichotomousModelSettings()
        elif isinstance(settings, DichotomousModelSettings):
            model_settings = settings
        else:
            model_settings = DichotomousModelSettings.parse_obj(settings)

        # get default values, may require further model customization
        if not isinstance(model_settings.priors, ModelPriors):
            prior_class = (
                model_settings.priors
                if isinstance(model_settings.priors, PriorClass)
                else self.get_default_prior_class()
            )
            model_settings.priors = get_dichotomous_prior(
                self.bmd_model_class, prior_class=prior_class
            )

        return model_settings

    def execute(self) -> DichotomousResult:
        # setup inputs
        inputs = DichotomousAnalysis(
            model=self.bmd_model_class,
            dataset=self.dataset,
            priors=self.settings.priors,
            BMD_type=self.settings.bmr_type,
            BMR=self.settings.bmr,
            alpha=self.settings.alpha,
            degree=self.settings.degree,
            samples=self.settings.samples,
            burnin=self.settings.burnin,
        )
        structs = inputs.to_c()
        self.structs = structs

        dll = BmdsLibraryManager.get_dll(bmds_version="BMDS330", base_name="libDRBMD")
        dll.runBMDSDichoAnalysis(
            ctypes.pointer(structs.analysis),
            ctypes.pointer(structs.result),
            ctypes.pointer(structs.gof),
            ctypes.pointer(structs.summary),
            ctypes.pointer(structs.aod),
        )
        self.results = DichotomousResult.from_model(self)
        return self.results

    def get_default_model_degree(self, dataset) -> int:
        return 2

    def get_default_prior_class(self) -> PriorClass:
        return PriorClass.frequentist_restricted

    def dr_curve(self, doses, params) -> np.ndarray:
        raise NotImplementedError()

    def get_param_names(self) -> List[str]:
        names = list(self.bmd_model_class.params)
        return names

    def serialize(self) -> "BmdModelDichotomousSchema":
        return BmdModelDichotomousSchema(
            name=self.name(),
            model_class=self.bmd_model_class,
            settings=self.settings,
            results=self.results,
        )

    def report(self) -> str:
        name = f"╒════════════════════╕\n│ {self.name():18} │\n╘════════════════════╛"
        if not self.has_results:
            return "\n\n".join([name, "Execution was not completed."])

        return "\n\n".join(
            [
                name,
                f"Summary:\n{self.results.tbl()}",
                f"Model Parameters:\n{self.results.parameters.tbl()}",
                f"Goodness of Fit:\n{self.results.gof.tbl(self.dataset)}",
                f"Analysis of Deviance:\n{self.results.deviance.tbl()}",
            ]
        )


class BmdModelDichotomousSchema(BmdModelSchema):
    name: str
    model_class: DichotomousModel
    settings: DichotomousModelSettings
    results: Optional[DichotomousResult]

    def deserialize(self, dataset: DichotomousDataset) -> BmdModelDichotomous:
        Model = bmd_model_map[self.model_class.id]
        model = Model(dataset=dataset, settings=self.settings)
        model.results = self.results
        return model


class Logistic(BmdModelDichotomous):
    bmd_model_class = DichotomousModelChoices.d_logistic.value

    def dr_curve(self, doses, params) -> np.ndarray:
        a = params[0]
        b = params[1]
        return 1 / (1 + np.exp(-a - b * doses))

    def get_default_prior_class(self) -> PriorClass:
        return PriorClass.frequentist_unrestricted


class LogLogistic(BmdModelDichotomous):
    bmd_model_class = DichotomousModelChoices.d_loglogistic.value

    def dr_curve(self, doses, params) -> np.ndarray:
        g = params[0]
        a = params[1]
        b = params[2]
        return g + (1 - g) * (1 / (1 + np.exp(-a - b * np.log(doses))))


class Probit(BmdModelDichotomous):
    bmd_model_class = DichotomousModelChoices.d_probit.value

    def dr_curve(self, doses, params) -> np.ndarray:
        a = params[0]
        b = params[1]
        return norm.cdf(a + b * doses)

    def get_default_prior_class(self) -> PriorClass:
        return PriorClass.frequentist_unrestricted


class LogProbit(BmdModelDichotomous):
    bmd_model_class = DichotomousModelChoices.d_logprobit.value

    def dr_curve(self, doses, params) -> np.ndarray:
        g = params[0]
        a = params[1]
        b = params[2]
        return g + (1 - g) * (1 / (1 + np.exp(-a - b * np.log(doses))))


class Gamma(BmdModelDichotomous):
    bmd_model_class = DichotomousModelChoices.d_gamma.value

    def dr_curve(self, doses, params) -> np.ndarray:
        g = params[0]
        a = params[1]
        b = params[2]
        return g + (1 - g) * gamma.cdf(b * doses, a)


class QuantalLinear(BmdModelDichotomous):
    bmd_model_class = DichotomousModelChoices.d_qlinear.value

    def dr_curve(self, doses, params) -> np.ndarray:
        g = params[0]
        a = params[1]
        return g + (1 - g) * 1 - np.exp(-a * doses)

    def get_default_prior_class(self) -> PriorClass:
        return PriorClass.frequentist_unrestricted


class Weibull(BmdModelDichotomous):
    bmd_model_class = DichotomousModelChoices.d_weibull.value

    def dr_curve(self, doses, params) -> np.ndarray:
        g = params[0]
        a = params[1]
        b = params[2]
        return g + (1 - g) * (1 - np.exp(-b * doses ** a))


class DichotomousHill(BmdModelDichotomous):
    bmd_model_class = DichotomousModelChoices.d_hill.value

    def dr_curve(self, doses, params) -> np.ndarray:
        g = params[0]
        n = params[1]
        a = params[2]
        b = params[3]
        return g + (1 - g) * n * (1 / (1 + np.exp(-a - b * np.log(doses))))


class Multistage(BmdModelDichotomous):
    bmd_model_class = DichotomousModelChoices.d_multistage.value

    def get_model_settings(
        self, dataset: DichotomousDataset, settings: InputModelSettings
    ) -> DichotomousModelSettings:
        model_settings = super().get_model_settings(dataset, settings)

        if model_settings.degree < 1:
            model_settings.degree = self.get_default_model_degree(dataset)

        return model_settings

    def name(self) -> str:
        return f"Multistage {self.settings.degree}°"

    def dr_curve(self, doses, params) -> np.ndarray:
        g = params[0]
        val = doses * 0
        for i in range(1, len(params)):
            val += params[i] * doses ** i
        return g + (1 - g) * (1 - np.exp(-1.0 * val))

    def get_param_names(self) -> List[str]:
        return [f"b{i}" for i in range(self.settings.degree + 1)]


bmd_model_map = {
    DichotomousModelIds.d_hill.value: DichotomousHill,
    DichotomousModelIds.d_gamma.value: Gamma,
    DichotomousModelIds.d_logistic.value: Logistic,
    DichotomousModelIds.d_loglogistic.value: LogLogistic,
    DichotomousModelIds.d_logprobit.value: LogProbit,
    DichotomousModelIds.d_multistage.value: Multistage,
    DichotomousModelIds.d_probit.value: Probit,
    DichotomousModelIds.d_qlinear.value: QuantalLinear,
    DichotomousModelIds.d_weibull.value: Weibull,
}
