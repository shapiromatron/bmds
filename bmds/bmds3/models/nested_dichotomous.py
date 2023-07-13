import numpy as np
from scipy.stats import gamma, norm

from ...datasets import NestedDichotomousDataset
from ..constants import (
    NestedDichotomousModel,
    NestedDichotomousModelChoices,
    NestedDichotomousModelIds,
    PriorClass,
)
from ..types.nested_dichotomous import (
    NestedDichotomousAnalysis,
    NestedDichotomousModelSettings,
    NestedDichotomousResult,
)
from .base import BmdModel, BmdModelSchema, InputModelSettings

# from ..types.priors import ModelPriors, get_dichotomous_prior


class BmdModelNestedDichotomous(BmdModel):
    bmd_model_class: NestedDichotomousModel
    model_version: str = "BMDS330"

    def get_model_settings(
        self, dataset: NestedDichotomousDataset, settings: InputModelSettings
    ) -> NestedDichotomousModelSettings:
        if settings is None:
            model_settings = NestedDichotomousModelSettings()
        elif isinstance(settings, NestedDichotomousModelSettings):
            model_settings = settings
        else:
            model_settings = NestedDichotomousModelSettings.parse_obj(settings)

        # # get default values, may require further model customization
        # if not isinstance(model_settings.priors, ModelPriors):
        #     prior_class = (
        #         model_settings.priors
        #     )
        #     model_settings.priors = get_dichotomous_prior(
        #         self.bmd_model_class, prior_class=prior_class
        #     )

        return model_settings

    def _build_inputs(self) -> NestedDichotomousAnalysis:
        return NestedDichotomousAnalysis(
            model=self.bmd_model_class,
            dataset=self.dataset,
            # priors=self.settings.priors,
            BMD_type=self.settings.bmr_type,
            BMR=self.settings.bmr,
            alpha=self.settings.alpha,
            # degree=self.settings.degree,
            # samples=self.settings.samples,
            # burnin=self.settings.burnin,
        )

    def execute(self) -> NestedDichotomousResult:
        inputs = self._build_inputs()
        structs = inputs.to_cpp()
        self.structs = structs
        self.structs.execute()
        self.results = NestedDichotomousResult.from_model(self)
        return self.results

    def get_default_model_degree(self, dataset) -> int:
        return 2

    def get_default_prior_class(self) -> PriorClass:
        return PriorClass.frequentist_restricted

    def get_param_names(self) -> list[str]:
        names = list(self.bmd_model_class.params)
        return names

    def serialize(self) -> "BmdModelNestedDichotomousSchema":
        return BmdModelNestedDichotomousSchema(
            name=self.name(),
            model_class=self.bmd_model_class,
            settings=self.settings,
            results=self.results,
        )

    def get_gof_pvalue(self) -> float:
        return self.results.gof.p_value

    def get_priors_list(self) -> list[list]:
        degree = self.settings.degree if self.degree_required else None
        return self.settings.priors.priors_list(degree=degree)


class BmdModelNestedDichotomousSchema(BmdModelSchema):
    name: str
    model_class: NestedDichotomousModel
    settings: NestedDichotomousModelSettings
    results: NestedDichotomousResult | None

    def deserialize(self, dataset: NestedDichotomousDataset) -> BmdModelNestedDichotomous:
        Model = bmd_model_map[self.model_class.id]
        model = Model(dataset=dataset, settings=self.settings)
        model.results = self.results
        return model


class Logistic(BmdModelNestedDichotomous):
    bmd_model_class = NestedDichotomousModelChoices.d_logistic.value

    def dr_curve(self, doses, params) -> np.ndarray:
        a = params[0]
        b = params[1]
        return 1 / (1 + np.exp(-a - b * doses))

    def get_default_prior_class(self) -> PriorClass:
        return PriorClass.frequentist_unrestricted


class LogLogistic(BmdModelNestedDichotomous):
    bmd_model_class = NestedDichotomousModelChoices.d_loglogistic.value

    def dr_curve(self, doses, params) -> np.ndarray:
        g = params[0]
        a = params[1]
        b = params[2]
        return g + (1 - g) * (1 / (1 + np.exp(-a - b * np.log(doses))))


class Probit(BmdModelNestedDichotomous):
    bmd_model_class = NestedDichotomousModelChoices.d_probit.value

    def dr_curve(self, doses, params) -> np.ndarray:
        a = params[0]
        b = params[1]
        return norm.cdf(a + b * doses)

    def get_default_prior_class(self) -> PriorClass:
        return PriorClass.frequentist_unrestricted


class LogProbit(BmdModelNestedDichotomous):
    bmd_model_class = NestedDichotomousModelChoices.d_logprobit.value

    def dr_curve(self, doses, params) -> np.ndarray:
        g = params[0]
        a = params[1]
        b = params[2]
        return g + (1 - g) * norm.cdf(a + b * np.log(doses))

    def get_default_prior_class(self) -> PriorClass:
        return PriorClass.frequentist_unrestricted


class Gamma(BmdModelNestedDichotomous):
    bmd_model_class = NestedDichotomousModelChoices.d_gamma.value

    def dr_curve(self, doses, params) -> np.ndarray:
        g = params[0]
        a = params[1]
        b = params[2]
        return g + (1 - g) * gamma.cdf(b * doses, a)


class QuantalLinear(BmdModelNestedDichotomous):
    bmd_model_class = NestedDichotomousModelChoices.d_qlinear.value

    def dr_curve(self, doses, params) -> np.ndarray:
        g = params[0]
        b = params[1]
        return g + (1 - g) * 1 - np.exp(-b * doses)

    def get_default_prior_class(self) -> PriorClass:
        return PriorClass.frequentist_unrestricted


class Weibull(BmdModelNestedDichotomous):
    bmd_model_class = NestedDichotomousModelChoices.d_weibull.value

    def dr_curve(self, doses, params) -> np.ndarray:
        g = params[0]
        a = params[1]
        b = params[2]
        return g + (1 - g) * (1 - np.exp(-b * doses**a))


class NestedDichotomousHill(BmdModelNestedDichotomous):
    bmd_model_class = NestedDichotomousModelChoices.d_hill.value

    def dr_curve(self, doses, params) -> np.ndarray:
        g = params[0]
        v = params[1]
        a = params[2]
        b = params[3]
        return g + (1 - g) * v * (1 / (1 + np.exp(-a - b * np.log(doses))))


class Multistage(BmdModelNestedDichotomous):
    bmd_model_class = NestedDichotomousModelChoices.d_multistage.value
    degree_required: bool = True

    def get_model_settings(
        self, dataset: NestedDichotomousDataset, settings: InputModelSettings
    ) -> NestedDichotomousModelSettings:
        model_settings = super().get_model_settings(dataset, settings)

        if model_settings.degree < 1:
            model_settings.degree = self.get_default_model_degree(dataset)

        # TODO - priors for cancer multistage must change

        return model_settings

    def name(self) -> str:
        return f"Multistage {self.settings.degree}°"

    def dr_curve(self, doses, params) -> np.ndarray:
        g = params[0]
        val = doses * 0
        for i in range(1, len(params)):
            val += params[i] * doses**i
        return g + (1 - g) * (1 - np.exp(-1.0 * val))

    def get_param_names(self) -> list[str]:
        names = [f"b{i}" for i in range(self.settings.degree + 1)]
        names[0] = "g"
        return names


bmd_model_map = {
    NestedDichotomousModelIds.d_hill.value: NestedDichotomousHill,
    NestedDichotomousModelIds.d_gamma.value: Gamma,
    NestedDichotomousModelIds.d_logistic.value: Logistic,
    NestedDichotomousModelIds.d_loglogistic.value: LogLogistic,
    NestedDichotomousModelIds.d_logprobit.value: LogProbit,
    NestedDichotomousModelIds.d_multistage.value: Multistage,
    NestedDichotomousModelIds.d_probit.value: Probit,
    NestedDichotomousModelIds.d_qlinear.value: QuantalLinear,
    NestedDichotomousModelIds.d_weibull.value: Weibull,
}
