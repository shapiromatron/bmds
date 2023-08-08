import numpy as np

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

        return model_settings

    def _build_inputs(self) -> NestedDichotomousAnalysis:
        return NestedDichotomousAnalysis(
            model=self.bmd_model_class,
            dataset=self.dataset,
            BMD_type=self.settings.bmr_type,
            BMR=self.settings.bmr,
            alpha=self.settings.alpha,
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


class NestedLogistic(BmdModelNestedDichotomous):
    bmd_model_class = NestedDichotomousModelChoices.d_logistic.value

    def dr_curve(self, doses, params) -> np.ndarray:
        return np.linspace(0, 1, doses.size)

    def get_default_prior_class(self) -> PriorClass:
        return PriorClass.frequentist_unrestricted


class Nctr(BmdModelNestedDichotomous):
    bmd_model_class = NestedDichotomousModelChoices.d_nctr.value

    def dr_curve(self, doses, params) -> np.ndarray:
        return np.linspace(0, 2, doses.size)

    def get_default_prior_class(self) -> PriorClass:
        return PriorClass.frequentist_unrestricted


bmd_model_map = {
    NestedDichotomousModelIds.d_nested_logistic.value: NestedLogistic,
    NestedDichotomousModelIds.d_nctr.value: Nctr,
}
