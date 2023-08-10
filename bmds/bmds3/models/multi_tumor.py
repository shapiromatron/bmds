import numpy as np

from ...datasets import NestedDichotomousDataset
from ..constants import PriorClass
from ..types.multi_tumor import (
    MultiTumorAnalysis,
    MultiTumorModelSettings,
    MultiTumorResult,
)
from .base import BmdModel, BmdModelSchema, InputModelSettings


class BmdModelMultiTumor(BmdModel):
    bmd_model_class: MultiTumorModel
    model_version: str = "BMDS330"

    def get_model_settings(
        self, dataset: NestedDichotomousDataset, settings: InputModelSettings
    ) -> MultiTumorModelSettings:
        if settings is None:
            model_settings = MultiTumorModelSettings()
        elif isinstance(settings, MultiTumorModelSettings):
            model_settings = settings
        else:
            model_settings = MultiTumorModelSettings.parse_obj(settings)

        # # get default values, may require further model customization
        # if not isinstance(model_settings.priors, ModelPriors):
        #     prior_class = (
        #         model_settings.priors
        #     )
        #     model_settings.priors = get_dichotomous_prior(
        #         self.bmd_model_class, prior_class=prior_class
        #     )

        return model_settings

    def _build_inputs(self) -> MultiTumorAnalysis:
        return MultiTumorAnalysis(
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

    def execute(self) -> MultiTumorResult:
        inputs = self._build_inputs()
        structs = inputs.to_cpp()
        self.structs = structs
        self.structs.execute()
        self.results = MultiTumorResult.from_model(self)
        return self.results

    def get_default_model_degree(self, dataset) -> int:
        return 2

    def get_default_prior_class(self) -> PriorClass:
        return PriorClass.frequentist_restricted

    def get_param_names(self) -> list[str]:
        names = list(self.bmd_model_class.params)
        return names

    def serialize(self) -> "BmdModelMultiTumorSchema":
        return BmdModelMultiTumorSchema(
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


class BmdModelMultiTumorSchema(BmdModelSchema):
    name: str
    model_class: MultiTumorModel
    settings: MultiTumorModelSettings
    results: MultiTumorResult | None

    def deserialize(self, dataset: NestedDichotomousDataset) -> BmdModelMultiTumor:
        Model = bmd_model_map[self.model_class.id]
        model = Model(dataset=dataset, settings=self.settings)
        model.results = self.results
        return model


class Logistic(BmdModelMultiTumor):
    bmd_model_class = 999

    def dr_curve(self, doses, params) -> np.ndarray:
        return np.linspace(0, 1, doses.size)

    def get_default_prior_class(self) -> PriorClass:
        ...
