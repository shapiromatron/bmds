import numpy as np

from ...datasets import NestedDichotomousDataset
from ...utils import multi_lstrip
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

    def name(self) -> str:
        return f"{super().name()} ({self.settings.litter_specific_covariate.text}{self.settings.intralitter_correlation.text})"

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

    def to_cpp(self) -> NestedDichotomousAnalysis:
        structs = NestedDichotomousAnalysis.blank()
        structs.analysis.model = self.bmd_model_class.id
        structs.analysis.restricted = self.settings.restricted
        structs.analysis.doses = self.dataset.doses
        structs.analysis.litterSize = self.dataset.litter_ns
        structs.analysis.incidence = self.dataset.incidences
        structs.analysis.lsc = self.dataset.litter_covariates
        structs.analysis.LSC_type = self.settings.litter_specific_covariate.value
        structs.analysis.ILC_type = self.settings.intralitter_correlation.value
        structs.analysis.BMD_type = self.settings.bmr_type.value
        structs.analysis.background = self.settings.background.value
        structs.analysis.BMR = self.settings.bmr
        structs.analysis.alpha = self.settings.alpha
        structs.analysis.iterations = self.settings.bootstrap_iterations
        structs.analysis.seed = self.settings.bootstrap_seed
        return structs

    def execute(self) -> NestedDichotomousResult:
        self.structs = self.to_cpp()
        self.structs.execute()
        self.results = NestedDichotomousResult.from_model(self)
        return self.results

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

    def get_gof_pvalue(self):
        ...

    def get_priors_list(self):
        ...

    def model_settings_text(self) -> str:
        input_tbl = self.settings.tbl()
        return multi_lstrip(
            f"""
        Input Summary:
        {input_tbl}
        """
        )


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
