import numpy as np
from pydantic import Field

from ... import plotting
from ...constants import ZEROISH
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
            model_settings = NestedDichotomousModelSettings.model_validate(settings)

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

    def _plot_bmr_lines(self, ax):
        plotting.add_bmr_lines(
            ax,
            self.results.summary.bmd,
            self.results.plotting.bmd_y,
            self.results.summary.bmdl,
            self.results.summary.bmdu,
        )

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
    bmd_model_class: NestedDichotomousModel = Field(alias="model_class")
    settings: NestedDichotomousModelSettings
    results: NestedDichotomousResult | None

    def deserialize(self, dataset: NestedDichotomousDataset) -> BmdModelNestedDichotomous:
        Model = bmd_model_map[self.bmd_model_class.id]
        model = Model(dataset=dataset, settings=self.settings)
        model.results = self.results
        return model


class NestedLogistic(BmdModelNestedDichotomous):
    bmd_model_class = NestedDichotomousModelChoices.d_logistic.value

    def get_param_names(self) -> list[str]:
        return ["alpha", "beta", "theta1", "theta2", "rho"] + [
            f"phi{i}" for i in range(1, self.dataset.num_dose_groups + 1)
        ]

    def dr_curve(self, doses: np.ndarray, params: dict, fixed_lsc: float) -> np.ndarray:
        alpha = params["alpha"]
        beta = params["beta"]
        theta1 = params["theta1"]
        theta2 = params["theta2"]
        rho = params["rho"]
        d = doses.copy()
        d[d < ZEROISH] = ZEROISH
        return (
            alpha
            + theta1 * fixed_lsc
            + (1 - alpha - theta1 * fixed_lsc)
            / (1 + np.exp(-1 * beta - theta2 * fixed_lsc - rho * np.log(d)))
        )

    def get_default_prior_class(self) -> PriorClass:
        # TODO - change
        return PriorClass.frequentist_unrestricted


class Nctr(BmdModelNestedDichotomous):
    bmd_model_class = NestedDichotomousModelChoices.d_nctr.value

    def execute(self) -> NestedDichotomousResult:
        # TODO - change
        raise NotImplementedError()

    def dr_curve(self, doses, params) -> np.ndarray:
        # TODO - change
        return np.linspace(0, 2, doses.size)

    def get_default_prior_class(self) -> PriorClass:
        # TODO - change
        return PriorClass.frequentist_unrestricted


bmd_model_map = {
    NestedDichotomousModelIds.d_nested_logistic.value: NestedLogistic,
    NestedDichotomousModelIds.d_nctr.value: Nctr,
}
