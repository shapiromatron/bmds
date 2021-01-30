import ctypes
import math
from typing import Optional

import numpy as np

from ...datasets import ContinuousDataset
from ..constants import (
    BMDS_BLANK_VALUE,
    ContinuousModel,
    ContinuousModelChoices,
    ContinuousModelIds,
    DistType,
    ModelPriors,
    PriorClass,
)
from ..types.common import residual_of_interest
from ..types.continuous import (
    ContinuousAnalysis,
    ContinuousBmdsResultsStruct,
    ContinuousModelResult,
    ContinuousModelSettings,
    ContinuousResult,
)
from ..types.priors import get_continuous_prior
from .base import BmdModel, BmdModelSchema, BmdsLibraryManager, InputModelSettings


def get_default_priors(
    model_class: ContinuousModel,
    settings: ContinuousModelSettings,
    prior_class: PriorClass = PriorClass.frequentist_restricted,
) -> ModelPriors:
    """Generate a default ModelPriors for an analysis; for some models, settings-dependent.

    Args:
        model_class (ContinuousModel): A model type
        settings (ContinuousModelSettings): Existing settings dataset
        prior_class (Optional[PriorClass], optional): Prior class; if known

    Raises:
        ValueError: If we cannot determine prior defaults from known information
    """
    # TODO -make this an easy api call for running new datasets; add FAQ?

    model_priors = get_continuous_prior(model_class, prior_class)

    # exp5
    if model_class == ContinuousModelChoices.c_exp_m5.value:
        if settings.is_increasing is True:
            model_priors.priors[2].min_value = 0
            model_priors.priors[2].max_value = 18
        elif settings.is_increasing is False:
            model_priors.priors[2].min_value = -18
            model_priors.priors[2].max_value = 0
        else:
            raise ValueError("Can't set prior; direction unknown")

    # hill
    elif model_class == ContinuousModelChoices.c_hill.value:
        if settings.disttype == DistType.normal:
            model_priors.priors[0].min_value = -1e8
            model_priors.priors[1].min_value = -1e8
            model_priors.priors[1].max_value = 1e8
            model_priors.priors[2].max_value = 30
            model_priors.priors[3].max_value = 18
        elif settings.disttype == DistType.normal_ncv:
            model_priors.priors[0].min_value = -1e8
            model_priors.priors[1].min_value = -1000
            model_priors.priors[1].max_value = 1000
            model_priors.priors[2].max_value = 30
            model_priors.priors[3].max_value = 18
        elif settings.disttype == DistType.log_normal:
            model_priors.priors[0].min_value = 1e-8
            model_priors.priors[1].min_value = -1e8
            model_priors.priors[1].max_value = 1e8
            model_priors.priors[2].max_value = 100
            model_priors.priors[3].max_value = 100
        else:
            raise ValueError("Can't set prior; disttype unknown")

    # linear/poly
    elif model_class == ContinuousModelChoices.c_power.value:
        if settings.is_increasing is True:
            model_priors.priors[1].min_value = 0
            model_priors.priors[1].max_value = 1e8
            model_priors.variance_priors[1].min_value = 0
            model_priors.variance_priors[1].max_value = 1e8
        elif settings.is_increasing is False:
            model_priors.priors[1].min_value = -1e8
            model_priors.priors[1].max_value = 0
            model_priors.variance_priors[1].min_value = -1e8
            model_priors.variance_priors[1].max_value = 0
        else:
            raise ValueError("Can't set prior; direction unknown")

    return model_priors


class BmdModelContinuous(BmdModel):
    bmd_model_class: ContinuousModel

    def get_model_settings(
        self, dataset: ContinuousDataset, settings: InputModelSettings
    ) -> ContinuousModelSettings:
        if settings is None:
            model = ContinuousModelSettings()
        elif isinstance(settings, ContinuousModelSettings):
            model = settings
        else:
            model = ContinuousModelSettings.parse_obj(settings)

        if model.degree == 0:
            model.degree = self.get_default_model_degree(dataset)

        if model.is_increasing is None:
            model.is_increasing = dataset.is_increasing

        if model.priors is None:
            model.priors = get_default_priors(self.bmd_model_class, model)

        return model

    def execute(self):
        inputs = ContinuousAnalysis(
            model=self.bmd_model_class,
            dataset=self.dataset,
            priors=self.settings.priors,
            BMD_type=self.settings.bmr_type,
            BMR=self.settings.bmr,
            alpha=self.settings.alpha,
            suff_stat=self.settings.suff_stat,
            is_increasing=self.settings.is_increasing,
            tail_prob=self.settings.tail_prob,
            disttype=self.settings.disttype,
            samples=self.settings.samples,
            burnin=self.settings.burnin,
            degree=self.settings.degree,
        )
        fit_results = ContinuousModelResult(
            model=self.bmd_model_class, dist_numE=200, num_params=inputs.num_params
        )

        inputs_struct = inputs.to_c()
        fit_results_struct = fit_results.to_c()
        bmds_results_struct = ContinuousBmdsResultsStruct.from_results(fit_results)

        # run the analysis
        dll = BmdsLibraryManager.get_dll(bmds_version="BMDS330", base_name="libDRBMD")
        dll.runBMDSContAnalysis(
            ctypes.pointer(inputs_struct),
            ctypes.pointer(fit_results_struct),
            ctypes.pointer(bmds_results_struct),
        )

        fit_results.from_c(fit_results_struct)
        dr_x = self.dataset.dose_linspace
        critical_xs = np.array(
            [bmds_results_struct.bmdl, bmds_results_struct.bmd, bmds_results_struct.bmdu]
        )
        dr_y = self.dr_curve(dr_x, fit_results.params)
        critical_ys = self.dr_curve(critical_xs, fit_results.params)
        residuals = [d + 1 for d in self.dataset.doses]  # TODO - use real version
        aic = (
            bmds_results_struct.aic if math.isfinite(bmds_results_struct.aic) else BMDS_BLANK_VALUE
        )  # TODO - after models fixed; remove this check?
        result = ContinuousResult(
            bmdl=bmds_results_struct.bmdl,
            bmd=bmds_results_struct.bmd,
            bmdu=bmds_results_struct.bmdu,
            aic=aic,
            roi=residual_of_interest(bmds_results_struct.bmd, self.dataset.doses, residuals),
            bounded=[bmds_results_struct.bounded[i] for i in range(fit_results.num_params)],
            fit=fit_results,
            dr_x=dr_x.tolist(),
            dr_y=dr_y.tolist(),
            bmdl_y=critical_ys[0] if bmds_results_struct.bmdl > 0 else BMDS_BLANK_VALUE,
            bmd_y=critical_ys[1] if bmds_results_struct.bmd > 0 else BMDS_BLANK_VALUE,
            bmdu_y=critical_ys[2] if bmds_results_struct.bmdu > 0 else BMDS_BLANK_VALUE,
        )
        return result

    def get_default_model_degree(self, dataset) -> int:
        return 0

    def dr_curve(self, doses, params) -> np.ndarray:
        raise NotImplementedError()

    def serialize(self) -> "BmdModelContinuousSchema":
        return BmdModelContinuousSchema(
            name=self.name(),
            model_class=self.bmd_model_class,
            settings=self.settings,
            results=self.results,
        )


class BmdModelContinuousSchema(BmdModelSchema):
    name: str
    model_class: ContinuousModel
    settings: ContinuousModelSettings
    results: Optional[ContinuousResult]

    def deserialize(self, dataset: ContinuousDataset) -> BmdModelContinuous:
        Model = bmd_model_map[self.model_class.id]
        model = Model(dataset=dataset, settings=self.settings)
        model.results = self.results
        return model


class Power(BmdModelContinuous):
    bmd_model_class = ContinuousModelChoices.c_power.value

    def dr_curve(self, doses, params) -> np.ndarray:
        g = params[0]
        v = params[1]
        n = params[2]
        return g + v * doses ** n


class Hill(BmdModelContinuous):
    bmd_model_class = ContinuousModelChoices.c_hill.value

    def dr_curve(self, doses, params) -> np.ndarray:
        g = params[0]
        v = params[1]
        k = params[2]
        n = params[3]
        return g + v * doses ** n / (k ** n + doses ** n)


class Polynomial(BmdModelContinuous):
    bmd_model_class = ContinuousModelChoices.c_polynomial.value

    def get_model_settings(
        self, dataset: ContinuousDataset, settings: InputModelSettings
    ) -> ContinuousModelSettings:
        model = super().get_model_settings(dataset, settings)

        if model.degree < 1:
            raise ValueError(f"Polynomial must be ≥ 1; got {model.degree}")

        return model

    def get_default_model_degree(self, dataset) -> int:
        return 2

    def dr_curve(self, doses, params) -> np.ndarray:
        val = doses * 0.0 + params[0]
        for i in range(1, self.settings.degree + 1):
            val += params[i] * doses ** i
        return val


class Linear(Polynomial):
    def get_default_model_degree(self, dataset) -> int:
        return 1

    def get_model_settings(
        self, dataset: ContinuousDataset, settings: InputModelSettings
    ) -> ContinuousModelSettings:
        model = super().get_model_settings(dataset, settings)
        if model.degree != 1:
            raise ValueError("Linear model must have degree of 1")
        return model


class ExponentialM3(BmdModelContinuous):
    bmd_model_class = ContinuousModelChoices.c_exp_m3.value

    def dr_curve(self, doses, params) -> np.ndarray:
        a = params[0]
        b = params[1]
        d = params[3]
        sign = 1.0 if self.settings.is_increasing else -1.0
        return a * np.exp(sign * ((b * doses) ** d))


class ExponentialM5(BmdModelContinuous):
    bmd_model_class = ContinuousModelChoices.c_exp_m5.value

    def dr_curve(self, doses, params) -> np.ndarray:
        a = params[0]
        b = params[1]
        c = params[2]
        d = params[3]
        return a * (np.exp(c) - (np.exp(c) - 1.0) * (np.exp(-((b * doses) ** d))))


bmd_model_map = {
    ContinuousModelIds.c_power.value: Power,
    ContinuousModelIds.c_hill.value: Hill,
    ContinuousModelIds.c_polynomial.value: Polynomial,
    ContinuousModelIds.c_exp_m3.value: ExponentialM3,
    ContinuousModelIds.c_exp_m5.value: ExponentialM5,
}
