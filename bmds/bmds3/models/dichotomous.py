import ctypes
from typing import Optional

import numpy as np
from scipy.stats import gamma, norm

from ...datasets import DichotomousDataset
from ..constants import (
    BMDS_BLANK_VALUE,
    DichotomousModel,
    DichotomousModelChoices,
    DichotomousModelIds,
    ModelPriors,
    PriorClass,
)
from ..types.common import residual_of_interest
from ..types.dichotomous import (
    DichotomousAnalysis,
    DichotomousModelResult,
    DichotomousModelSettings,
    DichotomousPgofResult,
    DichotomousResult,
)
from ..types.priors import get_dichotomous_prior
from ..types.structs import DichotomousModelResultStruct
from .base import BmdModel, BmdModelSchema, BmdsLibraryManager, InputModelSettings


class BmdModelDichotomous(BmdModel):
    bmd_model_class: DichotomousModel

    def get_model_settings(
        self, dataset: DichotomousDataset, settings: InputModelSettings
    ) -> DichotomousModelSettings:
        if settings is None:
            model = DichotomousModelSettings()
        elif isinstance(settings, DichotomousModelSettings):
            model = settings
        else:
            model = DichotomousModelSettings.parse_obj(settings)

        if model.degree == 0:
            model.degree = self.get_default_model_degree(dataset)

        if model.priors is None:
            model.priors = self.get_default_priors()

        return model

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

        dll = BmdsLibraryManager.get_dll(bmds_version="BMDS330", base_name="libDRBMD")
        dll.runBMDSDichoAnalysis(
            ctypes.pointer(structs.analysis),
            ctypes.pointer(structs.result),
            ctypes.pointer(structs.gof),
            ctypes.pointer(structs.summary),
        )

        fit_results = DichotomousModelResult.from_c(structs.result, self)
        gof_results = DichotomousPgofResult.from_c(structs.gof)
        dr_x = self.dataset.dose_linspace
        critical_xs = np.array([structs.summary.bmdl, structs.summary.bmd, structs.summary.bmdu])
        dr_y = self.dr_curve(dr_x, fit_results.params)
        critical_ys = self.dr_curve(critical_xs, fit_results.params)
        result = DichotomousResult(
            bmdl=structs.summary.bmdl,
            bmd=structs.summary.bmd,
            bmdu=structs.summary.bmdu,
            aic=structs.summary.aic,
            roi=residual_of_interest(structs.summary.bmd, self.dataset.doses, gof_results.residual),
            bounded=[structs.summary.bounded[i] for i in range(inputs.num_params)],
            fit=fit_results,
            gof=gof_results,
            dr_x=dr_x.tolist(),
            dr_y=dr_y.tolist(),
            bmdl_y=critical_ys[0] if structs.summary.bmdl > 0 else BMDS_BLANK_VALUE,
            bmd_y=critical_ys[1] if structs.summary.bmd > 0 else BMDS_BLANK_VALUE,
            bmdu_y=critical_ys[2] if structs.summary.bmdu > 0 else BMDS_BLANK_VALUE,
        )

        self.structs = structs
        self.results = result

        return result

    def get_default_model_degree(self, dataset) -> int:
        return self.bmd_model_class.num_params - 1

    def transform_params(self, struct: DichotomousModelResultStruct):
        return struct.parms[: struct.nparms]

    def get_default_priors(self) -> ModelPriors:
        raise NotImplementedError()

    def dr_curve(self, doses, params) -> np.ndarray:
        raise NotImplementedError()

    def serialize(self) -> "BmdModelDichotomousSchema":
        return BmdModelDichotomousSchema(
            name=self.name(),
            model_class=self.bmd_model_class,
            settings=self.settings,
            results=self.results,
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

    def get_default_priors(self) -> ModelPriors:
        return get_dichotomous_prior(self.bmd_model_class, PriorClass.frequentist_unrestricted)


class LogLogistic(BmdModelDichotomous):
    bmd_model_class = DichotomousModelChoices.d_loglogistic.value

    def transform_params(self, struct: DichotomousModelResultStruct):
        params = struct.parms
        return [1 / (1 + np.exp(-params[0])), params[1], params[2]]

    def dr_curve(self, doses, params) -> np.ndarray:
        g = params[0]
        a = params[1]
        b = params[2]
        return g + (1 - g) * (1 / (1 + np.exp(-a - b * np.log(doses))))

    def get_default_priors(self) -> ModelPriors:
        return get_dichotomous_prior(self.bmd_model_class, PriorClass.frequentist_restricted)


class Probit(BmdModelDichotomous):
    bmd_model_class = DichotomousModelChoices.d_probit.value

    def dr_curve(self, doses, params) -> np.ndarray:
        a = params[0]
        b = params[1]
        return norm.cdf(a + b * doses)

    def get_default_priors(self) -> ModelPriors:
        return get_dichotomous_prior(self.bmd_model_class, PriorClass.frequentist_unrestricted)


class LogProbit(BmdModelDichotomous):
    bmd_model_class = DichotomousModelChoices.d_logprobit.value

    def transform_params(self, struct: DichotomousModelResultStruct):
        params = struct.parms
        return [1 / (1 + np.exp(-params[0])), params[1], params[2]]

    def dr_curve(self, doses, params) -> np.ndarray:
        g = params[0]
        a = params[1]
        b = params[2]
        return g + (1 - g) * (1 / (1 + np.exp(-a - b * np.log(doses))))

    def get_default_priors(self) -> ModelPriors:
        return get_dichotomous_prior(self.bmd_model_class, PriorClass.frequentist_restricted)


class Gamma(BmdModelDichotomous):
    bmd_model_class = DichotomousModelChoices.d_gamma.value

    def transform_params(self, struct: DichotomousModelResultStruct):
        params = struct.parms
        return [1 / (1 + np.exp(-params[0])), params[1], params[2]]

    def dr_curve(self, doses, params) -> np.ndarray:
        g = params[0]
        a = params[1]
        b = params[2]
        return g + (1 - g) * gamma.cdf(b * doses, a)

    def get_default_priors(self) -> ModelPriors:
        return get_dichotomous_prior(self.bmd_model_class, PriorClass.frequentist_restricted)


class QuantalLinear(BmdModelDichotomous):
    bmd_model_class = DichotomousModelChoices.d_qlinear.value

    def transform_params(self, struct: DichotomousModelResultStruct):
        params = struct.parms
        return [1 / (1 + np.exp(-params[0])), params[1]]

    def dr_curve(self, doses, params) -> np.ndarray:
        g = params[0]
        a = params[1]
        return g + (1 - g) * 1 - np.exp(-a * doses)

    def get_default_priors(self) -> ModelPriors:
        return get_dichotomous_prior(self.bmd_model_class, PriorClass.frequentist_unrestricted)


class Weibull(BmdModelDichotomous):
    bmd_model_class = DichotomousModelChoices.d_weibull.value

    def transform_params(self, struct: DichotomousModelResultStruct):
        params = struct.parms
        return [1 / (1 + np.exp(-params[0])), params[1], params[2]]

    def dr_curve(self, doses, params) -> np.ndarray:
        g = params[0]
        a = params[1]
        b = params[2]
        return g + (1 - g) * (1 - np.exp(-b * doses ** a))

    def get_default_priors(self) -> ModelPriors:
        return get_dichotomous_prior(self.bmd_model_class, PriorClass.frequentist_restricted)


class DichotomousHill(BmdModelDichotomous):
    bmd_model_class = DichotomousModelChoices.d_hill.value

    def transform_params(self, struct: DichotomousModelResultStruct):
        params = struct.parms
        return [1 / (1 + np.exp(-params[0])), 1 / (1 + np.exp(-params[1])), params[2], params[3]]

    def dr_curve(self, doses, params) -> np.ndarray:
        g = params[0]
        n = params[1]
        a = params[2]
        b = params[3]
        return g + (1 - g) * n * (1 / (1 + np.exp(-a - b * np.log(doses))))

    def get_default_priors(self) -> ModelPriors:
        return get_dichotomous_prior(self.bmd_model_class, PriorClass.frequentist_restricted)


class Multistage(BmdModelDichotomous):
    bmd_model_class = DichotomousModelChoices.d_multistage.value

    def get_model_settings(
        self, dataset: DichotomousDataset, settings: InputModelSettings
    ) -> DichotomousModelSettings:
        model = super().get_model_settings(dataset, settings)

        if model.degree < 1:
            raise ValueError(f"Multistage must be ≥ 1; got {model.degree}")

        return model

    def name(self) -> str:
        return f"Multistage {self.settings.degree}°"

    def transform_params(self, struct: DichotomousModelResultStruct):
        params = super().transform_params(struct)
        params[0] = 1 / (1 + np.exp(-params[0]))
        return params

    def dr_curve(self, doses, params) -> np.ndarray:
        g = params[0]
        val = doses * 0
        for i in range(1, len(params)):
            val += params[i] * doses ** i
        return g + (1 - g) * (1 - np.exp(-1.0 * val))

    def get_default_priors(self) -> ModelPriors:
        return get_dichotomous_prior(self.bmd_model_class, PriorClass.frequentist_restricted)


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
