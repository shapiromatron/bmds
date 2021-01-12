import ctypes
from typing import List, Optional

import numpy as np
from scipy.stats import gamma, norm

from ...datasets import DichotomousDataset
from ..constants import (
    DichotomousModel,
    DichotomousModelChoices,
    DichotomousModelIds,
    Prior,
    PriorClass,
)
from ..types.common import residual_of_interest
from ..types.dichotomous import (
    DichotomousAnalysis,
    DichotomousBmdsResultsStruct,
    DichotomousModelResult,
    DichotomousModelResultStruct,
    DichotomousModelSettings,
    DichotomousPgofDataStruct,
    DichotomousPgofResult,
    DichotomousPgofResultStruct,
    DichotomousResult,
)
from ..types.priors import DichotomousPriorLookup
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

        return model

    def get_analysis_inputs(self) -> DichotomousAnalysis:
        # setup inputs
        priors = self.get_priors(self.settings.prior)
        return DichotomousAnalysis(
            model=self.bmd_model_class,
            dataset=self.dataset,
            priors=priors,
            BMD_type=self.settings.bmr_type,
            BMR=self.settings.bmr,
            alpha=self.settings.alpha,
            degree=self.settings.degree,
            samples=self.settings.samples,
            burnin=self.settings.burnin,
        )

    def execute(self, debug=False) -> DichotomousResult:
        # setup inputs
        inputs = self.get_analysis_inputs()
        inputs_struct = inputs.to_c()

        # setup outputs
        fit_results = DichotomousModelResult(dist_numE=200, num_params=inputs.num_params)
        fit_results_struct = fit_results.to_c(self.bmd_model_class.id)

        # can be used for model averaging
        self.inputs_struct = inputs_struct
        self.fit_results_struct = fit_results_struct

        dll = BmdsLibraryManager.get_dll(bmds_version="BMDS330", base_name="libDRBMD")

        if debug:
            print(inputs_struct)

        dll.estimate_sm_laplace_dicho(
            ctypes.pointer(inputs_struct), ctypes.pointer(fit_results_struct), True
        )
        fit_results.from_c(fit_results_struct, self)

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
        dr_x = self.dataset.dose_linspace
        result = DichotomousResult(
            bmdl=bmds_results_struct.bmdl,
            bmd=bmds_results_struct.bmd,
            bmdu=bmds_results_struct.bmdu,
            aic=bmds_results_struct.aic,
            roi=residual_of_interest(
                bmds_results_struct.bmd, self.dataset.doses, gof_results.residual
            ),
            bounded=[bmds_results_struct.bounded[i] for i in range(fit_results.num_params)],
            fit=fit_results,
            gof=gof_results,
            dr_x=dr_x.tolist(),
            dr_y=self.dr_curve(dr_x, fit_results.params).tolist(),
        )
        return result

    def get_priors(
        self, prior_class: PriorClass = PriorClass.frequentist_unrestricted
    ) -> List[Prior]:
        return DichotomousPriorLookup[(self.bmd_model_class.id, prior_class.value)]

    def get_default_model_degree(self, dataset) -> int:
        return self.bmd_model_class.num_params - 1

    def transform_params(self, struct: DichotomousModelResultStruct):
        return struct.parms[: struct.nparms]

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


class Probit(BmdModelDichotomous):
    bmd_model_class = DichotomousModelChoices.d_probit.value

    def dr_curve(self, doses, params) -> np.ndarray:
        a = params[0]
        b = params[1]
        return norm.cdf(a + b * doses)


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


class QuantalLinear(BmdModelDichotomous):
    bmd_model_class = DichotomousModelChoices.d_qlinear.value

    def transform_params(self, struct: DichotomousModelResultStruct):
        params = struct.parms
        return [1 / (1 + np.exp(-params[0])), params[1]]

    def dr_curve(self, doses, params) -> np.ndarray:
        g = params[0]
        a = params[1]
        return g + (1 - g) * 1 - np.exp(-a * doses)


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


class Multistage(BmdModelDichotomous):
    bmd_model_class = DichotomousModelChoices.d_multistage.value

    def get_default_model_degree(self, dataset) -> int:
        return dataset.num_dose_groups - 1

    def get_model_settings(
        self, dataset: DichotomousDataset, settings: InputModelSettings
    ) -> DichotomousModelSettings:
        model = super().get_model_settings(dataset, settings)

        if model.degree < 2:
            raise ValueError(f"Multistage must be ≥ 2; got {model.degree}")

        return model

    def model_name(self) -> str:
        return f"Multistage {self.settings.degree}°"

    def transform_params(self, struct: DichotomousModelResultStruct):
        params = super().transform_params(struct)
        params[0] = 1 / (1 + np.exp(-params[0]))
        return params

    def dr_curve(self, doses, params) -> np.ndarray:
        # TODO - test!
        # adapted from https://github.com/wheelemw/RBMDS/pull/11/files
        g = params[0]
        val = doses * 0
        for i in range(1, len(params)):
            val -= -params[i] * doses ** i
        return g + (1 - g) * 1 - np.exp(val)


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
