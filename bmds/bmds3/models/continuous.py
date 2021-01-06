import ctypes
from typing import List

import numpy as np

from ...datasets import ContinuousDataset
from ..constants import (
    ContinuousModel,
    ContinuousModelChoices,
    ContinuousModelIds,
    Prior,
    PriorClass,
)
from ..types.continuous import (
    ContinuousAnalysis,
    ContinuousBmdsResultsStruct,
    ContinuousModelResult,
    ContinuousModelSettings,
    ContinuousResult,
)
from ..types.priors import ContinuousPriorLookup
from .base import BmdModel, BmdModelSchema, BmdsLibraryManager, InputModelSettings


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

        return model

    def execute(self, debug=False):
        # setup inputs
        priors = self.get_priors(self.settings.prior)
        inputs = ContinuousAnalysis(
            model=self.bmd_model_class,
            dataset=self.dataset,
            priors=priors,
            BMD_type=self.settings.bmr_type,
            BMR=self.settings.bmr,
            alpha=self.settings.alpha,
            suff_stat=self.settings.suff_stat,
            isIncreasing=self.settings.isIncreasing,
            tail_prob=self.settings.tail_prob,
            disttype=self.settings.disttype,
            samples=self.settings.samples,
            burnin=self.settings.burnin,
            degree=self.settings.degree,
        )
        # setup outputs
        fit_results = ContinuousModelResult(
            model=self.bmd_model_class, dist_numE=200, num_params=inputs.num_params
        )
        fit_results_struct = fit_results.to_c()

        dll = BmdsLibraryManager.get_dll(bmds_version="BMDS330", base_name="libDRBMD")

        inputs_struct = inputs.to_c()
        if debug:
            print(inputs_struct)

        dll.estimate_sm_laplace_cont(
            ctypes.pointer(inputs_struct), ctypes.pointer(fit_results_struct)
        )

        fit_results.from_c(fit_results_struct)

        bmds_results_struct = ContinuousBmdsResultsStruct.from_results(fit_results)

        dll.collect_cont_bmd_values(
            ctypes.pointer(inputs_struct),
            ctypes.pointer(fit_results_struct),
            ctypes.pointer(bmds_results_struct),
        )

        dr_x = self.dataset.dose_linspace
        result = ContinuousResult(
            model_class=self.model_class(),
            model_name=self.model_name(),
            bmdl=bmds_results_struct.bmdl,
            bmd=bmds_results_struct.bmd,
            bmdu=bmds_results_struct.bmdu,
            aic=bmds_results_struct.aic,
            bounded=[bmds_results_struct.bounded[i] for i in range(fit_results.num_params)],
            fit=fit_results,
            dr_x=dr_x.tolist(),
            dr_y=self.dr_curve(dr_x, fit_results.params).tolist(),
        )
        return result

    def get_default_model_degree(self, dataset) -> int:
        return self.bmd_model_class.num_params - 1

    def model_class(self) -> str:
        return self.bmd_model_class.verbose

    def model_name(self) -> str:
        return self.model_class()

    def get_priors(
        self, prior_class: PriorClass = PriorClass.frequentist_unrestricted
    ) -> List[Prior]:
        return ContinuousPriorLookup[(self.bmd_model_class.id, prior_class.value)]

    def dr_curve(self, doses, params) -> np.ndarray:
        raise NotImplementedError()

    def serialize(self) -> "BmdModelContinuousSchema":
        return BmdModelContinuousSchema(
            model_class=self.bmd_model_class, settings=self.settings, results=self.results
        )


class BmdModelContinuousSchema(BmdModelSchema):
    model_class: ContinuousModel
    settings: ContinuousModelSettings
    results: ContinuousResult

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

    def dr_curve(self, doses, params) -> np.ndarray:
        # TODO - test!
        # adapted from https://github.com/wheelemw/RBMDS/pull/11/files
        val = params[0]
        for i in range(1, len(params)):
            val += params[i] * doses ** i
        return val


class Linear(Polynomial):
    # TODO - force degree
    pass


class ExponentialM2(BmdModelContinuous):
    bmd_model_class = ContinuousModelChoices.c_exp_m2.value

    def dr_curve(self, doses, params) -> np.ndarray:
        # TODO fix; remove np.nan_to_num
        a = params[0]
        b = params[1]
        return np.nan_to_num(a * np.exp(b * doses))


class ExponentialM3(BmdModelContinuous):
    bmd_model_class = ContinuousModelChoices.c_exp_m3.value

    def dr_curve(self, doses, params) -> np.ndarray:
        # TODO fix; remove np.nan_to_num
        a = params[0]
        b = params[1]
        d = params[3]
        return np.nan_to_num(a * np.exp((b * doses) ** d))


class ExponentialM4(BmdModelContinuous):
    bmd_model_class = ContinuousModelChoices.c_exp_m4.value

    def dr_curve(self, doses, params) -> np.ndarray:
        # TODO fix; remove np.nan_to_num
        a = params[0]
        b = params[1]
        c = params[2]
        return np.nan_to_num(a * (np.exp(c) - (np.exp(c) - 1.0) * (np.exp(-((b * doses))))))


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
    ContinuousModelIds.c_exp_m2.value: ExponentialM2,
    ContinuousModelIds.c_exp_m3.value: ExponentialM3,
    ContinuousModelIds.c_exp_m4.value: ExponentialM4,
    ContinuousModelIds.c_exp_m5.value: ExponentialM5,
}
