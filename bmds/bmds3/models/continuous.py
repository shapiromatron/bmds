import ctypes
from typing import List, Optional

import numpy as np

from ...datasets import ContinuousDatasets
from ..constants import (
    ContinuousModel,
    ContinuousModelChoices,
    ContinuousModelIds,
    DistType,
    ModelPriors,
    PriorClass,
)
from ..types.continuous import ContinuousAnalysis, ContinuousModelSettings, ContinuousResult
from ..types.priors import get_continuous_prior
from .base import BmdModel, BmdModelSchema, BmdsLibraryManager, InputModelSettings


class BmdModelContinuous(BmdModel):
    bmd_model_class: ContinuousModel

    def get_model_settings(
        self, dataset: ContinuousDatasets, settings: InputModelSettings,
    ) -> ContinuousModelSettings:
        if settings is None:
            model_settings = ContinuousModelSettings()
        elif isinstance(settings, ContinuousModelSettings):
            model_settings = settings
        else:
            model_settings = ContinuousModelSettings.parse_obj(settings)

        # only estimate direction if unspecified in settings
        if model_settings.is_increasing is None:
            model_settings.is_increasing = dataset.is_increasing

        # get default values, may require further model customization
        if not isinstance(model_settings.priors, ModelPriors):
            prior_class = (
                model_settings.priors
                if isinstance(model_settings.priors, PriorClass)
                else self.get_default_prior_class()
            )
            model_settings.priors = get_continuous_prior(
                self.bmd_model_class, prior_class=prior_class
            )

        return model_settings

    def execute(self):
        inputs = ContinuousAnalysis(
            model=self.bmd_model_class,
            dataset=self.dataset,
            priors=self.settings.priors,
            BMD_type=self.settings.bmr_type,
            BMR=self.settings.bmr,
            alpha=self.settings.alpha,
            is_increasing=self.settings.is_increasing,
            tail_prob=self.settings.tail_prob,
            disttype=self.settings.disttype,
            samples=self.settings.samples,
            burnin=self.settings.burnin,
            degree=self.settings.degree,
        )
        structs = inputs.to_c()
        self.structs = structs

        # run the analysis
        dll = BmdsLibraryManager.get_dll(bmds_version="BMDS330", base_name="libDRBMD-0")
        dll.excelCont(
            ctypes.pointer(structs.analysis),
            ctypes.pointer(structs.result),
            ctypes.pointer(structs.summary),
            ctypes.pointer(structs.aod),
            ctypes.pointer(structs.gof),
            ctypes.pointer(ctypes.c_bool(False)),
        )
        self.results = ContinuousResult.from_model(self)
        return self.results

    def get_default_model_degree(self, dataset) -> int:
        return 0

    def get_default_prior_class(self) -> PriorClass:
        return PriorClass.frequentist_restricted

    def dr_curve(self, doses, params) -> np.ndarray:
        raise NotImplementedError()

    def serialize(self) -> "BmdModelContinuousSchema":
        return BmdModelContinuousSchema(
            name=self.name(),
            model_class=self.bmd_model_class,
            settings=self.settings,
            results=self.results,
        )

    def get_param_names(self) -> List[str]:
        names = list(self.bmd_model_class.params)
        names.extend(self.get_variance_param_names())
        return names

    def get_variance_param_names(self):
        if self.settings.disttype == DistType.normal_ncv:
            return list(self.bmd_model_class.variance_params)
        else:
            return [self.bmd_model_class.variance_params[0]]

    def report(self) -> str:
        name = f"╒════════════════════╕\n│ {self.name():18} │\n╘════════════════════╛"
        if not self.has_results:
            return "\n\n".join([name, "Execution was not completed."])

        return "\n\n".join(
            [
                name,
                f"Summary:\n{self.results.tbl()}",
                f"Model Parameters:\n{self.results.parameters.tbl()}",
                f"Goodness of Fit:\n{self.results.gof.tbl()}",
                f"Analysis of Deviance:\n{self.results.deviance.tbl()}",
                f"Tests of Interest:\n{self.results.tests.tbl()}",
            ]
        )


class BmdModelContinuousSchema(BmdModelSchema):
    name: str
    model_class: ContinuousModel
    settings: ContinuousModelSettings
    results: Optional[ContinuousResult]

    def deserialize(self, dataset: ContinuousDatasets) -> BmdModelContinuous:
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

    def get_model_settings(
        self, dataset: ContinuousDatasets, settings: InputModelSettings
    ) -> ContinuousModelSettings:
        model_settings = super().get_model_settings(dataset, settings)

        if model_settings.priors.prior_class in [
            PriorClass.frequentist_unrestricted,
            PriorClass.frequentist_restricted,
        ]:
            v = model_settings.priors.get_prior("v")
            if model_settings.disttype is DistType.log_normal:
                model_settings.priors.get_prior("g").min_value = 1e-8
                model_settings.priors.get_prior("k").max_value = 100
                model_settings.priors.get_prior("n").max_value = 100
                v.min_value = -1e8
                v.max_value = 1e8
            else:
                model_settings.priors.get_prior("g").min_value = -1e8
                model_settings.priors.get_prior("k").max_value = 30
                model_settings.priors.get_prior("n").max_value = 18
                if model_settings.disttype is DistType.normal:
                    v.min_value = -1e8
                    v.max_value = 1e8
                elif model_settings.disttype is DistType.normal_ncv:
                    v.min_value = -1000
                    v.max_value = 1000

        return model_settings

    def dr_curve(self, doses, params) -> np.ndarray:
        g = params[0]
        v = params[1]
        k = params[2]
        n = params[3]
        return g + v * doses ** n / (k ** n + doses ** n)


class Polynomial(BmdModelContinuous):
    bmd_model_class = ContinuousModelChoices.c_polynomial.value

    def name(self) -> str:
        return f"Polynomial {self.settings.degree}°"

    def get_model_settings(
        self, dataset: ContinuousDatasets, settings: InputModelSettings
    ) -> ContinuousModelSettings:
        model_settings = super().get_model_settings(dataset, settings)

        if model_settings.degree < 1:
            model_settings.degree = self.get_default_model_degree(dataset)

        if model_settings.priors.prior_class is PriorClass.frequentist_restricted:
            if model_settings.is_increasing is True:
                model_settings.priors.get_prior("beta1").min_value = 0
                model_settings.priors.get_prior("betaN").min_value = 0
                model_settings.priors.get_prior("alpha").min_value = 0

            if model_settings.is_increasing is False:
                model_settings.priors.get_prior("beta1").max_value = 0
                model_settings.priors.get_prior("betaN").max_value = 0
                model_settings.priors.get_prior("alpha").max_value = 0

        return model_settings

    def get_default_model_degree(self, dataset) -> int:
        return 2

    def dr_curve(self, doses, params) -> np.ndarray:
        val = doses * 0.0 + params[0]
        for i in range(1, self.settings.degree + 1):
            val += params[i] * doses ** i
        return val

    def get_param_names(self) -> List[str]:
        names = [f"b{i}" for i in range(self.settings.degree + 1)]
        names.extend(self.get_variance_param_names())
        return names


class Linear(Polynomial):
    def name(self) -> str:
        return "Linear"

    def get_default_model_degree(self, dataset) -> int:
        return 1

    def get_default_prior_class(self) -> PriorClass:
        return PriorClass.frequentist_unrestricted

    def get_model_settings(
        self, dataset: ContinuousDatasets, settings: InputModelSettings
    ) -> ContinuousModelSettings:
        model_settings = super().get_model_settings(dataset, settings)
        if model_settings.degree != 1:
            raise ValueError("Linear model must have degree of 1")
        return model_settings


class ExponentialM3(BmdModelContinuous):
    bmd_model_class = ContinuousModelChoices.c_exp_m3.value

    def get_model_settings(
        self, dataset: ContinuousDatasets, settings: InputModelSettings
    ) -> ContinuousModelSettings:
        model_settings = super().get_model_settings(dataset, settings)

        if model_settings.priors.prior_class is PriorClass.frequentist_restricted:
            attr = "min_value" if model_settings.is_increasing else "max_value"
            setattr(model_settings.priors.get_prior("c"), attr, 0.0)

        return model_settings

    def dr_curve(self, doses, params) -> np.ndarray:
        a = params[0]
        b = params[1]
        d = params[3]
        sign = 1.0 if self.settings.is_increasing else -1.0
        return a * np.exp(sign * ((b * doses) ** d))


class ExponentialM5(BmdModelContinuous):
    bmd_model_class = ContinuousModelChoices.c_exp_m5.value

    def get_model_settings(
        self, dataset: ContinuousDatasets, settings: InputModelSettings
    ) -> ContinuousModelSettings:
        model_settings = super().get_model_settings(dataset, settings)

        if model_settings.priors.prior_class is PriorClass.frequentist_restricted:
            attr = "min_value" if model_settings.is_increasing else "max_value"
            setattr(model_settings.priors.get_prior("c"), attr, 0.0)

        return model_settings

    def dr_curve(self, doses, params) -> np.ndarray:
        a = params[0]
        b = params[1]
        c = params[2]
        d = params[3]
        return a * (c - (c - 1.0) * np.exp(-1.0 * np.power(b * doses, d)))


bmd_model_map = {
    ContinuousModelIds.c_power.value: Power,
    ContinuousModelIds.c_hill.value: Hill,
    ContinuousModelIds.c_polynomial.value: Polynomial,
    ContinuousModelIds.c_exp_m3.value: ExponentialM3,
    ContinuousModelIds.c_exp_m5.value: ExponentialM5,
}
