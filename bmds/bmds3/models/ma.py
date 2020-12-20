import ctypes

from ..types.dichotomous import (
    DichotomousAnalysis,
    DichotomousModelResult,
    DichotomousModelSettings,
)
from ..types.ma import DichotomousMAAnalysis, DichotomousMAResult
from . import dichotomous as dmodels
from .base import BaseModel, BmdsLibraryManager, InputModelSettings


class DichotomousMA(BaseModel):
    def get_model_settings(self, settings: InputModelSettings) -> DichotomousModelSettings:
        if settings is None:
            model = DichotomousModelSettings()
        elif isinstance(settings, DichotomousModelSettings):
            model = settings
        else:
            model = DichotomousModelSettings.parse_obj(settings)

        return model

    def execute(self, debug=False):
        dll = BmdsLibraryManager.get_dll(bmds_version="BMDS330", base_name="libDRBMD")

        # TODO add more models
        models = [dmodels.DichotomousHill(self.dataset), dmodels.Gamma(self.dataset)]

        analysis = DichotomousAnalysis(
            model=models[0].model,  # not used, needed for init
            dataset=self.dataset,
            priors=models[0].get_priors(),  # not used, needed for init
            BMD_type=self.settings.bmr_type,
            BMR=self.settings.bmr,
            alpha=self.settings.alpha,
            degree=self.settings.degree,
            samples=self.settings.samples,
            burnin=self.settings.burnin,
        )
        analysis_struct = analysis.to_c()

        priors = [model.get_priors() for model in models]

        ma_analysis = DichotomousMAAnalysis(
            models=[model.model for model in models], priors=priors,
        )
        ma_analysis_struct = ma_analysis.to_c()

        dist_numE = 300

        results = []

        for model in models:
            result = DichotomousModelResult(
                model=model.model, dist_numE=dist_numE, num_params=model.model.num_params
            )
            result_struct = result.to_c()
            results.append(result_struct)

        ma_result = DichotomousMAResult(
            results=results, num_models=len(models), dist_numE=dist_numE,
        )
        ma_result_struct = ma_result.to_c()

        dll.estimate_ma_laplace_dicho(
            ctypes.pointer(ma_analysis_struct),
            ctypes.pointer(analysis_struct),
            ctypes.pointer(ma_result_struct),
        )

        # TODO return results
        return None

    def to_dict(self, _):
        return {"results": self.results}
