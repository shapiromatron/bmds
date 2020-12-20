import ctypes

from ..types.dichotomous import (
    DichotomousAnalysis,
    DichotomousModelSettings,
)
from ..types.ma import DichotomousMAAnalysisStruct, DichotomousMAResult
from .base import BaseModelAveraging, BmdsLibraryManager, InputModelSettings


class DichotomousMA(BaseModelAveraging):
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

        # assumes inputs are the same for first model as the inputs for this analysis
        analysis_struct = self.models[0].get_analysis_inputs().to_c()

        ma_analysis_struct = DichotomousMAAnalysisStruct.from_python(
            models=[model.model for model in self.models],
            priors=[model.get_priors() for model in self.models],
        )

        ma_result = DichotomousMAResult(
            results=[model.fit_results_struct for model in self.models],
            num_models=len(self.models),
            dist_numE=200,
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
