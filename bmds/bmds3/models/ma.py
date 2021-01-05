import ctypes

from ...datasets import DichotomousDataset
from ..types.dichotomous import DichotomousModelSettings
from ..types.ma import (
    DichotomousMAAnalysisStruct,
    DichotomousMAResultStruct,
    DichotomousModelAverageResult,
)
from .base import BaseModelAveraging, BmdsLibraryManager, InputModelSettings


class DichotomousMA(BaseModelAveraging):
    def get_model_settings(
        self, dataset: DichotomousDataset, settings: InputModelSettings
    ) -> DichotomousModelSettings:
        if settings is None:
            model = DichotomousModelSettings()
        elif isinstance(settings, DichotomousModelSettings):
            model = settings
        else:
            model = DichotomousModelSettings.parse_obj(settings)

        return model

    def execute(self, session, debug=False):
        dll = BmdsLibraryManager.get_dll(bmds_version="BMDS330", base_name="libDRBMD")

        models = [session.models[idx] for idx in self.models]

        ma_analysis_struct = DichotomousMAAnalysisStruct.from_python(
            models=[model.inputs_struct for model in models]
        )
        ma_inputs_struct = models[0].inputs_struct
        ma_result_struct = DichotomousMAResultStruct.from_python(
            models=[model.fit_results_struct for model in models]
        )

        dll.estimate_ma_laplace_dicho(
            ctypes.pointer(ma_analysis_struct),
            ctypes.pointer(ma_inputs_struct),
            ctypes.pointer(ma_result_struct),
        )

        return DichotomousModelAverageResult.from_execution(
            ma_analysis_struct, ma_result_struct, models
        )
