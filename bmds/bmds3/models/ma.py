import ctypes
from typing import Dict

from ..types.dichotomous import DichotomousModelSettings
from ..types.ma import (
    DichotomousMAAnalysisStruct,
    DichotomousMAResultStruct,
    DichotomousModelAverageResult,
)
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

        ma_analysis_struct = DichotomousMAAnalysisStruct.from_python(
            models=[model.inputs_struct for model in self.models]
        )
        ma_inputs_struct = self.models[0].inputs_struct
        ma_result_struct = DichotomousMAResultStruct.from_python(
            models=[model.fit_results_struct for model in self.models]
        )

        dll.estimate_ma_laplace_dicho(
            ctypes.pointer(ma_analysis_struct),
            ctypes.pointer(ma_inputs_struct),
            ctypes.pointer(ma_result_struct),
        )

        return DichotomousModelAverageResult.from_execution(
            ma_analysis_struct, ma_result_struct, self.models
        )

    def to_dict(self, model_index: int) -> Dict:
        """
        Return a summary of the model in a dictionary format for serialization.

        Args:
            model_index (int): numeric model index in a list of models, should be unique

        Returns:
            A dictionary of model inputs, and raw and parsed outputs
        """
        return dict(
            model_index=model_index,
            model_class=-1,
            model_name="Model average",
            model_version=self.model_version,
            has_output=self.output_created,
            execution_halted=self.execution_halted,
            settings=self.settings.dict(),
            results=self.results.dict() if self.results else None,
        )
