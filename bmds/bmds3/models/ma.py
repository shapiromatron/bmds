import ctypes
from typing import List

from ...datasets import DichotomousDataset
from ..types.dichotomous import DichotomousModelSettings
from ..types.ma import (
    DichotomousMAAnalysisStruct,
    DichotomousMAResultStruct,
    DichotomousModelAverageResult,
)
from .base import BmdModelAveraging, BmdModelAveragingSchema, BmdsLibraryManager, InputModelSettings


class BmdModelAveragingDichotomous(BmdModelAveraging):
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

    def execute(self) -> DichotomousModelAverageResult:
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

    def serialize(self, session) -> "BmdModelAveragingDichotomousSchema":
        model_indexes = [session.models.index(model) for model in self.models]
        return BmdModelAveragingDichotomousSchema(
            settings=self.settings, model_indexes=model_indexes, results=self.results
        )


class BmdModelAveragingDichotomousSchema(BmdModelAveragingSchema):
    settings: DichotomousModelSettings
    results: DichotomousModelAverageResult
    model_indexes: List[int]

    def deserialize(self, session) -> BmdModelAveragingDichotomous:
        models = [session.models[idx] for idx in self.model_indexes]
        ma = BmdModelAveragingDichotomous(
            dataset=session.dataset, models=models, settings=self.settings
        )
        ma.results = self.results
        return ma
