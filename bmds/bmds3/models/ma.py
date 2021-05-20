import ctypes
from typing import List

from ...datasets import DichotomousDataset
from ..types.dichotomous import DichotomousModelSettings
from ..types.ma import DichotomousModelAverageResult
from ..types.structs import DichotomousMAAnalysisStruct, MAResultsStruct, DichotomousMAResultStruct
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
            models=[model.structs.analysis for model in self.models]
        )
        ma_inputs_struct = self.models[0].structs.analysis
        dich_ma_result_struct = DichotomousMAResultStruct.from_python(
            models=[model.structs.result for model in self.models]
        )
        ma_result_struct = MAResultsStruct(n_models=len(self.models))

        dll.runBMDSDichoMA(
            ctypes.pointer(ma_analysis_struct),
            ctypes.pointer(ma_inputs_struct),
            ctypes.pointer(dich_ma_result_struct),
            ctypes.pointer(ma_result_struct),
        )

        model_results = [model.results for model in self.models]
        return DichotomousModelAverageResult.from_execution(
            ma_analysis_struct, dich_ma_result_struct, model_results, ma_result_struct
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
