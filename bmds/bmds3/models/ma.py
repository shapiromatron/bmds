import ctypes
from typing import List

from ..types.dichotomous import DichotomousModelSettings
from ..types.ma import DichotomousModelAverageResult
from ..types.structs import DichotomousMAStructs
from .base import BmdModelAveraging, BmdModelAveragingSchema, BmdsLibraryManager, InputModelSettings


class BmdModelAveragingDichotomous(BmdModelAveraging):
    def get_model_settings(self, settings: InputModelSettings) -> DichotomousModelSettings:
        if settings is None:
            return DichotomousModelSettings()
        elif isinstance(settings, DichotomousModelSettings):
            return settings
        else:
            return DichotomousModelSettings.parse_obj(settings)

    def execute(self) -> DichotomousModelAverageResult:
        structs = DichotomousMAStructs.from_session(
            self.session.dataset, self.models, self.session.ma_weights
        )
        self.structs = structs

        dll = BmdsLibraryManager.get_dll(bmds_version="BMDS330", base_name="libDRBMD")
        dll.runBMDSDichoMA(
            ctypes.pointer(structs.analysis),
            ctypes.pointer(structs.inputs),
            ctypes.pointer(structs.dich_result),
            ctypes.pointer(structs.result),
        )
        self.results = DichotomousModelAverageResult.from_structs(
            structs, [model.results for model in self.models]
        )
        return self.results

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
        ma = BmdModelAveragingDichotomous(session=session, models=models, settings=self.settings)
        ma.results = self.results
        return ma
