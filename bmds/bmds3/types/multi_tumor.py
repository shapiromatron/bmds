from typing import NamedTuple, Self

from pydantic import BaseModel, Field

from ... import bmdscore
from ...datasets.dichotomous import DichotomousDatasetSchema
from ...utils import multi_lstrip, pretty_table
from .dichotomous import (
    DichotomousAnalysisCPPStructs,
    DichotomousModelSettings,
    DichotomousResult,
    DichotomousRiskType,
)
from .sessions import VersionSchema


class MultitumorAnalysis(NamedTuple):
    analysis: bmdscore.python_multitumor_analysis
    result: bmdscore.python_multitumor_result

    def execute(self):
        bmdscore.pythonBMDSMultitumor(self.analysis, self.result)


class MultitumorSettings(BaseModel):
    degrees: list[int]
    bmr: float = Field(default=0.1, gt=0)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    bmr_type: DichotomousRiskType = DichotomousRiskType.ExtraRisk


class MultitumorResult(BaseModel):
    bmd: float
    bmdl: float
    bmdu: float
    ll: float
    ll_constant: float
    models: list[list[DichotomousResult]]  # all degrees for all datasets
    settings: list[list[DichotomousModelSettings]]
    selected_model_indexes: list[int]
    slope_factor: float
    valid_result: list[bool]

    @classmethod
    def from_model(cls, model) -> Self:
        result: bmdscore.python_multitumor_result = model.structs.result
        i_models = []
        i_settings = []
        for i, models in enumerate(model.models):
            j_models = []
            j_settings = []
            i_models.append(j_models)
            i_settings.append(j_settings)
            for j, m in enumerate(models):
                m.structs = DichotomousAnalysisCPPStructs(
                    analysis=model.structs.analysis.models[i][j],
                    result=model.structs.result.models[i][j],
                )
                m.results = DichotomousResult.from_model(m)
                j_models.append(m.results)
                j_settings.append(m.settings)
        return cls(
            bmd=result.BMD,
            bmdl=result.BMDL,
            bmdu=result.BMDU,
            ll=result.combined_LL,
            ll_constant=result.combined_LL_const,
            models=i_models,
            settings=i_settings,
            selected_model_indexes=result.selectedModelIndex,
            slope_factor=result.slopeFactor,
            valid_result=result.validResult,
        )

    def text(self, datasets, models) -> str:
        texts = []
        for i, dataset in enumerate(datasets):
            model_idx = self.selected_model_indexes[i]
            texts.append("\n" + dataset._get_dataset_name() + "\n" + "═" * 80)
            texts.append("\n" + dataset.tbl() + "\n")
            texts.append(models[i][model_idx].text())
        fitted = "\n".join(texts)

        return multi_lstrip(
            f"""
        Summary:
        {self.tbl()}

        {fitted}
        """
        )

    def tbl(self) -> str:
        data = [
            ["BMD", self.bmd],
            ["BMDL", self.bmdl],
            ["BMDU", self.bmdu],
            ["Slope Factor", self.slope_factor],
            ["Combined Log-likelihood", self.ll],
            ["Combined Log-likelihood Constant", self.ll_constant],
        ]
        return pretty_table(data, "")


class MultitumorSchema(BaseModel):
    version: VersionSchema
    id: int | str | None
    datasets: list[DichotomousDatasetSchema]
    settings: MultitumorSettings
    results: MultitumorResult | None
