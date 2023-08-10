from typing import NamedTuple, Self

from pydantic import BaseModel

from ... import bmdscore
from ...utils import multi_lstrip, pretty_table


class MultitumorAnalysis(NamedTuple):
    analysis: bmdscore.python_multitumor_analysis
    result: bmdscore.python_multitumor_result

    def execute(self):
        bmdscore.pythonBMDSMultitumor(self.analysis, self.result)


class MultitumorConfig(BaseModel):
    x: int


class MultitumorResult(BaseModel):
    bmd: float
    bmdl: float
    bmdu: float
    ll: float
    ll_constant: float
    # models: list[list[python_dichotomous_model_result]]  # all degrees for all datasets
    selected_model_index: list[int]
    slope_factor: float
    valid_result: list[bool]

    @classmethod
    def from_model(cls, model) -> Self:
        result: bmdscore.python_multitumor_result = model.structs.result
        return cls(
            bmd=result.BMD,
            bmdl=result.BMDL,
            bmdu=result.BMDU,
            ll=result.combined_LL,
            ll_constant=result.combined_LL_const,
            selected_model_index=result.selectedModelIndex,
            slope_factor=result.slopeFactor,
            valid_result=result.validResult,
        )

    def text(self) -> str:
        return multi_lstrip(
            f"""
        Summary:
        {self.tbl()}
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
