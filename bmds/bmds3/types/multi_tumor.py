from enum import IntEnum
from textwrap import dedent
from typing import NamedTuple, Self

from pydantic import BaseModel, Field, confloat, conint

from bmds import bmdscore

from ...datasets import DichotomousDataset
from ...utils import multi_lstrip, pretty_table


class MultiTumorRiskType(IntEnum):
    AddedRisk = 0
    ExtraRisk = 1


class MultiTumorLSCType(IntEnum):
    OverallMean = 0
    ControlGroupMean = 1


class MultiTumorBackgroundType(IntEnum):
    Zero = 0
    Estimated = 1


_bmr_text_map = {
    MultiTumorRiskType.ExtraRisk: "{:.0%} extra risk",
    MultiTumorRiskType.AddedRisk: "{:.0%} added risk",
}


class MultiTumorModelSettings(BaseModel):
    bmr: float = Field(default=0.1, gt=0)
    bmr_type: MultiTumorRiskType = MultiTumorRiskType.ExtraRisk
    alpha: float = Field(default=0.05, gt=0, lt=1)

    def bmr_text(self) -> str:
        return _bmr_text_map[self.bmr_type].format(self.bmr)

    @property
    def confidence_level(self) -> float:
        return 1 - self.alpha

    def tbl(self, show_degree: bool = True) -> str:
        data = [
            ["BMR", self.bmr_text],
            ["Confidence Level", self.confidence_level],
        ]
        return pretty_table(data, "")


class MultiTumorAnalysis2(BaseModel):
    """
    Purpose - Contains all of the information for a multi tumor dichotomous analysis.
    """

    def getMultitumorPrior(degree, prior_cols):
        prG, prB, pr = [], [], []
        for i in range(prior_cols):
            pr.append(prG[i])
            for j in range(degree):
                pr.append(prB[i])
        return pr

    def to_cpp(Self):
        """_summary_

        Returns:
            _type_: _description_
        """

        doses1, doses2, doses3 = [], [], []
        Y1, Y2, Y3 = [], [], []
        n_group1, n_group2, n_group3 = [], [], []
        doses, Y, n_group = [], [], []

        doses.append(doses1)
        doses.append(doses2)
        doses.append(doses3)
        Y.append(Y1)
        Y.append(Y2)
        Y.append(Y3)
        n_group.append(n_group1)
        n_group.append(n_group2)
        n_group.append(n_group3)

        dist_numE = 200
        ndatasets = 3
        BMR = 0.1
        BMD_type = 1
        alpha = 0.05
        prior_cols = 5

        n = []
        degree = []
        mt_analysis = bmdscore.python_multitumor_analysis()
        mt_analysis.ndatasets = ndatasets
        mt_analysis.n = n
        mt_analysis.degree = degree
        mt_analysis.BMR = BMR
        mt_analysis.BMD_type = BMD_type
        mt_analysis.alpha = alpha
        mt_analysis.prior_cols = prior_cols

        pyRes = bmdscore.python_multitumor_result()
        pyRes.ndatasets = ndatasets

        models = []
        resModels = []
        nmodels = []
        for dataset in range(mt_analysis.ndatasets):
            models.append([])
            resModels.append([])
            count = 0
            if degree[dataset] == 0:
                for deg in range(2, mt_analysis.n[dataset]):
                    models[dataset].append(bmdscore.python_dichotomous_analysis())
                    models[dataset][count].model = bmdscore.dich_model.d_multistage
                    models[dataset][count].prior = self.getMultitumorPrior(
                        deg, mt_analysis.prior_cols
                    )
                    models[dataset][count].degree = deg
                    models[dataset][count].parms = deg + 1
                    models[dataset][count].Y = Y[dataset]
                    models[dataset][count].n_group = n_group[dataset]
                    models[dataset][count].doses = doses[dataset]
                    models[dataset][count].n = len(Y[dataset])
                    models[dataset][count].BMR = BMR
                    models[dataset][count].BMD_type = BMD_type
                    models[dataset][count].alpha = alpha
                    models[dataset][count].prior_cols = prior_cols
                    resModels[dataset].append(bmdscore.python_dichotomous_model_result())
                    resModels[dataset][count].model = bmdscore.dich_model.d_multistage
                    resModels[dataset][count].nparms = deg + 1
                    resModels[dataset][count].dist_numE = dist_numE
                    gof = bmdscore.dichotomous_GOF()
                    bmdsRes = bmdscore.BMDS_results()
                    aod = bmdscore.dicho_AOD()
                    resModels[dataset][count].gof = gof
                    resModels[dataset][count].bmdsRes = bmdsRes
                    resModels[dataset][count].aod = aod
                    count = count + 1
            else:
                models[dataset].append(bmdscore.python_dichotomous_analysis())
                models[dataset][count].model = bmdscore.dich_model.d_multistage
                models[dataset][count].prior = self.getMultitumorPrior(
                    degree[dataset], mt_analysis.prior_cols
                )
                models[dataset][count].degree = degree[dataset]
                models[dataset][count].parms = degree[dataset] + 1
                models[dataset][count].Y = Y[dataset]
                models[dataset][count].n_group = n_group[dataset]
                models[dataset][count].doses = doses[dataset]
                models[dataset][count].n = len(Y[dataset])
                models[dataset][count].BMR = BMR
                models[dataset][count].BMD_type = BMD_type
                models[dataset][count].alpha = alpha
                models[dataset][count].prior_cols = prior_cols
                resModels[dataset].append(bmdscore.python_dichotomous_model_result())
                resModels[dataset][count].model = bmdscore.dich_model.d_multistage
                resModels[dataset][count].nparms = degree[dataset] + 1
                resModels[dataset][count].dist_numE = dist_numE
                gof = bmdscore.dichotomous_GOF()
                bmdsRes = bmdscore.BMDS_results()
                aod = bmdscore.dicho_AOD()
                resModels[dataset][count].gof = gof
                resModels[dataset][count].bmdsRes = bmdsRes
                resModels[dataset][count].aod = aod
                count = 1
            nmodels.append(count)

        mt_analysis.models = models
        pyRes.models = resModels
        mt_analysis.nmodels = nmodels
        pyRes.nmodels = nmodels

        bmdscore.pythonBMDSMultitumor(mt_analysis, pyRes)

        return MultiTumorAnalysisCPPStructs(mt_analysis, pyRes)


class MultiTumorAnalysisCPPStructs(NamedTuple):
    analysis: bmdscore.python_dichotomousMA_analysis
    result: bmdscore.python_dichotomousMA_result

    def execute(self):
        bmdscore.pythonBMDSMultitumor(self.analysis, self.result)


class MultitumorAnalysis(BaseModel):
    BMD_type: int
    BMR: float
    alpha: float
    degree: list[int]
    # models: list[list[python_dichotomous_analysis]]
    n: list[int]
    ndatasets: int
    nmodels: list[int]
    prior: list[list[float]]
    prior_cols: int


class MultitumorResult(BaseModel):
    bmd: float
    bmdl: float
    bmdu: float
    ll: float
    ll_const: float
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
            ll_const=result.combined_LL_const,
            selected_model_index=result.selectedModelIndex,
            slope_factor=result.slopeFactor,
            valid_result=result.validResult,
        )
