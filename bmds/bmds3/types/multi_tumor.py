from enum import IntEnum
from textwrap import dedent
from typing import NamedTuple, Self

from pydantic import BaseModel, confloat, conint

from bmds import bmdscore

from ...datasets import MultiTumorDataset
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
    bmr: confloat(gt=0) = 0.1
    alpha: confloat(gt=0, lt=1) = 0.05
    bmr_type: MultiTumorRiskType = MultiTumorRiskType.ExtraRisk
    litter_specific_covariate: MultiTumorLSCType = MultiTumorLSCType.ControlGroupMean
    background: MultiTumorBackgroundType = MultiTumorBackgroundType.Estimated
    bootstrap_iterations: conint(gt=0) = 1
    bootstrap_seed: conint(gt=0) = 0

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

        # if show_degree:
        #     data.append(["Degree", self.degree])

        # if self.priors.is_bayesian:
        #     data.extend((["Samples", self.samples], ["Burn-in", self.burnin]))

        return pretty_table(data, "")


class MultiTumorAnalysis(BaseModel):
    """
    Purpose - Contains all of the information for a multi tumor dichotomous analysis.
    """


    def getMultitumorPrior(degree, prior_cols):
        prG, prB, pr = [],[],[]
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

        doses1,doses2,doses3 = [],[],[]
        Y1,Y2,Y3 = [],[],[]
        n_group1, n_group2, n_group3 = [],[],[]
        doses, Y , n_group = [],[],[]

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
                    models[dataset][count].prior = self.getMultitumorPrior(deg, mt_analysis.prior_cols)
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
                models[dataset][count].prior = self.getMultitumorPrior(degree[dataset], mt_analysis.prior_cols)
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
    analysis: bmdscore.foo
    result: bmdscore.foo

    def execute(self):
        bmdscore.pythonBMDSMultitumor(self.analysis, self.result)

    def __str__(self):
        return dedent(
            f"""
            Analysis:
            {self.analysis}

            Result:
            {self.result}
            """
        )


class MultiTumorResult(BaseModel):
  
    @classmethod
    def from_model(cls, model) -> Self:
        result = model.structs.result
        summary = result.bmdsRes
        # fit = MultiTumorModelResult.from_model(model)
        # gof = MultiTumorPgofResult.from_model(model)
        # parameters = MultiTumorParameters.from_model(model)
        # deviance = MultiTumorAnalysisOfDeviance.from_model(model)
        # plotting = MultiTumorPlotting.from_model(model, parameters.values)
        return cls(
            # bmdl=summary.BMDL,
            bmd=summary.BMD,
            # bmdu=summary.BMDU,
            # has_completed=summary.validResult,
            # fit=fit,
            # gof=gof,
            # parameters=parameters,
            # deviance=deviance,
            # plotting=plotting,
        )

    def text(
        self, dataset: MultiTumorDataset, settings: MultiTumorModelSettings
    ) -> str:
        return multi_lstrip(
            f"""
        Summary:
        {self.tbl()}
        """
        )

    def tbl(self) -> str:
        data = [
            ["BMD", self.bmd],
            # ["BMDL", self.bmdl],
            # ["BMDU", self.bmdu],
            # ["AIC", self.fit.aic],
            # ["Log Likelihood", self.fit.loglikelihood],
            # ["P-Value", self.gof.p_value],
            # ["Overall DOF", self.gof.df],
            # ["Chi²", self.fit.chisq],
        ]
        return pretty_table(data, "")
