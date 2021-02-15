import ctypes
from textwrap import dedent
from typing import List, NamedTuple

import numpy as np

from .. import constants
from .common import list_t_c

# DICHOTOMOUS MODELS
# ------------------


class DichotomousAnalysisStruct(ctypes.Structure):

    _fields_ = [
        ("model", ctypes.c_int),  # Model Type as listed in DichModel
        ("n", ctypes.c_int),  # total number of observations obs/n
        ("Y", ctypes.POINTER(ctypes.c_double)),  # observed +
        ("doses", ctypes.POINTER(ctypes.c_double)),
        ("n_group", ctypes.POINTER(ctypes.c_double)),  # size of the group
        ("prior", ctypes.POINTER(ctypes.c_double)),  # a column order matrix parms X prior_cols
        ("BMD_type", ctypes.c_int),  # 1 = extra ; added otherwise
        ("BMR", ctypes.c_double),
        ("alpha", ctypes.c_double),  # alpha of the analysis
        ("degree", ctypes.c_int),  # degree of polynomial used only multistage
        ("samples", ctypes.c_int),  # number of MCMC samples
        ("burnin", ctypes.c_int),  # size of burnin
        ("parms", ctypes.c_int),  # number of parameters in the model
        ("prior_cols", ctypes.c_int),  # columns in the prior
    ]

    def __str__(self) -> str:
        return dedent(
            f"""
            model: {self.model}
            n: {self.n}
            Y: {self.Y[:self.n]}
            doses: {self.doses[:self.n]}
            n_group: {self.n_group[:self.n]}
            prior: {self.prior[:self.parms*self.prior_cols]}
            BMD_type: {self.BMD_type}
            BMR: {self.BMR}
            alpha: {self.alpha}
            degree: {self.degree}
            samples: {self.samples}
            burnin: {self.burnin}
            parms: {self.parms}
            prior_cols: {self.prior_cols}
            """
        )


class DichotomousModelResultStruct(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_int),  # dichotomous model specification
        ("nparms", ctypes.c_int),  # number of parameters in the model
        ("parms", ctypes.POINTER(ctypes.c_double)),  # parameter estimate
        ("cov", ctypes.POINTER(ctypes.c_double)),  # covariance estimate
        ("max", ctypes.c_double),  # value of the likelihood/posterior at the maximum
        ("dist_numE", ctypes.c_int),  # number of entries in rows for the bmd_dist
        ("model_df", ctypes.c_double),  # Used model degrees of freedom
        ("total_df", ctypes.c_double),  # Total degrees of freedom
        ("bmd_dist", ctypes.POINTER(ctypes.c_double),),  # bmd distribution (dist_numE x 2) matrix
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # reference same memory in struct and numpy
        # https://stackoverflow.com/a/23330369/906385
        self.np_parms = np.zeros(kwargs["nparms"], dtype=np.float64)
        self.parms = np.ctypeslib.as_ctypes(self.np_parms)
        self.np_cov = np.zeros(kwargs["nparms"] ** 2, dtype=np.float64)
        self.cov = np.ctypeslib.as_ctypes(self.np_cov)
        self.np_bmd_dist = np.zeros(kwargs["dist_numE"] * 2, dtype=np.float64)
        self.bmd_dist = np.ctypeslib.as_ctypes(self.np_bmd_dist)

    def __str__(self) -> str:
        return dedent(
            f"""
            model: {self.model}
            nparms: {self.nparms}
            parms: {self.parms[:self.nparms]}
            cov: {self.cov[:self.nparms**2]}
            max: {self.max}
            dist_numE: {self.dist_numE}
            model_df: {self.model_df}
            total_df: {self.total_df}
            bmd_dist: {self.bmd_dist[:self.dist_numE*2]}
            """
        )


class DichotomousPgofResultStruct(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_int),  # total number of observations obs/n
        ("expected", ctypes.POINTER(ctypes.c_double)),
        ("residual", ctypes.POINTER(ctypes.c_double)),
        ("test_statistic", ctypes.c_double),
        ("p_value", ctypes.c_double),
        ("df", ctypes.c_double),
    ]

    def __str__(self) -> str:
        return dedent(
            f"""
            n: {self.n}
            expected: {self.expected[:self.n]}
            residual: {self.residual[:self.n]}
            test_statistic: {self.test_statistic}
            p_value: {self.p_value}
            df: {self.df}
            """
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.np_expected = np.zeros(self.n, dtype=np.float64)
        self.expected = np.ctypeslib.as_ctypes(self.np_expected)
        self.np_residual = np.zeros(self.n, dtype=np.float64)
        self.residual = np.ctypeslib.as_ctypes(self.np_residual)


class DichotomousAodStruct(ctypes.Structure):

    _fields_ = [
        ("fullLL", ctypes.c_double),
        ("nFull", ctypes.c_int),
        ("redLL", ctypes.c_double),
        ("nRed", ctypes.c_int),
        ("fittedLL", ctypes.c_double),
        ("nFit", ctypes.c_int),
        ("devFit", ctypes.c_double),
        ("devRed", ctypes.c_double),
        ("dfFit", ctypes.c_int),
        ("dfRed", ctypes.c_int),
        ("pvFit", ctypes.c_int),
        ("pvRed", ctypes.c_int),
    ]

    def __str__(self) -> str:
        return dedent(
            f"""
            fullLL: {self.fullLL}
            nFull: {self.nFull}
            redLL: {self.redLL}
            nRed: {self.nRed}
            fittedLL: {self.fittedLL}
            nFit: {self.nFit}
            devFit: {self.devFit}
            devRed: {self.devRed}
            dfFit: {self.dfFit}
            dfRed: {self.dfRed}
            pvFit: {self.pvFit}
            pvRed: {self.pvRed}
            """
        )


class DichotomousBmdsResultsStruct(ctypes.Structure):
    _fields_ = [
        ("bmd", ctypes.c_double),
        ("bmdl", ctypes.c_double),
        ("bmdu", ctypes.c_double),
        ("aic", ctypes.c_double),
        ("chisq", ctypes.c_double),
        ("bounded", ctypes.POINTER(ctypes.c_bool)),
    ]

    def __str__(self) -> str:
        return dedent(
            f"""
            bmd: {self.bmd}
            bmdl: {self.bmdl}
            bmdu: {self.bmdu}
            aic: {self.aic}
            bounded: {self.bounded[:self.n]}
            """
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = kwargs["num_params"]
        self.bmd = constants.BMDS_BLANK_VALUE
        self.bmdl = constants.BMDS_BLANK_VALUE
        self.bmdu = constants.BMDS_BLANK_VALUE
        self.aic = constants.BMDS_BLANK_VALUE
        self.np_bounded = np.zeros(self.n, dtype=np.bool_)
        self.bounded = np.ctypeslib.as_ctypes(self.np_bounded)


class DichotomousStructs(NamedTuple):
    analysis: DichotomousAnalysisStruct
    result: DichotomousModelResultStruct
    gof: DichotomousPgofResultStruct
    summary: DichotomousBmdsResultsStruct
    aod: DichotomousAodStruct

    def __str__(self):
        return dedent(
            f"""
            Analysis:
            {self.analysis}

            Result:
            {self.result}

            GoF:
            {self.gof}

            Summary:
            {self.summary}

            AoD:
            {self.aod}
            """
        )


# DICHOTOMOUS MODEL AVERAGING
# ---------------------------


class DichotomousMAAnalysisStruct(ctypes.Structure):
    _fields_ = [
        ("nmodels", ctypes.c_int),
        ("priors", ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),),
        ("nparms", ctypes.POINTER(ctypes.c_int)),
        ("actual_parms", ctypes.POINTER(ctypes.c_int)),
        ("prior_cols", ctypes.POINTER(ctypes.c_int),),
        ("models", ctypes.POINTER(ctypes.c_int)),
        ("modelPriors", ctypes.POINTER(ctypes.c_double)),
    ]

    @classmethod
    def from_python(cls, models: List[DichotomousAnalysisStruct]):

        # list of floats
        priors = [
            list_t_c(model.prior[: model.parms * model.prior_cols], ctypes.c_double,)
            for model in models
        ]

        # pointer of pointers
        priors2 = list_t_c(
            [ctypes.cast(el, ctypes.POINTER(ctypes.c_double)) for el in priors],
            ctypes.POINTER(ctypes.c_double),
        )

        return cls(
            nmodels=ctypes.c_int(len(models)),
            priors=priors2,
            nparms=list_t_c([model.parms for model in models], ctypes.c_int),
            actual_parms=list_t_c([model.parms for model in models], ctypes.c_int),
            prior_cols=list_t_c([model.prior_cols for model in models], ctypes.c_int),
            models=list_t_c([model.model for model in models], ctypes.c_int),
            modelPriors=list_t_c([1 / len(models)] * len(models), ctypes.c_double),
        )


class DichotomousMAResultStruct(ctypes.Structure):
    _fields_ = [
        ("nmodels", ctypes.c_int),
        ("models", ctypes.POINTER(ctypes.POINTER(DichotomousModelResultStruct))),
        ("dist_numE", ctypes.c_int),
        ("post_probs", ctypes.POINTER(ctypes.c_double)),
        ("bmd_dist", ctypes.POINTER(ctypes.c_double)),
    ]

    @classmethod
    def from_python(cls, models: List[DichotomousModelResultStruct]):
        _results = [ctypes.pointer(model) for model in models]
        nmodels = len(models)
        dist_numE = 200
        return DichotomousMAResultStruct(
            nmodels=nmodels,
            models=list_t_c(_results, ctypes.POINTER(DichotomousModelResultStruct)),
            dist_numE=ctypes.c_int(dist_numE),
            post_probs=(ctypes.c_double * nmodels)(),
            bmd_dist=(ctypes.c_double * (dist_numE * 2))(),
        )


# CONTINUOUS MODELS
# -----------------


class ContinuousAnalysisStruct(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_int),
        ("n", ctypes.c_int),
        ("suff_stat", ctypes.c_bool),  # true if continuous summary, false if individual data
        ("Y", ctypes.POINTER(ctypes.c_double)),  # observed data means or actual data
        ("doses", ctypes.POINTER(ctypes.c_double)),
        (
            "sd",
            ctypes.POINTER(ctypes.c_double),
        ),  # SD of the group if suff_stat = true, null otherwise
        (
            "n_group",
            ctypes.POINTER(ctypes.c_double),
        ),  # N for each group if suff_stat = true, null otherwise
        (
            "prior",
            ctypes.POINTER(ctypes.c_double),
        ),  # a column order matrix px5 where p is the number of parameters
        ("BMD_type", ctypes.c_int),  # type of BMD
        ("isIncreasing", ctypes.c_bool),  # if the BMD is defined increasing or decreasing
        ("BMR", ctypes.c_double),  # benchmark response related to the BMD type
        ("tail_prob", ctypes.c_double),  # tail probability
        ("disttype", ctypes.c_int),  # distribution type defined in the enum distribution
        ("alpha", ctypes.c_double),  # specified alpha
        ("samples", ctypes.c_int),  # number of MCMC samples
        ("degree", ctypes.c_int),
        ("burnin", ctypes.c_int),
        ("parms", ctypes.c_int),  # number of parameters
        ("prior_cols", ctypes.c_int),
    ]

    def __str__(self) -> str:
        sd = self.sd[: self.n] if self.suff_stat else []
        n_group = self.n_group[: self.n] if self.suff_stat else []
        return dedent(
            f"""
            model: {self.model}
            n: {self.n}
            suff_stat: {self.suff_stat}
            Y: {self.Y[:self.n]}
            doses: {self.doses[:self.n]}
            sd: {sd}
            n_group: {n_group}
            prior: {self.prior[:self.parms*self.prior_cols]}
            BMD_type: {self.BMD_type}
            isIncreasing: {self.isIncreasing}
            BMR: {self.BMR}
            tail_prob: {self.tail_prob}
            disttype: {self.disttype}
            alpha: {self.alpha}
            samples: {self.samples}
            degree: {self.degree}
            burnin: {self.burnin}
            parms: {self.parms}
            prior_cols: {self.prior_cols}
            """
        )


class ContinuousModelResultStruct(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_int),  # continuous model specification
        ("dist", ctypes.c_int),  # distribution type
        ("nparms", ctypes.c_int),  # number of parameters in the model
        ("parms", ctypes.POINTER(ctypes.c_double)),  # parameter estimate
        ("cov", ctypes.POINTER(ctypes.c_double)),  # covariance estimate
        ("max", ctypes.c_double),  # value of the likelihood/posterior at the maximum
        ("dist_numE", ctypes.c_int),  # number of entries in rows for the bmd_dist
        ("model_df", ctypes.c_double),
        ("total_df", ctypes.c_double),
        ("bmd_dist", ctypes.POINTER(ctypes.c_double),),  # bmd distribution (dist_numE x 2) matrix
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # this is modified by Exp3
        self.initial_n = self.nparms
        # reference same memory in struct and numpy
        # https://stackoverflow.com/a/23330369/906385
        self.np_parms = np.zeros(kwargs["nparms"], dtype=np.float64)
        self.parms = np.ctypeslib.as_ctypes(self.np_parms)
        self.np_cov = np.zeros(kwargs["nparms"] ** 2, dtype=np.float64)
        self.cov = np.ctypeslib.as_ctypes(self.np_cov)
        self.np_bmd_dist = np.zeros(kwargs["dist_numE"] * 2, dtype=np.float64)
        self.bmd_dist = np.ctypeslib.as_ctypes(self.np_bmd_dist)

    def __str__(self) -> str:
        return dedent(
            f"""
            model: {self.model}
            dist: {self.dist}
            nparms: {self.nparms}
            nparms <initial>: {self.initial_n}
            parms: {self.parms[:self.initial_n]}
            cov: {self.cov[:self.initial_n**2]}
            max: {self.max}
            dist_numE: {self.dist_numE}
            model_df: {self.model_df}
            total_df: {self.total_df}
            bmd_dist: {self.bmd_dist[:self.dist_numE*2]}
            """
        )


class ContinuousGofStruct(ctypes.Structure):
    _fields_ = [
        ("dose", ctypes.POINTER(ctypes.c_double)),
        ("size", ctypes.POINTER(ctypes.c_double)),
        ("estMean", ctypes.POINTER(ctypes.c_double)),
        ("calcMean", ctypes.POINTER(ctypes.c_double)),
        ("obsMean", ctypes.POINTER(ctypes.c_double)),
        ("estSD", ctypes.POINTER(ctypes.c_double)),
        ("calcSD", ctypes.POINTER(ctypes.c_double)),
        ("obsSD", ctypes.POINTER(ctypes.c_double)),
        ("res", ctypes.POINTER(ctypes.c_double)),
        ("n", ctypes.c_int),
    ]

    def __str__(self) -> str:
        return dedent(
            f"""
            dose: {self.dose[:self.n]}
            size: {self.size[:self.n]}
            estMean: {self.estMean[:self.n]}
            calcMean: {self.calcMean[:self.n]}
            obsMean: {self.obsMean[:self.n]}
            estSD: {self.estSD[:self.n]}
            calcSD: {self.calcSD[:self.n]}
            obsSD: {self.obsSD[:self.n]}
            res: {self.res[:self.n]}
            n: {self.n}
            """
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.np_dose = np.zeros(self.n, dtype=np.float64)
        self.dose = np.ctypeslib.as_ctypes(self.np_dose)
        self.np_size = np.zeros(self.n, dtype=np.float64)
        self.size = np.ctypeslib.as_ctypes(self.np_size)
        self.np_estMean = np.zeros(self.n, dtype=np.float64)
        self.estMean = np.ctypeslib.as_ctypes(self.np_estMean)
        self.np_calcMean = np.zeros(self.n, dtype=np.float64)
        self.calcMean = np.ctypeslib.as_ctypes(self.np_calcMean)
        self.np_obsMean = np.zeros(self.n, dtype=np.float64)
        self.obsMean = np.ctypeslib.as_ctypes(self.np_obsMean)
        self.np_estSD = np.zeros(self.n, dtype=np.float64)
        self.estSD = np.ctypeslib.as_ctypes(self.np_estSD)
        self.np_calcSD = np.zeros(self.n, dtype=np.float64)
        self.calcSD = np.ctypeslib.as_ctypes(self.np_calcSD)
        self.np_obsSD = np.zeros(self.n, dtype=np.float64)
        self.obsSD = np.ctypeslib.as_ctypes(self.np_obsSD)
        self.np_res = np.zeros(self.n, dtype=np.float64)
        self.res = np.ctypeslib.as_ctypes(self.np_res)


class ContinuousToiStruct(ctypes.Structure):
    _fields_ = [
        ("llRatio", ctypes.POINTER(ctypes.c_double)),
        ("DF", ctypes.POINTER(ctypes.c_double)),
        ("pVal", ctypes.POINTER(ctypes.c_double)),
    ]

    def __str__(self) -> str:
        return dedent(
            f"""
            llRatio: {self.llRatio[:4]}
            DF: {self.DF[:4]}
            pVal: {self.pVal[:4]}
            """
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.np_llRatio = np.zeros(4, dtype=np.float64)
        self.llRatio = np.ctypeslib.as_ctypes(self.np_llRatio)
        self.np_DF = np.zeros(4, dtype=np.float64)
        self.DF = np.ctypeslib.as_ctypes(self.np_DF)
        self.np_pVal = np.zeros(4, dtype=np.float64)
        self.pVal = np.ctypeslib.as_ctypes(self.np_pVal)


class ContinuousAodStruct(ctypes.Structure):
    _fields_ = [
        ("LL", ctypes.POINTER(ctypes.c_double)),
        ("nParms", ctypes.POINTER(ctypes.c_double)),
        ("AIC", ctypes.POINTER(ctypes.c_double)),
        ("addConst", ctypes.c_double),
        ("TOI", ctypes.POINTER(ContinuousToiStruct)),
    ]

    def __str__(self) -> str:
        return (
            dedent(
                f"""
                LL: {self.LL[:5]}
                nParms: {self.nParms[:5]}
                AIC: {self.AIC[:5]}
                addConst: {self.addConst}
                """
            )
            + str(self.TOI[0])
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.np_LL = np.zeros(5, dtype=np.float64)
        self.LL = np.ctypeslib.as_ctypes(self.np_LL)
        self.np_nParms = np.zeros(5, dtype=np.float64)
        self.nParms = np.ctypeslib.as_ctypes(self.np_nParms)
        self.np_AIC = np.zeros(5, dtype=np.float64)
        self.AIC = np.ctypeslib.as_ctypes(self.np_AIC)
        self.TOI = ctypes.pointer(ContinuousToiStruct())


class ContinuousBmdsResultsStruct(ctypes.Structure):
    _fields_ = [
        ("bmd", ctypes.c_double),
        ("bmdl", ctypes.c_double),
        ("bmdu", ctypes.c_double),
        ("aic", ctypes.c_double),
        ("chisq", ctypes.c_double),
        ("bounded", ctypes.POINTER(ctypes.c_bool)),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = kwargs["nparms"]
        self.bmd = constants.BMDS_BLANK_VALUE
        self.bmdl = constants.BMDS_BLANK_VALUE
        self.bmdu = constants.BMDS_BLANK_VALUE
        self.aic = constants.BMDS_BLANK_VALUE
        self.np_bounded = np.zeros(self.n, dtype=np.bool_)
        self.bounded = np.ctypeslib.as_ctypes(self.np_bounded)

    def __str__(self) -> str:
        return dedent(
            f"""
            bmd: {self.bmd}
            bmdl: {self.bmdl}
            bmdu: {self.bmdu}
            aic: {self.aic}
            bounded: {self.bounded[:self.n]}
            """
        )


class ContinuousStructs(NamedTuple):
    analysis: ContinuousAnalysisStruct
    result: ContinuousModelResultStruct
    summary: ContinuousBmdsResultsStruct
    aod: ContinuousAodStruct
    gof: ContinuousGofStruct

    def __str__(self):
        return dedent(
            f"""
            Analysis:
            {self.analysis}

            Result:
            {self.result}

            Summary:
            {self.summary}

            AoD:
            {self.aod}

            GoF:
            {self.gof}
            """
        )
