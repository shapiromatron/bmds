import ctypes
from textwrap import dedent
from typing import NamedTuple, Self

import numpy as np
import numpy.typing as npt

from .. import constants
from .common import list_t_c


def get_version(dll: ctypes.CDLL) -> str:
    """Get the version from the bmds shared object"""
    buffer = ctypes.create_string_buffer(32)
    dll.version(ctypes.pointer(buffer))
    return buffer.value.decode("utf8")


# DICHOTOMOUS MODELS
# ------------------
class DichotomousAnalysisStruct(ctypes.Structure):
    _fields_ = (
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
    )

    def __init__(self, *args, **kwargs):
        self.np_Y = np.array(kwargs.pop("Y"), dtype=np.double)
        self.np_doses = np.array(kwargs.pop("doses"), dtype=np.double)
        self.np_n_group = np.array(kwargs.pop("n_group"), dtype=np.double)
        self.np_prior = kwargs.pop("prior")
        super().__init__(
            *args,
            Y=self.np_Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            doses=self.np_doses.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n_group=self.np_n_group.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            prior=self.np_prior.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            **kwargs,
        )

    def __str__(self) -> str:
        txt = dedent(
            f"""
            model: {self.model}
            n: {self.n}
            Y: {self.Y[:self.n]}
            doses: {self.doses[:self.n]}
            n_group: {self.n_group[:self.n]}
            prior<{self.parms},{self.prior_cols}>:
            <PRIOR>
            priors <unprocessed>: {", ".join(str(v) for v in self.np_prior.tolist())}
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
        txt = txt.replace("<PRIOR>", str(self.np_prior.reshape(self.prior_cols, self.parms).T))
        return txt


class DichotomousModelResultStruct(ctypes.Structure):
    _fields_ = (
        ("model", ctypes.c_int),  # dichotomous model specification
        ("nparms", ctypes.c_int),  # number of parameters in the model
        ("parms", ctypes.POINTER(ctypes.c_double)),  # parameter estimate
        ("cov", ctypes.POINTER(ctypes.c_double)),  # covariance estimate
        ("max", ctypes.c_double),  # value of the likelihood/posterior at the maximum
        ("dist_numE", ctypes.c_int),  # number of entries in rows for the bmd_dist
        ("model_df", ctypes.c_double),  # Used model degrees of freedom
        ("total_df", ctypes.c_double),  # Total degrees of freedom
        ("bmd_dist", ctypes.POINTER(ctypes.c_double)),  # bmd distribution (dist_numE x 2) matrix
        ("bmd", ctypes.c_double),  # the central estimate of the BMD
    )

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
            bmd: {self.bmd}
            """
        )


class DichotomousPgofResultStruct(ctypes.Structure):
    _fields_ = (
        ("n", ctypes.c_int),  # total number of observations obs/n
        ("expected", ctypes.POINTER(ctypes.c_double)),
        ("residual", ctypes.POINTER(ctypes.c_double)),
        ("test_statistic", ctypes.c_double),
        ("p_value", ctypes.c_double),
        ("df", ctypes.c_double),
        ("ebLower", ctypes.POINTER(ctypes.c_double)),
        ("ebUpper", ctypes.POINTER(ctypes.c_double)),
    )

    def __str__(self) -> str:
        return dedent(
            f"""
            n: {self.n}
            expected: {self.expected[:self.n]}
            residual: {self.residual[:self.n]}
            test_statistic: {self.test_statistic}
            p_value: {self.p_value}
            df: {self.df}
            ebLower: {self.ebLower[:self.n]}
            ebUpper: {self.ebUpper[:self.n]}
            """
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.np_expected = np.zeros(self.n, dtype=np.float64)
        self.expected = np.ctypeslib.as_ctypes(self.np_expected)
        self.np_residual = np.zeros(self.n, dtype=np.float64)
        self.residual = np.ctypeslib.as_ctypes(self.np_residual)
        self.np_ebLower = np.zeros(self.n, dtype=np.float64)
        self.ebLower = np.ctypeslib.as_ctypes(self.np_ebLower)
        self.np_ebUpper = np.zeros(self.n, dtype=np.float64)
        self.ebUpper = np.ctypeslib.as_ctypes(self.np_ebUpper)


class DichotomousAodStruct(ctypes.Structure):
    # Dichotomous Analysis of Deviance Struct

    _fields_ = (
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
        ("pvFit", ctypes.c_double),
        ("pvRed", ctypes.c_double),
    )

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


class BmdsResultsStruct(ctypes.Structure):
    # used for both continuous and dichotomous data
    _fields_ = (
        ("bmd", ctypes.c_double),
        ("bmdl", ctypes.c_double),
        ("bmdu", ctypes.c_double),
        ("aic", ctypes.c_double),
        ("BIC_equiv", ctypes.c_double),
        ("chisq", ctypes.c_double),
        ("bounded", ctypes.POINTER(ctypes.c_bool)),
        ("stdErr", ctypes.POINTER(ctypes.c_double)),
        ("lowerConf", ctypes.POINTER(ctypes.c_double)),
        ("upperConf", ctypes.POINTER(ctypes.c_double)),
        ("validResult", ctypes.c_bool),
    )

    def __str__(self) -> str:
        return dedent(
            f"""
            bmd: {self.bmd}
            bmdl: {self.bmdl}
            bmdu: {self.bmdu}
            aic: {self.aic}
            BIC_equiv: {self.BIC_equiv}
            chisq: {self.chisq}
            bounded: {self.bounded[:self.n]}
            stdErr: {self.stdErr[:self.n]}
            lowerConf: {self.lowerConf[:self.n]}
            upperConf: {self.upperConf[:self.n]}
            validResult: {self.validResult}
            """
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = kwargs["num_params"]
        self.bmd = constants.BMDS_BLANK_VALUE
        self.bmdl = constants.BMDS_BLANK_VALUE
        self.bmdu = constants.BMDS_BLANK_VALUE
        self.aic = constants.BMDS_BLANK_VALUE
        self.BIC_equiv = constants.BMDS_BLANK_VALUE
        self.chisq = constants.BMDS_BLANK_VALUE
        self.np_bounded = np.zeros(self.n, dtype=np.bool_)
        self.np_stdErr = np.zeros(self.n, dtype=np.float64)
        self.np_lowerConf = np.zeros(self.n, dtype=np.float64)
        self.np_upperConf = np.zeros(self.n, dtype=np.float64)
        self.bounded = np.ctypeslib.as_ctypes(self.np_bounded)
        self.stdErr = np.ctypeslib.as_ctypes(self.np_stdErr)
        self.lowerConf = np.ctypeslib.as_ctypes(self.np_lowerConf)
        self.upperConf = np.ctypeslib.as_ctypes(self.np_upperConf)
        self.validResult = ctypes.c_bool()


class DichotomousStructs(NamedTuple):
    analysis: DichotomousAnalysisStruct
    result: DichotomousModelResultStruct
    gof: DichotomousPgofResultStruct
    aod: DichotomousAodStruct
    summary: BmdsResultsStruct

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
    _fields_ = (
        ("nmodels", ctypes.c_int),
        ("priors", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
        ("nparms", ctypes.POINTER(ctypes.c_int)),
        ("actual_parms", ctypes.POINTER(ctypes.c_int)),
        ("prior_cols", ctypes.POINTER(ctypes.c_int)),
        ("models", ctypes.POINTER(ctypes.c_int)),
        ("modelPriors", ctypes.POINTER(ctypes.c_double)),
    )

    def __init__(self, models: list[DichotomousAnalysisStruct], model_weights: npt.NDArray):
        self.np_nparms = np.array([model.parms for model in models], dtype=np.int32)
        self.np_actual_parms = self.np_nparms.copy()
        self.np_prior_cols = np.array([model.prior_cols for model in models], dtype=np.int32)
        self.np_models = np.array([model.model for model in models], dtype=np.int32)
        self.np_modelPriors = model_weights

        # list of floats
        _priors = [
            list_t_c(
                model.prior[: model.parms * model.prior_cols],
                ctypes.c_double,
            )
            for model in models
        ]

        super().__init__(
            nmodels=len(models),
            nparms=self.np_nparms.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            actual_parms=self.np_actual_parms.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            prior_cols=self.np_prior_cols.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            models=self.np_models.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            modelPriors=model_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            priors=list_t_c(
                [ctypes.cast(el, ctypes.POINTER(ctypes.c_double)) for el in _priors],
                ctypes.POINTER(ctypes.c_double),
            ),
        )

    def __str__(self) -> str:
        return dedent(
            f"""
            nmodels: {self.nmodels}
            priors: {self.priors}
            nparms: {self.np_nparms}
            actual_parms: {self.np_actual_parms}
            prior_cols: {self.np_prior_cols}
            models: {self.np_models}
            modelPriors: {self.np_modelPriors}
            """
        )


class DichotomousMAResultStruct(ctypes.Structure):
    _fields_ = (
        ("nmodels", ctypes.c_int),
        ("models", ctypes.POINTER(ctypes.POINTER(DichotomousModelResultStruct))),
        ("dist_numE", ctypes.c_int),
        ("post_probs", ctypes.POINTER(ctypes.c_double)),
        ("bmd_dist", ctypes.POINTER(ctypes.c_double)),
    )

    def __init__(self, models: list[DichotomousModelResultStruct]):
        self.nmodels = len(models)
        self.models = list_t_c(
            [ctypes.pointer(model) for model in models],
            ctypes.POINTER(DichotomousModelResultStruct),
        )
        self.dist_numE = 200
        self.np_post_probs = np.zeros(self.nmodels, dtype=np.float64)
        self.post_probs = np.ctypeslib.as_ctypes(self.np_post_probs)
        self.np_bmd_dist = np.zeros(self.dist_numE * 2, dtype=np.float64)
        self.bmd_dist = np.ctypeslib.as_ctypes(self.np_bmd_dist)

    def __str__(self) -> str:
        return dedent(
            f"""
            nmodels: {self.nmodels}
            models: {self.models}
            dist_numE: {self.dist_numE}
            post_probs: {self.np_post_probs}
            bmd_dist: {self.np_bmd_dist}
            """
        )


class MAResultsStruct(ctypes.Structure):
    _fields_ = (
        ("bmd_ma", ctypes.c_double),
        ("bmdl_ma", ctypes.c_double),
        ("bmdu_ma", ctypes.c_double),
        ("bmd", ctypes.POINTER(ctypes.c_double)),
        ("bmdl", ctypes.POINTER(ctypes.c_double)),
        ("bmdu", ctypes.POINTER(ctypes.c_double)),
        ("ebLower", ctypes.POINTER(ctypes.c_double)),
        ("ebUpper", ctypes.POINTER(ctypes.c_double)),
    )

    def __str__(self) -> str:
        return dedent(
            f"""
            bmd_ma: {self.bmd_ma}
            bmdl_ma: {self.bmdl_ma}
            bmdu_ma: {self.bmdu_ma}
            bmd: {self.np_bmd}
            bmdl: {self.np_bmdl}
            bmdu: {self.np_bmdu}
            ebLower: {self.np_ebLower}
            ebUpper: {self.np_ebUpper}
            """
        )

    def __init__(self, n_dose_groups: int, n_models: int):
        super().__init__()
        self.np_bmd = np.zeros(n_models, dtype=np.float64)
        self.np_bmdl = np.zeros(n_models, dtype=np.float64)
        self.np_bmdu = np.zeros(n_models, dtype=np.float64)
        self.np_ebLower = np.zeros(n_dose_groups, dtype=np.float64)
        self.np_ebUpper = np.zeros(n_dose_groups, dtype=np.float64)
        self.bmd = np.ctypeslib.as_ctypes(self.np_bmd)
        self.bmdl = np.ctypeslib.as_ctypes(self.np_bmdl)
        self.bmdu = np.ctypeslib.as_ctypes(self.np_bmdu)
        self.ebLower = np.ctypeslib.as_ctypes(self.np_ebLower)
        self.ebUpper = np.ctypeslib.as_ctypes(self.np_ebUpper)


class DichotomousMAStructs(NamedTuple):
    analysis: DichotomousMAAnalysisStruct
    inputs: DichotomousAnalysisStruct
    dich_result: DichotomousMAResultStruct
    result: MAResultsStruct

    @classmethod
    def from_session(cls, dataset, models, weights) -> Self:
        return cls(
            analysis=DichotomousMAAnalysisStruct(
                [model.structs.analysis for model in models], weights
            ),
            inputs=models[0].structs.analysis,
            dich_result=DichotomousMAResultStruct([model.structs.result for model in models]),
            result=MAResultsStruct(n_dose_groups=dataset.num_dose_groups, n_models=len(models)),
        )

    def __str__(self):
        return dedent(
            f"""
            MA Analysis:
            {self.analysis}

            Analysis:
            {self.inputs}

            Dichotomous Result:
            {self.dich_result}

            Result:
            {self.result}
            """
        )


# CONTINUOUS MODELS
# -----------------
class ContinuousAnalysisStruct(ctypes.Structure):
    _fields_ = (
        ("model", ctypes.c_int),
        ("n", ctypes.c_int),
        ("suff_stat", ctypes.c_bool),  # true if continuous summary, false if individual data
        ("Y", ctypes.POINTER(ctypes.c_double)),  # observed data means or actual data
        ("doses", ctypes.POINTER(ctypes.c_double)),
        ("sd", ctypes.POINTER(ctypes.c_double)),  # SD of the group if suff_stat = true else null
        ("n_group", ctypes.POINTER(ctypes.c_double)),  # group N if suff_stat = true else null
        ("prior", ctypes.POINTER(ctypes.c_double)),  # column order matrix px5 where p # params
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
        ("transform_dose", ctypes.c_int),
    )

    def __init__(self, *args, **kwargs):
        self.np_Y = np.array(kwargs.pop("Y"), dtype=np.double)
        self.np_doses = np.array(kwargs.pop("doses"), dtype=np.double)
        self.np_sd = np.array(kwargs.pop("sd"), dtype=np.double)
        self.np_n_group = np.array(kwargs.pop("n_group"), dtype=np.double)
        self.np_prior = kwargs.pop("prior")
        super().__init__(
            *args,
            Y=self.np_Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            doses=self.np_doses.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            sd=self.np_sd.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n_group=self.np_n_group.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            prior=self.np_prior.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            **kwargs,
        )

    def __str__(self) -> str:
        txt = dedent(
            f"""
            model: {self.model}
            n: {self.n}
            suff_stat: {self.suff_stat}
            Y: {self.np_Y}
            doses: {self.np_doses}
            sd: {self.np_sd}
            n_group: {self.np_n_group}
            prior<{self.parms, self.prior_cols}>
            <PRIOR>
            priors <unprocessed>: {", ".join(str(v) for v in self.np_prior.tolist())}
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
            transform_dose: {self.transform_dose}
            """
        )
        txt = txt.replace("<PRIOR>", str(self.np_prior.reshape(self.prior_cols, self.parms).T))
        return txt


class ContinuousModelResultStruct(ctypes.Structure):
    _fields_ = (
        ("model", ctypes.c_int),  # continuous model specification
        ("dist", ctypes.c_int),  # distribution type
        ("nparms", ctypes.c_int),  # number of parameters in the model
        ("parms", ctypes.POINTER(ctypes.c_double)),  # parameter estimate
        ("cov", ctypes.POINTER(ctypes.c_double)),  # covariance estimate
        ("max", ctypes.c_double),  # value of the likelihood/posterior at the maximum
        ("dist_numE", ctypes.c_int),  # number of entries in rows for the bmd_dist
        ("model_df", ctypes.c_double),  # Used model degrees of freedom
        ("total_df", ctypes.c_double),  # Total degrees of freedom
        ("bmd", ctypes.c_double),  # The bmd at the maximum
        ("bmd_dist", ctypes.POINTER(ctypes.c_double)),  # bmd distribution (dist_numE x 2) matrix
    )

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
            bmd: {self.bmd}
            bmd_dist: {self.bmd_dist[:self.dist_numE*2]}
            """
        )


class ContinuousGofStruct(ctypes.Structure):
    _fields_ = (
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
        ("ebLower", ctypes.POINTER(ctypes.c_double)),
        ("ebUpper", ctypes.POINTER(ctypes.c_double)),
    )

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
            ebLower: {self.ebLower[:self.n]}
            ebUpper: {self.ebUpper[:self.n]}
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
        self.np_ebLower = np.zeros(self.n, dtype=np.float64)
        self.ebLower = np.ctypeslib.as_ctypes(self.np_ebLower)
        self.np_ebUpper = np.zeros(self.n, dtype=np.float64)
        self.ebUpper = np.ctypeslib.as_ctypes(self.np_ebUpper)


class ContinuousToiStruct(ctypes.Structure):
    _fields_ = (
        ("llRatio", ctypes.POINTER(ctypes.c_double)),
        ("DF", ctypes.POINTER(ctypes.c_double)),
        ("pVal", ctypes.POINTER(ctypes.c_double)),
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
    _fields_ = (
        ("LL", ctypes.POINTER(ctypes.c_double)),
        ("nParms", ctypes.POINTER(ctypes.c_int)),
        ("AIC", ctypes.POINTER(ctypes.c_double)),
        ("addConst", ctypes.c_double),
        ("TOI", ctypes.POINTER(ContinuousToiStruct)),
    )

    def __str__(self) -> str:
        toi = self.TOI[0]
        return dedent(
            f"""
            LL: {self.LL[:5]}
            nParms: {self.nParms[:5]}
            AIC: {self.AIC[:5]}
            addConst: {self.addConst}
            llRatio: {toi.llRatio[:4]}
            DF: {toi.DF[:4]}
            pVal: {toi.pVal[:4]}
            """
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.np_LL = np.zeros(5, dtype=np.float64)
        self.LL = np.ctypeslib.as_ctypes(self.np_LL)
        self.np_nParms = np.zeros(5, dtype=np.int32)
        self.nParms = np.ctypeslib.as_ctypes(self.np_nParms)
        self.np_AIC = np.zeros(5, dtype=np.float64)
        self.AIC = np.ctypeslib.as_ctypes(self.np_AIC)
        self.toi_struct = ContinuousToiStruct()
        self.TOI = ctypes.pointer(self.toi_struct)


class ContinuousStructs(NamedTuple):
    analysis: ContinuousAnalysisStruct
    result: ContinuousModelResultStruct
    summary: BmdsResultsStruct
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
