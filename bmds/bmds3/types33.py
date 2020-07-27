import ctypes
from enum import IntEnum
from typing import List, Any
from pydantic import BaseModel


class EstMethod(IntEnum):
    est_mle = 1
    est_laplace = 2
    est_mcmc = 3


class DichModel(IntEnum):
    d_hill = 1
    d_gamma = 2
    d_logistic = 3
    d_loglogistic = 4
    d_logprobit = 5
    d_multistage = 6
    d_probit = 7
    d_qlinear = 8
    d_weibull = 9


class ContModel(IntEnum):
    hill = 6
    exp_3 = 3
    exp_5 = 5
    power = 8
    gain_loss_model = 10
    polynomial = 666


class Distribution(IntEnum):
    normal = 1
    normal_ncv = 2
    log_normal = 3


def _list_to_c(list: List[Any], ctype):
    return (ctype * len(list))(*list)


########################
# Dichotomous Structures
########################


class DichotomousAnalysis(BaseModel):
    """
    Purpose - Contains all of the information for a dichotomous analysis.
    It is used do describe a single model analysis, in which all of the
    informationis used, or a MA analysis, in which all the information
    save prior, degree, parms and prior_cols are used.
    """

    model: int
    n: int
    Y: List[float]
    doses: List[float]
    n_group: List[float]
    prior: List[float]
    BMD_type: int
    BMR: float
    alpha: float
    degree: int
    samples: int
    burnin: int
    parms: int
    prior_cols: int

    class Struct(ctypes.Structure):

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

    def to_c(self):
        return self.Struct(
            model=ctypes.c_int(self.model),
            n=ctypes.c_int(self.n),
            Y=_list_to_c(self.Y, ctypes.c_double),
            doses=_list_to_c(self.doses, ctypes.c_double),
            n_group=_list_to_c(self.n_group, ctypes.c_double),
            prior=_list_to_c(self.prior, ctypes.c_double),
            BMD_type=ctypes.c_int(self.BMD_type),
            BMR=ctypes.c_double(self.BMR),
            alpha=ctypes.c_double(self.alpha),
            degree=ctypes.c_int(self.degree),
            samples=ctypes.c_int(self.samples),
            burnin=ctypes.c_int(self.burnin),
            parms=ctypes.c_int(self.parms),
            prior_cols=ctypes.c_int(self.prior_cols),
        )


class DichotomousModelResult(BaseModel):
    """
    Purpose: Data structure that is populated with all of the necessary
    information for a single model fit.
    """

    model: int
    nparms: int
    parms: List[float]
    cov: List[float]
    max: float
    dist_numE: int
    bmd_dist: List[float]

    class Struct(ctypes.Structure):

        _fields_ = [
            ("model", ctypes.c_int),  # dichotomous model specification
            ("nparms", ctypes.c_int),  # number of parameters in the model
            ("parms", ctypes.POINTER(ctypes.c_double)),  # parameter estimate
            ("cov", ctypes.POINTER(ctypes.c_double)),  # covariance estimate
            ("max", ctypes.c_double),  # value of the likelihood/posterior at the maximum
            ("dist_numE", ctypes.c_int),  # number of entries in rows for the bmd_dist
            ("bmd_dist", ctypes.POINTER(ctypes.c_double)),  # bmd distribution (dist_numE x 2) matrix
        ]

    def to_c(self):
        return self.Struct(
            model=ctypes.c_int(self.model),
            nparms=ctypes.c_int(self.nparms),
            parms=_list_to_c(self.parms, ctypes.c_double),
            cov=_list_to_c(self.cov, ctypes.c_double),
            max=ctypes.c_double(self.max),
            dist_numE=ctypes.c_int(self.dist_numE),
            bmd_dist=_list_to_c(self.bmd_dist, ctypes.c_double),
        )


class DichotomousMAAnalysis(BaseModel):
    """
    Purpose: Fill out all of the information for a dichotomous model average.
    """

    nmodels: int
    priors: List[List[float]]
    nparms: List[int]
    actual_parms: List[int]
    prior_cols: List[int]
    models: List[int]
    modelPriors: List[float]

    class Struct(ctypes.Structure):
        _fields_ = [
            ("nmodels", ctypes.c_int),  # number of models for the model average
            ("priors", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),  # list of pointers to prior arrays
            ("nparms", ctypes.POINTER(ctypes.c_int)),  # parameters in each model
            ("actual_parms", ctypes.POINTER(ctypes.c_int)),  # actual number of parameters in the model
            ("prior_cols", ctypes.POINTER(ctypes.c_int)),  # columns in the prior if there are "more" in the future
            ("models", ctypes.POINTER(ctypes.c_int)),  # list of models defined by DichModel
            ("modelPriors", ctypes.POINTER(ctypes.c_double)),  # prior probability on the model
        ]

    def to_c(self):
        return self.Struct(
            nmodels=ctypes.c_int(self.nmodels),
            priors=_list_to_c(_list_to_c(self.priors, ctypes.c_double), ctypes.POINTER(ctypes.c_double)),
            nparms=_list_to_c(self.nparms, ctypes.c_int),
            actual_parms=_list_to_c(self.actual_parms, ctypes.c_int),
            prior_cols=_list_to_c(self.prior_cols, ctypes.c_int),
            models=_list_to_c(self.models, ctypes.c_int),
            modelPriors=_list_to_c(self.modelPriors, ctypes.c_double),
        )


class DichotomousMAResult(BaseModel):
    nmodels: int
    models: List[List[DichotomousModelResult]]
    dist_numE: int
    post_probs: List[float]
    bmd_dist: List[float]

    class Struct(ctypes.Structure):
        _fields_ = [
            ("nmodels", ctypes.c_int),  # number of models for each
            (
                "models",
                ctypes.POINTER(ctypes.POINTER(DichotomousModelResult.Struct)),
            ),  # individual model fits for each model average
            ("dist_numE", ctypes.c_int),  # number of entries in rows for the bmd_dist
            ("post_probs", ctypes.POINTER(ctypes.c_double)),  # posterior probabilities
            ("bmd_dist", ctypes.POINTER(ctypes.c_double)),  # bmd ma distribution (dist_numE x 2) matrix
        ]

    def to_c(self):
        return self.Struct(
            nmodels=ctypes.c_int(self.nmodels),
            models=_list_to_c(
                _list_to_c(list(self.models.map(lambda x: x.to_c())), DichotomousModelResult.Struct),
                ctypes.POINTER(DichotomousModelResult.Struct),
            ),
            dist_numE=ctypes.c_int(self.dist_numE),
            post_probs=_list_to_c(self.post_probs, ctypes.c_double),
            bmd_dist=_list_to_c(self.bmd_dist, ctypes.c_double),
        )


#######################
# Continuous Structures
#######################


class ContinuousAnalysis(BaseModel):
    model: int
    n: int
    suff_stat: bool
    Y: List[float]
    doses: List[float]
    sd: List[float]
    n_group: List[float]
    prior: List[float]
    BMD_type: int
    isIncreasing: bool
    BMR: float
    tail_prob: float
    disttype: int
    alpha: float
    samples: int
    burnin: int
    parms: int
    prior_cols: int

    class Struct(ctypes.Structure):
        _fields_ = [
            ("model", ctypes.c_int),
            ("n", ctypes.c_int),
            ("suff_stat", ctypes.c_bool),  # true if the data are in sufficient statistics format
            ("Y", ctypes.POINTER(ctypes.c_double)),  # observed data means or actual data
            ("doses", ctypes.POINTER(ctypes.c_double)),
            ("sd", ctypes.POINTER(ctypes.c_double)),  # SD of the group if suff_stat = true, null otherwise
            ("n_group", ctypes.POINTER(ctypes.c_double)),  # N for each group if suff_stat = true, null otherwise
            ("prior", ctypes.POINTER(ctypes.c_double)),  # a column order matrix px5 where p is the number of parameters
            ("BMD_type", ctypes.c_int),  # type of BMD
            ("isIncreasing", ctypes.c_bool),  # if the BMD is defined increasing or decreasing
            ("BMR", ctypes.c_double),  # benchmark response related to the BMD type
            ("tail_prob", ctypes.c_double),  # tail probability
            ("disttype", ctypes.c_int),  # distribution type defined in the enum distribution
            ("alpha", ctypes.c_double),  # specified alpha
            ("samples", ctypes.c_int),  # number of MCMC samples
            ("burnin", ctypes.c_int),  # burn in
            ("parms", ctypes.c_int),  # number of parameters
            ("prior_cols", ctypes.c_int),
        ]

    def to_c(self):
        return self.Struct(
            model=ctypes.c_int(self.model),
            n=ctypes.c_int(self.n),
            suff_stat=ctypes.c_bool(self.suff_stat),
            Y=_list_to_c(self.Y, ctypes.c_double),
            doses=_list_to_c(self.doses, ctypes.c_double),
            sd=_list_to_c(self.sd, ctypes.c_double),
            n_group=_list_to_c(self.n_group, ctypes.c_double),
            prior=_list_to_c(self.prior, ctypes.c_double),
            BMD_type=ctypes.c_int(self.BMD_type),
            isIncreasing=ctypes.c_bool(self.isIncreasing),
            BMR=ctypes.c_double(self.BMR),
            tail_prob=ctypes.c_double(self.tail_prob),
            disttype=ctypes.c_int(self.disttype),
            alpha=ctypes.c_double(self.alpha),
            samples=ctypes.c_int(self.samples),
            burnin=ctypes.c_int(self.burnin),
            parms=ctypes.c_int(self.parms),
            prior_cols=ctypes.c_int(self.prior_cols),
        )


class ContinuousMAAnalysis(BaseModel):

    nmodels: int
    priors: List[List[float]]
    nparms: List[int]
    actual_parms: List[int]
    prior_cols: List[int]
    models: List[int]
    disttype: List[int]
    modelPriors: List[float]

    class Struct(ctypes.Structure):
        _fields_ = [
            ("nmodels", ctypes.c_int),  # number of models for each
            ("priors", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),  # pointer to pointer arrays for the prior
            ("nparms", ctypes.POINTER(ctypes.c_int)),  # parameters in each model
            ("actual_parms", ctypes.POINTER(ctypes.c_int)),  # actual number of parameters in the model
            ("prior_cols", ctypes.POINTER(ctypes.c_int)),  # columns in the prior if there are "more" in the future
            ("models", ctypes.POINTER(ctypes.c_int)),  # given model
            ("disttype", ctypes.POINTER(ctypes.c_int)),  # given distribution type
            ("modelPriors", ctypes.POINTER(ctypes.c_double)),  # prior probability on the model
        ]

    def to_c(self):
        return self.Struct(
            nmodels=ctypes.c_int(self.nmodels),
            priors=_list_to_c(_list_to_c(self.priors, ctypes.c_double), ctypes.POINTER(ctypes.c_double)),
            nparms=_list_to_c(self.nparms, ctypes.c_int),
            actual_parms=_list_to_c(self.actual_parms, ctypes.c_int),
            prior_cols=_list_to_c(self.prior_cols, ctypes.c_int),
            models=_list_to_c(self.models, ctypes.c_int),
            disttype=_list_to_c(self.disttype, ctypes.c_int),
            modelPriors=_list_to_c(self.modelPriors, ctypes.c_double),
        )


class ContinuousModelResult(BaseModel):

    model: int
    dist: int
    nparms: int
    parms: List[float]
    cov: List[float]
    max: float
    dist_numE: int
    bmd_dist: List[float]

    class Struct(ctypes.Structure):
        _fields_ = [
            ("model", ctypes.c_int),  # continuous model specification
            ("dist", ctypes.c_int),  # distribution type
            ("nparms", ctypes.c_int),  # number of parameters in the model
            ("parms", ctypes.POINTER(ctypes.c_double)),  # parameter estimate
            ("cov", ctypes.POINTER(ctypes.c_double)),  # covariance estimate
            ("max", ctypes.c_double),  # value of the likelihood/posterior at the maximum
            ("dist_numE", ctypes.c_int),  # number of entries in rows for the bmd_dist
            ("bmd_dist", ctypes.POINTER(ctypes.c_double)),  # bmd distribution (dist_numE x 2) matrix
        ]

    def to_c(self):
        return self.Struct(
            model=ctypes.c_int(self.model),
            dist=ctypes.c_int(self.dist),
            nparms=ctypes.c_int(self.nparms),
            parms=_list_to_c(self.parms, ctypes.c_double),
            cov=_list_to_c(self.cov, ctypes.c_double),
            max=ctypes.c_double(self.max),
            dist_numE=ctypes.c_int(self.dist_numE),
            bmd_dist=_list_to_c(self.bmd_dist, ctypes.c_double),
        )


class ContinuousMAResult(BaseModel):

    nmodels: int
    models: List[List[ContinuousModelResult]]
    dist_numE: int
    post_probs: List[float]
    bmd_dist: List[float]

    class Struct(ctypes.Structure):
        _fields_ = [
            ("nmodels", ctypes.c_int),  # number of models for each
            ("models", ctypes.POINTER(ctypes.POINTER(ContinuousModelResult.Struct)),),  # priors
            ("dist_numE", ctypes.c_int),  # number of entries in rows for the bmd_dist
            ("post_probs", ctypes.POINTER(ctypes.c_double)),  # posterior probabilities
            ("bmd_dist", ctypes.POINTER(ctypes.c_double)),  # bmd ma distribution (dist_numE x 2) matrix
        ]

    def to_c(self):
        return self.Struct(
            nmodels=ctypes.c_int(self.nmodels),
            models=_list_to_c(
                _list_to_c(list(self.models.map(lambda x: x.to_c())), DichotomousModelResult.Struct),
                ctypes.POINTER(ContinuousModelResult.Struct),
            ),
            dist_numE=ctypes.c_int(self.dist_numE),
            post_probs=_list_to_c(self.post_probs, ctypes.c_double),
            bmd_dist=_list_to_c(self.bmd_dist, ctypes.c_double),
        )


#################
# MCMC Structures
#################


class BMDAnalysisMCMC(BaseModel):

    model: int
    burnin: int
    samples: int
    nparms: int
    BMDS: List[float]
    parms: List[float]

    class Struct(ctypes.Structure):
        _fields_ = [
            ("model", ctypes.c_int),  # model used in the analysis
            ("burnin", ctypes.c_uint),  # burnin samples
            ("samples", ctypes.c_uint),  # total samples including burnin
            ("nparms", ctypes.c_uint),  # parameters in the model
            ("BMDS", ctypes.POINTER(ctypes.c_double)),  # array of samples of BMDS length (samples)
            ("parms", ctypes.POINTER(ctypes.c_double)),  # array of parameters length (samples X parms)
        ]

    def to_c(self):
        return self.Struct(
            model=ctypes.c_int(self.model),
            burnin=ctypes.c_uint(self.burnin),
            samples=ctypes.c_uint(self.samples),
            nparms=ctypes.c_uint(self.nparms),
            BMDS=_list_to_c(self.BMDS, ctypes.c_double),
            parms=_list_to_c(self.parms, ctypes.c_double),
        )


class MAMCMSFits(BaseModel):

    nfits: int
    analyses: List[List[BMDAnalysisMCMC]]

    class Struct(ctypes.Structure):
        _fields_ = [("nfits", ctypes.c_uint), ("analyses", ctypes.POINTER(ctypes.POINTER(BMDAnalysisMCMC.Struct)))]

    def to_c(self):
        return self.Struct(
            nfits=ctypes.c_uint(self.nfits),
            analyses=_list_to_c(_list_to_c(self.analyses,), ctypes.POINTER(BMDAnalysisMCMC.Struct)),
        )

