import ctypes
from enum import IntEnum


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


########################
# Dichotomous Structures
########################


class DichotomousAnalysis(ctypes.Structure):
    """
    Purpose - Contains all of the information for a dichotomous analysis.
    It is used do describe a single model analysis, in which all of the
    informationis used, or a MA analysis, in which all the information
    save prior, degree, parms and prior_cols are used.
    """

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


class DichotomousModelResult(ctypes.Structure):
    """
    Purpose: Data structure that is populated with all of the necessary
    information for a single model fit.
    """

    _fields_ = [
        ("model", ctypes.c_int),  # dichotomous model specification
        ("nparms", ctypes.c_int),  # number of parameters in the model
        ("parms", ctypes.POINTER(ctypes.c_double)),  # parameter estimate
        ("cov", ctypes.POINTER(ctypes.c_double)),  # covariance estimate
        ("max", ctypes.c_double),  # value of the likelihood/posterior at the maximum
        ("dist_numE", ctypes.c_int),  # number of entries in rows for the bmd_dist
        ("bmd_dist", ctypes.POINTER(ctypes.c_double)),  # bmd distribution (dist_numE x 2) matrix
    ]


class DichotomousMAAnalysis(ctypes.Structure):
    """
    Purpose: Fill out all of the information for a dichotomous model average.
    """

    _fields_ = [
        ("nmodels", ctypes.c_int),  # number of models for the model average
        ("priors", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),  # list of pointers to prior arrays
        ("nparms", ctypes.POINTER(ctypes.c_int)),  # parameters in each model
        ("actual_parms", ctypes.POINTER(ctypes.c_int)),  # actual number of parameters in the model
        ("prior_cols", ctypes.POINTER(ctypes.c_int)),  # columns in the prior if there are "more" in the future
        ("models", ctypes.POINTER(ctypes.c_int)),  # list of models defined by DichModel
        ("modelPriors", ctypes.POINTER(ctypes.c_double)),  # prior probability on the model
    ]


class DichotomousMAResult(ctypes.Structure):
    _fields_ = [
        ("nmodels", ctypes.c_int),  # number of models for each
        (
            "models",
            ctypes.POINTER(ctypes.POINTER(DichotomousModelResult)),
        ),  # individual model fits for each model average
        ("dist_numE", ctypes.c_int),  # number of entries in rows for the bmd_dist
        ("post_probs", ctypes.POINTER(ctypes.c_double)),  # posterior probabilities
        ("bmd_dist", ctypes.POINTER(ctypes.c_double)),  # bmd ma distribution (dist_numE x 2) matrix
    ]


#######################
# Continuous Structures
#######################


class ContinuousAnalysis(ctypes.Structure):
    _fields_ = [
        ("model", ContModel),
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


class ContinuousMAAnalysis(ctypes.Structure):
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


class ContinuousModelResult(ctypes.Structure):
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


class ContinuousMAResult(ctypes.Structure):
    _fields_ = [
        ("nmodels", ctypes.c_int),  # number of models for each
        ("models", ctypes.POINTER(ctypes.POINTER(ContinuousModelResult)),),  # priors
        ("dist_numE", ctypes.c_int),  # number of entries in rows for the bmd_dist
        ("post_probs", ctypes.POINTER(ctypes.c_double)),  # posterior probabilities
        ("bmd_dist", ctypes.POINTER(ctypes.c_double)),  # bmd ma distribution (dist_numE x 2) matrix
    ]


#################
# MCMC Structures
#################


class BMDAnalysisMCMC(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_int),  # model used in the analysis
        ("burnin", ctypes.c_uint),  # burnin samples
        ("samples", ctypes.c_uint),  # total samples including burnin
        ("nparms", ctypes.c_uint),  # parameters in the model
        ("BMDS", ctypes.POINTER(ctypes.c_double)),  # array of samples of BMDS length (samples)
        ("parms", ctypes.POINTER(ctypes.c_double)),  # array of parameters length (samples X parms)
    ]


class MAMCMSFits(ctypes.Structure):
    _fields_ = [("nfits", ctypes.c_uint), ("analyses", ctypes.POINTER(ctypes.POINTER(BMDAnalysisMCMC)))]

