import ctypes
from typing import Union
from enum import Enum


BMDS_BLANK_VALUE = -9999
NUM_PRIOR_COLS = 5
CDF_TABLE_SIZE = 99
MY_MAX_PARMS = 16
NUM_LIKELIHOODS_OF_INTEREST = 5
NUM_TESTS_OF_INTEREST = 4


class VarType_t(Enum):
    eVarTypeNone = 0
    eConstant = 1
    eModeled = 2


class BMDS_C_Options_t(ctypes.Structure):
    __fields__ = [
        ("bmr", ctypes.c_double),
        ("alpha", ctypes.c_double),
        ("background", ctypes.c_double),
        ("tailProb", ctypes.c_double),  # Valid only for hybrid bmr type
        ("bmrType", ctypes.c_int),
        ("degree", ctypes.c_int),  # Valid for polynomial type models; for exponential, identifies the submodel
        ("adverseDirection", ctypes.c_int),  # Direction of adversity: 0=auto, 1=up, -1=down
        ("restriction", ctypes.c_int),  # Restriction on parameters for certain models
        ("varType", ctypes.c_int),  # VarType_t
        ("bLognormal", ctypes.c_bool),  # Valid only for continuous models
        ("bUserParmInit", ctypes.c_bool),  # Use specified priors instead of calculated values
    ]


class CModelID_t(Enum):
    eExp2 = 2
    eExp3 = 3
    eExp4 = 4
    eExp5 = 5
    eHill = 6
    ePoly = 7
    ePow = 8


class DModelID_t(Enum):
    eDHill = 1
    eGamma = 2
    eLogistic = 3
    eLogLogistic = 4
    eLogProbit = 5
    eMultistage = 6
    eProbit = 7
    eQLinear = 8
    eWeibull = 9


class BMDSPrior_t(Enum):
    eNone = 0
    eNormal = 1
    eLognormal = 2


class BMRType_t(Enum):
    eAbsoluteDev = 1
    eStandardDev = 2
    eRelativeDev = 3
    ePointEstimate = 4
    eExtra = 5  # Not used
    eHybrid_Extra = 6
    eHybrid_Added = 7


class RiskType_t(Enum):
    eExtraRisk = 1
    eAddedRisk = 2


class BMDSInputType_t(Enum):
    unused = 0
    eCont_2 = 1  # Individual dose-responses
    eCont_4 = 2  # Summarized dose-responses
    eDich_3 = 3  # Regular dichotomous dose-responses
    eDich_4 = 4  # Dichotomous d-r with covariate (e.g., nested)


class cGoFRow_t(ctypes.Structure):
    _fields_ = [
        ("dose", ctypes.c_double),
        ("obsMean", ctypes.c_double),
        ("obsStDev", ctypes.c_double),
        ("calcMedian", ctypes.c_double),
        ("calcGSD", ctypes.c_double),
        ("estMean", ctypes.c_double),
        ("estStDev", ctypes.c_double),
        ("size", ctypes.c_double),
        ("scaledResidual", ctypes.c_double),
        ("ebLower", ctypes.c_double),
        ("ebUpper", ctypes.c_double),
    ]


class GoFRow_t(ctypes.Structure):
    _fields_ = [
        ("dose", ctypes.c_double),
        ("estProb", ctypes.c_double),  # Model-estimated probability for dose
        ("expected", ctypes.c_double),  # Expected dose-response according to the model
        ("observed", ctypes.c_double),
        ("size", ctypes.c_double),
        ("scaledResidual", ctypes.c_double),
        ("ebLower", ctypes.c_double),  # Error bar lower bound
        ("ebUpper", ctypes.c_double),  # Error bar upper bound
    ]


class dGoF_t(ctypes.Structure):
    _fields_ = [
        ("chiSquare", ctypes.c_double),
        ("pvalue", ctypes.c_double),
        ("pzRow", ctypes.POINTER(GoFRow_t)),
        ("df", ctypes.c_int),
        ("n", ctypes.c_int),
    ]


class BMDSInputData_t(ctypes.Structure):
    _fields_ = [
        ("dose", ctypes.c_double),
        ("response", ctypes.c_double),  # Mean value for summary data
        ("groupSize", ctypes.c_double),
        ("col4", ctypes.c_double),  # stddev for cont_4 or covariate for dich_4
    ]


class Prior(ctypes.Structure):
    _fields_ = [
        (
            "type",
            ctypes.c_double,
        ),  # 0= None (frequentist), 1=  normal (Bayesian), 2= log-normal (Bayesian)
        ("initalValue", ctypes.c_double),
        ("stdDev", ctypes.c_double),  # Only used for type= 1 or 2
        ("minValue", ctypes.c_double),
        ("maxValue", ctypes.c_double),
    ]


class BMDS_D_Opts1_t(ctypes.Structure):
    _fields_ = [
        ("bmr", ctypes.c_double),
        ("alpha", ctypes.c_double),
        ("background", ctypes.c_double),
    ]


class BMDS_D_Opts2_t(ctypes.Structure):
    _fields_ = [
        ("bmrType", ctypes.c_int),
        ("degree", ctypes.c_int),  # Polynomial degree for the multistage model
    ]


class DichotomousDeviance_t(ctypes.Structure):
    _fields_ = [
        ("llFull", ctypes.c_double),  # Full model log-likelihood
        ("llReduced", ctypes.c_double),  # Reduced model log-likelihood
        ("devFit", ctypes.c_double),  # Fit model deviance
        ("devReduced", ctypes.c_double),  # Reduced model deviance
        ("pvFit", ctypes.c_double),  # Fit model p-value
        ("pvReduced", ctypes.c_double),  # Reduced model p-value
        ("nparmFull", ctypes.c_int),
        ("nparmFit", ctypes.c_int),
        ("dfFit", ctypes.c_int),
        ("nparmReduced", ctypes.c_int),
        ("dfReduced", ctypes.c_int),
    ]


class BMD_ANAL(ctypes.Structure):
    _fields_ = [
        ("model_id", ctypes.POINTER(ctypes.c_char)),
        ("MAP", ctypes.c_double),  # Equals the -LL for frequentist runs
        ("BMD", ctypes.c_double),
        ("BMDL", ctypes.c_double),
        ("BMDU", ctypes.c_double),
        ("AIC", ctypes.c_double),
        ("BIC_Equiv", ctypes.c_double),  # BIC equivalent for Bayesian runs
        ("PARMS", ctypes.POINTER(ctypes.c_double)),
        (
            "aCDF",
            ctypes.POINTER(ctypes.c_double),
        ),  # Array of cumulative density function values for BMD
        ("deviance", ctypes.POINTER(DichotomousDeviance_t)),
        ("gof", ctypes.POINTER(dGoF_t)),  # Goodness of Fit
        ("boundedParms", ctypes.POINTER(ctypes.c_bool)),
        ("nparms", ctypes.c_int),
        ("nCDF", ctypes.c_int),  # Requested number of aCDF elements to return
    ]


class LLRow_t(ctypes.Structure):
    _fields_ = [
        ("ll", ctypes.c_double),  # Log-likelihood
        ("aic", ctypes.c_double),
        ("model", ctypes.c_int),  # Data model number for test
        ("nParms", ctypes.c_int),  # Count of model parameters
    ]


class TestRow_t(ctypes.Structure):
    _fields_ = [
        ("deviance", ctypes.c_double),  # -2*log-likelihood ratio
        ("pvalue", ctypes.c_double),  # test p-value
        ("testNumber", ctypes.c_int),
        ("df", ctypes.c_int),  # test degrees of freedom
    ]


class ContinuousDeviance_t(ctypes.Structure):
    _fields_ = [
        ("llRows", ctypes.POINTER(LLRow_t)),
        ("testRows", ctypes.POINTER(TestRow_t)),
    ]


class BMD_C_ANAL(ctypes.Structure):
    _fields_ = [
        ("model_id", ctypes.POINTER(ctypes.c_char)),
        ("PARMS", ctypes.POINTER(ctypes.c_double)),
        ("deviance", ContinuousDeviance_t),
        ("gofRow", ctypes.POINTER(cGoFRow_t)),  # Goodness of Fit
        ("boundedParms", ctypes.POINTER(ctypes.c_bool)),
        ("MAP", ctypes.c_double),
        ("BMD", ctypes.c_double),
        ("BMDL", ctypes.c_double),
        ("BMDU", ctypes.c_double),
        ("AIC", ctypes.c_double),
        ("BIC_Equiv", ctypes.c_double),  # BIC equivalent for Bayesian runs
        ("ll_const", ctypes.c_double),  # LL "additive" constant term
        (
            "aCDF",
            ctypes.POINTER(ctypes.c_double)
        ),  # Array of cumulative density function values for BMD
        ("nCDF", ctypes.c_int),  # Requested number of aCDF elements to return
        ("nparms", ctypes.c_int),
        ("bAdverseUp", ctypes.c_bool),
    ]


class PRIOR(ctypes.Structure):
    _fields_ = [
        (
            "type",
            ctypes.c_double,
        ),  # 0= None (frequentist), 1=  normal (Bayesian), 2= log-normal (Bayesian)
        ("initialValue", ctypes.c_double),
        ("stdDev", ctypes.c_double),  # Only used for type= 1 or 2
        ("minValue", ctypes.c_double),
        ("maxValue", ctypes.c_double),
    ]


RESULT_TYPES = Union[BMD_ANAL, BMD_C_ANAL]
