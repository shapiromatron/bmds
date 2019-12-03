import ctypes
from enum import Enum


class BMDSInputType_t(Enum):
    unused = 0
    eCont_2 = 1  # Individual dose-responses
    eCont_4 = 2 # Summarized dose-responses
    eDich_3 = 3 # Regular dichotomous dose-responses
    eDich_4 = 4  # Dichotomous d-r with covariate (e.g., nested)


class cGoFRow_t(ctypes.Structure):
    _fields = [
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


class dGoF_t(ctypes.Structure):
    _fields = [
        ("chiSquare", ctypes.c_double),
        ("pvalue", ctypes.c_double),
        ("pzRow", ctypes.POINTER(cGoFRow_t)),
        ("df", ctypes.c_int),
        ("n", ctypes.c_int)
    ]


class BMDSInputData_t(ctypes.Structure):
    _fields = [
        ("dose", ctypes.c_double),
        ("response", ctypes.c_double), # Mean value for summary data
        ("groupSize", ctypes.c_double),
        ("col4", ctypes.c_double)  # stddev for cont_4 or covariate for dich_4
    ]


class Prior(ctypes.Structure):
    _fields = [
        ("type", ctypes.c_double), # 0= None (frequentist), 1=  normal (Bayesian), 2= log-normal (Bayesian)
        ("initalValue", ctypes.c_double),
        ("stdDev", ctypes.c_double), # Only used for type= 1 or 2
        ("minValue", ctypes.c_double),
        ("maxValue", ctypes.c_double),
    ]


class BMDS_D_Opts1_t(ctypes.Structure):
    _fields = [
        ("bmr", ctypes.c_double),
        ("alpha", ctypes.c_double),
        ("background", ctypes.c_double),
    ]


class BMDS_D_Opts2_t(ctypes.Structure):
    _fields = [
        ("bmrType", ctypes.c_int),
        ("degree", ctypes.c_int),  # Polynomial degree for the multistage model
    ]


class dGoF_t(ctypes.Structure):
    _fields = [
        ("dose", ctypes.c_double),
        ("estProb", ctypes.c_double),  # Model-estimated probability for dose
        ("expected", ctypes.c_double), # Expected dose-response according to the model
        ("observed", ctypes.c_double),
        ("size", ctypes.c_double),
        ("scaledResidual", ctypes.c_double),
        ("ebLower", ctypes.c_double), # Error bar lower bound
        ("ebUpper", ctypes.c_double), # Error bar upper bound
    ]


class DichotomousDeviance_t(ctypes.Structure):
    _fields = [
        ("llFull", ctypes.c_double), # Full model log-likelihood
        ("llReduced", ctypes.c_double), # Reduced model log-likelihood
        ("devFit", ctypes.c_double), # Fit model deviance
        ("devReduced", ctypes.c_double), # Reduced model deviance
        ("pvFit", ctypes.c_double), # Fit model p-value
        ("pvReduced", ctypes.c_double), # Reduced model p-value
        ("nparmFull", ctypes.c_int),
        ("nparmFit", ctypes.c_int),
        ("dfFit", ctypes.c_int),
        ("nparmReduced", ctypes.c_int),
        ("dfReduced", ctypes.c_int),
    ]


class BMD_ANAL(ctypes.Structure):
    _fields = [
        ("model_id", ctypes.POINTER(ctypes.c_char)),
        ("MAP", ctypes.c_double), # Equals the -LL for frequentist runs
        ("BMD", ctypes.c_double),
        ("BMDL", ctypes.c_double),
        ("BMDU", ctypes.c_double),
        ("AIC", ctypes.c_double),
        ("BIC_Equiv", ctypes.c_double), # BIC equivalent for Bayesian runs
        ("PARMS", ctypes.POINTER(ctypes.c_double)),
        ("aCDF", ctypes.POINTER(ctypes.c_double)), # Array of cumulative density function values for BMD
        ("deviance", ctypes.POINTER(DichotomousDeviance_t)),
        ("gof",  ctypes.POINTER(dGoF_t)), # Goodness of Fit
        ("boundedParms", ctypes.POINTER(ctypes.c_bool)),
        ("nparms", ctypes.c_int),
        ("nCDF", ctypes.c_int),     # Requested number of aCDF elements to return
    ]

class PRIOR(ctypes.Structure):
    _fields = [
      ("type", ctypes.c_double), # 0= None (frequentist), 1=  normal (Bayesian), 2= log-normal (Bayesian)
      ("initalValue", ctypes.c_double),
      ("stdDev", ctypes.c_double), # Only used for type= 1 or 2
      ("minValue", ctypes.c_double),
      ("maxValue", ctypes.c_double),
    ]