# -*- coding: utf-8 -*-

# dataset types
DICHOTOMOUS = 'D'
DICHOTOMOUS_CANCER = 'DC'
CONTINUOUS = 'C'
CONTINUOUS_INDIVIDUAL = 'CI'
DTYPES = (
    DICHOTOMOUS,
    DICHOTOMOUS_CANCER,
    CONTINUOUS,
    CONTINUOUS_INDIVIDUAL,
)
DICHOTOMOUS_DTYPES = (
    DICHOTOMOUS,
    DICHOTOMOUS_CANCER,
)
CONTINUOUS_DTYPES = (
    CONTINUOUS,
    CONTINUOUS_INDIVIDUAL,
)

# bmds versions
BMDS231 = 'BMDS231'
BMDS240 = 'BMDS240'
BMDS260 = 'BMDS260'
BMDS2601 = 'BMDS2601'
BMDS270 = 'BMDS270'

# model names
M_Weibull = 'Weibull'
M_LogProbit = 'LogProbit'
M_Probit = 'Probit'
M_QuantalLinear = 'Quantal Linear'
M_Multistage = 'Multistage'
M_Gamma = 'Gamma'
M_Logistic = 'Logistic'
M_LogLogistic = 'LogLogistic'
M_DichotomousHill = 'Dichotomous-Hill'
M_MultistageCancer = 'Multistage-Cancer'
M_Linear = 'Linear'
M_Polynomial = 'Polynomial'
M_Power = 'Power'
M_ExponentialM2 = 'Exponential-M2'
M_ExponentialM3 = 'Exponential-M3'
M_ExponentialM4 = 'Exponential-M4'
M_ExponentialM5 = 'Exponential-M5'
M_Hill = 'Hill'

VARIABLE_POLYNOMIAL = (
    M_Multistage,
    M_MultistageCancer,
    M_Polynomial,
)

# BMR types
DICHOTOMOUS_BMRS = [
    {'type': 'Extra', 'value': 0.1, 'confidence_level': 0.95},
    {'type': 'Added', 'value': 0.1, 'confidence_level': 0.95},
]
CONTINUOUS_BMRS = [
    {'type': 'Std. Dev.', 'value': 1.0, 'confidence_level': 0.95},
    {'type': 'Abs. Dev.', 'value': 0.1, 'confidence_level': 0.95},
    {'type': 'Rel. Dev.', 'value': 0.1, 'confidence_level': 0.95},
    {'type': 'Point', 'value': 1.0, 'confidence_level': 0.95},
    {'type': 'Extra', 'value': 1.0, 'confidence_level': 0.95},
]
BMR_CROSSWALK = {
    DICHOTOMOUS: {
        'Extra': 0,
        'Added': 1
    },
    DICHOTOMOUS_CANCER: {
        'Extra': 0
    },
    CONTINUOUS: {
        'Abs. Dev.': 0,
        'Std. Dev.': 1,
        'Rel. Dev.': 2,
        'Point': 3,
        'Extra': 4
    }
}
BMR_CROSSWALK[CONTINUOUS_INDIVIDUAL] = BMR_CROSSWALK[CONTINUOUS]

# go from integer to human-readable
BMR_INVERTED_CROSSALK = {
    DICHOTOMOUS: {v: k for k, v in BMR_CROSSWALK[DICHOTOMOUS].items()},
    DICHOTOMOUS_CANCER: {v: k for k, v in BMR_CROSSWALK[DICHOTOMOUS_CANCER].items()},
    CONTINUOUS: {v: k for k, v in BMR_CROSSWALK[CONTINUOUS].items()},
}
BMR_INVERTED_CROSSALK[CONTINUOUS_INDIVIDUAL] = BMR_INVERTED_CROSSALK[CONTINUOUS]

# field types
FT_INTEGER = 'i'
FT_DECIMAL = 'd'
FT_BOOL = 'b'
FT_DROPDOSE = 'dd'
FT_RESTRICTPOLY = 'rp'
FT_PARAM = 'p'

# field category
FC_OPTIMIZER = 'op'
FC_OTHER = 'ot'
FC_PARAM = 'p'
FC_BMR = 'b'

# param types
P_DEFAULT = 'd'
P_SPECIFIED = 's'
P_INITIALIZED = 'i'

# logic bins
BIN_NO_CHANGE = 0
BIN_WARNING = 1
BIN_FAILURE = 2
BIN_TYPES = (
    BIN_NO_CHANGE,
    BIN_WARNING,
    BIN_FAILURE,
)
BIN_TEXT = {
    BIN_NO_CHANGE: 'valid',
    BIN_WARNING: 'warning',
    BIN_FAILURE: 'failure',
}
BINMOJI = {
    0: u'✓',
    1: u'?',
    2: u'✕',
}

NULL = '-'
