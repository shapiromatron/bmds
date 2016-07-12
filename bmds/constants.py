# dataset types
DICHOTOMOUS = 'D'
DICHOTOMOUS_CANCER = 'DC'
CONTINUOUS = 'C'

# bmr types
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
