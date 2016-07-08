DICHOTOMOUS = 'D'
DICHOTOMOUS_CANCER = 'DC'
CONTINUOUS = 'C'

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
