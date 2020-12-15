from typing import Dict, List, Tuple

from ..constants import DichotomousModelChoices, Prior, PriorClass

DichotomousPriorLookup: Dict[Tuple, List[Prior]] = {
    # loglogistic
    (DichotomousModelChoices.d_loglogistic.value.id, PriorClass.frequentist_unrestricted): [
        Prior.parse_args(0, -2, 0, -18, 18),
        Prior.parse_args(0, -2, 0, -18, 18),
        Prior.parse_args(0, 1, 0, 0.0001, 18),
    ],
    (DichotomousModelChoices.d_loglogistic.value.id, PriorClass.frequentist_restricted): [
        Prior.parse_args(0, -2, 0, -18, 18),
        Prior.parse_args(0, -2, 0, -18, 18),
        Prior.parse_args(0, 1.2, 0, 1, 18),
    ],
    (DichotomousModelChoices.d_loglogistic.value.id, PriorClass.bayesian): [
        Prior.parse_args(1, 0, 2, -20, 20),
        Prior.parse_args(1, 0, 1, -40, 40),
        Prior.parse_args(2, 0.693147, 0.5, 0.0001, 20),
    ],
    # gamma
    (DichotomousModelChoices.d_gamma.value.id, PriorClass.frequentist_unrestricted): [
        Prior.parse_args(0, -2, 0, -18, 18),
        Prior.parse_args(0, 1, 0, 0.2, 18),
        Prior.parse_args(0, 0.1, 0, 0, 100),
    ],
    (DichotomousModelChoices.d_gamma.value.id, PriorClass.frequentist_restricted): [
        Prior.parse_args(0, -2, 0, -18, 18),
        Prior.parse_args(0, 1, 0, 1, 18),
        Prior.parse_args(0, 1, 0, 0, 100),
    ],
    (DichotomousModelChoices.d_gamma.value.id, PriorClass.bayesian): [
        Prior.parse_args(1, 0, 2, -18, 18),
        Prior.parse_args(2, 0.693147, 0.424264, 0.2, 20),
        Prior.parse_args(2, 0, 1, 0.0001, 100),
    ],
    # logistic
    (DichotomousModelChoices.d_logistic.value.id, PriorClass.frequentist_unrestricted): [
        Prior.parse_args(0, -2, 0, -18, 18),
        Prior.parse_args(0, 0.1, 0, 0, 100),
    ],
    (DichotomousModelChoices.d_logistic.value.id, PriorClass.bayesian): [
        Prior.parse_args(1, 0, 2, -20, 20),
        Prior.parse_args(2, 0.1, 1, 1e-12, 100),
    ],
    # probit
    (DichotomousModelChoices.d_probit.value.id, PriorClass.frequentist_unrestricted): [
        Prior.parse_args(0, -2, 0, -18, 18),
        Prior.parse_args(0, 0.1, 0, 0, 18),
    ],
    (DichotomousModelChoices.d_probit.value.id, PriorClass.bayesian): [
        Prior.parse_args(1, 0, 2, -8, 8),
        Prior.parse_args(2, 0.1, 1, 0, 40),
    ],
    # qlinear
    (DichotomousModelChoices.d_qlinear.value.id, PriorClass.frequentist_unrestricted): [
        Prior.parse_args(0, -2, 0, -18, 18),
        Prior.parse_args(0, 0.5, 0, 0, 100),
    ],
    (DichotomousModelChoices.d_qlinear.value.id, PriorClass.bayesian): [
        Prior.parse_args(1, 0, 2, -20, 20),
        Prior.parse_args(2, 0.5, 1, 0, 100),
    ],
    # logprobit
    (DichotomousModelChoices.d_logprobit.value.id, PriorClass.frequentist_unrestricted): [
        Prior.parse_args(0, -2, 0, -18, 18),
        Prior.parse_args(0, -3, 0, -18, 18),
        Prior.parse_args(0, 1, 0, 0.0001, 18),
    ],
    (DichotomousModelChoices.d_logprobit.value.id, PriorClass.frequentist_restricted): [
        Prior.parse_args(0, -2, 0, -18, 18),
        Prior.parse_args(0, -3, 0, -18, 18),
        Prior.parse_args(0, 1.2, 0, 1, 18),
    ],
    (DichotomousModelChoices.d_logprobit.value.id, PriorClass.bayesian): [
        Prior.parse_args(1, 0, 2, -20, 20),
        Prior.parse_args(1, 0, 1, -8, 8),
        Prior.parse_args(2, 0.693147, 0.5, 0.0001, 40),
    ],
    # weibull
    (DichotomousModelChoices.d_weibull.value.id, PriorClass.frequentist_unrestricted): [
        Prior.parse_args(0, -2, 0, -18, 18),
        Prior.parse_args(0, 0.5, 0, 0.000001, 18),
        Prior.parse_args(0, 0.1, 0, 0.000001, 100),
    ],
    (DichotomousModelChoices.d_weibull.value.id, PriorClass.frequentist_restricted): [
        Prior.parse_args(0, -2, 0, -18, 18),
        Prior.parse_args(0, 1, 0, 1, 18),
        Prior.parse_args(0, 0.1, 0, 0.000001, 100),
    ],
    (DichotomousModelChoices.d_weibull.value.id, PriorClass.bayesian): [
        Prior.parse_args(1, 0, 2, -20, 20),
        Prior.parse_args(2, 0.693147, 0.424264, 0.0001, 18),
        Prior.parse_args(2, 0, 1, 0.0001, 100),
    ],
    # multistage
    (DichotomousModelChoices.d_multistage.value.id, PriorClass.frequentist_unrestricted): [
        Prior.parse_args(0, -17, 0, -18, 18),
        Prior.parse_args(0, 0.1, 0, -18, 100),
        Prior.parse_args(0, 0.1, 0, -18, 10000),
    ],
    (DichotomousModelChoices.d_multistage.value.id, PriorClass.frequentist_restricted): [
        Prior.parse_args(0, -17, 0, -18, 18),
        Prior.parse_args(0, 0.1, 0, 0, 100),
        Prior.parse_args(0, 0.1, 0, 0, 10000),
    ],
    (DichotomousModelChoices.d_multistage.value.id, PriorClass.bayesian): [
        Prior.parse_args(1, 0, 2, -20, 20),
        Prior.parse_args(2, 0, 0.5, 0.0001, 100),
        Prior.parse_args(2, 0, 1, 0.0001, 1000000),
    ],
    # hill
    (DichotomousModelChoices.d_hill.value.id, PriorClass.frequentist_unrestricted): [
        Prior.parse_args(0, -2, 0, -18, 18),
        Prior.parse_args(0, 0, 0, -18, 18),
        Prior.parse_args(0, 0, 0, -18, 18),
        Prior.parse_args(0, 1, 0, 0.00000001, 18),
    ],
    (DichotomousModelChoices.d_hill.value.id, PriorClass.frequentist_restricted): [
        Prior.parse_args(0, -2, 0, -18, 18),
        Prior.parse_args(0, 0, 0, -18, 18),
        Prior.parse_args(0, 0, 0, -18, 18),
        Prior.parse_args(0, 1, 0, 1, 18),
    ],
    (DichotomousModelChoices.d_hill.value.id, PriorClass.bayesian): [
        Prior.parse_args(1, -1, 2, -40, 40),
        Prior.parse_args(1, 0, 3, -40, 40),
        Prior.parse_args(1, -3, 3.3, -40, 40),
        Prior.parse_args(2, 0.693147, 0.5, 0.00000001, 40),
    ],
}
