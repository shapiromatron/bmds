from . import session, models, constants

from collections import OrderedDict


class BMDS_v230(session.Session):
    version = constants.BMDS230
    model_options = {
        constants.DICHOTOMOUS: OrderedDict([
            (constants.M_Logistic, models.Logistic_213),
            (constants.M_LogLogistic, models.LogLogistic_213),
            (constants.M_Probit, models.Probit_32),
            (constants.M_LogProbit, models.LogProbit_32),
            (constants.M_Multistage, models.Multistage_32),
            (constants.M_Gamma, models.Gamma_215),
            (constants.M_Weibull, models.Weibull_215),
        ]),
        constants.DICHOTOMOUS_CANCER: OrderedDict([
            (constants.M_MultistageCancer, models.MultistageCancer_19),
        ]),
        constants.CONTINUOUS: OrderedDict([
            (constants.M_Linear, models.Linear_216),
            (constants.M_Polynomial, models.Polynomial_216),
            (constants.M_Power, models.Power_216),
            (constants.M_Hill, models.Hill_216),
            (constants.M_ExponentialM2, models.Exponential_M2_17),
            (constants.M_ExponentialM3, models.Exponential_M3_17),
            (constants.M_ExponentialM4, models.Exponential_M4_17),
            (constants.M_ExponentialM5, models.Exponential_M5_17),
        ]),
    }


class BMDS_v231(BMDS_v230):
    version = constants.BMDS231


class BMDS_v240(BMDS_v231):
    version = constants.BMDS240
    model_options = {
        constants.DICHOTOMOUS: OrderedDict([
            (constants.M_Logistic, models.Logistic_214),
            (constants.M_LogLogistic, models.LogLogistic_214),
            (constants.M_Probit, models.Probit_33),
            (constants.M_LogProbit, models.LogProbit_33),
            (constants.M_Multistage, models.Multistage_33),
            (constants.M_Gamma, models.Gamma_216),
            (constants.M_Weibull, models.Weibull_216),
        ]),
        constants.DICHOTOMOUS_CANCER: OrderedDict([
            (constants.M_MultistageCancer, models.MultistageCancer_110),
        ]),
        constants.CONTINUOUS: OrderedDict([
            (constants.M_Linear, models.Linear_217),
            (constants.M_Polynomial, models.Polynomial_217),
            (constants.M_Power, models.Power_217),
            (constants.M_Hill, models.Hill_217),
            (constants.M_ExponentialM2, models.Exponential_M2_19),
            (constants.M_ExponentialM3, models.Exponential_M3_19),
            (constants.M_ExponentialM4, models.Exponential_M4_19),
            (constants.M_ExponentialM5, models.Exponential_M5_19),
        ]),
    }


class BMDS_v260(BMDS_v240):
    version = constants.BMDS260
    model_options = {
        constants.DICHOTOMOUS: OrderedDict([
            (constants.M_Logistic, models.Logistic_214),
            (constants.M_LogLogistic, models.LogLogistic_214),
            (constants.M_Probit, models.Probit_33),
            (constants.M_LogProbit, models.LogProbit_33),
            (constants.M_Multistage, models.Multistage_34),
            (constants.M_Gamma, models.Gamma_216),
            (constants.M_Weibull, models.Weibull_216),
        ]),
        constants.DICHOTOMOUS_CANCER: OrderedDict([
            (constants.M_MultistageCancer, models.MultistageCancer_110),
        ]),
        constants.CONTINUOUS: OrderedDict([
            (constants.M_Linear, models.Linear_220),
            (constants.M_Polynomial, models.Polynomial_220),
            (constants.M_Power, models.Power_218),
            (constants.M_Hill, models.Hill_217),
            (constants.M_ExponentialM2, models.Exponential_M2_110),
            (constants.M_ExponentialM3, models.Exponential_M3_110),
            (constants.M_ExponentialM4, models.Exponential_M4_110),
            (constants.M_ExponentialM5, models.Exponential_M5_110),
        ]),
    }


class BMDS_v2601(BMDS_v260):
    version = constants.BMDS2601


VERSIONS = {
    constants.BMDS230: BMDS_v230,
    constants.BMDS231: BMDS_v231,
    constants.BMDS240: BMDS_v240,
    constants.BMDS260: BMDS_v260,
    constants.BMDS2601: BMDS_v2601,
}


def get_versions():
    return VERSIONS.keys()


def get_session(version):
    return VERSIONS[version]


def get_models_for_version(version):
    collection = VERSIONS.get(version)
    if collection is None:
        raise ValueError('Unknown BMDS version')
    return collection.model_options
