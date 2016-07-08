from . import models, constants


class Session(object):

    bmrs = {
        constants.DICHOTOMOUS: constants.DICHOTOMOUS_BMRS,
        constants.DICHOTOMOUS_CANCER: constants.DICHOTOMOUS_BMRS,
        constants.CONTINUOUS: constants.CONTINUOUS_BMRS
    }

    @classmethod
    def get_default_params(cls, dtype):
        models = cls.models[dtype]
        return [model.get_default() for model in models]


class BMDS_v230(Session):

    models = {
        constants.DICHOTOMOUS: {
            'Weibull': models.Weibull_215,
            'LogProbit': models.LogProbit_32,
            'Probit': models.Probit_32,
            'Multistage': models.Multistage_32,
            'Gamma': models.Gamma_215,
            'Logistic': models.Logistic_213,
            'LogLogistic': models.LogLogistic_213
        },
        constants.DICHOTOMOUS_CANCER: {
            'Multistage-Cancer': models.MultistageCancer_19
        },
        constants.CONTINUOUS: {
            'Linear': models.Linear_216,
            'Polynomial': models.Polynomial_216,
            'Power': models.Power_216,
            'Exponential-M2': models.Exponential_M2_17,
            'Exponential-M3': models.Exponential_M3_17,
            'Exponential-M4': models.Exponential_M4_17,
            'Exponential-M5': models.Exponential_M5_17,
            'Hill': models.Hill_216,
        },
        }


class BMDS_v231(BMDS_v230):
    pass


class BMDS_v240(BMDS_v231):

    models = {
        constants.DICHOTOMOUS: {
            'Weibull': models.Weibull_216,
            'LogProbit': models.LogProbit_33,
            'Probit': models.Probit_33,
            'Multistage': models.Multistage_33,
            'Gamma': models.Gamma_216,
            'Logistic': models.Logistic_214,
            'LogLogistic': models.LogLogistic_214,
        },
        constants.DICHOTOMOUS_CANCER: {
            'Multistage-Cancer': models.MultistageCancer_110,
        },
        constants.CONTINUOUS: {
            'Linear': models.Linear_217,
            'Polynomial': models.Polynomial_217,
            'Power': models.Power_217,
            'Exponential-M2': models.Exponential_M2_19,
            'Exponential-M3': models.Exponential_M3_19,
            'Exponential-M4': models.Exponential_M4_19,
            'Exponential-M5': models.Exponential_M5_19,
            'Hill': models.Hill_217,
        },
    }


VERSIONS = {
    '2.30': BMDS_v230,
    '2.31': BMDS_v231,
    '2.40': BMDS_v240
}


def get_versions():
    return VERSIONS.keys()


def get_models_for_version(version):
    collection = VERSIONS.get(version)()
    if collection is None:
        raise ValueError('Unknown BMDS version')
    return collection.models
