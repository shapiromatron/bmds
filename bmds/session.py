from copy import deepcopy
from collections import OrderedDict
import os
import pandas as pd

from . import constants, logic, models, utils


__all__ = ('BMDS', )


class BMDS(object):
    """
    A single dataset, related models, outputs, and model recommendations.
    """

    @utils.classproperty
    def versions(cls):
        """
        Return all available BMDS software versions.

        Example
        -------

        import bmds

        Returns
        -------
        OrderedDict of available BMDS versions.
        """
        return _BMDS_VERSIONS

    @classmethod
    def get_model(cls, version, model_name):
        """
        Return BMDS model class given BMDS version and model-name.
        """
        models = cls.versions[version].model_options
        for keystore in models.values():
            if model_name in keystore:
                return keystore[model_name]
        raise ValueError('Unknown model name')

    @classmethod
    def get_versions(cls):
        return cls.versions.keys()

    @classmethod
    def latest_version(cls, *args, **kwargs):
        # return the latest version of BMDS. If arguments are provided, create
        # an instance and return, otherwise return the class
        cls = list(cls.versions.values())[-1]
        if len(args) > 0 or len(kwargs) > 0:
            return cls(*args, **kwargs)
        return cls

    bmr_options = {
        constants.DICHOTOMOUS: constants.DICHOTOMOUS_BMRS,
        constants.DICHOTOMOUS_CANCER: constants.DICHOTOMOUS_BMRS,
        constants.CONTINUOUS: constants.CONTINUOUS_BMRS,
        constants.CONTINUOUS_INDIVIDUAL: constants.CONTINUOUS_BMRS
    }

    def __init__(self, dtype, dataset=None):
        self.dtype = dtype
        if self.dtype not in constants.DTYPES:
            raise ValueError('Invalid data type')
        self.models = []
        self.dataset = dataset

    @property
    def model_options(self):
        raise NotImplementedError('Abstract method requires implementation')

    def get_bmr_options(self):
        return self.bmr_options[self.dtype]

    def get_model_options(self):
        return [
            model.get_default()
            for model in self.model_options[self.dtype].values()
        ]

    @property
    def has_models(self):
        return len(self.models) > 0

    def add_default_models(self, global_overrides=None):
        for name in self.model_options[self.dtype].keys():
            overrides = deepcopy(global_overrides) \
                if global_overrides is not None \
                else None

            if name in constants.VARIABLE_POLYNOMIAL:
                start_idx = 2 if name == constants.M_Polynomial else 1
                for i in range(start_idx, min(self.dataset.num_dose_groups, 8)):
                    overrides = {} \
                        if overrides is None \
                        else deepcopy(overrides)
                    overrides['degree_poly'] = i
                    self.add_model(name, overrides=overrides)
            else:
                self.add_model(name, overrides=overrides)

    def add_model(self, name, overrides=None, id=None):
        if self.dataset is None:
            raise ValueError('Add dataset to session before adding models')
        Model = self.model_options[self.dtype][name]
        instance = Model(
            dataset=self.dataset,
            overrides=overrides,
            id=id,
        )
        self.models.append(instance)

    def execute(self):
        for model in self.models:
            model.execute()

    @property
    def recommendation_enabled(self):
        return hasattr(self, 'recommended_model')

    def add_recommender(self, overrides=None):
        self.recommender = logic.Recommender(self.dtype, overrides)

    def recommend(self):
        if not hasattr(self, 'recommender'):
            self.add_recommender()
        self.recommended_model = self.recommender.recommend(self.dataset, self.models)
        self.recommended_model_index = self.models.index(self.recommended_model) \
            if self.recommended_model is not None \
            else None
        return self.recommended_model

    @staticmethod
    def _df_ordered_dict(include_io=True):
        # return an ordered defaultdict list
        keys = [
            'dataset_index', 'model_name', 'model_index',
            'model_version', 'has_output',

            'BMD', 'BMDL', 'BMDU', 'CSF',
            'AIC', 'pvalue1', 'pvalue2', 'pvalue3', 'pvalue4',
            'Chi2', 'df', 'residual_of_interest',
            'warnings',

            'logic_bin', 'logic_cautions', 'logic_warnings', 'logic_failures',
            'recommended', 'recommended_variable',
        ]

        if include_io:
            keys.extend(['dfile', 'outfile'])

        return OrderedDict([(key, list()) for key in keys])

    def _add_to_to_ordered_dict(self, d, dataset_index, recommended_only=False):

        for model_index, model in enumerate(self.models):

            # determine if model should be presented, or if a null-model should
            # be presented (if no model is recommended.)
            show_null = False
            if recommended_only:
                if self.recommendation_enabled:
                    if self.recommended_model is None:
                        if model_index == 0:
                            show_null = True
                        else:
                            continue
                    elif self.recommended_model == model:
                        pass
                    else:
                        continue
                else:
                    if model_index == 0:
                        show_null = True
                    else:
                        continue

            d['dataset_index'].append(dataset_index)
            model._to_df(d, model_index, show_null)

    def to_dict(self, dataset_index):
        return dict(
            dataset_index=dataset_index,
            dataset=self.dataset.to_dict(),
            models=[model.to_dict(i) for i, model in enumerate(self.models)],
            recommended_model_index=getattr(self, 'recommended_model_index', None)
        )

    def to_excel(self, filename):
        d = self._df_ordered_dict()
        self._add_to_to_ordered_dict(d, 0)
        df = pd.DataFrame(d)
        filename = os.path.expanduser(filename)
        df.to_excel(filename, index=False)

    def save_plots(self, directory, prefix=None, format='png', recommended_only=False):
        directory = os.path.expanduser(directory)
        if not os.path.exists(directory):
            raise ValueError('Directory not found: {}'.format(directory))

        for model in self.models:
            if recommended_only and (self.recommendation_enabled is False or
                                     model.recommended is False):
                continue

            fn = '{}.{}'.format(model.name, format)
            if prefix is not None:
                fn = '{}-{}'.format(prefix, fn)

            fig = model.plot()
            fig.savefig(os.path.join(directory, fn))


class BMDS_v231(BMDS):
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
        constants.CONTINUOUS_INDIVIDUAL: OrderedDict([
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
            (constants.M_DichotomousHill, models.DichotomousHill_13),
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
        constants.CONTINUOUS_INDIVIDUAL: OrderedDict([
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


_BMDS_VERSIONS = OrderedDict((
    (constants.BMDS231, BMDS_v231),
    (constants.BMDS240, BMDS_v240),
    (constants.BMDS260, BMDS_v260),
    (constants.BMDS2601, BMDS_v2601),
))
