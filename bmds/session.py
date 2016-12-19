from copy import deepcopy
from . import datasets, constants, logic, models, utils

from collections import OrderedDict

__all__ = ('BMDS', )


class BMDS(object):

    @utils.classproperty
    def versions(cls):
        return _BMDS_VERSIONS

    @classmethod
    def get_model(cls, version, model_name):
        # given a bmds version and model_name, return model class
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

    def add_dataset(self, **kwargs):
        if self.dtype == constants.CONTINUOUS:
            ds = datasets.ContinuousDataset(**kwargs)
        elif self.dtype == constants.CONTINUOUS_INDIVIDUAL:
            ds = datasets.ContinuousIndividualDataset(**kwargs)
        elif self.dtype in constants.DICHOTOMOUS_DTYPES:
            ds = datasets.DichotomousDataset(**kwargs)
        else:
            raise ValueError('Invalid dtype')
        self.dataset = ds

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
                for i in range(start_idx, min(self.dataset.doses_used, 8)):
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

    def _to_df(self, d, dataset_index, recommended_only):

        def _nullify(show_null, value):
            return '-' if show_null else value

        for model_index, model in enumerate(self.models):

            # special case for determining what to present.
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

            # add general model information
            d['dataset_index'].append(dataset_index)

            d['model_name'].append(_nullify(show_null, model.model_name))
            d['model_index'].append(_nullify(show_null, model_index))
            d['model_version'].append(_nullify(show_null, model.version))
            d['has_output'].append(_nullify(show_null, model.output_created))

            # add model outputs
            outputs = {} \
                if show_null \
                else getattr(model, 'output', {})

            d['BMD'].append(outputs.get('BMD', '-'))
            d['BMDL'].append(outputs.get('BMDL', '-'))
            d['BMDU'].append(outputs.get('BMDU', '-'))
            d['CSF'].append(outputs.get('CSF', '-'))
            d['AIC'].append(outputs.get('AIC', '-'))
            d['pvalue1'].append(outputs.get('p_value1', '-'))
            d['pvalue2'].append(outputs.get('p_value2', '-'))
            d['pvalue3'].append(outputs.get('p_value3', '-'))
            d['pvalue4'].append(outputs.get('p_value4', '-'))
            d['Chi2'].append(outputs.get('Chi2', '-'))
            d['df'].append(outputs.get('df', '-'))
            d['residual_of_interest'].append(outputs.get('residual_of_interest', '-'))
            d['warnings'].append('; '.join(outputs.get('warnings', ['-'])))

            # add logic bin and warnings
            logics = getattr(model, 'logic_notes', {})
            bin_ = constants.BIN_TEXT[model.logic_bin] \
                if hasattr(model, 'logic_bin') \
                else '-'
            d['logic_bin'].append(_nullify(show_null, bin_))

            txt = '; '.join(logics.get(constants.BIN_NO_CHANGE, ['-']))
            d['logic_cautions'].append(_nullify(show_null, txt))
            txt = '; '.join(logics.get(constants.BIN_WARNING, ['-']))
            d['logic_warnings'].append(_nullify(show_null, txt))
            txt = '; '.join(logics.get(constants.BIN_FAILURE, ['-']))
            d['logic_failures'].append(_nullify(show_null, txt))

            # add recommendation and recommendation variable
            txt = getattr(model, 'recommended', '-')
            d['recommended'].append(_nullify(show_null, txt))
            txt = getattr(model, 'recommended_variable', '-')
            d['recommended_variable'].append(_nullify(show_null, txt))

            # add verbose outputs if specified
            if 'dfile' in d:
                txt = model.as_dfile()
                d['dfile'].append(_nullify(show_null, txt))
            if 'outfile' in d:
                txt = getattr(model, 'outfile', '-')
                d['outfile'].append(_nullify(show_null, txt))

    def _to_dict(self, dataset_index):
        return dict(
            dataset_index=dataset_index,
            dataset=self.dataset._to_dict(),
            models=[model._to_dict(i) for i, model in enumerate(self.models)],
            recommended_model_index=getattr(self, 'recommended_model_index', None)
        )


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
