import asyncio
from copy import deepcopy
from collections import OrderedDict
import os
import pandas as pd
from simple_settings import settings
import sys

from . import __version__, constants, logic, models, utils, reporter


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
        """
        Return the class of the latest version of the BMDS. If additional
        arguments are provided, an instance of this class is generated.
        """
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

        max_poly_order = min(
            self.dataset.num_dose_groups,
            settings.MAXIMUM_POLYNOMIAL_ORDER + 1)

        for name in self.model_options[self.dtype].keys():
            overrides = deepcopy(global_overrides) \
                if global_overrides is not None \
                else None

            if name in constants.VARIABLE_POLYNOMIAL:
                min_poly_order = 2 if name == constants.M_Polynomial else 1
                for i in range(min_poly_order, max_poly_order):
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

    async def execute_models(self):
        tasks = [
            model.execute_job()
            for model in self.models
        ]
        await asyncio.wait(tasks)

    def execute(self):
        if sys.platform == 'win32':
            loop = asyncio.ProactorEventLoop()
            asyncio.set_event_loop(loop)
        else:
            loop = asyncio.get_event_loop()
        loop.run_until_complete(self.execute_models())
        loop.close()

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

    def execute_and_recommend(self, drop_doses=False):
        """
        Execute and recommend a best-fitting model. If drop_doses and no model
        is recommended, drop the highest dose-group and repeat until either:

        1. a model is recommended, or
        2. the dataset is exhausted (i.e., only 3 dose-groups remain).

        This method adds two new attributes to Session:

        1. original_dataset - the unchanged original dataset. The dataset used
            on the object is mutated if doses were dropped
        2. doses_dropped - the number of doses that were dropped

        """
        original_dataset = deepcopy(self.dataset)
        doses_dropped = 0

        self.execute()
        self.recommend()

        if not drop_doses:
            return

        while self.recommended_model is None and \
                self.dataset.num_dose_groups > 3:
            doses_dropped += 1
            self.dataset.drop_dose()
            self.execute()
            self.recommend()

        self.original_dataset = original_dataset
        self.doses_dropped = doses_dropped

    @staticmethod
    def _df_ordered_dict(include_io=True):
        # return an ordered defaultdict list
        keys = [
            'dataset_index', 'model_name', 'model_index',
            'model_version', 'has_output', 'execution_halted',

            'BMD', 'BMDL', 'BMDU', 'CSF',
            'AIC', 'pvalue1', 'pvalue2', 'pvalue3', 'pvalue4',
            'Chi2', 'df', 'residual_of_interest',
            'warnings',

            'logic_bin', 'logic_cautions', 'logic_warnings', 'logic_failures',
            'recommended', 'recommended_variable',
        ]

        if include_io:
            keys.extend(['dfile', 'outfile', 'stdout', 'stderr'])

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
            bmds_version=self.version,
            bmds_python_version=__version__,
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

    def save_plots(self, directory, prefix=None,
                   format='png', dpi=None, recommended_only=False):

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
            fig.savefig(os.path.join(directory, fn), dpi=dpi)
            fig.clear()

    def to_docx(self, filename=None, title=None,
                input_dataset=True, summary_table=True,
                recommendation_details=True, recommended_model=True,
                all_models=False):
        """
        Write session outputs to a Word file.

        Parameters
        ----------
        filename : str or None
            If provided, the file is saved to this location, otherwise this
            method returns a docx.Document
        input_dataset : bool
            Include input dataset data table
        summary_table : bool
            Include model summary table
        recommendation_details : bool
            Include model recommendation details table
        recommended_model : bool
            Include the recommended model output and dose-response plot, if
            one exists
        all_models : bool
            Include all models output and dose-response plots

        Returns
        -------
        bmds.Reporter
            The bmds.Reporter object.

        """
        rep = reporter.Reporter()
        rep.add_session(self, input_dataset, summary_table,
                        recommendation_details, recommended_model, all_models)

        if filename:
            rep.save(filename)

        return rep

    def _group_models(self):
        """
        If AIC and BMD are numeric and identical, then treat models as
        identical. Returns a list of lists. The outer list is a list of related
        models, the inner list contains each individual model, sorted by the
        number of parameters in ascending order.

        This is required because in some cases, a higher-order model may not use
        some parameters and can effectively collapse to a lower order model
        (for example, a 2nd order polynomial and power model may collapse to a
        linear model). In summary outputs, we may want to present all models in
        one row, since they are the same model effectively.
        """
        od = OrderedDict()

        # Add models to appropriate list. We only aggregate models which
        # completed successfully and have a valid AIC and BMD.
        for i, model in enumerate(self.models):
            output = getattr(model, 'output', {})
            if output.get('AIC') and output.get('BMD') and output['BMD'] > 0:
                key = '{}-{}'.format(output['AIC'], output['BMD'])
                if key in od:
                    od[key].append(model)
                else:
                    od[key] = [model]
            else:
                od[i] = [model]

        # Sort each list by the number of parameters
        def _get_num_params(model):
            return len(model.output['parameters']) \
                if hasattr(model, 'output') and 'parameters' in model.output \
                else 0

        for key, _models in od.items():
            _models.sort(key=_get_num_params)

        return list(od.values())


class BMDS_v231(BMDS):
    version = constants.BMDS231
    version_pretty = 'BMDS v2.3.1'


class BMDS_v240(BMDS_v231):
    version = constants.BMDS240
    version_pretty = 'BMDS v2.4.0'
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
    version_pretty = 'BMDS v2.6.0'
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
    version_pretty = 'BMDS v2.6.0.1'


class BMDS_v270(BMDS_v2601):
    version = constants.BMDS270
    version_pretty = 'BMDS v2.7.0'
    model_options = {
        constants.DICHOTOMOUS: OrderedDict([
            (constants.M_Logistic, models.Logistic_215),
            (constants.M_LogLogistic, models.LogLogistic_215),
            (constants.M_Probit, models.Probit_34),
            (constants.M_LogProbit, models.LogProbit_34),
            (constants.M_Multistage, models.Multistage_34),
            (constants.M_Gamma, models.Gamma_217),
            (constants.M_Weibull, models.Weibull_217),
            (constants.M_DichotomousHill, models.DichotomousHill_13),
        ]),
        constants.DICHOTOMOUS_CANCER: OrderedDict([
            (constants.M_MultistageCancer, models.MultistageCancer_34),
        ]),
        constants.CONTINUOUS: OrderedDict([
            (constants.M_Linear, models.Linear_221),
            (constants.M_Polynomial, models.Polynomial_221),
            (constants.M_Power, models.Power_219),
            (constants.M_Hill, models.Hill_218),
            (constants.M_ExponentialM2, models.Exponential_M2_111),
            (constants.M_ExponentialM3, models.Exponential_M3_111),
            (constants.M_ExponentialM4, models.Exponential_M4_111),
            (constants.M_ExponentialM5, models.Exponential_M5_111),
        ]),
        constants.CONTINUOUS_INDIVIDUAL: OrderedDict([
            (constants.M_Linear, models.Linear_221),
            (constants.M_Polynomial, models.Polynomial_221),
            (constants.M_Power, models.Power_219),
            (constants.M_Hill, models.Hill_218),
            (constants.M_ExponentialM2, models.Exponential_M2_111),
            (constants.M_ExponentialM3, models.Exponential_M3_111),
            (constants.M_ExponentialM4, models.Exponential_M4_111),
            (constants.M_ExponentialM5, models.Exponential_M5_111),
        ]),
    }


_BMDS_VERSIONS = OrderedDict((
    (constants.BMDS231, BMDS_v231),
    (constants.BMDS240, BMDS_v240),
    (constants.BMDS260, BMDS_v260),
    (constants.BMDS2601, BMDS_v2601),
    (constants.BMDS270, BMDS_v270),
))
