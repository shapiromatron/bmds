from .. import constants
from . import rules


class Recommender(object):
    # Recommendation logic for a specified data-type.

    SUFFICIENTLY_CLOSE_BMDL = 3

    def __init__(self, dtype, overrides=None):

        rule_args = dict()
        if dtype == constants.DICHOTOMOUS:
            pass
        elif dtype in constants.CONTINUOUS_DTYPES:
            rule_args['continuous'] = True
        elif dtype == constants.DICHOTOMOUS_CANCER:
            rule_args['cancer'] = True
        else:
            raise ValueError('Invalid data type')

        self.dtype = dtype
        self.rules = self._get_rule_defaults(**rule_args)

        if overrides:
            raise NotImplementedError('Overrides not implemented (yet).')

    @classmethod
    def _get_rule_defaults(cls, continuous=False, cancer=False):
        continous_only = True if continuous else False
        cancer_only = True if cancer else False
        ggof_threshold = 0.05 if cancer else 0.1
        return [
            rules.BmdExists(
                failure_bin=constants.BIN_FAILURE,
            ),
            rules.BmdlExists(
                failure_bin=constants.BIN_FAILURE,
            ),
            rules.BmduExists(
                failure_bin=constants.BIN_NO_CHANGE,
                enabled=cancer_only,
            ),
            rules.AicExists(
                failure_bin=constants.BIN_FAILURE,
            ),
            rules.RoiExists(
                failure_bin=constants.BIN_WARNING,
            ),
            rules.CorrectVarianceModel(
                failure_bin=constants.BIN_WARNING,
                threshold=0.1,
                enabled=continous_only,
            ),
            rules.VarianceFit(
                failure_bin=constants.BIN_WARNING,
                threshold=0.1,
                enabled=continous_only,
            ),
            rules.GlobalFit(
                failure_bin=constants.BIN_WARNING,
                threshold=ggof_threshold,
            ),
            rules.BmdBmdlRatio(
                failure_bin=constants.BIN_NO_CHANGE,
                threshold=5.,
                rule_name='BMD/BMDL (warning)',
            ),
            rules.BmdBmdlRatio(
                failure_bin=constants.BIN_WARNING,
                threshold=20.,
            ),
            rules.RoiFit(
                failure_bin=constants.BIN_NO_CHANGE,
                threshold=2.,
            ),
            rules.Warnings(
                failure_bin=constants.BIN_NO_CHANGE,
            ),
            rules.HighBmd(
                failure_bin=constants.BIN_NO_CHANGE,
                threshold=1.,
            ),
            rules.HighBmdl(
                failure_bin=constants.BIN_WARNING,
                threshold=1.,
            ),
            rules.LowBmd(
                failure_bin=constants.BIN_NO_CHANGE,
                threshold=3.,
                rule_name='Low BMD (warning)',
            ),
            rules.LowBmdl(
                failure_bin=constants.BIN_NO_CHANGE,
                threshold=3.,
                rule_name='Low BMDL (warning)',
            ),
            rules.LowBmd(
                failure_bin=constants.BIN_WARNING,
                threshold=10.,
            ),
            rules.LowBmdl(
                failure_bin=constants.BIN_WARNING,
                threshold=10.,
            ),
            rules.ControlResidual(
                failure_bin=constants.BIN_WARNING,
                threshold=2.,
                enabled=continous_only,
            ),
            rules.ControlStdevResiduals(
                failure_bin=constants.BIN_WARNING,
                threshold=1.5,
                enabled=continous_only,
            ),
        ]

    def recommend(self, dataset, models):
        # apply rules to each model
        for model in models:

            # set global recommendations
            model.recommended = False
            model.recommended_variable = None

            # set no warnings by default (innocent until proven guilty)
            model.logic_notes = {
                constants.BIN_NO_CHANGE: [],
                constants.BIN_WARNING: [],
                constants.BIN_FAILURE: [],
            }

            # if no output is created, place model in failure bin
            if model.output_created:
                model.logic_bin = constants.BIN_NO_CHANGE
            else:
                model.logic_bin = constants.BIN_FAILURE
                continue

            # apply tests for each model
            for rule in self.rules:
                bin_, notes = rule.check(dataset, model.output)
                model.logic_bin = max(bin_, model.logic_bin)
                if notes:
                    model.logic_notes[bin_].append(notes)

        # get only models in highest bin-category
        model_subset = [
            model for model in models
            if model.logic_bin == constants.BIN_NO_CHANGE
        ]

        # exit early if there are no models left to recommend
        if len(model_subset) == 0:
            return

        # determine which approach to use for best-fitting model
        bmd_ratio = self._get_bmdl_ratio(model_subset)
        if bmd_ratio <= self.SUFFICIENTLY_CLOSE_BMDL:
            fld_name = 'AIC'
        else:
            fld_name = 'BMDL'

        # get and set recommended model
        model = self._get_min_model(model_subset, fld_name)
        model.recommended = True
        model.recommended_variable = fld_name
        return model

    def show_rules(self):
        return u'\n'.join([rule.__unicode__() for rule in self.rules])

    def _get_bmdl_ratio(self, models):
        """Return BMDL ratio in list of models."""
        bmdls = [
            model.output['BMDL']
            for model in models
            if model.output['BMDL'] > 0
        ]
        return max(bmdls) / min(bmdls)

    def _get_min_model(self, models, fld_name):
        """Return model with minimum value for specified field."""
        min_ = float('inf')
        idx = -1

        for i, model in enumerate(models):
            val = model.output[fld_name]
            if val != -999 and val < min_:
                idx = i
                min_ = val

        return models[idx]
