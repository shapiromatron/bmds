from .. import constants, rules


class Recommender(object):

    SUFFICIENTLY_CLOSE_BMDL = 3

    def __init__(self, dtype):
        self.dtype = dtype
        if self.dtype not in constants.DTYPES:
            raise ValueError('Invalid data type')
        self._load_default_rules()

    def _load_default_rules(self):
        # load default rules based on self.dtype
        self.rules = rules  # todo: load from JSON

    def recommend(self, dataset, models):
        # apply rules to each model
        for model in models:

            # set global recommendations
            model.recommended = False
            model.recommended_notes = ''

            # set no warnings by default (innocent until proven guilty)
            model.logic_bin = constants.BIN_NO_CHANGE
            model.logic_notes = {
                constants.BIN_NO_CHANGE: [],
                constants.BIN_WARNING: [],
                constants.BIN_FAILURE: [],
            }

            # apply tests for each model
            for rule in self.rules:
                bin_, notes = rule.apply_rule(model.outputs)
                model.bin = max(bin_, model.bin)
                if notes:
                    model.logic_notes[bin_].append(notes)

        # get only models in highest bin-category
        model_subset = [
            model for model in models
            if model.logic_bin == constants.BIN_NO_CHANGE
        ]

        # determine which approach to use for best-fitting model
        bmd_ratio = self._get_bmdl_ratio(model_subset)
        if bmd_ratio <= self.SUFFICIENTLY_CLOSE_BMDL:
            fld_name = 'AIC'
        else:
            fld_name = 'BMDL'

        # get and set recommended model
        model = self._get_min_model(model_subset, fld_name)
        if model:
            model.recommended = True
            model.recommended_variable = fld_name

    def _get_bmdl_ratio(self, models):
        """Return BMDL ratio in list of models."""
        bmdls = [
            model.output['BMDL']
            for model in models
            if model.output['BMDL'] > 0
        ]
        return max(bmdls) / min(bmdls)

    def _get_min_model(models, fld_name):
        """Get model with minimum value for specified field."""
        min_ = float('inf')
        idx = -1

        for i, model in enumerate(models):
            val = model.output[fld_name]
            if val != -999 and val < min_:
                idx = i
                min_ = val

        return models[idx] if idx >= 0 else None
