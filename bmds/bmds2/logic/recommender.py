import pandas as pd
from simple_settings import settings

from ... import constants
from . import rules


class Recommender:
    """
    Recommendation logic for a specified data-type.
    """

    def __init__(self, dtype):
        rule_args = dict()
        if dtype == constants.DICHOTOMOUS:
            pass
        elif dtype in constants.CONTINUOUS_DTYPES:
            rule_args["continuous"] = True
        elif dtype == constants.DICHOTOMOUS_CANCER:
            rule_args["cancer"] = True
        else:
            raise ValueError("Invalid data type")

        self.dtype = dtype
        self.rules = self._get_rule_defaults(**rule_args)
        self.SUFFICIENTLY_CLOSE_BMDL = settings.SUFFICIENTLY_CLOSE_BMDL

    @classmethod
    def _get_rule_defaults(cls, continuous=False, cancer=False):
        continous_only = True if continuous else False
        cancer_only = True if cancer else False
        ggof_threshold = 0.05 if cancer else 0.1
        return [
            rules.BmdExists(failure_bin=constants.BIN_FAILURE),
            rules.BmdlExists(failure_bin=constants.BIN_FAILURE),
            rules.BmduExists(failure_bin=constants.BIN_NO_CHANGE, enabled=cancer_only),
            rules.AicExists(failure_bin=constants.BIN_FAILURE),
            rules.RoiExists(failure_bin=constants.BIN_WARNING),
            rules.CorrectVarianceModel(failure_bin=constants.BIN_WARNING, enabled=continous_only),
            rules.VarianceModelFit(failure_bin=constants.BIN_WARNING, enabled=continous_only),
            rules.GlobalFit(failure_bin=constants.BIN_WARNING, threshold=ggof_threshold),
            rules.RoiFit(failure_bin=constants.BIN_WARNING, threshold=2.0),
            rules.BmdBmdlRatio(
                failure_bin=constants.BIN_NO_CHANGE,
                threshold=5.0,
                rule_name="BMD to BMDL ratio (warning)",
            ),
            rules.BmdBmdlRatio(failure_bin=constants.BIN_WARNING, threshold=20.0),
            rules.NoDegreesOfFreedom(failure_bin=constants.BIN_WARNING),
            rules.Warnings(failure_bin=constants.BIN_NO_CHANGE),
            rules.HighBmd(failure_bin=constants.BIN_NO_CHANGE, threshold=1.0),
            rules.HighBmdl(failure_bin=constants.BIN_NO_CHANGE, threshold=1.0),
            rules.LowBmd(
                failure_bin=constants.BIN_NO_CHANGE, threshold=3.0, rule_name="Low BMD (warning)"
            ),
            rules.LowBmdl(
                failure_bin=constants.BIN_NO_CHANGE, threshold=3.0, rule_name="Low BMDL (warning)"
            ),
            rules.LowBmd(failure_bin=constants.BIN_WARNING, threshold=10.0),
            rules.LowBmdl(failure_bin=constants.BIN_WARNING, threshold=10.0),
            rules.ControlResidual(
                failure_bin=constants.BIN_WARNING, threshold=2.0, enabled=continous_only
            ),
            rules.ControlStdevResiduals(
                failure_bin=constants.BIN_WARNING, threshold=1.5, enabled=continous_only
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
        model_subset = [model for model in models if model.logic_bin == constants.BIN_NO_CHANGE]

        # exit early if there are no models left to recommend
        if len(model_subset) == 0:
            return

        # determine which approach to use for best-fitting model
        bmd_ratio = self._get_bmdl_ratio(model_subset)
        if bmd_ratio <= self.SUFFICIENTLY_CLOSE_BMDL:
            fld_name = "AIC"
        else:
            fld_name = "BMDL"

        # get and set recommended model
        model_subset = self._get_recommended_models(model_subset, fld_name)
        model = self._get_parsimonious_model(model_subset)
        model.recommended = True
        model.recommended_variable = fld_name
        return model

    def show_rules(self):
        return "\n".join([rule.__unicode__() for rule in self.rules])

    def rules_df(self):
        df = pd.DataFrame(
            data=[rule.as_row() for rule in self.rules],
            columns=["rule_name", "enabled", "failure_bin", "threshold"],
        )
        df = df[df.enabled == True]  # noqa: E712
        return df

    def _get_bmdl_ratio(self, models):
        """Return BMDL ratio in list of models."""
        bmdls = [model.output["BMDL"] for model in models if model.output["BMDL"] > 0]

        return max(bmdls) / min(bmdls) if len(bmdls) > 0 else 0

    @staticmethod
    def _get_recommended_models(models, fld_name):
        """
        Returns a list of models which have the minimum target field value
        for a given field name (AIC or BMDL).
        """
        target_value = min([model.output[fld_name] for model in models])
        return [model for model in models if model.output[fld_name] == target_value]

    @staticmethod
    def _get_parsimonious_model(models):
        """
        Return the most parsimonious model of all available models. The most
        parsimonious model is defined as the model with the fewest number of
        parameters.
        """
        params = [len(model.output["parameters"]) for model in models]
        idx = params.index(min(params))
        return models[idx]
