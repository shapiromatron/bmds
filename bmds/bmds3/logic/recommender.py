from typing import Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, validator
from simple_settings import settings

from ... import constants
from .constants import DEFAULT_RULE_ARGS, RULE_TYPES
from .rules import Rule


class Recommender(BaseModel):
    """
    Recommendation logic for a specified data-type.
    """

    dtype: str
    rule_args: Dict[str, Dict] = DEFAULT_RULE_ARGS
    rules: Optional[List[Rule]]

    @validator("dtype")
    def validate_dtype(cls, dtype):
        if dtype not in constants.DTYPES:
            raise ValueError("Invalid data type")
        return dtype

    @validator("rules", always=True)
    def validate_rules(cls, rules, values):
        if rules is None:
            rules = cls.rules_from_args(values["rule_args"])
        return rules

    @classmethod
    def rules_from_args(cls, rule_args):
        return [RULE_TYPES[type](**args) for (type, args) in rule_args.items()]

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
                bin_, notes = rule.check(self.dtype, model.settings, dataset, model.results.dict())
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
        if bmd_ratio <= settings.SUFFICIENTLY_CLOSE_BMDL:
            fld_name = "aic"
        else:
            fld_name = "bmdl"

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
            data=[rule.as_row() for rule in self.rules if rule.enabled(self.dtype)],
            columns=[
                "rule_name",
                "enabled_nested",
                "enabled_continuous",
                "enabled_dichotomous",
                "failure_bin",
                "threshold",
            ],
        )
        return df

    def _get_bmdl_ratio(self, models):
        """Return BMDL ratio in list of models."""
        bmdls = [model.results.bmdl for model in models if getattr(model.results, "bmdl", 0) > 0]

        return max(bmdls) / min(bmdls) if len(bmdls) > 0 else 0

    @staticmethod
    def _get_recommended_models(models, fld_name):
        """
        Returns a list of models which have the minimum target field value
        for a given field name (AIC or BMDL).
        """
        target_value = min([getattr(model.results, fld_name) for model in models])
        return [model for model in models if getattr(model.results, fld_name) == target_value]

    @staticmethod
    def _get_parsimonious_model(models):
        """
        Return the most parsimonious model of all available models. The most
        parsimonious model is defined as the model with the fewest number of
        parameters.
        """
        params = [len(model.results.fit.params) for model in models]
        idx = params.index(min(params))
        return models[idx]
