from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, validator

from ...constants import BIN_ICON, BIN_TEXT, LogicBin
from ...datasets import DatasetBase
from ..models.base import BmdModel
from .checks import RULE_MAP, CheckResponse
from .constants import RuleClass


class Rule(BaseModel):
    rule_class: RuleClass
    failure_bin: LogicBin
    threshold: Optional[float] = None
    enabled_dichotomous: bool = True
    enabled_continuous: bool = True
    enabled_nested: bool = True

    def bin_icon(self) -> str:
        return BIN_ICON[self.failure_bin]

    def bin_text(self) -> str:
        return BIN_TEXT[self.failure_bin]

    def to_row(self):
        return [
            self.rule_class,
            self.failure_bin,
            self.threshold,
            self.enabled_dichotomous,
            self.enabled_continuous,
            self.enabled_nested,
        ]


class RecommenderSettings(BaseModel):
    enabled: bool = True
    recommend_questionable: bool = False
    recommend_viable: bool = True
    sufficiently_close_bmdl: float = 3
    rules: List[Rule]

    _default: Optional[str] = None

    @validator("rules")
    def rules_all_classes(cls, rules):
        rule_classes = set(rule.rule_class for rule in rules)
        all_rule_classes = set(RuleClass.__members__)
        missing = all_rule_classes - rule_classes
        if missing:
            raise ValueError(f"Rule list must be complete; missing {missing}")
        return rules

    @classmethod
    def build_default(cls) -> "RecommenderSettings":
        if cls._default is None:
            path = Path(__file__).parent / "default.json"
            cls._default = path.read_text()
        return cls.parse_raw(cls._default)

    def to_df(self) -> pd.DataFrame:
        df = pd.DataFrame(
            data=[rule.to_row() for rule in self.rules],
            columns=[
                "rule_class",
                "failure_bin",
                "threshold",
                "enabled_dichotomous",
                "enabled_continuous",
                "enabled_nested",
            ],
        )
        return df


class RecommenderResults(BaseModel):
    recommended_model_index: Optional[int]
    recommended_model_variable: Optional[str]
    model_bin: List[LogicBin] = []
    model_notes: List[Dict[int, List[str]]] = []


class RecommenderSchema(BaseModel):
    settings: RecommenderSettings
    results: RecommenderResults

    def deserialize(self) -> "Recommender":
        recommender = Recommender(self.settings)
        recommender.results = self.results
        return recommender


class Recommender:
    """
    Recommendation logic for a specified data-type.
    """

    def __init__(self, settings: Optional[RecommenderSettings] = None):
        if settings is None:
            settings = RecommenderSettings.build_default()
        self.settings: RecommenderSettings = settings
        self.results: Optional[RecommenderResults] = None

    def check(self, dataset: DatasetBase, model, rule: Rule) -> CheckResponse:
        CheckClass = RULE_MAP[rule.rule_class]
        return CheckClass.check(dataset, model, rule)

    def recommend(self, dataset: DatasetBase, models: List[BmdModel]):
        self.results = RecommenderResults()

        if not self.settings.enabled:
            return

        # apply rules to each model
        model_bins = []
        model_notes = []
        for model in models:
            # set defaults
            current_bin = LogicBin.NO_CHANGE
            notes: Dict[int, List[str]] = {
                LogicBin.NO_CHANGE: [],
                LogicBin.WARNING: [],
                LogicBin.FAILURE: [],
            }

            if model.has_results:
                # apply tests for each model
                for rule in self.settings.rules:
                    response = self.check(dataset, model, rule)
                    current_bin = max(response.logic_bin, current_bin)
                    if response.message:
                        notes[response.logic_bin].append(response.message)
            else:
                current_bin = LogicBin.FAILURE
                notes[LogicBin.FAILURE].append("Did not successfully execute.")

            model_bins.append(current_bin)
            model_notes.append(notes)

        self.results.model_bin = model_bins
        self.results.model_notes = model_notes

        # get only models in highest bin-category
        valid_model_indicies = []
        for idx, model_bin in enumerate(model_bins):
            if (self.settings.recommend_viable and model_bin == LogicBin.NO_CHANGE) or (
                self.settings.recommend_questionable and model_bin == LogicBin.WARNING
            ):
                valid_model_indicies.append(idx)

        model_subset = [models[idx] for idx in valid_model_indicies]

        # exit early if there are no models left to recommend
        if len(model_subset) == 0:
            return

        # determine which approach to use for best-fitting model
        bmd_ratio = self._get_bmdl_ratio(model_subset)
        if bmd_ratio <= self.settings.sufficiently_close_bmdl:
            field = "aic"
        else:
            field = "bmdl"

        self.results.recommended_model_variable = field

        # get and set recommended model
        model_subset = self._get_recommended_models(model_subset, field)
        model = self._get_parsimonious_model(model_subset)
        self.results.recommended_model_index = models.index(model)

    def _get_bmdl_ratio(self, models: List[BmdModel]) -> float:
        """Return BMDL ratio in list of models."""
        bmdls = [model.results.bmdl for model in models if model.results.bmdl > 0]
        return max(bmdls) / min(bmdls)

    def _get_recommended_models(self, models: List[BmdModel], field: str) -> List[BmdModel]:
        """
        Returns a list of models which have the minimum target field value
        for a given field name (AIC or BMDL).
        """
        target_value = min([getattr(model.results, field) for model in models])
        return [model for model in models if getattr(model.results, field) == target_value]

    def _get_parsimonious_model(self, models: List[BmdModel]) -> BmdModel:
        """
        Return the most parsimonious model of all available models. The most
        parsimonious model is defined as the model with the fewest number of
        parameters.
        """
        params = [len(model.results.fit.params) for model in models]
        idx = params.index(min(params))
        return models[idx]

    def serialize(self) -> RecommenderSchema:
        return RecommenderSchema(settings=self.settings, results=self.results)
