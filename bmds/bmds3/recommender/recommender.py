import itertools
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, validator

from ...constants import BIN_ICON, BIN_TEXT, BIN_TEXT_BMDS3, LogicBin
from ...datasets import DatasetBase
from ..models.base import BmdModel
from .checks import RULE_MAP, CheckResponse
from .constants import RuleClass


class Rule(BaseModel):
    rule_class: RuleClass
    failure_bin: LogicBin
    threshold: float | None = None
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
    rules: list[Rule]

    _default: str | None = None

    @validator("rules")
    def rules_all_classes(cls, rules):
        rule_classes = set(rule.rule_class for rule in rules)
        all_rule_classes = set(RuleClass.__members__)
        missing = all_rule_classes - rule_classes
        if missing:
            raise ValueError(f"Rule list must be complete; missing {missing}")
        return rules

    @classmethod
    def build_default(cls) -> Self:
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
    recommended_model_index: int | None
    recommended_model_variable: str | None
    model_bin: list[LogicBin] = []
    model_notes: list[dict[int, list[str]]] = []

    def bin_text(self, index: int) -> str:
        if self.recommended_model_index == index:
            return f"Recommended - Lowest {self.recommended_model_variable.upper()}"
        return BIN_TEXT_BMDS3[self.model_bin[index]]

    def notes_text(self, index: int) -> str:
        notes = self.model_notes[index].values()
        return "\n".join(sorted([text for text in itertools.chain(*notes)], reverse=True))

    def update_record(self, d: dict, index: int) -> None:
        """Update data record for a tabular-friendly export"""
        d.update(
            recommended=(index == self.recommended_model_index),
            recommendation_bin=self.bin_text(index),
            recommendation_notes=self.notes_text(index),
        )


class RecommenderSchema(BaseModel):
    settings: RecommenderSettings
    results: RecommenderResults | None

    def deserialize(self) -> "Recommender":
        recommender = Recommender(self.settings)
        recommender.results = self.results
        return recommender


class Recommender:
    """
    Recommendation logic for a specified data-type.
    """

    def __init__(self, settings: RecommenderSettings | None = None):
        if settings is not None:
            settings = RecommenderSettings.parse_obj(settings)
        else:
            settings = RecommenderSettings.build_default()
        self.settings: RecommenderSettings = settings
        self.results: RecommenderResults | None = None

    def recommend(self, dataset: DatasetBase, models: list[BmdModel]):
        self.results = RecommenderResults()

        if not self.settings.enabled:
            return

        # apply rules to each model
        model_bins = []
        model_notes = []
        for model in models:
            # set defaults
            current_bin = LogicBin.NO_CHANGE
            notes: dict[int, list[str]] = {
                LogicBin.NO_CHANGE: [],
                LogicBin.WARNING: [],
                LogicBin.FAILURE: [],
            }

            if model.has_results:
                # apply tests for each model
                for rule in self.settings.rules:
                    response: CheckResponse = RULE_MAP[rule.rule_class].check(dataset, model, rule)
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

    def _get_bmdl_ratio(self, models: list[BmdModel]) -> float:
        """Return BMDL ratio in list of models."""
        bmdls = [model.results.bmdl for model in models if model.results.bmdl > 0]
        return max(bmdls) / min(bmdls)

    def _get_recommended_models(self, models: list[BmdModel], field: str) -> list[BmdModel]:
        """
        Returns a list of models which have the minimum target field value
        for a given field name (AIC or BMDL).
        """
        if field == "aic":
            values = np.array([getattr(model.results.fit, field) for model in models])
        elif field == "bmdl":
            values = np.array([getattr(model.results, field) for model in models])
        else:
            raise ValueError(f"Unknown target field: {field}")

        matches = np.where(values == values.min())[0].tolist()
        return [models[i] for i in matches]

    def _get_parsimonious_model(self, models: list[BmdModel]) -> BmdModel:
        """
        Return the most parsimonious model of all available models. The most
        parsimonious model is defined as the model with the fewest number of
        parameters.
        """
        params = [len(model.results.parameters.values) for model in models]
        idx = params.index(min(params))
        return models[idx]

    def serialize(self) -> RecommenderSchema:
        return RecommenderSchema(settings=self.settings, results=self.results)
