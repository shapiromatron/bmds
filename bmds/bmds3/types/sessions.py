from typing import List, Optional, Tuple

from pydantic import BaseModel

from ...datasets.base import DatasetSchemaBase
from ..models.base import BmdModelAveragingSchema, BmdModelSchema
from ..recommender import RecommenderSchema
from ..selected import SelectedModelSchema


class VersionSchema(BaseModel):
    string: str
    pretty: str
    numeric: Tuple[int, ...]
    python: str
    dll: str


class SessionSchemaBase(BaseModel):
    version: VersionSchema
    dataset: DatasetSchemaBase
    models: List[BmdModelSchema]
    model_average: Optional[BmdModelAveragingSchema]
    recommender: Optional[RecommenderSchema]
    selected: SelectedModelSchema
