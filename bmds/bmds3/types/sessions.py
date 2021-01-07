from typing import List, Optional, Tuple

from pydantic import BaseModel

from ...datasets.base import DatasetSchemaBase
from ..models.base import BmdModelAveragingSchema, BmdModelSchema
from ..recommender import RecommenderSchema


class VersionSchema(BaseModel):
    string: str
    pretty: str
    numeric: Tuple[int, ...]


class SessionSchemaBase(BaseModel):
    version: VersionSchema
    dataset: DatasetSchemaBase
    models: List[BmdModelSchema]
    model_average: Optional[BmdModelAveragingSchema]
    recommender: Optional[RecommenderSchema]
