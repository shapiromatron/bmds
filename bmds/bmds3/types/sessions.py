from pydantic import BaseModel

from ...datasets.base import DatasetSchemaBase
from ..models.base import BmdModelAveragingSchema, BmdModelSchema
from ..recommender import RecommenderSchema
from ..selected import SelectedModelSchema


class VersionSchema(BaseModel):
    string: str
    pretty: str
    numeric: tuple[int, ...]
    python: str
    dll: str


class SessionSchemaBase(BaseModel):
    version: VersionSchema
    dataset: DatasetSchemaBase
    models: list[BmdModelSchema]
    bmds_model_average: BmdModelAveragingSchema | None = None
    recommender: RecommenderSchema | None = None
    selected: SelectedModelSchema
