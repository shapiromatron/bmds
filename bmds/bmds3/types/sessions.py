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
    id: int | str | None
    version: VersionSchema
    dataset: DatasetSchemaBase
    models: list[BmdModelSchema]
    model_average: BmdModelAveragingSchema | None
    recommender: RecommenderSchema | None
    selected: SelectedModelSchema
