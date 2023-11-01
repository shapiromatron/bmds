from pydantic import BaseModel, Field, SerializeAsAny

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
    dataset: SerializeAsAny[DatasetSchemaBase]
    models: list[SerializeAsAny[BmdModelSchema]]
    bmds_model_average: SerializeAsAny[BmdModelAveragingSchema] | None = Field(
        default=None, alias="model_average"
    )
    recommender: RecommenderSchema | None = None
    selected: SelectedModelSchema
