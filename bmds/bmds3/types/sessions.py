from typing import List, TypeVar

from pydantic import BaseModel

from ...datasets.schema import DatasetSchema
from ..models.schema import ModelSchema


class SessionSchemaBase(BaseModel):
    pass


SessionSchema = TypeVar("SessionSchema", bound=SessionSchemaBase)  # noqa


class Bmds330Schema(SessionSchemaBase):
    bmds_version: str
    bmds_python_version: str
    dataset: DatasetSchema
    models: List[ModelSchema]
