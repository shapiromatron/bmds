from typing import List, Tuple

from pydantic import BaseModel

from ...datasets.base import DatasetSchemaBase
from ..models.base import BmdModelSchema


class VersionSchema(BaseModel):
    string: str
    pretty: str
    numeric: Tuple[int, ...]


class SessionSchemaBase(BaseModel):
    version: VersionSchema
    dataset: DatasetSchemaBase
    models: List[BmdModelSchema]
