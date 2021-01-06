from typing import Tuple

from pydantic import BaseModel

from ...datasets.base import DatasetSchemaBase


class VersionSchema(BaseModel):
    string: str
    pretty: str
    numeric: Tuple[int, ...]


class SessionSchemaBase(BaseModel):
    version: VersionSchema
    dataset: DatasetSchemaBase
