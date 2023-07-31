from enum import StrEnum

from bmds.bmds2.models.base import BMDModel as BmdModel2
from bmds.bmds3.models.base import BmdModel as BmdModel3
from bmds.constants import Version

BmdModel = BmdModel2 | BmdModel3


class BaseAttribute(StrEnum):
    def get_value(self, model: BmdModel, version: Version) -> float:
        raise NotImplementedError()
