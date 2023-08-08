from enum import StrEnum

from .base import BaseAttribute, BmdModel, Version


class Attribute(BaseAttribute):
    BMD = "bmd"
    BMDL = "bmdl"
    BMDU = "bmdu"
    AIC = "aic"

    def get_value(self, model: BmdModel, version: Version) -> float:
        if version == Version.BMDS270:
            return BMDS270Attribute(self).get_value(model)
        if version == Version.BMDS330:
            return BMDS330Attribute(self).get_value(model)
        raise ValueError(f'Invalid BMDS version "{version}".')


class BMDS270Attribute(StrEnum):
    BMD = "bmd"
    BMDL = "bmdl"
    BMDU = "bmdu"
    AIC = "aic"

    def get_value(self, model: BmdModel) -> float:
        if self == self.BMD:
            return model.output["BMD"]
        if self == self.BMDL:
            return model.output["BMDL"]
        if self == self.BMDU:
            return model.output["BMDU"]
        if self == self.AIC:
            return model.output["AIC"]
        raise NotImplementedError()


class BMDS330Attribute(StrEnum):
    BMD = "bmd"
    BMDL = "bmdl"
    BMDU = "bmdu"
    AIC = "aic"

    def get_value(self, model: BmdModel) -> float:
        if self == self.BMD:
            return model.results.bmd
        if self == self.BMDL:
            return model.results.bmdl
        if self == self.BMDU:
            return model.results.bmdu
        if self == self.AIC:
            return model.results.fit.aic
        raise NotImplementedError()
