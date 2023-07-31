from enum import StrEnum
from bmds import constants
from bmds.bmds2.models.base import BMDModel as BmdModel2
from bmds.bmds3.models.base import BmdModel as BmdModel3

BmdModel = BmdModel2 | BmdModel3

class ScalarAttribute(StrEnum):
    # TODO: rename to ScalarAttribute, find better pattern?
    BMD = "bmd"
    BMDL = "bmdl"
    BMDU = "bmdu"
    AIC = "aic"

    def get_value(self,model:BmdModel,version:constants.Version)->float:
        if version == constants.BMDS270:
            return BMDS270Attribute(self).get_value(model)
        if version == constants.BMDS330:
            return BMDS330Attribute(self).get_value(model)
        raise ValueError(f'Invalid BMDS version "{version}".')

class BMDS270Attribute(StrEnum):
    BMD = "bmd"
    BMDL = "bmdl"
    BMDU = "bmdu"
    AIC = "aic"

    def get_value(self,model:BmdModel)->float:
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

    def get_value(self,model:BmdModel)->float:
        if self == self.BMD:
            return model.results.bmd
        if self == self.BMDL:
            return model.results.bmdl
        if self == self.BMDU:
            return model.results.bmdu
        if self == self.AIC:
            return model.results.fit.aic
        raise NotImplementedError()

