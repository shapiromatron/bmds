import json

import numpy as np
from sqlalchemy import Column, ForeignKey, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.types import Boolean, Float, Integer, Text, TypeDecorator

from .. import datasets

Base = declarative_base()


class JSONEncodedDict(TypeDecorator):
    impl = Text

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = json.loads(value)
        return value


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)

    dataset = Column(Text, index=True)
    dataset_id = Column(Text)
    dataset_metadata = Column(JSONEncodedDict())

    dtype = Column(Text)
    doses = Column(Text)
    ns = Column(Text)
    incidences = Column(Text)
    means = Column(Text)
    stdevs = Column(Text)

    __table_args__ = (UniqueConstraint(dataset, dataset_id),)

    def to_bmds(self):
        if self.dtype == "D":
            return datasets.DichotomousDataset(
                doses=np.array(self.doses.split(","), dtype=np.float64).tolist(),
                ns=np.array(self.ns.split(","), dtype=np.float64).tolist(),
                incidences=np.array(self.incidences.split(","), dtype=np.float64).tolist(),
                id=self.id,
            )
        elif self.dtype == "C":
            return datasets.ContinuousDataset(
                doses=np.array(self.doses.split(","), dtype=np.float64).tolist(),
                ns=np.array(self.ns.split(","), dtype=np.float64).tolist(),
                means=np.array(self.means.split(","), dtype=np.float64).tolist(),
                stdevs=np.array(self.stdevs.split(","), dtype=np.float64).tolist(),
                id=self.id,
            )
        else:
            raise ValueError(f"Unknown type: {self.dtype}")


class ModelResult(Base):
    __tablename__ = "model_results"

    id = Column(Integer, primary_key=True, index=True)

    analysis = Column(Integer, index=True)
    bmds_version = Column(Text, index=True)
    model = Column(Text, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), index=True)

    completed = Column(Boolean)
    inputs = Column(JSONEncodedDict())
    outputs = Column(JSONEncodedDict())
    bmd = Column(Float)
    bmdl = Column(Float)
    bmdu = Column(Float)
    aic = Column(Float)

    dataset = relationship("Dataset", backref="model_results")

    __table_args__ = (UniqueConstraint(analysis, bmds_version, model, dataset_id),)
