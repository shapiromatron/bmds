import json

import numpy as np
from sqlalchemy import Column, ForeignKey, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.types import Boolean, Float, Integer, Text, TypeDecorator

from ... import datasets

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


class DichotomousDataset(Base):
    __tablename__ = "dichotomous_datasets"

    id = Column(Integer, primary_key=True, index=True)

    dataset = Column(Text)
    dataset_id = Column(Text)
    dataset_metadata = Column(JSONEncodedDict())

    doses = Column(Text)
    ns = Column(Text)
    incidences = Column(Text)

    results = relationship("DichotomousResult", back_populates="dataset")

    __table_args__ = (UniqueConstraint(dataset, dataset_id),)

    def to_bmds(self):
        return datasets.DichotomousDataset(
            doses=np.array(self.doses.split(","), dtype=np.float).tolist(),
            ns=np.array(self.ns.split(","), dtype=np.float).tolist(),
            incidences=np.array(self.incidences.split(","), dtype=np.float).tolist(),
            id=self.id,
        )


class ContinuousDataset(Base):
    __tablename__ = "continuous_datasets"

    id = Column(Integer, primary_key=True, index=True)

    dataset = Column(Text)
    dataset_id = Column(Text)
    dataset_metadata = Column(JSONEncodedDict())

    doses = Column(Text)
    ns = Column(Text)
    means = Column(Text)
    stdevs = Column(Text)

    results = relationship("ContinuousResult", back_populates="dataset")

    __table_args__ = (UniqueConstraint(dataset, dataset_id),)

    def to_bmds(self):
        return datasets.ContinuousDataset(
            doses=np.array(self.doses.split(","), dtype=np.float).tolist(),
            ns=np.array(self.ns.split(","), dtype=np.float).tolist(),
            means=np.array(self.means.split(","), dtype=np.float).tolist(),
            stdevs=np.array(self.stdevs.split(","), dtype=np.float).tolist(),
            id=self.id,
        )


class DichotomousResult(Base):
    __tablename__ = "dichotomous_results"

    id = Column(Integer, primary_key=True, index=True)

    bmds_version = Column(Text, index=True)
    model = Column(Text, index=True)
    completed = Column(Boolean)
    inputs = Column(JSONEncodedDict())
    outputs = Column(JSONEncodedDict())
    bmd = Column(Float)
    bmdl = Column(Float)
    bmdu = Column(Float)
    aic = Column(Float)

    dataset_id = Column(Integer, ForeignKey("dichotomous_datasets.id"), index=True)
    dataset = relationship("DichotomousDataset", back_populates="results")

    __table_args__ = (UniqueConstraint(bmds_version, model, dataset_id),)


class ContinuousResult(Base):
    __tablename__ = "continuous_results"

    id = Column(Integer, primary_key=True, index=True)

    bmds_version = Column(Text, index=True)
    model = Column(Text, index=True)
    completed = Column(Boolean)
    inputs = Column(JSONEncodedDict())
    outputs = Column(JSONEncodedDict())
    bmd = Column(Float)
    bmdl = Column(Float)
    bmdu = Column(Float)
    aic = Column(Float)

    dataset_id = Column(Integer, ForeignKey("continuous_datasets.id"), index=True)
    dataset = relationship("ContinuousDataset", back_populates="results")

    __table_args__ = (UniqueConstraint(bmds_version, model, dataset_id),)
