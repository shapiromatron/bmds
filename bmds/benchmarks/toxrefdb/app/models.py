import json

import numpy as np

import bmds

from . import db


class JSONEncodedDict(db.TypeDecorator):
    impl = db.Text

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = json.loads(value)
        return value


class DichotomousDataset(db.Model):
    __tablename__ = "dichotomous_datasets"

    id = db.Column(db.Text, primary_key=True, index=True)

    doses = db.Column(db.Text)
    ns = db.Column(db.Text)
    incidences = db.Column(db.Text)

    results = db.relationship("DichotomousResult", back_populates="dataset")

    def to_bmds(self):
        return bmds.DichotomousDataset(
            doses=np.array(self.doses.split(","), dtype=np.float).tolist(),
            ns=np.array(self.ns.split(","), dtype=np.float).tolist(),
            incidences=np.array(self.incidences.split(","), dtype=np.float).tolist(),
            id=self.id,
        )


class ContinuousDataset(db.Model):
    __tablename__ = "continuous_datasets"

    id = db.Column(db.Text, primary_key=True, index=True)

    doses = db.Column(db.Text)
    ns = db.Column(db.Text)
    means = db.Column(db.Text)
    stdevs = db.Column(db.Text)

    results = db.relationship("ContinuousResult", back_populates="dataset")

    def to_bmds(self):
        return bmds.ContinuousDataset(
            doses=np.array(self.doses.split(","), dtype=np.float).tolist(),
            ns=np.array(self.ns.split(","), dtype=np.float).tolist(),
            means=np.array(self.means.split(","), dtype=np.float).tolist(),
            stdevs=np.array(self.stdevs.split(","), dtype=np.float).tolist(),
            id=self.id,
        )


class DichotomousResult(db.Model):
    __tablename__ = "dichotomous_results"

    id = db.Column(db.Integer, primary_key=True, index=True)

    bmds_version = db.Column(db.Text, index=True)
    model = db.Column(db.Text, index=True)
    completed = db.Column(db.Boolean)
    inputs = db.Column(JSONEncodedDict())
    outputs = db.Column(JSONEncodedDict())
    bmd = db.Column(db.Float)
    bmdl = db.Column(db.Float)
    bmdu = db.Column(db.Float)
    aic = db.Column(db.Float)

    dataset_id = db.Column(db.Text, db.ForeignKey("dichotomous_datasets.id"))
    dataset = db.relationship("DichotomousDataset", back_populates="results")

    __table_args__ = (db.UniqueConstraint(bmds_version, model, dataset_id),)


class ContinuousResult(db.Model):
    __tablename__ = "continuous_results"

    id = db.Column(db.Integer, primary_key=True, index=True)

    bmds_version = db.Column(db.Text, index=True)
    model = db.Column(db.Text, index=True)
    completed = db.Column(db.Boolean)
    inputs = db.Column(JSONEncodedDict())
    outputs = db.Column(JSONEncodedDict())
    bmd = db.Column(db.Float)
    bmdl = db.Column(db.Float)
    bmdu = db.Column(db.Float)
    aic = db.Column(db.Float)

    dataset_id = db.Column(db.Text, db.ForeignKey("continuous_datasets.id"))
    dataset = db.relationship("ContinuousDataset", back_populates="results")

    __table_args__ = (db.UniqueConstraint(bmds_version, model, dataset_id),)
