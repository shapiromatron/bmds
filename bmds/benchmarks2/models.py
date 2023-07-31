import pandas as pd
from sqlmodel import Column, Field, PickleType, Relationship, SQLModel

from bmds.bmds2.sessions import BMDS_v270 as BMDS270Session
from bmds.bmds3.sessions import BmdsSession as BMDS330Session
from bmds.constants import Version

from . import constants, db

BmdsSession = BMDS270Session | BMDS330Session


class TblSession(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)

    value: BmdsSession = Field(sa_column=Column(PickleType))

    bmds_version: Version
    os: str

    session_name: str
    dataset_name: str
    dtype: str

    models: list["TblModel"] = Relationship(back_populates="session")

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_bmds(cls, session: BmdsSession, attrs: list[str] = None) -> "TblSession":
        if attrs is None:
            attrs = [attr for attr in constants.ScalarAttribute]
        tbl_session = cls(
            value=session,
            bmds_version=session._bmds_version,
            os=session._os,
            session_name=session._session_name,
            dataset_name=session.dataset.metadata.name,
            dtype=session.dataset.dtype,
        )
        for tbl_model in TblModel.from_session(tbl_session):
            TblModelResultScalar.from_model(tbl_model, attrs)
        return tbl_session

    @classmethod
    def get_df(cls, objs: "TblSession", follow: bool = True):
        records = [
            obj.dict(include={"id", "bmds_version", "os", "dataset_name", "dtype"}) for obj in objs
        ]
        df = pd.DataFrame.from_records(records).rename(columns={"id": "session_id"})
        return df


class TblModel(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)

    session_id: int = Field(foreign_key="tblsession.id")
    session: TblSession = Relationship(back_populates="models")

    index: int

    model_name: str
    settings_name: str

    scalar_results: list["TblModelResultScalar"] = Relationship(back_populates="model")

    @property
    def value(self):
        return self.session.value.models[self.index]

    @classmethod
    def from_session(cls, session: TblSession) -> list["TblModel"]:
        objs = []
        for index, model in enumerate(session.value.models):
            obj = cls(
                session=session,
                index=index,
                model_name=model._model_name,
                settings_name=model._settings_name,
            )
            objs.append(obj)
        return objs

    @classmethod
    def get_df(cls, objs: "TblModel", follow: bool = True):
        records = [
            obj.dict(include={"id", "session_id", "model_name", "settings_name"}) for obj in objs
        ]
        df = pd.DataFrame.from_records(records).rename(columns={"id": "model_id"})
        if follow:
            _objs = {obj.session.id: obj.session for obj in objs}.values()
            _df = TblSession.get_df(_objs, follow=True)
            return df.merge(_df, on="session_id")
        return df


class TblModelResultScalar(SQLModel, table=True):
    # TODO make a superclass for result? and add "attribute", "value", and class methods
    id: int | None = Field(default=None, primary_key=True)

    model_id: int = Field(foreign_key="tblmodel.id")
    model: TblModel = Relationship(back_populates="scalar_results")

    attribute: constants.ScalarAttribute
    value: float

    @classmethod
    def from_model(
        cls, model: TblModel, attrs: list[constants.ScalarAttribute]
    ) -> list["TblModelResultScalar"]:
        bmds_version = model.session.bmds_version
        objs = []
        for attr in attrs:
            obj = cls(
                model=model,
                attribute=attr,
                value=attr.get_value(model.value, bmds_version),
            )
            objs.append(obj)
        return objs

    @classmethod
    def get_df(cls, objs: "TblModelResultScalar", follow: bool = True):
        records = [obj.dict(include={"id", "model_id", "attribute", "value"}) for obj in objs]
        df = pd.DataFrame.from_records(records).rename(columns={"id": "model_result_scalar_id"})
        if follow:
            _objs = {obj.model.id: obj.model for obj in objs}.values()
            _df = TblModel.get_df(_objs, follow=True)
            return df.merge(_df, on="model_id")
        return df


SQLModel.metadata.create_all(db._engine)
