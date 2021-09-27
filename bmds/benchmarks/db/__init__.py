from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..config import Config
from .models import Base

_engine = create_engine(Config.DB_URI)
Session = sessionmaker(_engine, autocommit=True)

Base.metadata.bind = _engine


def setup_db():
    Base.metadata.create_all(checkfirst=True)


def reset_db():
    Base.metadata.drop_all()
