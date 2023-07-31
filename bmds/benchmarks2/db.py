from contextlib import contextmanager

from sqlmodel import Session, create_engine

from .config import Config

_engine = create_engine(Config.DB_URI)


@contextmanager
def SQLSession():
    session = Session(_engine)
    try:
        yield session
    finally:
        session.close()
