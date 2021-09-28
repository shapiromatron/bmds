from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .config import Config
from .models import Base

_engine = create_engine(Config.DB_URI)
Session = sessionmaker(_engine, autocommit=True)

Base.metadata.bind = _engine


def setup_db():
    """Create database, tables, indexes, etc."""
    Base.metadata.create_all(checkfirst=True)


def reset_db():
    """Drop all tables, indexes, etc."""
    Base.metadata.drop_all()


@contextmanager
def transaction():
    """Provide a transactional scope around a series of operations."""
    sess = Session()
    try:
        yield sess
        sess.commit()
    except Exception:
        sess.rollback()
    finally:
        sess.close()
