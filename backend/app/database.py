"""Database lifecycle helpers for the local POC."""

from __future__ import annotations

from pathlib import Path

from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from alembic import command
from app.config import PROJECT_ROOT, get_settings


class Base(DeclarativeBase):
    """Base class for future persistence models."""


def build_engine():
    """Create the configured SQLAlchemy engine."""

    settings = get_settings()
    is_sqlite = settings.database_url.startswith("sqlite")
    connect_args = {"check_same_thread": False} if is_sqlite else {}
    return create_engine(settings.database_url, connect_args=connect_args)


engine = build_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def initialise_database() -> None:
    """Apply all local database migrations before serving requests."""

    alembic_config = Config(str(Path(PROJECT_ROOT) / "alembic.ini"))
    alembic_config.set_main_option("sqlalchemy.url", get_settings().database_url)
    command.upgrade(alembic_config, "head")


def get_db():
    """Yield a request-scoped database session."""

    database = SessionLocal()
    try:
        yield database
    finally:
        database.close()
