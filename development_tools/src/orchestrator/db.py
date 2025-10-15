from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from .config import settings


engine = create_engine(settings.orch_db_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


@contextmanager
def session_scope():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _sqlite_has_column(table: str, column: str) -> bool:
    with engine.connect() as conn:
        res = conn.exec_driver_sql(f"PRAGMA table_info('{table}')").fetchall()
        cols = {row[1] for row in res}
        return column in cols


def apply_migrations() -> None:
    # Best-effort additive migrations for SQLite dev usage
    with engine.begin() as conn:
        if _sqlite_has_column("tasks", "pr_url") is False:
            conn.exec_driver_sql("ALTER TABLE tasks ADD COLUMN pr_url TEXT")
        if _sqlite_has_column("tasks", "paths") is False:
            conn.exec_driver_sql("ALTER TABLE tasks ADD COLUMN paths TEXT")
        if _sqlite_has_column("feature_requests", "dag_json_path") is False:
            conn.exec_driver_sql("ALTER TABLE feature_requests ADD COLUMN dag_json_path TEXT")


