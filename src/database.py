"""Persistent prediction trail; uploaded image contents are never stored."""
from pathlib import Path

from sqlalchemy import (Column, Float, MetaData, String, Table, create_engine,
                        desc, insert, select, text)

metadata = MetaData()
predictions = Table(
    "predictions", metadata,
    Column("id", String(36), primary_key=True),
    Column("created_at", String(40), nullable=False, index=True),
    Column("status", String(24), nullable=False),
    Column("filename", String(200), nullable=False),
    Column("image_sha256", String(64)),
    Column("label", String(32)),
    Column("confidence", Float),
    Column("ai_probability", Float),
    Column("threshold", Float),
    Column("model_name", String(128)),
    Column("model_version", String(64)),
    Column("model_sha256", String(64)),
    Column("duration_ms", Float),
    Column("error", String(400)),
)


class PredictionStore:
    def __init__(self, url):
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+psycopg://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        if url.startswith("sqlite:///") and url != "sqlite:///:memory:":
            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        kwargs = {"pool_pre_ping": True}
        if url.startswith("sqlite:"):
            kwargs["connect_args"] = {"check_same_thread": False}
        elif url.startswith("postgresql"):
            kwargs["connect_args"] = {"connect_timeout": 10}
        self.engine = create_engine(url, **kwargs)
        metadata.create_all(self.engine)
        print("Startup 3/4: prediction database ready", flush=True)

    def write(self, record):
        with self.engine.begin() as connection:
            connection.execute(insert(predictions).values(**record))

    def history(self, limit=100, offset=0):
        statement = select(predictions).order_by(desc(predictions.c.created_at)).limit(
            min(max(int(limit), 1), 100)).offset(max(int(offset), 0))
        with self.engine.connect() as connection:
            return [dict(row) for row in connection.execute(statement).mappings()]

    def get(self, identifier):
        with self.engine.connect() as connection:
            row = connection.execute(select(predictions).where(
                predictions.c.id == identifier)).mappings().first()
            return dict(row) if row else None

    def health(self):
        with self.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
