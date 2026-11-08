"""Persistent prediction trail; uploaded image contents are never stored."""
from pathlib import Path
import ssl

from sqlalchemy import (Column, Float, MetaData, String, Table, create_engine,
                        desc, insert, select, text)
from sqlalchemy.engine import make_url

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


def connection_settings(url):
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    parsed = make_url(url)
    kwargs = {"pool_pre_ping": True}
    if parsed.get_backend_name() == "sqlite":
        if parsed.database and parsed.database != ":memory:":
            Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
        kwargs["connect_args"] = {"check_same_thread": False}
    elif parsed.get_backend_name() == "postgresql":
        # pg8000 avoids the native libpq crash observed after TensorFlow loads on Render.
        mode = parsed.query.get("sslmode", "prefer")
        tls = mode in {"require", "verify-ca", "verify-full"} or (
            parsed.host or "").endswith(".neon.tech")
        parsed = parsed.set(drivername="postgresql+pg8000")
        parsed = parsed.difference_update_query(["sslmode", "channel_binding", "connect_timeout"])
        kwargs["connect_args"] = {"timeout": 10}
        if tls:
            kwargs["connect_args"]["ssl_context"] = ssl.create_default_context()
    return parsed, kwargs


class PredictionStore:
    def __init__(self, url):
        url, kwargs = connection_settings(url)
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
