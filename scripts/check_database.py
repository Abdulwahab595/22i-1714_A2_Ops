"""Verify PostgreSQL and initialize the prediction table without exposing its connection URL."""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text

from src.database import PredictionStore


if __name__ == "__main__":
    load_dotenv()
    uri = os.getenv("DATABASE_URL", "")
    if not uri.startswith(("postgresql://", "postgresql+psycopg://", "postgres://")):
        print("DATABASE_URL must contain the connection URL beginning with postgresql://")
        sys.exit(1)
    try:
        store = PredictionStore(uri)
        store.health()
        with store.engine.connect() as connection:
            version = connection.execute(text("SELECT version()")).scalar_one()
            database = connection.execute(text("SELECT current_database()")).scalar_one()
        record = {"project": "22i-1714_A2_Ops", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                  "status": "connected", "database": database, "server_version": version,
                  "prediction_schema": "initialized"}
        Path("artifacts").mkdir(exist_ok=True)
        Path("artifacts/database_health.json").write_text(json.dumps(record, indent=2))
        print(json.dumps(record, indent=2))
    except Exception as error:
        print("Database check failed:", type(error).__name__)
        print("SQL state:", getattr(getattr(error, "orig", None), "sqlstate", None))
        sys.exit(1)
