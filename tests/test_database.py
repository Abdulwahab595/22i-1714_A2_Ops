import ssl

from src.database import connection_settings


def test_neon_uses_python_driver_and_verified_tls():
    url, options = connection_settings(
        "postgresql+psycopg://user:pass@example.neon.tech/db?sslmode=require&channel_binding=require")
    assert url.drivername == "postgresql+pg8000"
    assert url.host == "example.neon.tech" and url.database == "db"
    assert "sslmode" not in url.query and "channel_binding" not in url.query
    context = options["connect_args"]["ssl_context"]
    assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
    assert options["connect_args"]["timeout"] == 10


def test_compose_database_keeps_internal_service_name():
    url, options = connection_settings("postgres://user:pass@db:5432/visionguard")
    assert url.drivername == "postgresql+pg8000"
    assert url.host == "db" and url.port == 5432
    assert options["connect_args"] == {"timeout": 10}
