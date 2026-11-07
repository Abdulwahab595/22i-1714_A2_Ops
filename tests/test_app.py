import io

import pytest
from PIL import Image

from app import create_app
from src.database import PredictionStore


class FixedEngine:
    """Predictable test double; it is never used in scientific evaluation or serving."""
    loads = 1
    metadata = {"model_version": "test-only"}

    def predict(self, _content):
        return {"label": "ai_generated", "confidence": 0.8, "ai_probability": 0.8,
                "threshold": 0.7, "model_name": "test-only", "model_version": "test-only",
                "model_sha256": "a" * 64}


def image_bytes(format="PNG"):
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), "navy").save(buffer, format=format)
    return buffer.getvalue()


@pytest.fixture
def configured(tmp_path):
    store = PredictionStore(f"sqlite:///{(tmp_path / 'history.db').as_posix()}")
    app = create_app({"TESTING": True, "APP_MODE": "local", "API_KEY": "",
                       "MAX_UPLOAD_BYTES": 1024 * 1024}, engine=FixedEngine(), store=store)
    return app, store, tmp_path


def test_prediction_is_durable_without_saving_image(configured):
    app, store, directory = configured
    with app.test_client() as client:
        response = client.post("/api/predict", data={"image": (io.BytesIO(image_bytes()),
                                                               "../../safe.png")})
        assert response.status_code == 200
        result = response.get_json()
        assert result["filename"] == "safe.png"
        assert result["model_version"] == "test-only"
        assert result["confidence"] == 0.8
        assert len(result["image_sha256"]) == 64
        assert client.get(f"/result/{result['id']}").status_code == 200
        assert client.get("/history").status_code == 200
    reopened = PredictionStore(f"sqlite:///{(directory / 'history.db').as_posix()}")
    assert reopened.get(result["id"])["label"] == "ai_generated"
    assert len(store.history()) == 1
    assert not list(directory.glob("*.png"))


@pytest.mark.parametrize("content,filename,status", [
    (b"text", "notes.txt", 415),
    (b"not actually a PNG", "looks_like_image.png", 400),
    (image_bytes("JPEG"), "renamed.png", 415),
    (b"", "empty.png", 400),
    (image_bytes(), "", 400),
])
def test_bad_uploads_are_rejected_and_logged(configured, content, filename, status):
    app, store, _ = configured
    response = app.test_client().post("/api/predict",
                                     data={"image": (io.BytesIO(content), filename)})
    assert response.status_code == status
    assert response.get_json()["error"]
    assert store.history()[0]["status"] == "rejected"


def test_oversize_upload_is_friendly(configured):
    app, store, _ = configured
    app.config["MAX_UPLOAD_BYTES"] = 20
    response = app.test_client().post("/", data={"image": (io.BytesIO(image_bytes()), "big.png")})
    assert response.status_code == 413
    assert b"too large" in response.data
    assert store.history()[0]["status"] == "rejected"


def test_model_is_reused_and_health_checks_database(configured):
    app, _, _ = configured
    client = app.test_client()
    for _ in range(2):
        assert client.post("/api/predict", data={"image": (io.BytesIO(image_bytes()),
                                                            "example.png")}).status_code == 200
    assert client.get("/health").get_json()["model_loads"] == 1


def test_api_authentication(configured):
    app, _, _ = configured
    app.config["API_KEY"] = "test-secret"
    client = app.test_client()
    assert client.get("/api/history").status_code == 401
    assert client.get("/api/history", headers={"Authorization": "Bearer test-secret"}).status_code == 200


def test_inference_failure_is_logged(configured):
    app, store, _ = configured
    def fail(_content):
        raise RuntimeError("test-only simulated inference failure")
    app.extensions["inference_engine"].predict = fail
    response = app.test_client().post("/api/predict", data={"image": (io.BytesIO(image_bytes()),
                                                                     "valid.png")})
    assert response.status_code == 503
    assert store.history()[0]["status"] == "failed"


def test_remote_mode_starts_without_a_local_model():
    app = create_app({"TESTING": True, "APP_MODE": "remote",
                       "MODEL_API_URL": "https://example.invalid"})
    assert app.extensions["inference_engine"] is None
    assert app.test_client().get("/").status_code == 200
