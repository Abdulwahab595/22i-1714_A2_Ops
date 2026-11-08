"""Flask UI and image API. APP_MODE=remote connects the local UI to the cloud API."""
import hashlib
import hmac
import io
import logging
import os
import time
import uuid
import warnings
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from flask import Flask, Request, abort, jsonify, redirect, render_template, request, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from src.data import DISPLAY_LABELS
from src.database import PredictionStore

CAVEAT = ("This is an experimental model score, not proof of authenticity. "
          "Edited photos, artwork, compression, and unfamiliar generators can cause errors. "
          "The score has not been calibrated as a probability of correctness.")


class MemoryOnlyRequest(Request):
    def _get_file_stream(self, total_content_length, content_type, filename=None,
                         content_length=None):
        # Werkzeug normally spools larger uploads to a temporary disk file.
        # The global request-size bound makes a memory-only stream safe here.
        return io.BytesIO()


class UploadError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def validate_upload(upload, max_bytes, max_pixels):
    if upload is None or not upload.filename:
        raise UploadError("Choose a JPEG or PNG image to continue.")
    filename = secure_filename(upload.filename)[:200]
    if not filename or "." not in filename:
        raise UploadError("Use a file ending in .jpg, .jpeg, or .png.", 415)
    extension = filename.rsplit(".", 1)[-1].lower()
    if extension not in {"jpg", "jpeg", "png"}:
        raise UploadError("That file type is not supported. Please upload JPEG or PNG.", 415)
    content = upload.stream.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise UploadError(f"Your image is too large. The limit is {max_bytes // 1024**2} MB.", 413)
    if not content:
        raise UploadError("The selected image is empty.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise UploadError("The file contents must be JPEG or PNG.", 415)
                if (image.format == "PNG") != (extension == "png"):
                    raise UploadError("The file extension does not match the image contents.", 415)
                if image.width * image.height > max_pixels:
                    raise UploadError("This image has too many pixels. Please resize it first.", 413)
                image.verify()
            with Image.open(io.BytesIO(content)) as image:
                image.load()
    except UploadError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise UploadError("We could not read this image. Please choose a valid JPEG or PNG.") from error
    return filename, content


def create_app(config=None, engine=None, store=None):
    load_dotenv()
    app = Flask(__name__)
    app.request_class = MemoryOnlyRequest
    app.config.update(
        APP_MODE=os.getenv("APP_MODE", "local"),
        MODEL_PATH=os.getenv("MODEL_PATH", "models/champion.keras"),
        MODEL_METADATA_PATH=os.getenv("MODEL_METADATA_PATH", "models/champion.json"),
        DECISION_THRESHOLD=os.getenv("DECISION_THRESHOLD") or None,
        DATABASE_URL=os.getenv("DATABASE_URL", "sqlite:///artifacts/visionguard.db"),
        MODEL_API_URL=os.getenv("MODEL_API_URL", "").rstrip("/"),
        MODEL_API_TOKEN=os.getenv("MODEL_API_TOKEN", ""), API_KEY=os.getenv("API_KEY", ""),
        API_TIMEOUT_SECONDS=float(os.getenv("API_TIMEOUT_SECONDS", "120")),
        MAX_UPLOAD_BYTES=int(float(os.getenv("MAX_UPLOAD_MB", "5")) * 1024**2),
        MAX_IMAGE_PIXELS=int(os.getenv("MAX_IMAGE_PIXELS", "2000000")),
        JSON_SORT_KEYS=False,
    )
    if config:
        app.config.update(config)
    # Include a small allowance for multipart headers; validate image bytes separately.
    app.config["MAX_CONTENT_LENGTH"] = app.config["MAX_UPLOAD_BYTES"] + 64 * 1024
    if app.config["APP_MODE"] not in {"local", "remote"}:
        raise ValueError("APP_MODE must be local or remote")
    remote = app.config["APP_MODE"] == "remote"
    if remote and not app.config["MODEL_API_URL"]:
        raise ValueError("MODEL_API_URL is required when APP_MODE=remote")
    if not remote:
        if engine is None:
            from src.inference import InferenceEngine
            engine = InferenceEngine(app.config["MODEL_PATH"], app.config["MODEL_METADATA_PATH"],
                                     app.config["DECISION_THRESHOLD"])
        store = store or PredictionStore(app.config["DATABASE_URL"])
    app.extensions.update(inference_engine=engine, prediction_store=store)
    session = requests.Session()
    if app.config["MODEL_API_TOKEN"]:
        session.headers["Authorization"] = "Bearer " + app.config["MODEL_API_TOKEN"]

    def remote_request(method, path, **kwargs):
        response = session.request(method, app.config["MODEL_API_URL"] + path,
                                   timeout=app.config["API_TIMEOUT_SECONDS"], **kwargs)
        if response.status_code >= 400:
            try:
                message = response.json().get("error", "The prediction service could not respond.")
            except ValueError:
                message = "The prediction service is starting or temporarily unavailable."
            raise UploadError(message, response.status_code if response.status_code < 500 else 503)
        return response.json()

    def authorized():
        key = app.config["API_KEY"]
        if not key:
            return True
        supplied = request.headers.get("Authorization", "").removeprefix("Bearer ")
        supplied = request.headers.get("X-API-Key", supplied)
        return hmac.compare_digest(key, supplied)

    def trail(status, filename="", content=None, prediction=None, error=None, duration_ms=None):
        record = {"id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc).isoformat(),
                  "status": status, "filename": filename,
                  "image_sha256": hashlib.sha256(content).hexdigest() if content else None,
                  "duration_ms": duration_ms, "error": error[:400] if error else None}
        if prediction:
            record.update(prediction)
        store.write(record)
        return record

    def analyze():
        started = time.monotonic()
        upload = request.files.get("image")
        try:
            filename, content = validate_upload(upload, app.config["MAX_UPLOAD_BYTES"],
                                                 app.config["MAX_IMAGE_PIXELS"])
        except UploadError as error:
            if not remote:
                trail("rejected", secure_filename(upload.filename)[:200] if upload else "",
                      error=str(error), duration_ms=(time.monotonic() - started) * 1000)
            raise
        if remote:
            return remote_request("POST", "/api/predict", files={"image": (filename, content)})
        try:
            prediction = engine.predict(content)
        except Exception:
            app.logger.exception("Inference failed")
            trail("failed", filename, content, error="Inference failed",
                  duration_ms=(time.monotonic() - started) * 1000)
            raise UploadError("We could not analyze this image. Please try again.", 503)
        record = trail("success", filename, content, prediction,
                       duration_ms=(time.monotonic() - started) * 1000)
        return {**record, "caveat": CAVEAT}

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            try:
                result = analyze()
                return redirect(url_for("result_page", identifier=result["id"]))
            except UploadError as error:
                return render_template("index.html", error=str(error)), error.status
            except Exception:
                app.logger.exception("Prediction request could not be completed")
                return render_template("error.html", title="Service unavailable",
                                       message="We could not save or retrieve the prediction. "
                                               "Please try again shortly."), 503
        return render_template("index.html")

    @app.get("/result/<identifier>")
    def result_page(identifier):
        try:
            result = (remote_request("GET", f"/api/predictions/{identifier}") if remote
                      else store.get(identifier))
            if not result:
                abort(404)
            return render_template("result.html", result=result, labels=DISPLAY_LABELS,
                                   caveat=CAVEAT)
        except (requests.RequestException, UploadError):
            return render_template("error.html", title="Service unavailable",
                                   message="The prediction service is temporarily unavailable."), 503

    @app.get("/history")
    def history_page():
        try:
            rows = remote_request("GET", "/api/history")["predictions"] if remote else store.history()
            return render_template("history.html", predictions=rows, labels=DISPLAY_LABELS)
        except Exception:
            app.logger.exception("History unavailable")
            return render_template("error.html", title="History unavailable",
                                   message="We could not retrieve history. Please try again shortly."), 503

    @app.post("/api/predict")
    def predict_api():
        if not authorized():
            return jsonify(error="A valid API key is required."), 401
        try:
            return jsonify(analyze())
        except UploadError as error:
            return jsonify(error=str(error)), error.status
        except Exception:
            app.logger.exception("Prediction could not be committed to the database")
            return jsonify(error="The service could not complete and save the prediction."), 503

    @app.get("/api/history")
    def history_api():
        if not authorized():
            return jsonify(error="A valid API key is required."), 401
        try:
            if remote:
                return jsonify(remote_request("GET", "/api/history"))
            return jsonify(predictions=store.history(request.args.get("limit", 100, type=int),
                                                     request.args.get("offset", 0, type=int)))
        except Exception:
            app.logger.exception("History query failed")
            return jsonify(error="History is temporarily unavailable."), 503

    @app.get("/api/predictions/<identifier>")
    def prediction_api(identifier):
        if not authorized():
            return jsonify(error="A valid API key is required."), 401
        result = (remote_request("GET", f"/api/predictions/{identifier}") if remote
                  else store.get(identifier))
        return (jsonify(result) if result else (jsonify(error="Prediction not found."), 404))

    @app.get("/health")
    def health():
        try:
            if remote:
                upstream = remote_request("GET", "/health")
                return jsonify(status="healthy", project="22i-1714_A2_Ops", mode="remote",
                               upstream=upstream.get("status"))
            store.health()
            return jsonify(status="healthy", project="22i-1714_A2_Ops", mode="local",
                           database="connected", model="loaded", model_loads=engine.loads,
                           model_version=str(engine.metadata.get("model_version", "unregistered")))
        except Exception:
            app.logger.exception("Readiness check failed")
            return jsonify(status="unhealthy", project="22i-1714_A2_Ops"), 503

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        message = f"Your upload is too large. The image limit is {app.config['MAX_UPLOAD_BYTES'] // 1024**2} MB."
        if not remote:
            try:
                trail("rejected", error=message)
            except Exception:
                app.logger.exception("Could not record oversized request")
        if request.path.startswith("/api/"):
            return jsonify(error=message), 413
        return render_template("error.html", title="Image too large", message=message), 413

    @app.errorhandler(404)
    def not_found(_error):
        if request.path.startswith("/api/"):
            return jsonify(error="Not found."), 404
        return render_template("error.html", title="Page not found",
                               message="That page or prediction could not be found."), 404

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_app().run(host=os.getenv("HOST", "127.0.0.1"),
                     port=int(os.getenv("PORT", "8000")), debug=False)
