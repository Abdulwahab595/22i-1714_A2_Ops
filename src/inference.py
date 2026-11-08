"""Load the champion once and reuse the exact training preprocessing."""
import hashlib
import json
import os
import threading
from pathlib import Path

from src.data import preprocess_bytes


class InferenceEngine:
    def __init__(self, model_path, metadata_path, threshold=None):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
        os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
        from tensorflow import keras

        model_path, metadata_path = Path(model_path), Path(metadata_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Champion model missing: {model_path}. Export it before startup.")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Champion metadata missing: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text())
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        expected = self.metadata.get("model_sha256")
        if expected and digest != expected:
            raise ValueError("Model hash differs from champion metadata; deploy matching files")
        self.metadata["model_sha256"] = digest
        self.threshold = float(threshold if threshold is not None else self.metadata["threshold"])
        if not 0 <= self.threshold <= 1:
            raise ValueError("DECISION_THRESHOLD must be between zero and one")
        self.image_size = int(self.metadata["image_size"])
        self.model = keras.models.load_model(model_path, compile=False)
        self._lock = threading.Lock()
        self.loads = 1

    def predict(self, content):
        with self._lock:
            # Bound concurrent decoder/resize memory as well as model execution.
            batch = preprocess_bytes(content, self.image_size)
            probability = float(self.model(batch, training=False).numpy().reshape(-1)[0])
        is_ai = probability >= self.threshold
        return {"label": "ai_generated" if is_ai else "real",
                "confidence": probability if is_ai else 1 - probability,
                "ai_probability": probability, "threshold": self.threshold,
                "model_name": self.metadata.get("registered_model_name", "22i-1714_A2_Ops"),
                "model_version": str(self.metadata.get("model_version", "unregistered")),
                "model_sha256": self.metadata["model_sha256"]}
