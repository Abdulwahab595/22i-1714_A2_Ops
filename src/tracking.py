"""Tracking helpers: no account names or credentials are embedded in source."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from tensorflow import keras


def configure_tracking(experiment="22i-1714_A2_Ops"):
    load_dotenv()
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        Path("artifacts").mkdir(exist_ok=True)
        uri = "sqlite:///artifacts/mlflow.db"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return uri


def lineage(seed=42):
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=False).stdout.strip() or "uncommitted"
    pointer = Path("data/raw.dvc")
    split = Path("data/splits/train.csv")
    data_version = (hashlib.sha256(pointer.read_bytes()).hexdigest() if pointer.exists()
                    else hashlib.sha256(split.read_bytes()).hexdigest() if split.exists()
                    else "not_available")
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                           text=True, check=False).stdout.strip()
    return {"project": "22i-1714_A2_Ops", "git_commit": commit, "dvc_data_version": data_version,
            "git_dirty": str(bool(dirty)).lower(),
            "data_version_kind": "dvc_pointer_sha256" if pointer.exists() else "split_sha256",
            "seed": str(seed), "positive_class": "ai_generated"}


class ManualMetrics(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        values = {key: float(value) for key, value in (logs or {}).items()}
        values["learning_rate"] = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        mlflow.log_metrics(values, step=epoch + 1)


def log_common_artifacts(output):
    for path in Path(output).iterdir():
        if path.is_file() and path.suffix not in {".keras", ".h5"}:
            mlflow.log_artifact(str(path), artifact_path="analysis")
    for directory in ["data/splits"]:
        if Path(directory).exists():
            mlflow.log_artifacts(directory, artifact_path="splits")
    for filename in ["artifacts/packages.txt", "artifacts/environment.json", "data/raw.dvc"]:
        if Path(filename).exists():
            mlflow.log_artifact(filename, artifact_path="reproducibility")


def log_keras_model(model):
    import mlflow.tensorflow

    return mlflow.tensorflow.log_model(model, artifact_path="model", pip_requirements=[
        "tensorflow==2.19.0", "keras==3.10.0", "numpy==2.1.3"])


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
