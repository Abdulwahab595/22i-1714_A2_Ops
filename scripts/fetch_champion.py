"""Download a pinned model once before server startup; never called by request handlers."""
import hashlib
import json
import os
import shutil
from pathlib import Path


def main():
    model_path = Path(os.getenv("MODEL_PATH", "models/champion.keras"))
    metadata_path = Path(os.getenv("MODEL_METADATA_PATH", "models/champion.json"))
    run_id = os.getenv("CHAMPION_RUN_ID")
    if not model_path.exists():
        if not run_id or not os.getenv("MLFLOW_TRACKING_URI"):
            raise RuntimeError(f"Champion model missing at {model_path}. Mount it, or set "
                               "CHAMPION_RUN_ID and MLflow credentials")
        import mlflow.artifacts
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = model_path.parent / "download"
        temporary.mkdir(exist_ok=True)
        for name, target in [("champion.keras", model_path), ("champion.json", metadata_path)]:
            fetched = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=f"deployment/{name}", dst_path=str(temporary))
            shutil.copy2(fetched, target)
    metadata = json.loads(metadata_path.read_text())
    actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual != metadata["model_sha256"]:
        raise RuntimeError("Downloaded champion model hash does not match metadata")
    print(f"Champion ready: version={metadata['model_version']}, sha256={actual}", flush=True)


if __name__ == "__main__":
    main()
