"""Register both evaluated finalists, assign champion, and log deployment files."""
import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.tracking import configure_tracking, dump_json


def main():
    configure_tracking()
    selected = json.loads(Path("artifacts/evaluation/locked_selection.json").read_text())
    client = MlflowClient()
    versions = {}
    for family in ["cnn", "transfer"]:
        result = selected[family]
        version = mlflow.register_model(f"runs:/{result['run_id']}/model", "22i-1714_A2_Ops")
        client.set_model_version_tag("22i-1714_A2_Ops", version.version,
                                     "training_run", result["name"])
        versions[result["name"]] = str(version.version)
    champion = selected["champion"]
    version = versions[champion["name"]]
    client.set_registered_model_alias("22i-1714_A2_Ops", "champion", version)
    metadata_path = Path("models/champion.json")
    metadata = json.loads(metadata_path.read_text())
    metadata["model_version"] = version
    metadata["registry_model_uri"] = "models:/22i-1714_A2_Ops@champion"
    dump_json(metadata_path, metadata)
    with mlflow.start_run(run_id=champion["run_id"]):
        mlflow.log_artifact("models/champion.keras", artifact_path="deployment")
        mlflow.log_artifact("models/champion.json", artifact_path="deployment")
    dump_json("artifacts/evaluation/registry.json", {"versions": versions,
              "champion_version": version, "champion_run_id": champion["run_id"]})
    print(json.dumps({"versions": versions, "champion": version}, indent=2))


if __name__ == "__main__":
    main()
