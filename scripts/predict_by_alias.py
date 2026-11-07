"""A standalone registry consumer, intentionally separate from Flask serving."""
import argparse
import json
from pathlib import Path

import mlflow.tensorflow
from mlflow.tracking import MlflowClient

from src.data import preprocess_bytes
from src.tracking import configure_tracking


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()
    configure_tracking()
    uri = "models:/22i-1714_A2_Ops@champion"
    version = MlflowClient().get_model_version_by_alias("22i-1714_A2_Ops", "champion")
    metadata = json.loads(Path("models/champion.json").read_text())
    model = mlflow.tensorflow.load_model(uri)
    probability = float(model(preprocess_bytes(Path(args.image).read_bytes(),
                                               metadata["image_size"]), training=False).numpy()[0, 0])
    print(json.dumps({"model_uri": uri, "version": version.version,
                      "ai_probability": probability, "threshold": metadata["threshold"],
                      "label": "ai_generated" if probability >= metadata["threshold"] else "real"},
                     indent=2))
