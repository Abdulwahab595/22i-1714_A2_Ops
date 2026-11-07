"""Record genuine environment versions and TensorFlow GPU visibility."""
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import tensorflow as tf

packages = ["tensorflow", "tensorflow-cpu", "keras", "numpy", "pandas", "Pillow", "mlflow",
            "Flask", "SQLAlchemy", "psycopg", "keras-tuner", "dvc", "scikit-learn"]
versions = {}
for name in packages:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        continue
record = {"project": "22i-1714_A2_Ops", "student": "Abdul Wahab", "roll_number": "22i-1714",
          "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
          "executable": sys.executable, "platform": platform.platform(), "versions": versions,
          "tensorflow_gpu_devices": [d.name for d in tf.config.list_physical_devices("GPU")]}
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/environment.json").write_text(json.dumps(record, indent=2))
freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                        check=True, text=True).stdout
Path("artifacts/packages.txt").write_text(freeze)
print(json.dumps(record, indent=2))
