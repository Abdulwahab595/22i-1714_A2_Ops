"""Recover genuine saved Keras metrics after the documented autolog parameter conflict."""
import json
import time
from pathlib import Path

from mlflow.tracking import MlflowClient


if __name__ == "__main__":
    client = MlflowClient(tracking_uri="sqlite:///artifacts/mlflow.db")
    for name in ["03_mobilenet_frozen_lr001", "04_mobilenet_frozen_lr0003"]:
        output = Path("artifacts/runs") / name
        result = json.loads((output / "result.json").read_text())
        history = json.loads((output / "history.json").read_text())
        run_id = result["run_id"]
        recovered = []
        for metric, values in history.items():
            if client.get_metric_history(run_id, metric):
                continue
            for step, value in enumerate(values, 1):
                # Timestamp is the honest recovery time, not an invented original epoch time.
                client.log_metric(run_id, metric, float(value), timestamp=int(time.time() * 1000),
                                  step=step)
            recovered.append(metric)
        client.set_tag(run_id, "logging_approach", "autolog_attempt_then_history_recovery")
        client.set_tag(run_id, "logging_issue", "batch_size=32 conflicted with fit(batch_size=None)")
        client.set_tag(run_id, "metric_recovery_source", "analysis/history.json")
        client.set_tag(run_id, "recovered_timestamps", "recovery time, not original epoch time")
        client.log_param(run_id, "pipeline_batch_size", 32)
        result["logging_recovery"] = "Genuine saved history; original epoch timestamps unavailable"
        (output / "result.json").write_text(json.dumps(result, indent=2))
        print(f"Recovered {name}: {recovered}")
