"""Publish genuine locally recorded runs if remote setup was unavailable during training.

Original metric values, timestamps, and steps are retained. Imported runs are explicitly tagged;
this is not a replacement training run and never invents scores or a cloud execution history.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="sqlite:///artifacts/mlflow.db")
    parser.add_argument("--experiment", default="22i-1714_A2_Ops")
    args = parser.parse_args()
    load_dotenv()
    destination = os.getenv("MLFLOW_TRACKING_URI", "")
    if not destination.startswith("https://dagshub.com/"):
        raise ValueError("Set the DagsHub MLFLOW_TRACKING_URI in the untracked .env first")
    source = MlflowClient(tracking_uri=args.source)
    target = MlflowClient(tracking_uri=destination)
    original_experiment = source.get_experiment_by_name(args.experiment)
    remote_experiment = target.get_experiment_by_name(args.experiment)
    experiment_id = (remote_experiment.experiment_id if remote_experiment
                     else target.create_experiment(args.experiment))
    mapping_file = Path("artifacts/mlflow_remote_mapping.json")
    mapping = json.loads(mapping_file.read_text()) if mapping_file.exists() else {}
    for original in source.search_runs([original_experiment.experiment_id], max_results=1000):
        if original.info.run_id in mapping:
            continue
        existing = target.search_runs([experiment_id], filter_string=
            f"tags.original_local_run_id = '{original.info.run_id}'", max_results=1)
        if existing:
            remote = existing[0]
        else:
            tags = {key: value for key, value in original.data.tags.items()
                    if not key.startswith("mlflow.")}
            tags.update(original_local_run_id=original.info.run_id,
                         execution_location="local", publication="imported genuine local run")
            remote = target.create_run(experiment_id, start_time=original.info.start_time,
                                       tags=tags, run_name=original.data.tags.get("mlflow.runName"))
            target.log_batch(remote.info.run_id,
                             params=[Param(k, str(v)) for k, v in original.data.params.items()],
                             tags=[RunTag("source_tracking_store", args.source)])
            for name in original.data.metrics:
                metrics = source.get_metric_history(original.info.run_id, name)
                for start in range(0, len(metrics), 500):
                    target.log_batch(remote.info.run_id, metrics=[
                        Metric(m.key, m.value, m.timestamp, m.step) for m in metrics[start:start+500]])
            with tempfile.TemporaryDirectory(dir="artifacts") as temporary:
                artifacts = source.download_artifacts(original.info.run_id, "", temporary)
                target.log_artifacts(remote.info.run_id, artifacts)
            target.set_terminated(remote.info.run_id, original.info.status,
                                   end_time=original.info.end_time)
        mapping[original.info.run_id] = remote.info.run_id
        mapping_file.write_text(json.dumps(mapping, indent=2))
        print(f"Published {original.info.run_id} -> {remote.info.run_id}", flush=True)
    # Point selection records to the published copies for remote registration and final test logging.
    for path in Path("artifacts/runs").glob("*/result.json"):
        record = json.loads(path.read_text())
        if record.get("run_id") in mapping:
            record["original_local_run_id"] = record["run_id"]
            record["run_id"] = mapping[record["run_id"]]
            path.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
