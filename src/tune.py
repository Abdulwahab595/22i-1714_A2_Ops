"""Eight real-data Keras Tuner trials, each independently tracked in MLflow."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "6")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import keras_tuner as kt
import mlflow
import mlflow.tensorflow
import pandas as pd
from tensorflow import keras

from src.data import make_dataset
from src.metrics import choose_threshold, classification_metrics, plot_curves
from src.models import set_seed, unfreeze
from src.tracking import (configure_tracking, dump_json, lineage,
                          log_common_artifacts, log_keras_model)
from src.train import callbacks, describe_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()
    if args.trials < 8:
        raise ValueError("The assignment requires at least eight VisionGuard trials")
    configure_tracking()
    mlflow.tensorflow.autolog(disable=True)
    train, val = [pd.read_csv(f"data/splits/{name}.csv") for name in ["train", "validation"]]
    train_data = make_dataset(train, "data/raw", training=True,
                               cache_dir="data/processed/tf128/train")
    val_data = make_dataset(val, "data/raw", cache_dir="data/processed/tf128/validation")
    candidates = [json.loads(path.read_text())
                  for path in Path("artifacts/runs").glob("*/result.json")]
    candidates = [r for r in candidates if r.get("stage") == "frozen"
                  and r.get("architecture") == "mobilenet_v2"]
    if not candidates:
        raise FileNotFoundError("Complete a frozen-backbone experiment before tuning")
    parent_result = max(candidates, key=lambda r: r["validation"]["roc_auc"])
    parent = parent_result["checkpoint"]
    print(f"Fixed tuner parent selected on validation ROC-AUC: {parent_result['name']}", flush=True)

    def builder(hp):
        set_seed(42)
        model = keras.models.load_model(parent)
        model.get_layer("regularization").rate = hp.Choice("dropout", [0.2, 0.35, 0.5])
        return unfreeze(model, lr=hp.Choice("learning_rate", [1e-5, 3e-5, 1e-4]),
                        last_layers=hp.Choice("unfrozen_layers", [4, 8, 12]))

    class TrackedSearch(kt.RandomSearch):
        def run_trial(self, trial, *fit_args, **fit_kwargs):
            output = Path("artifacts/runs") / f"07_tuner_{trial.trial_id}"
            output.mkdir(parents=True, exist_ok=True)
            with mlflow.start_run(run_name=f"07_tuner_{trial.trial_id}") as run:
                mlflow.set_tags({**lineage(), "logging_approach": "manual", "tool": "KerasTuner",
                                 "test_set_used": "false"})
                mlflow.log_params({**trial.hyperparameters.values, "image_size": 128,
                                   "stage": "tuner_fine_tune", "max_epochs": args.epochs,
                                   "parent_checkpoint": parent, "seed": 42})
                fit_kwargs["callbacks"] = callbacks(output, patience=2)
                histories = super().run_trial(trial, *fit_args, **fit_kwargs)
                history = histories[0].history if isinstance(histories, list) else histories.history
                model = keras.models.load_model(output / "best.keras")
                probabilities = model.predict(val_data, verbose=0).reshape(-1)
                threshold, policy = choose_threshold(val["label"], probabilities)
                scores = classification_metrics(val["label"], probabilities, threshold)
                dump_json(output / "history.json", history)
                best_epoch = plot_curves(history, output)
                describe_model(model, output)
                result = {"name": f"07_tuner_{trial.trial_id}", "architecture": "mobilenet_v2",
                          "stage": "tuner_fine_tune", "logging": "manual",
                          **trial.hyperparameters.values, "run_id": run.info.run_id,
                          "checkpoint": str(output / "best.keras"), "image_size": 128,
                          "seed": 42, "best_epoch": best_epoch, "validation": scores,
                          "threshold_policy": policy}
                dump_json(output / "result.json", result)
                mlflow.log_metrics({f"validation_{k}": float(v) for k, v in scores.items()})
                mlflow.log_metric("best_epoch", best_epoch)
                log_common_artifacts(output)
                log_keras_model(model)
                return histories

    tuner = TrackedSearch(builder, objective="val_loss", max_trials=args.trials, seed=42,
                          directory="artifacts/tuner", project_name="22i-1714_A2_Ops",
                          overwrite=False)
    tuner.search(train_data, validation_data=val_data, epochs=args.epochs, verbose=2)
    rows = []
    for trial in tuner.oracle.trials.values():
        rows.append({"trial_id": trial.trial_id, **trial.hyperparameters.values,
                     "best_val_loss": trial.score, "status": trial.status})
    output = Path("artifacts/tuner")
    pd.DataFrame(rows).to_csv(output / "trial_results.csv", index=False)
    dump_json(output / "best_hyperparameters.json", tuner.get_best_hyperparameters(1)[0].values)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
