"""Run six controlled experiments. Test labels are never opened by this module."""
import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "6")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("KERAS_HOME", str(Path(".keras-cache").resolve()))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
from tensorflow import keras

from src.data import make_dataset
from src.metrics import choose_threshold, classification_metrics, plot_curves
from src.models import (augmentation, build_cnn, build_transfer, set_seed,
                        trainable_layer_names, unfreeze)
from src.tracking import (ManualMetrics, configure_tracking, dump_json, lineage,
                          log_common_artifacts, log_keras_model)


def callbacks(output, logging="manual", patience=3):
    items = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                           restore_best_weights=True),
             keras.callbacks.ModelCheckpoint(str(output / "best.keras"), monitor="val_loss",
                                               save_best_only=True),
             keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                patience=2, min_lr=1e-7)]
    if logging == "manual":
        items.append(ManualMetrics())
    return items


def describe_model(model, output):
    with (output / "model_summary.txt").open("w", encoding="utf-8") as file, contextlib.redirect_stdout(file):
        model.summary(expand_nested=True)
    layers = []
    for layer in model.layers:
        try:
            shape = str(layer.output.shape)
        except AttributeError:
            shape = "multiple"
        layers.append({"name": layer.name, "type": type(layer).__name__, "output_shape": shape,
                       "parameters": layer.count_params(), "trainable": layer.trainable})
    pd.DataFrame(layers).to_csv(output / "layers.csv", index=False)
    dump_json(output / "trainable_layers.json", trainable_layer_names(model))


def save_augmented_batch(dataset, output):
    pixels, labels = next(iter(dataset))
    augmented = augmentation(42)(pixels[:12], training=True).numpy()
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for ax, image, label in zip(axes.flat, augmented, labels.numpy()):
        ax.imshow(np.clip(image / 255, 0, 1))
        ax.set_title("AI-generated" if label else "Real")
        ax.axis("off")
    fig.suptitle("22i-1714_A2_Ops | Training-only random augmentation")
    fig.tight_layout()
    fig.savefig(output / "augmented_batch.png", dpi=150)
    plt.close(fig)


def fit_experiment(config, train_data, val_data, val_frame, epochs, image_size, seed,
                   parent_checkpoint=None):
    set_seed(seed)
    keras.backend.clear_session()
    output = Path("artifacts/runs") / config["name"]
    if (output / "result.json").exists():
        return json.loads((output / "result.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    if config["architecture"] == "cnn":
        model = build_cnn(image_size, config["lr"], config["dropout"], config["augmentation"], seed)
    elif config["stage"] == "fine_tune":
        if not parent_checkpoint:
            raise ValueError("Fine tuning requires the frozen-stage checkpoint")
        model = keras.models.load_model(parent_checkpoint)
        unfreeze(model, config["lr"])
    else:
        model = build_transfer(image_size, config["lr"], config["dropout"],
                               config["augmentation"], seed)
    describe_model(model, output)
    dump_json(output / "config.json", config)
    with mlflow.start_run(run_name=config["name"]) as run:
        mlflow.set_tags({**lineage(seed), "logging_approach": config["logging"],
                         "stage": config["stage"], "test_set_used": "false"})
        mlflow.log_params({**config, "image_size": image_size, "pipeline_batch_size": 32,
                           "max_epochs": epochs, "seed": seed,
                           "resize": "bilinear_resize_with_pad", "loss": "binary_crossentropy",
                           "optimizer": "Adam"})
        if config["logging"] == "autolog":
            mlflow.tensorflow.autolog(log_models=False, checkpoint=False,
                                      log_datasets=False, silent=False)
        else:
            mlflow.tensorflow.autolog(disable=True)
        started = time.monotonic()
        history = model.fit(train_data, validation_data=val_data, epochs=epochs,
                            callbacks=callbacks(output, config["logging"]), verbose=2)
        elapsed = time.monotonic() - started
        mlflow.tensorflow.autolog(disable=True)
        model = keras.models.load_model(output / "best.keras")
        probabilities = model.predict(val_data, verbose=0).reshape(-1)
        threshold, policy = choose_threshold(val_frame["label"].to_numpy(), probabilities)
        scores = classification_metrics(val_frame["label"], probabilities, threshold)
        history_values = {key: [float(v) for v in values] for key, values in history.history.items()}
        dump_json(output / "history.json", history_values)
        best_epoch = plot_curves(history_values, output)
        prediction_table = val_frame[["path", "class", "label", "sha256"]].copy()
        prediction_table["ai_probability"] = probabilities
        prediction_table.to_csv(output / "validation_predictions.csv", index=False)
        result = {**config, "run_id": run.info.run_id, "checkpoint": str(output / "best.keras"),
                  "best_epoch": best_epoch, "epochs_completed": len(history.history["loss"]),
                  "training_seconds": elapsed, "threshold_policy": policy,
                  "validation": scores, "image_size": image_size, "seed": seed}
        dump_json(output / "result.json", result)
        mlflow.log_metrics({f"validation_{key}": float(value) for key, value in scores.items()})
        mlflow.log_metrics({"best_epoch": best_epoch, "training_seconds": elapsed})
        log_common_artifacts(output)
        log_keras_model(model)
        print(f"FINISHED {config['name']}: {json.dumps(scores)}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--only", default="", help="Comma-separated experiment numbers, e.g. 1,2")
    parser.add_argument("--limit", type=int, default=0, help="Smoke test only; never use in final runs")
    args = parser.parse_args()
    print("Tracking:", configure_tracking(), flush=True)
    train_frame, val_frame = [pd.read_csv(f"data/splits/{name}.csv")
                              for name in ["train", "validation"]]
    if args.limit:
        train_frame = train_frame.groupby("label", group_keys=False).head(args.limit // 2)
        val_frame = val_frame.groupby("label", group_keys=False).head(max(4, args.limit // 4))
    datasets = [make_dataset(frame, "data/raw", args.image_size, training=training,
                             cache_dir=None if args.limit else
                             f"data/processed/tf{args.image_size}/{name}")
                for frame, training, name in [(train_frame, True, "train"),
                                              (val_frame, False, "validation")]]
    Path("artifacts").mkdir(exist_ok=True)
    save_augmented_batch(datasets[0], Path("artifacts"))
    configs = [
        {"name": "01_cnn_no_augmentation", "architecture": "cnn", "stage": "scratch",
         "lr": 1e-3, "dropout": 0.3, "augmentation": False, "logging": "manual"},
        {"name": "02_cnn_augmentation", "architecture": "cnn", "stage": "scratch",
         "lr": 1e-3, "dropout": 0.3, "augmentation": True, "logging": "manual"},
        {"name": "03_mobilenet_frozen_lr001", "architecture": "mobilenet_v2", "stage": "frozen",
         "lr": 1e-3, "dropout": 0.3, "augmentation": True, "logging": "autolog"},
        {"name": "04_mobilenet_frozen_lr0003", "architecture": "mobilenet_v2", "stage": "frozen",
         "lr": 3e-4, "dropout": 0.3, "augmentation": True, "logging": "autolog"},
        {"name": "05_mobilenet_last_block_lr1e5", "architecture": "mobilenet_v2",
         "stage": "fine_tune", "lr": 1e-5, "dropout": 0.3, "augmentation": True,
         "logging": "manual"},
        {"name": "06_mobilenet_last_block_lr3e5", "architecture": "mobilenet_v2",
         "stage": "fine_tune", "lr": 3e-5, "dropout": 0.3, "augmentation": True,
         "logging": "manual"},
        {"name": "08_mobilenet_autolog_corrected", "architecture": "mobilenet_v2",
         "stage": "frozen", "lr": 3e-4, "dropout": 0.3, "augmentation": True,
         "logging": "autolog"},
        {"name": "09_mobilenet_last_block_corrected_lr1e5", "architecture": "mobilenet_v2",
         "stage": "fine_tune", "lr": 1e-5, "dropout": 0.3, "augmentation": True,
         "logging": "manual"},
        {"name": "10_mobilenet_last_block_corrected_lr3e5", "architecture": "mobilenet_v2",
         "stage": "fine_tune", "lr": 3e-5, "dropout": 0.3, "augmentation": True,
         "logging": "manual"},
    ]
    selected = {int(n) for n in args.only.split(",") if n} if args.only else set(range(1, 7))
    results = []
    parent = "artifacts/runs/04_mobilenet_frozen_lr0003/best.keras"
    for index, config in enumerate(configs, 1):
        if index in selected:
            result = fit_experiment(config, datasets[0], datasets[1], val_frame,
                                    args.epochs, args.image_size, 42, parent)
            results.append(result)
    Path("artifacts/experiment_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
