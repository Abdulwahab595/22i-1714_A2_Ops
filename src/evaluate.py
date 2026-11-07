"""Lock finalists on validation, evaluate each test set once, and export the champion."""
import hashlib
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tensorflow import keras

from src.data import make_dataset
from src.metrics import save_evaluation
from src.tracking import configure_tracking, dump_json, lineage


def ranking(result):
    scores = result["validation"]
    return (-scores["roc_auc"], scores["weighted_error"], -scores["f1"])


def error_grid(frame, raw, output):
    errors = frame[frame["label"] != frame["prediction"]].copy()
    errors = errors.sort_values("confidence", ascending=False).head(10)
    # Hypotheses are deliberately left for visual interpretation, never asserted as causes.
    errors["hypothesis"] = "Pending visual inspection; appearance alone does not establish cause."
    errors.to_csv(output / "misclassified.csv", index=False)
    fig, axes = plt.subplots(2, 5, figsize=(14, 7))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, entry) in zip(axes.flat, errors.iterrows()):
        with Image.open(raw / entry["path"]) as image:
            ax.imshow(ImageOps.exif_transpose(image).convert("RGB"))
        truth = "AI" if entry["label"] else "Real"
        predicted = "AI" if entry["prediction"] else "Real"
        ax.set_title(f"True: {truth} | Predicted: {predicted}\n"
                     f"AI score: {entry['ai_probability']:.3f}", fontsize=9)
    fig.suptitle("22i-1714_A2_Ops | Highest-confidence test errors (up to ten)")
    fig.tight_layout()
    fig.savefig(output / "error_grid.png", dpi=170)
    plt.close(fig)


def main():
    configure_tracking()
    output = Path("artifacts/evaluation")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "evaluation_complete.json").exists():
        print("Final test evaluation already exists; it will not be repeated.")
        return
    results = [json.loads(path.read_text()) for path in Path("artifacts/runs").glob("*/result.json")]
    if len(results) < 14:
        raise ValueError("Complete six controlled runs and at least eight tuner trials first")
    comparison = [{"name": r["name"], "architecture": r["architecture"],
                    "stage": r["stage"], "best_epoch": r["best_epoch"],
                    **{f"val_{k}": v for k, v in r["validation"].items()}}
                   for r in results]
    pd.DataFrame(comparison).sort_values("val_roc_auc", ascending=False).to_csv(
        output / "validation_comparison.csv", index=False)
    cnn = min([r for r in results if r["architecture"] == "cnn"], key=ranking)
    transfer = min([r for r in results if r["architecture"] == "mobilenet_v2"], key=ranking)
    champion = min([cnn, transfer], key=ranking)
    # Selection is persisted before this module opens any test labels.
    dump_json(output / "locked_selection.json", {"cnn": cnn, "transfer": transfer,
              "champion": champion, "selection_basis": "validation ROC-AUC; weighted error tie-break"})
    test = pd.read_csv("data/splits/test.csv")
    dataset = make_dataset(test, "data/raw", cache_dir="data/processed/tf128/test")
    evaluated = []
    for result in [cnn, transfer]:
        subdir = output / result["architecture"]
        subdir.mkdir(exist_ok=True)
        if (subdir / "test_predictions.csv").exists():
            prediction_table = pd.read_csv(subdir / "test_predictions.csv")
            metrics = json.loads((subdir / "metrics.json").read_text())
        else:
            model = keras.models.load_model(result["checkpoint"], compile=False)
            probabilities = model.predict(dataset, verbose=0).reshape(-1)
            threshold = result["validation"]["threshold"]
            metrics = save_evaluation(test["label"], probabilities, threshold, subdir)
            prediction_table = test.copy()
            prediction_table["ai_probability"] = probabilities
            prediction_table["prediction"] = (probabilities >= threshold).astype(int)
            prediction_table["confidence"] = np.where(prediction_table["prediction"],
                                                       probabilities, 1 - probabilities)
            prediction_table.to_csv(subdir / "test_predictions.csv", index=False)
            error_grid(prediction_table, Path("data/raw"), subdir)
        with mlflow.start_run(run_id=result["run_id"]):
            mlflow.set_tag("finalist_test_evaluated_once", "true")
            mlflow.log_metrics({f"test_{k}": float(v) for k, v in metrics.items()})
            mlflow.log_artifacts(str(subdir), artifact_path="final_test")
        evaluated.append({"name": result["name"], "architecture": result["architecture"], **metrics})
    Path("models").mkdir(exist_ok=True)
    shutil.copy2(champion["checkpoint"], "models/champion.keras")
    metadata = {"project": "22i-1714_A2_Ops", "registered_model_name": "22i-1714_A2_Ops",
                "model_version": "unregistered", "model_sha256": hashlib.sha256(
                    Path("models/champion.keras").read_bytes()).hexdigest(),
                "run_id": champion["run_id"], "architecture": champion["architecture"],
                "image_size": champion["image_size"],
                "threshold": champion["validation"]["threshold"], "positive_class": "ai_generated",
                "class_mapping": {"0": "real", "1": "ai_generated"},
                "preprocessing": "Pillow EXIF transpose, RGB, tf.image.resize_with_pad bilinear antialias",
                "normalization": "Included inside the saved Keras model",
                "threshold_policy": "Validation only: minimize (2*FP + FN)/N; AI is positive",
                "selection_basis": "Validation ROC-AUC, weighted-error and F1 tie-breaks",
                "lineage": lineage(), "confidence_calibrated": False}
    dump_json("models/champion.json", metadata)
    pd.DataFrame(evaluated).to_csv(output / "test_comparison.csv", index=False)
    dump_json(output / "evaluation_complete.json", {"champion": champion["name"],
              "test_results": evaluated, "model_metadata": metadata})
    print(json.dumps({"champion": champion["name"], "test_results": evaluated}, indent=2))


if __name__ == "__main__":
    main()
