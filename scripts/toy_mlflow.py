"""Required tiny Keras lab: two learning rates, per-epoch metrics, plot, and model."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_HOME", str(Path(".keras-cache").resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from tensorflow import keras

from src.models import set_seed
from src.tracking import ManualMetrics, configure_tracking, log_keras_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true")
    args = parser.parse_args()
    if not args.remote:
        os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///artifacts/local_mlflow_lab.db"
    configure_tracking("22i-1714_A2_Ops_MLflow_Lab")
    (pixels, labels), _ = keras.datasets.fashion_mnist.load_data()
    pixels = pixels[:4000].astype("float32") / 255
    labels = labels[:4000]
    for lr in [1e-3, 1e-2]:
        keras.backend.clear_session()
        set_seed(42)
        model = keras.Sequential([keras.Input((28, 28)), keras.layers.Flatten(),
                                  keras.layers.Dense(32, activation="relu"),
                                  keras.layers.Dense(10, activation="softmax")])
        model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        with mlflow.start_run(run_name=f"toy_lr_{lr}"):
            mlflow.log_params({"learning_rate": lr, "units": 32, "dataset": "Fashion-MNIST",
                               "training_examples": 4000, "seed": 42})
            mlflow.set_tag("lab", "remote" if args.remote else "local")
            history = model.fit(pixels, labels, validation_split=0.2, epochs=3, batch_size=64,
                                callbacks=[ManualMetrics()], verbose=2)
            output = Path("artifacts/local-mlflow") / f"lr_{lr}"
            output.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(np.arange(1, 4), history.history["loss"], label="train")
            ax.plot(np.arange(1, 4), history.history["val_loss"], label="validation")
            ax.set(xlabel="Epoch", ylabel="Loss", title=f"22i-1714_A2_Ops | Toy LR {lr}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output / "loss.png", dpi=150)
            plt.close(fig)
            mlflow.log_artifact(str(output / "loss.png"))
            log_keras_model(model)


if __name__ == "__main__":
    main()
