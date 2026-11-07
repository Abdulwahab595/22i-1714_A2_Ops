"""Required built-in dataset Keras Tuner warm-up. Test images are not used for tuning."""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_HOME", str(Path(".keras-cache").resolve()))

import keras_tuner as kt
import pandas as pd
from tensorflow import keras

from src.models import set_seed

set_seed(42)
(pixels, labels), _ = keras.datasets.fashion_mnist.load_data()
pixels = pixels[:4000].astype("float32") / 255
labels = labels[:4000]


def builder(hp):
    model = keras.Sequential([keras.Input((28, 28)), keras.layers.Flatten(),
                              keras.layers.Dense(hp.Choice("units", [32, 64, 128]),
                                                 activation="relu"),
                              keras.layers.Dense(10, activation="softmax")])
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice("learning_rate", [1e-3, 1e-2])),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    tuner = kt.RandomSearch(builder, objective="val_accuracy", max_trials=4, seed=42,
                            directory="artifacts/toy-tuner", project_name="22i-1714_A2_Ops")
    tuner.search(pixels, labels, validation_split=0.2, epochs=3, batch_size=64, verbose=2)
    rows = [{"trial": t.trial_id, **t.hyperparameters.values, "val_accuracy": t.score}
            for t in tuner.oracle.trials.values()]
    pd.DataFrame(rows).to_csv("artifacts/toy-tuner/results.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("Best settings:", tuner.get_best_hyperparameters(1)[0].values)
