from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tensorflow import keras

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PARAMS_PATH = Path(__file__).resolve().parent.parent / "params.yaml"


def main():
    params = yaml.safe_load(open(PARAMS_PATH))["train"]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    x_train = np.load(PROCESSED_DIR / "x_train.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    x_val = np.load(PROCESSED_DIR / "x_val.npy")
    y_val = np.load(PROCESSED_DIR / "y_val.npy")

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=x_train.shape[1:]),
        keras.layers.Dense(params["dense_units"], activation="relu"),
        keras.layers.Dropout(params["dropout_rate"]),
        keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
    )

    model.save(MODELS_DIR / "model.h5")
    pd.DataFrame(history.history).to_csv(MODELS_DIR / "history.csv", index=False)

    print(f"Saved model to {MODELS_DIR / 'model.h5'}")
    print(f"Saved history to {MODELS_DIR / 'history.csv'}")


if __name__ == "__main__":
    main()
