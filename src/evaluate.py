import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow import keras

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ROOT_DIR = Path(__file__).resolve().parent.parent


def main():
    x_test = np.load(PROCESSED_DIR / "x_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    model = keras.models.load_model(MODELS_DIR / "model.h5")

    loss, accuracy = model.evaluate(x_test, y_test)

    y_pred = np.argmax(model.predict(x_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    ConfusionMatrixDisplay(cm).plot()
    plt.savefig(ROOT_DIR / "confusion_matrix.png")

    metrics = {"loss": float(loss), "accuracy": float(accuracy)}
    with open(ROOT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Test loss: {loss}, Test accuracy: {accuracy}")


if __name__ == "__main__":
    main()
