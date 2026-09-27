from pathlib import Path

import numpy as np
from tensorflow import keras

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    np.save(RAW_DIR / "x_train.npy", x_train)
    np.save(RAW_DIR / "y_train.npy", y_train)
    np.save(RAW_DIR / "x_test.npy", x_test)
    np.save(RAW_DIR / "y_test.npy", y_test)

    print(f"Saved raw Fashion-MNIST splits to {RAW_DIR}")
    print(f"  x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"  x_test:  {x_test.shape}, y_test:  {y_test.shape}")


if __name__ == "__main__":
    main()
