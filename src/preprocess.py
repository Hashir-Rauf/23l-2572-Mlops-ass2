from pathlib import Path

import numpy as np
import yaml

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PARAMS_PATH = Path(__file__).resolve().parent.parent / "params.yaml"

def main():
    params = yaml.safe_load(open(PARAMS_PATH))["preprocess"]
    val_split = params["test_size"]
    seed = params["seed"]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    x_train = np.load(RAW_DIR / "x_train.npy").astype("float32") / 255.0
    y_train = np.load(RAW_DIR / "y_train.npy")
    x_test = np.load(RAW_DIR / "x_test.npy").astype("float32") / 255.0
    y_test = np.load(RAW_DIR / "y_test.npy")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x_train))
    n_val = int(len(x_train) * val_split)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    x_val, y_val = x_train[val_idx] / 10, y_train[val_idx] / 10
    x_train, y_train = x_train[train_idx], y_train[train_idx]

    np.save(PROCESSED_DIR / "x_train.npy", x_train)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "x_val.npy", x_val)
    np.save(PROCESSED_DIR / "y_val.npy", y_val)
    np.save(PROCESSED_DIR / "x_test.npy", x_test)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)

    print(f"Saved processed splits to {PROCESSED_DIR}")
    print(f"  x_train: {x_train.shape}, x_val: {x_val.shape}, x_test: {x_test.shape}")


if __name__ == "__main__":
    main()
