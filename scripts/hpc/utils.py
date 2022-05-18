import tensorflow as tf
from typing import Any, Tuple
import argparse
import os
import pandas as pd
import numpy as np


def save_model(model: tf.keras.Model, save_path: str, history: Any):
    # Save the model
    model.save(save_path)

    # Save the history object
    if hasattr(history, "history"):
        history = history.history

    hist_csv_file = os.path.join(save_path, "history.csv")
    hist_df = pd.DataFrame(history)
    with open(hist_csv_file, mode="w") as f:
        hist_df.to_csv(f)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="HPC Training Script",
        description="Run the Training Script",
    )
    parser.add_argument(
        "--data_path",
        "-dp",
        help="Relative data path of the folder containing the csv files.",
    )
    parser.add_argument(
        "--save_path",
        "-sp",
        help="Relative path of the folder to save the models in.",
    )

    return parser.parse_args()


def load_training_datasets(
    root_path: str,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    files = [
        pd.read_csv(os.path.join(root_path, f))
        for f in ["BS_train_input.csv", "BS_train_output.csv"]
    ]

    X_train, y_train = tuple(files)

    # Modify so that the volatility becomes the target
    y_train_iv = X_train["sigma"]
    X_train_iv = X_train.copy()
    X_train_iv["price"] = y_train
    X_train_iv.drop("sigma", axis=1, inplace=True)

    return (X_train.values, y_train.values), (
        X_train_iv.values,
        y_train_iv.values,
    )
