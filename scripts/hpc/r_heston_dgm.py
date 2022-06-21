"""
This Python script trains multiple simple MLP's with the same configuration
on the Black-Scholes and Implied Volatility datasets. It should be run with Slurm
as a batch job on DHPC.
"""

import logging
import tensorflow as tf
import os
import numpy as np
from aragog.logger import configure_logger
from aragog.callbacks.timing import TimingCallback
from aragog.networks.dgm import DGM
from utils import load_training_datasets, save_model, parse_args


LOGGER = logging.getLogger(__name__)


def build_model(
    input_shape: int,
    nodes: int,
    layers: int,
    loss: tf.keras.losses.Loss = tf.keras.losses.MeanSquaredError(),
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_shape,))
    dgm_wrapper = DGM(units=nodes, n_layers=layers)
    outputs = dgm_wrapper(inputs)
    model = tf.keras.Model(inputs, outputs)
    model.compile(loss=loss, optimizer=optimizer)
    return model


def train_model(
    nodes: int,
    layers: int,
    X: np.ndarray,
    y: np.ndarray,
    save_path: str,
    type: str,
    name: str,
):
    model_save_path = os.path.join(
        save_path, f"{type}_{name}_{layers}l_{nodes}n_equal_params"
    )
    os.makedirs(model_save_path, exist_ok=True)

    model = build_model(input_shape=X.shape[1], layers=layers, nodes=nodes)
    LOGGER.info(model.summary())

    timing_cb = TimingCallback(save_path=model_save_path)
    history = model.fit(
        X,
        y,
        epochs=200,
        batch_size=64,
        callbacks=[timing_cb],
        validation_split=0.2,
    )

    save_model(model, model_save_path, history)
    del model


def runner(args):
    configure_logger()

    LOGGER.warning(f"GPU: {tf.test.is_gpu_available()}")

    nodes = 50
    layers = 2

    X_train_heston, y_train_heston = load_training_datasets(
        args.data_path,
        include_iv=False,
        filenames=["X_train_heston.csv", "y_train_heston.csv"],
    )

    train_model(
        nodes,
        layers,
        X_train_heston,
        y_train_heston,
        args.save_path,
        type="heston",
        name="dgm",
    )


if __name__ == "__main__":
    args = parse_args()
    runner(args)
