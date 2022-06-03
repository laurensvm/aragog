import tensorflow as tf
from typing import Tuple
from aragog.networks.layers.dgm import DGMLayer
from aragog.networks.dgm import DGMSpaceTime, DGMParametric
from aragog.networks.layers.highway import HighwayLayer
from aragog.networks.layers.dense_concat import (
    DenseConcatThreeInputs,
    DenseConcatTwoInputs,
)


def create_spacetime_dgm_network(
    dimension_x: int,
    units: int = 50,
    layers: int = 3,
    layer_instance: DGMLayer = DGMLayer,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    x = tf.keras.Input(shape=(dimension_x,))
    t = tf.keras.Input(shape=(1,))
    # Customize the network here
    dgm_wrapper = DGMSpaceTime(
        units=units, n_layers=layers, layer_instance=layer_instance
    )
    dgm_wrapper.build(input_shape=(None, dimension_x + 1))
    outputs = dgm_wrapper(t, x)

    # These are actually KerasTensors.
    # They are treated as layers and should be passed into a tf.keras.Model class
    return t, x, outputs


def create_spacetime_highway_network(
    dimension_x: int, units: int = 50, layers: int = 3
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    x = tf.keras.Input(shape=(dimension_x,))
    t = tf.keras.Input(shape=(1,))

    dense_1 = DenseConcatTwoInputs(units)
    dense_1.build(input_shape=(None, dimension_x + 1))

    outputs = dense_1(t, x)
    outputs = tf.concat([t, x, outputs], axis=1)
    for i in range(layers):
        outputs = HighwayLayer(
            units=units + dimension_x + 1, activation_func=tf.nn.tanh
        )(outputs)
    outputs = tf.keras.layers.Dense(1)(outputs)
    return t, x, outputs


def create_spacetime_mlp(
    dimension_x: int, units: int = 50, layers: int = 3
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    x = tf.keras.Input(shape=(dimension_x,))
    t = tf.keras.Input(shape=(1,))

    dense_1 = DenseConcatTwoInputs(units)
    dense_1.build(input_shape=(None, dimension_x + 1))

    outputs = dense_1(t, x)
    for i in range(layers):
        outputs = tf.keras.layers.Dense(units=units, activation=tf.nn.tanh)(
            outputs
        )
    outputs = tf.keras.layers.Dense(1)(outputs)
    return t, x, outputs


def create_variance_process_dgm_network(
    dimension_x: int, units: int = 50, layers: int = 3, layer_instance=DGMLayer
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    x = tf.keras.Input(shape=(dimension_x,))
    t = tf.keras.Input(shape=(1,))
    v = tf.keras.Input(shape=(1,))
    # Customize the network here
    dgm_wrapper = DGMParametric(
        units=units, n_layers=layers, layer_instance=layer_instance
    )
    dgm_wrapper.build(input_shape=(None, dimension_x + 2))
    outputs = dgm_wrapper(t, x, v)
    # These are actually KerasTensors.
    # They are treated as layers and should be passed into a tf.keras.Model class
    return t, x, v, outputs


def create_variance_process_highway_network(
    dimension_x: int,
    units: int = 50,
    layers: int = 3,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    x = tf.keras.Input(shape=(dimension_x,))
    t = tf.keras.Input(shape=(1,))
    v = tf.keras.Input(shape=(1,))

    dense_1 = DenseConcatThreeInputs(units)
    dense_1.build(input_shape=(None, dimension_x + 2))

    outputs = dense_1(t, x, v)
    outputs = tf.concat([t, x, v, outputs], axis=1)
    for i in range(layers):
        outputs = HighwayLayer(
            units=units + dimension_x + 2, activation_func=tf.nn.tanh
        )(outputs)
    outputs = tf.keras.layers.Dense(1)(outputs)
    return t, x, v, outputs


def create_variance_process_mlp(
    dimension_x: int, units: int = 50, layers: int = 3
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    x = tf.keras.Input(shape=(dimension_x,))
    t = tf.keras.Input(shape=(1,))
    v = tf.keras.Input(shape=(1,))

    dense_1 = DenseConcatThreeInputs(units)
    dense_1.build(input_shape=(None, dimension_x + 2))

    outputs = dense_1(t, x, v)
    for i in range(layers):
        outputs = tf.keras.layers.Dense(units=units, activation=tf.nn.tanh)(
            outputs
        )
    outputs = tf.keras.layers.Dense(1)(outputs)
    return t, x, v, outputs
