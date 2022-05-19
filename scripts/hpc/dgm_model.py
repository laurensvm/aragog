import os
import logging
import tensorflow as tf
from typing import Tuple
from utils import save_model, parse_args
from aragog.logger import configure_logger
from aragog.pde_models.black_scholes.european import (
    EuropeanBlackScholesPDEModel,
)
from aragog.networks.dgm import DGMSpaceTime
from aragog.networks.layers.dgm import DGMLayer
from aragog.generators.space_time import SpaceTimeGenerator
from aragog.callbacks.timing import TimingCallback

LOGGER = logging.getLogger(__name__)


def create_dgm_network(
    dimension_x: int, units: int = 50, layers: int = 2, layer_instance=DGMLayer
) -> Tuple[tf.Tensor, tf.Tensor]:
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


def runner(args):
    configure_logger()
    LOGGER.info(f"Available devices: {tf.config.list_physical_devices('GPU')}")
    LOGGER.info(f"Model save path: {args.save_path}")

    # CONSTANTS
    batch_size = 10000
    dimension_x = 10
    T = 2.0
    x_range = [0 + 1e-10, 4]
    t_range = [0 + 1e-10, T]

    K = 1.0
    volatilities = tf.fill((dimension_x,), 0.25)
    correlations = tf.fill((dimension_x, dimension_x), 0.6)
    correlations = tf.linalg.set_diag(
        correlations, tf.fill((dimension_x,), 1.0)
    )
    riskfree_rate = 0.1

    layers = 3
    units = 256

    name = f"european_bs_{dimension_x}d_{units}n_{layers}l"
    save_path = os.path.join(args.save_path, name)

    t, x, outputs = create_dgm_network(
        dimension_x, units, layers, layer_instance=DGMLayer
    )

    # tf.strategy.MirroredStrategy here
    generator = SpaceTimeGenerator(
        dimension_x=dimension_x,
        batch_size=batch_size,
        x_range=x_range,
        t_range=t_range,
    )

    @tf.function
    def g_arithmetic(x: tf.Tensor, K: tf.constant) -> tf.Tensor:
        # g(x) = max(0, 1 / d * \sum_{i=1}^d x_i - K)
        return tf.math.maximum(tf.math.reduce_mean(x, axis=1) - K, 0)

    model = EuropeanBlackScholesPDEModel(
        dimension_x=dimension_x,
        K=K,
        volatilities=volatilities,
        correlations=correlations,
        riskfree_rate=riskfree_rate,
        inputs=[t, x],
        outputs=outputs,
        g_terminal=g_arithmetic,
    )

    boundaries = [7000, 10000, 15000, 25000, 50000, 75000]
    values = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values, name=None
    )

    timing_cb = TimingCallback(save_path=save_path)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    )

    LOGGER.info(model.summary())

    history = model.fit(
        x=generator, epochs=5000, steps_per_epoch=20, callbacks=[timing_cb]
    )

    save_model(model, save_path, history)


if __name__ == "__main__":
    args = parse_args()
    runner(args)
