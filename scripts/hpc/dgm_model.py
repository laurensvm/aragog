import os
import logging
import tensorflow as tf
from utils import save_model, parse_args
from aragog.logger import configure_logger
from aragog.pde_models.black_scholes.european import (
    EuropeanBlackScholesPDEModel,
)
from aragog.generators.space_time import SpaceTimeGenerator
from aragog.networks.factories import create_spacetime_dgm_network
from aragog.callbacks.timing import TimingCallback
from aragog.pde_models.payoffs import g_arithmetic
from aragog.schedules.piecewise import build_piecewise_decay_schedule

LOGGER = logging.getLogger(__name__)


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

    epochs = 5000
    steps_per_epoch = 20

    name = f"european_bs_{dimension_x}d_{units}n_{layers}l"
    save_path = os.path.join(args.save_path, name)

    t, x, outputs = create_spacetime_dgm_network(dimension_x, units, layers)

    # tf.strategy.MirroredStrategy here
    generator = SpaceTimeGenerator(
        dimension_x=dimension_x,
        batch_size=batch_size,
        x_range=x_range,
        t_range=t_range,
    )

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

    learning_rate = build_piecewise_decay_schedule(epochs * steps_per_epoch)

    timing_cb = TimingCallback(save_path=save_path)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    )

    LOGGER.info(model.summary())

    history = model.fit(
        x=generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[timing_cb],
    )

    save_model(model, save_path, history)


if __name__ == "__main__":
    args = parse_args()
    runner(args)
