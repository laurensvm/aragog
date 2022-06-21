import os
import logging
import tensorflow as tf

# import numpy as np
from aragog.logger import configure_logger
from aragog.generators.space_time import SpaceTimeGenerator
from aragog.pde_models.black_scholes.european import (
    EuropeanBlackScholesPDEModel,
)
from aragog.callbacks.timing import TimingCallback
from aragog.pde_models.payoffs import g_arithmetic
from aragog.networks.factories import (
    # create_spacetime_mlp,
    # create_spacetime_dgm_network,
    create_spacetime_highway_network,
)
from aragog.schedules.piecewise import build_piecewise_decay_schedule
from scripts.hpc.utils import save_model, parse_args

LOGGER = logging.getLogger(__name__)


def runner(args):
    configure_logger()

    # Constants
    units = 100
    layers = 4
    steps_per_epoch = 20

    batch_size = 5000
    epochs = 1000

    K = 1.0
    T = 2.0
    dimension_x = 20
    t_range = [0 + 1e-10, T]
    x_range = [0 + 1e-10, 4.0]

    volatilities = tf.fill((dimension_x,), 0.25)
    # correlations = np.array([[1, 0.5], [0.5, 1]])
    correlations = tf.fill([dimension_x, dimension_x], 0.5)
    correlations = tf.linalg.set_diag(
        correlations, tf.ones(shape=(dimension_x,))
    )
    riskfree_rate = 0.1

    name = f"european_bs_{dimension_x}d_{units}n_{layers}l_hw_arith"
    save_path = os.path.join(args.save_path, name)

    learning_rate = build_piecewise_decay_schedule(epochs * steps_per_epoch)
    timing_cb = TimingCallback(save_path=save_path)

    generator = SpaceTimeGenerator(
        batch_size=batch_size,
        dimension_x=dimension_x,
        t_range=t_range,
        x_range=x_range,
    )

    t, x, outputs = create_spacetime_highway_network(
        dimension_x=dimension_x, units=units, layers=layers
    )

    model = EuropeanBlackScholesPDEModel(
        dimension_x=dimension_x,
        K=K,
        correlations=correlations,
        volatilities=volatilities,
        riskfree_rate=riskfree_rate,
        g_terminal=g_arithmetic,
        inputs=[t, x],
        outputs=outputs,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    )

    print(model.summary())

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
