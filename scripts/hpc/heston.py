import os
import logging
import tensorflow as tf
import numpy as np
from aragog.logger import configure_logger
from aragog.generators.variance import HestonSpaceTimeVarianceGenerator
from aragog.pde_models.heston.european import EuropeanHestonPDEModel
from aragog.callbacks.timing import TimingCallback
from aragog.pde_models.payoffs import g_minimum
from aragog.networks.factories import (
    create_variance_process_mlp,
    # create_variance_process_dgm_network,
    # create_variance_process_highway_network,
)
from aragog.schedules.piecewise import build_piecewise_decay_schedule
from scripts.hpc.utils import save_model, parse_args

LOGGER = logging.getLogger(__name__)


def runner(args):
    configure_logger()

    # Constants
    units = 75
    layers = 4
    steps_per_epoch = 20

    batch_size = 5000
    epochs = 1000

    K = 1.0
    T = 2.0
    dimension_x = 2
    t_range = [0 + 1e-10, T]
    x_range = [0 + 1e-10, 4.0]
    v_range = [0 + 1e-10, 0.3]
    correlations = np.array([[1, 0.5], [0.5, 1]])
    variance_correlations = np.array([0.25, 0.25])
    riskfree_rate = 0.0
    reversion_speed = 1.0
    long_average_variance = 0.04
    vol_of_vol = 0.05
    initial_variance = 0.04

    name = f"european_heston_{dimension_x}d_{units}n_{layers}l_mlp"
    save_path = os.path.join(args.save_path, name)

    learning_rate = build_piecewise_decay_schedule(epochs * steps_per_epoch)
    timing_cb = TimingCallback(save_path=save_path)

    generator = HestonSpaceTimeVarianceGenerator(
        batch_size=batch_size,
        dimension_x=dimension_x,
        t_range=t_range,
        x_range=x_range,
        v_range=v_range,
    )

    t, x, v, outputs = create_variance_process_mlp(
        dimension_x=dimension_x, units=units, layers=layers
    )

    model = EuropeanHestonPDEModel(
        dimension_x=dimension_x,
        K=K,
        correlations=correlations,
        variance_correlations=variance_correlations,
        riskfree_rate=riskfree_rate,
        reversion_speed=reversion_speed,
        long_average_variance=long_average_variance,
        vol_of_vol=vol_of_vol,
        initial_variance=initial_variance,
        g_terminal=g_minimum,
        inputs=[t, x, v],
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
