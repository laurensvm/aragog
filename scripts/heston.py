import tensorflow as tf
import numpy as np
import logging
from aragog.generators.variance import HestonSpaceTimeVarianceGenerator
from aragog.pde_models.heston.european import EuropeanHestonPDEModel
from aragog.pde_models.payoffs import g_minimum
from aragog.networks.factories import create_variance_process_dgm_network
from scripts.hpc.utils import save_model


LOGGER = logging.getLogger(__name__)


def runner():
    # Constants
    units = 50
    layers = 3
    learning_rate = 1e-4
    steps_per_epoch = 40

    batch_size = 5000
    epochs = 500
    K = 100.0
    T = 1.0
    dimension_x = 2
    t_range = [0 + 1e-10, T]
    x_range = [85.0, 115.0]
    v_range = [0 + 1e-10, 0.2]
    correlations = np.array([[1, 0.5], [0.5, 1]])
    variance_correlations = np.array([0.25, 0.25])
    riskfree_rate = 0.0
    reversion_speed = 1.0
    long_average_variance = 0.04
    vol_of_vol = 0.05
    initial_variance = 0.04

    save_path = "./output/heston_2d_dgm"

    generator = HestonSpaceTimeVarianceGenerator(
        batch_size=batch_size,
        dimension_x=dimension_x,
        t_range=t_range,
        x_range=x_range,
        v_range=v_range,
    )

    t, x, v, outputs = create_variance_process_dgm_network(
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
    history = model.fit(
        x=generator, epochs=epochs, steps_per_epoch=steps_per_epoch
    )
    save_model(model, save_path, history)


if __name__ == "__main__":
    runner()
