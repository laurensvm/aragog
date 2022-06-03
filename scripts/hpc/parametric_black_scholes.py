import os
import logging
import tensorflow as tf
from aragog.logger import configure_logger
from aragog.generators.parametric import (
    BlackScholesParametricSpaceTimeGenerator,
)
from aragog.pde_models.black_scholes.parametric_european import (
    ParametricEuropeanBlackScholesPDEModel,
)
from aragog.callbacks.timing import TimingCallback
from aragog.pde_models.payoffs import g_arithmetic
from aragog.networks.factories import create_parametric_dgm
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
    riskfree_rate_range = [0 + 1e-10, 0.1]
    volatility_range = [0.01, 0.2]
    correlation_range = [-0.5, 0.5]

    name = f"parametric_european_bs_{dimension_x}d_{units}n_{layers}l_hw"
    save_path = os.path.join(args.save_path, name)

    learning_rate = build_piecewise_decay_schedule(epochs * steps_per_epoch)
    timing_cb = TimingCallback(save_path=save_path)

    generator = BlackScholesParametricSpaceTimeGenerator(
        batch_size=batch_size,
        dimension_x=dimension_x,
        t_range=t_range,
        x_range=x_range,
        riskfree_rate_range=riskfree_rate_range,
        volatility_range=volatility_range,
        correlation_range=correlation_range,
    )

    t, x, params, outputs = create_parametric_dgm(
        dimension_x=dimension_x,
        dimension_params=2 * dimension_x,
        units=units,
        layers=layers,
    )

    model = ParametricEuropeanBlackScholesPDEModel(
        dimension_x=dimension_x,
        batch_size=batch_size,
        K=K,
        g_terminal=g_arithmetic,
        inputs=[t, x, params],
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
