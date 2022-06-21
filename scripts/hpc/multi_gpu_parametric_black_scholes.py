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
from aragog.networks.factories import (
    create_parametric_dgm,
    # create_parametric_highway_network,
    # create_parametric_mlp,
)
from aragog.schedules.piecewise import build_piecewise_decay_schedule
from scripts.hpc.utils import save_model, parse_args

LOGGER = logging.getLogger(__name__)


def runner(args):
    configure_logger()

    strategy = tf.distribute.MirroredStrategy()
    LOGGER.info(f"Number of GPUs detected: {strategy.num_replicas_in_sync}")

    # Constants
    units = 75
    layers = 3
    steps_per_epoch = 50

    batch_size = 5000
    global_batch_size = batch_size * strategy.num_replicas_in_sync
    LOGGER.info(f"Global Batch Size: {global_batch_size}")
    epochs = 10000
    LOGGER.info(f"Number of Epochs: {epochs}")

    K = 1.0
    T = 2.0
    dimension_x = 2
    t_range = [0 + 1e-10, T]
    x_range = [0 + 1e-10, 3.0]
    riskfree_rate_range = [0 + 1e-10, 0.2]
    volatility_range = [0.01, 0.4]
    correlation_range = [-0.7, 0.7]

    name = f"par_european_bs_{dimension_x}d_{units}n_{layers}l_dgm_arith_{epochs}e_mw"
    save_path = os.path.join(args.save_path, name)

    learning_rate = build_piecewise_decay_schedule(
        epochs * steps_per_epoch, increments=8
    )
    timing_cb = TimingCallback(save_path=save_path)

    generator = BlackScholesParametricSpaceTimeGenerator(
        batch_size=global_batch_size,
        dimension_x=dimension_x,
        t_range=t_range,
        x_range=x_range,
        riskfree_rate_range=riskfree_rate_range,
        volatility_range=volatility_range,
        correlation_range=correlation_range,
    )

    # dataset = tf.data.Dataset.from_generator(
    #     generator=generator,
    #     output_types=tuple(generator.output_types),
    #     output_shapes=tuple(generator.output_shapes)
    # )

    with strategy.scope():
        t, x, params, outputs = create_parametric_dgm(
            dimension_x=dimension_x,
            dimension_params=2 * dimension_x,
            units=units,
            layers=layers,
        )

        model = ParametricEuropeanBlackScholesPDEModel(
            dimension_x=dimension_x,
            batch_size=global_batch_size,
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
