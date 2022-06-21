import tensorflow as tf


def build_piecewise_decay_schedule(
    steps: int,
    initial_lr: float = 1e-3,
    increments: int = 5,
) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    step_increment = steps // increments
    boundaries = [i * step_increment for i in range(1, increments + 1)]

    dividers = [2, 10, 20, 100, 200, 1000, 2000, 10000, 20000][:increments]
    values = [initial_lr] + [initial_lr / divider for divider in dividers]

    return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values
    )
