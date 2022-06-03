from typing import List
import tensorflow as tf
from aragog.generators.space_time import SpaceTimeGenerator
from aragog.generators.samplers import (
    sample_interior,
    sample_terminal,
    sample_boundary,
    sample_variance,
)


class HestonSpaceTimeVarianceGenerator(SpaceTimeGenerator):
    def __init__(self, v_range: List[float], *args, **kwargs):
        super(HestonSpaceTimeVarianceGenerator, self).__init__(*args, **kwargs)
        self.v_min = tf.constant(v_range[0], dtype=tf.float32)
        self.v_max = tf.constant(v_range[1], dtype=tf.float32)

    def __getitem__(self, *args) -> List[tf.Tensor]:
        interior = sample_interior(
            self.batch_size,
            self.dimension_x,
            self.x_min,
            self.x_max * self.oversampling_multiplier,
            self.t_min,
            self.t_max,
        )

        variance = sample_variance(self.batch_size, self.v_min, self.v_max)

        terminal = sample_terminal(
            self.batch_size,
            self.dimension_x,
            self.x_min,
            self.x_max * self.oversampling_multiplier,
            self.t_max,
        )

        variance_terminal = sample_variance(
            self.batch_size, self.v_min, self.v_max
        )

        if self.include_boundary:
            boundary = sample_boundary(
                self.batch_size,
                self.dimension_x,
                self.x_min,
                self.x_max,
                self.t_min,
                self.t_max,
            )

            variance_boundary = sample_variance(
                self.batch_size, self.v_min, self.v_max
            )

            return (
                interior
                + variance
                + terminal
                + variance_terminal
                + boundary
                + variance_boundary
            )
        return interior + variance + terminal + variance_terminal
