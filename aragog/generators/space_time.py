from typing import List
import tensorflow as tf
from aragog.generators.samplers import (
    sample_interior,
    sample_terminal,
    sample_boundary,
)


class SpaceTimeGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        batch_size: int,
        dimension_x: int,
        x_range: List[float] = [0.0, 1.0],
        t_range: List[float] = [0.0, 1.0],
        oversampling_multiplier: float = 1.0,
        include_boundary: bool = False,
    ):
        self.batch_size = tf.constant(batch_size, dtype=tf.int32)
        self.batch_size_terminal = tf.constant(batch_size, dtype=tf.int32)
        self.dimension_x = tf.constant(dimension_x, dtype=tf.int32)
        self.x_min = tf.constant(x_range[0], dtype=tf.float32)
        self.x_max = tf.constant(x_range[1], dtype=tf.float32)
        self.t_min = tf.constant(t_range[0], dtype=tf.float32)
        self.t_max = tf.constant(t_range[1], dtype=tf.float32)
        self.oversampling_multiplier = tf.constant(
            oversampling_multiplier, dtype=tf.float32
        )
        self.include_boundary = include_boundary

    def __len__(self):
        """Describes the number of points to create"""
        return self.batch_size

    def __getitem__(self, *args) -> List[tf.Tensor]:
        interior = sample_interior(
            self.batch_size,
            self.dimension_x,
            self.x_min,
            self.x_max * self.oversampling_multiplier,
            self.t_min,
            self.t_max,
        )

        terminal = sample_terminal(
            self.batch_size,
            self.dimension_x,
            self.x_min,
            self.x_max * self.oversampling_multiplier,
            self.t_max,
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
            return interior + terminal + boundary
        return interior + terminal

    @property
    def output_types(self) -> List[tf.TensorSpec]:
        return [tensor.dtype for tensor in self.__getitem__()]

    @property
    def output_shapes(self) -> List[tf.TensorShape]:
        return [tensor.shape for tensor in self.__getitem__()]
