from typing import Optional, Union
import tensorflow as tf
import numpy as np


class PDEModel(tf.keras.Model):
    def __init__(
        self,
        K: Union[float, np.float64],
        dimension_x: Union[int, np.int64],
        g_terminal: tf.function,
        g_boundary: Optional[tf.function] = None,
        *args,
        **kwargs
    ):
        super(PDEModel, self).__init__(*args, **kwargs)
        self.K = tf.constant(K, dtype=tf.float32)
        self.dimension_x = tf.constant(dimension_x, dtype=tf.int32)
        self.g_terminal = g_terminal
        self.g_boundary = g_boundary
        self.include_boundary = g_boundary is not None

    def train_step(self, data):
        raise NotImplementedError("This class cannot be used for training.")
