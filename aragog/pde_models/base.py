from typing import Optional
import tensorflow as tf


class PDEModel(tf.keras.Model):
    def __init__(
        self,
        g_terminal: tf.function,
        g_boundary: Optional[tf.function] = None,
        *args,
        **kwargs
    ):
        super(PDEModel, self).__init__(*args, **kwargs)
        self.g_terminal = g_terminal
        self.g_boundary = g_boundary
        self.include_boundary = g_boundary is not None


class ParametricBlackScholesPDEModel(PDEModel):
    def __init__(self, dimension_x: int, K: float, *args, **kwargs):
        super(ParametricBlackScholesPDEModel, self).__init__(*args, **kwargs)
        self.K = tf.constant(K, dtype=tf.float32)
        self.dimension_x = tf.constant(dimension_x, dtype=tf.int32)

    def train_step(self, data):
        raise NotImplementedError("This class cannot be used for training.")
