from typing import List, Union
import numpy as np
import tensorflow as tf
from aragog.pde_models.base import PDEModel


class HestonPDEModel(PDEModel):
    def __init__(
        self,
        correlations: Union[List[float], np.ndarray],
        variance_correlations: Union[List[float], np.ndarray],
        riskfree_rate: float,
        reversion_speed: float,
        long_average_variance: float,
        vol_of_vol: float,
        initial_variance: float,
        *args,
        **kwargs
    ):
        super(HestonPDEModel, self).__init__(*args, **kwargs)
        self.correlations = tf.reshape(
            tf.constant(correlations, dtype=tf.float32),
            [self.dimension_x, self.dimension_x],
        )
        self.variance_correlations = tf.constant(
            variance_correlations, dtype=tf.float32
        )
        self.reversion_speed = tf.constant(reversion_speed, dtype=tf.float32)
        self.long_average_variance = tf.constant(
            long_average_variance, dtype=tf.float32
        )
        self.vol_of_vol = tf.constant(vol_of_vol, dtype=tf.float32)
        self.initial_variance = tf.constant(initial_variance, dtype=tf.float32)
        self.riskfree_rate = tf.constant(riskfree_rate, dtype=tf.float32)

    @tf.function
    def compute_interior_loss(
        self, t: tf.Tensor, x: tf.Tensor, v: tf.Tensor
    ) -> tf.Tensor:
        u_interior = self([t, x, v], training=True)[:, 0]

        u_t = tf.gradients(u_interior, t)[0]
        u_x = tf.gradients(u_interior, x)[0]
        u_v = tf.gradients(u_interior, v)[0]

        u_xx = tf.transpose(
            tf.map_fn(
                lambda i: tf.gradients(u_x[:, i], x)[0],
                elems=tf.range(self.dimension_x),
                fn_output_signature=tf.float32,
            ),
            perm=[1, 0, 2],
        )

        u_xv = tf.squeeze(
            tf.transpose(
                tf.map_fn(
                    lambda i: tf.gradients(u_x[:, i], v)[0],
                    elems=tf.range(self.dimension_x),
                    fn_output_signature=tf.float32,
                ),
                perm=[1, 0, 2],
            )
        )

        u_vv = tf.gradients(u_v, v)[0]

        second_derivative_weight_matrix = (
            tf.einsum("ij,ik->ijk", x, x) * self.correlations
        )

        residual_interior = (
            u_t[:, 0]
            + tf.reduce_sum(self.riskfree_rate * x * u_x, axis=1)
            + self.reversion_speed
            * (self.long_average_variance - v[:, 0])
            * u_v[:, 0]
            + 0.5
            * v[:, 0]
            * tf.reduce_sum(
                second_derivative_weight_matrix * u_xx, axis=[1, 2]
            )
            + self.vol_of_vol
            * v[:, 0]
            * tf.reduce_sum(u_xv * x * self.variance_correlations, axis=1)
            + 0.5 * tf.square(self.vol_of_vol) * v[:, 0] * u_vv[:, 0]
            - self.riskfree_rate * u_interior
        )

        loss_interior = tf.reduce_mean(tf.square(residual_interior))
        return loss_interior

    @tf.function
    def compute_terminal_loss(
        self, t: tf.Tensor, x: tf.Tensor, v: tf.Tensor
    ) -> tf.Tensor:
        payoff = self.g_terminal(x, self.K)

        v_initial = self([t, x, v], training=True)[:, 0]
        loss_initial = tf.reduce_mean(tf.square(v_initial - payoff))
        return loss_initial

    def predict(self, t, x, v, *args, **kwargs):
        return super(HestonPDEModel, self).predict([t, x, v], *args, **kwargs)

    def train_step(self, data):
        raise NotImplementedError("This class cannot be used for training.")
