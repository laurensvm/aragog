from typing import List
import tensorflow as tf
from aragog.pde_models.base import PDEModel


class BlackScholesPDEModel(PDEModel):
    """
    Abstract class for Black-Scholes PDE models
    """

    def __init__(
        self,
        volatilities: List[float],
        correlations: List[float],
        riskfree_rate: float,
        *args,
        **kwargs
    ):
        super(BlackScholesPDEModel, self).__init__(*args, **kwargs)
        self.volatilities = tf.constant(volatilities, dtype=tf.float32)
        self.correlations = tf.reshape(
            tf.constant(correlations, dtype=tf.float32),
            [self.dimension_x, self.dimension_x],
        )
        self.riskfree_rate = tf.constant(riskfree_rate, dtype=tf.float32)

    def train_step(self, data):
        raise NotImplementedError("This class cannot be used for training.")

    @tf.function
    def compute_interior_loss(self, t: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        # noinspection PyCallingNonCallable
        v_interior = self([t, x], training=True)[:, 0]

        v_t = tf.gradients(v_interior, t)[0]
        v_x = tf.gradients(v_interior, x)[0]

        """
        This is pure TensorFlow code. It should be used instead of the python loop below.
        The problem is that my notebook keeps crashing (probably a memory issue).
        Should look into this. Also uncomment @tf.function
        """
        second_grads = tf.transpose(
            tf.map_fn(
                lambda i: tf.gradients(v_x[:, i], x)[0],
                elems=tf.range(self.dimension_x),
                fn_output_signature=tf.float32,
            ),
            perm=[1, 0, 2],
        )

        # Keep this code as a reference to what happens above. Should document clearly
        # second_grads = tf.concat(
        #     [tf.gradients(v_x[:, i], x)[0] for i in range(self.dimension_x)],
        #     axis=1,
        # )
        # second_grads = tf.reshape(
        #     second_grads, (-1, self.dimension_x, self.dimension_x)
        # )

        # \rho_{ij} * \sigma_i * x_i * sigma_j * x_j
        # We obtain a matrix to multiply with the second derivatives which we sum over
        second_derivative_weight_matrix = (
            tf.einsum(
                "ij,ik->ijk",
                self.volatilities * x,
                self.volatilities * x,
            )
            * self.correlations
        )

        residual_interior = (
            v_t[:, 0]
            + tf.reduce_sum(self.riskfree_rate * x * v_x, axis=1)
            - self.riskfree_rate * v_interior
            + 0.5
            * tf.reduce_sum(
                second_derivative_weight_matrix * second_grads, axis=[1, 2]
            )
        )

        # compute average L2-norm of differential operator
        loss_interior = tf.reduce_mean(tf.square(residual_interior))
        return loss_interior

    @tf.function
    def compute_terminal_loss(self, t: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        payoff = self.g_terminal(x, self.K)

        # noinspection PyCallingNonCallable
        v_initial = self([t, x], training=True)[:, 0]
        loss_initial = tf.reduce_mean(tf.square(v_initial - payoff))
        return loss_initial

    def predict(self, t, x, *args, **kwargs):
        return super(BlackScholesPDEModel, self).predict(
            [t, x], *args, **kwargs
        )
