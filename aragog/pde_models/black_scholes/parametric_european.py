import tensorflow as tf
from aragog.pde_models.base import ParametricBlackScholesPDEModel


class ParametricEuropeanBlackScholesPDEModel(ParametricBlackScholesPDEModel):
    def __init__(self, batch_size: int, *args, **kwargs):
        super(ParametricEuropeanBlackScholesPDEModel, self).__init__(
            *args, **kwargs
        )
        self.batch_size = tf.constant(batch_size, dtype=tf.int32)

    @tf.function
    def to_correlation_matrix(
        self, pairwise_correlations: tf.Tensor
    ) -> tf.Tensor:
        return tf.reshape(
            tf.concat(
                [
                    tf.ones(shape=(self.batch_size, 1)),
                    pairwise_correlations,
                    pairwise_correlations,
                    tf.ones(shape=(self.batch_size, 1)),
                ],
                axis=1,
            ),
            [-1, self.dimension_x, self.dimension_x],
        )

    @tf.function
    def train_step(self, data):
        (
            x_interior,
            t_interior,
            x_terminal,
            t_terminal,
            rfr,
            volas,
            pairwise_corrs,
        ) = data[0]
        parameters = tf.concat([rfr, volas, pairwise_corrs], axis=1)

        with tf.GradientTape() as tape:
            # Interior loss
            v_interior = self(
                [t_interior, x_interior, parameters], training=True
            )[:, 0]

            v_t = tf.gradients(v_interior, t_interior)[0]
            v_x = tf.gradients(v_interior, x_interior)[0]

            second_grads = tf.transpose(
                tf.map_fn(
                    lambda i: tf.gradients(v_x[:, i], x_interior)[0],
                    elems=tf.range(self.dimension_x),
                    fn_output_signature=tf.float32,
                ),
                perm=[1, 0, 2],
            )

            # \rho_{ij} * \sigma_i * x_i * sigma_j * x_j
            # We obtain a matrix to multiply with the second derivatives which we sum over
            second_derivative_weight_matrix = tf.einsum(
                "ij,ik->ijk",
                volas * x_interior,
                volas * x_interior,
            ) * self.to_correlation_matrix(pairwise_corrs)

            residual_interior = (
                v_t[:, 0]
                + tf.reduce_sum(rfr * x_interior * v_x, axis=1)
                - rfr[:, 0] * v_interior
                + 0.5
                * tf.reduce_sum(
                    second_derivative_weight_matrix * second_grads, axis=[1, 2]
                )
            )

            loss_interior = tf.reduce_mean(tf.square(residual_interior))

            # Loss terminal
            payoff = self.g_terminal(x_terminal, self.K)
            v_terminal = self(
                [t_terminal, x_terminal, parameters], training=True
            )[:, 0]
            loss_terminal = tf.reduce_mean(tf.square(v_terminal - payoff))

            loss = loss_interior + loss_terminal

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )

        return {
            "loss": loss,
            "loss_interior": loss_interior,
            "loss_terminal": loss_terminal,
        }
