import tensorflow as tf
from aragog.pde_models.black_scholes.black_scholes import BlackScholesPDEModel


class AmericanBlackScholesPDEModel(BlackScholesPDEModel):
    def __init__(self, *args, **kwargs):
        super(AmericanBlackScholesPDEModel, self).__init__(*args, **kwargs)

    @tf.function
    def compute_boundary_loss(self, t: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        # noinspection PyCallingNonCallable
        v_boundary = self([t, x], training=True)[:, 0]

        loss_boundary = tf.reduce_mean(
            tf.math.square(
                tf.math.maximum(self.g_terminal(x, self.K) - v_boundary, 0)
            )
        )

        # In case there are no samples {(x, t): f(t, x; theta) <= g(x)}, we have a nan boundary.
        # This should be avoided because the total loss will be nan, which we cannot optimize
        loss_boundary = tf.where(
            tf.math.is_nan(loss_boundary),
            tf.constant(0, dtype=tf.float32),
            loss_boundary,
        )

        return loss_boundary

    def train_step(self, data):
        (
            x_interior,
            t_interior,
            x_terminal,
            t_terminal,
            x_boundary,
            t_boundary,
        ) = data[0]

        with tf.GradientTape() as tape:
            # Loss term #1: PDE
            # compute function value and derivatives at current sampled points

            # noinspection PyCallingNonCallable
            v_interior = self([t_interior, x_interior], training=False)[:, 0]

            # Keep only values {(x, t): f(t, x; theta) > g(x)}
            indices = tf.where(
                v_interior[:, 0] > self.g_terminal(x_interior, self.K)
            )[:, 0]
            x_interior = tf.gather(x_interior, indices)
            t_interior = tf.gather(t_interior, indices)

            loss_interior = self.compute_interior_loss(
                t=t_interior, x=x_interior
            )

            # Loss term #2: boundary condition
            # noinspection PyCallingNonCallable
            v_boundary = self([t_boundary, x_boundary], training=False)[:, 0]

            # Keep only {(x, t): f(t, x; theta) <= g(x)}
            indices = tf.where(
                v_boundary <= self.g_terminal(x_boundary, self.K)
            )[:, 0]
            x_boundary = tf.gather(x_boundary, indices)
            t_boundary = tf.gather(t_boundary, indices)

            loss_boundary = self.compute_boundary_loss(
                t=t_boundary, x=x_boundary
            )

            # Loss term #3: initial/terminal condition
            loss_terminal = self.compute_terminal_loss(
                t=t_terminal, x=x_terminal
            )

            loss = loss_interior + loss_terminal + loss_boundary

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )

        return {
            "loss_interior": loss_interior,
            "loss_boundary": loss_boundary,
            "loss_terminal": loss_terminal,
            "loss": loss,
        }
