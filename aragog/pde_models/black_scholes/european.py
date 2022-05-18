import tensorflow as tf
from aragog.pde_models.black_scholes.black_scholes import BlackScholesPDEModel


class EuropeanBlackScholesPDEModel(BlackScholesPDEModel):
    def __init__(self, *args, **kwargs):
        super(EuropeanBlackScholesPDEModel, self).__init__(*args, **kwargs)

    def train_step(self, data):
        x_interior, t_interior, x_terminal, t_terminal = data[0]
        with tf.GradientTape() as tape:
            # Loss term #1: PDE
            # compute function value and derivatives at current sampled points
            loss_interior = self.compute_interior_loss(
                t=t_interior, x=x_interior
            )

            # Loss term #2: boundary condition
            # no boundary condition for this problem

            # Loss term #3: initial/terminal condition
            loss_terminal = self.compute_terminal_loss(
                t=t_terminal, x=x_terminal
            )

            loss = loss_interior + loss_terminal

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )

        return {
            "loss_interior": loss_interior,
            "loss_terminal": loss_terminal,
            "loss": loss,
        }
