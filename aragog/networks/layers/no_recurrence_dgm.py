import tensorflow as tf
from aragog.networks.layers.dgm import DGMLayer


class NoRecurrenceDGMLayer(DGMLayer):
    def __init__(self, *args, **kwargs):
        super(NoRecurrenceDGMLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        # Input shape is a pair of (x, S). We extract s and then get the dimension of x

        # DGM-Layer weights
        # Z
        self.WZ = self.add_weight(
            name="WZ",
            shape=(self.units, self.units),
            initializer=self.init_func,
            trainable=True,
        )
        self.bZ = self.add_weight(
            name="bZ",
            shape=(self.units,),
            initializer=self.init_func,
            trainable=True,
        )
        # G
        self.WG = self.add_weight(
            name="WG",
            shape=(self.units, self.units),
            initializer=self.init_func,
            trainable=True,
        )
        self.bG = self.add_weight(
            name="bG",
            shape=(self.units,),
            initializer=self.init_func,
            trainable=True,
        )
        # R
        self.WR = self.add_weight(
            name="WR",
            shape=(self.units, self.units),
            initializer=self.init_func,
            trainable=True,
        )
        self.bR = self.add_weight(
            name="bR",
            shape=(self.units,),
            initializer=self.init_func,
            trainable=True,
        )
        # H
        self.WH = self.add_weight(
            name="WH",
            shape=(self.units, self.units),
            initializer=self.init_func,
            trainable=True,
        )
        self.bH = self.add_weight(
            name="bH",
            shape=(self.units,),
            initializer=self.init_func,
            trainable=True,
        )
        self.built = True

    def call(self, inputs, *args, **kwargs):
        _, S = inputs

        # calculate Z
        Z = tf.matmul(S, self.WZ) + self.bZ
        Z = self.activation_func(Z)

        # calculate G
        G = tf.matmul(S, self.WG) + self.bG
        G = self.activation_func(G)

        # calculate R
        R = tf.matmul(S, self.WR) + self.bR
        R = self.activation_func(R)

        # calculate H
        H = tf.matmul(tf.math.multiply(S, R), self.WH) + self.bH
        H = self.activation_func(H)

        # merge
        y = tf.math.multiply(tf.ones(tf.shape(G)) - G, H) + tf.math.multiply(
            Z, S
        )

        return y
