import tensorflow as tf
from aragog.networks.layers.dgm import DGMLayer


class DeepDGMLayer(DGMLayer):
    def __init__(self, depth: int, *args, **kwargs):
        super(DeepDGMLayer, self).__init__(*args, **kwargs)
        self.sub_layers = [
            HighwayDGMSubLayer(
                init_func=self.init_func,
                activation_func=self.activation_func,
                units=self.units,
            )
            for _ in range(depth)
        ]

    def build(self, input_shape):
        # Input shape is a pair of (x, S). We extract S and then get the dimension of x

        # DGM-Layer weights
        # Z
        self.UZ = self.add_weight(
            name="UZ",
            shape=(input_shape[0][-1], self.units),
            initializer=self.init_func,
            trainable=True,
        )
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
        self.UG = self.add_weight(
            name="UG",
            shape=(input_shape[0][-1], self.units),
            initializer=self.init_func,
            trainable=True,
        )
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
        self.UR = self.add_weight(
            name="UR",
            shape=(input_shape[0][-1], self.units),
            initializer=self.init_func,
            trainable=True,
        )
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

        self.built = True

    def call(self, inputs, *args, **kwargs):
        x, S = inputs

        # calculate Z
        Z = tf.matmul(x, self.UZ) + tf.matmul(S, self.WZ) + self.bZ
        Z = self.activation_func(Z)

        # calculate G
        G = tf.matmul(x, self.UG) + tf.matmul(S, self.WG) + self.bG
        G = self.activation_func(G)

        # calculate R
        R = tf.matmul(x, self.UR) + tf.matmul(S, self.WR) + self.bR
        R = self.activation_func(R)

        # calculate H
        for sub_layer in self.sub_layers:
            R = sub_layer((x, S, R))

        # merge
        y = tf.math.multiply(tf.ones(tf.shape(G)) - G, R) + tf.math.multiply(
            Z, S
        )

        return y


class HighwayDGMSubLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        init_func: tf.keras.initializers.Initializer,
        activation_func: tf.keras.activations,
        units: int,
        *args,
        **kwargs
    ):
        super(HighwayDGMSubLayer, self).__init__(*args, **kwargs)
        self.init_func = init_func
        self.activation_func = activation_func
        self.units = units

    def build(self, input_shape):
        # H
        self.UH = self.add_weight(
            name="UH",
            shape=(input_shape[0][-1], self.units),
            initializer=self.init_func,
            trainable=True,
        )
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

    def call(self, inputs):
        x, S, R = inputs
        H = (
            tf.matmul(x, self.UH)
            + tf.matmul(tf.math.multiply(S, R), self.WH)
            + self.bH
        )
        H = self.activation_func(H)
        return H
