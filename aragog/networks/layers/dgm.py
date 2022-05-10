import tensorflow as tf


class DGMLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        units: int = 32,
        init_func: tf.keras.initializers.Initializer = tf.keras.initializers.GlorotNormal(),
        activation_func: tf.keras.activations = tf.keras.activations.tanh,
        *args,
        **kwargs
    ):
        super(DGMLayer, self).__init__(*args, **kwargs)
        self.units = units
        # The DGM paper uses Xavier Initialization
        self.init_func = init_func
        # The DGM paper uses a tanh activation function
        self.activation_func = activation_func

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
        H = (
            tf.matmul(x, self.UH)
            + tf.matmul(tf.math.multiply(S, R), self.WH)
            + self.bH
        )
        H = self.activation_func(H)

        # merge
        y = tf.math.multiply(tf.ones(tf.shape(G)) - G, H) + tf.math.multiply(
            Z, S
        )

        return y
