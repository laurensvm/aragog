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
        self.init_func = init_func
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


class DGMWrapper(tf.keras.layers.Layer):
    def __init__(
        self,
        units: int = 32,
        n_layers: int = 4,
        init_func: tf.keras.initializers.Initializer = tf.keras.initializers.GlorotNormal(),
        activation_func: tf.keras.activations = tf.keras.activations.tanh,
        layer_instance: tf.keras.layers.Layer = DGMLayer,
        depth: int = 1,
        *args,
        **kwargs
    ):
        super(DGMWrapper, self).__init__(*args, **kwargs)
        self.units = units
        self.init_func = init_func
        self.activation_func = activation_func
        self.dgm_layers = [
            layer_instance(
                units=units,
                init_func=self.init_func,
                activation_func=self.activation_func,
                # depth=depth,
            )
            for _ in range(n_layers)
        ]
        self.dense_final = tf.keras.layers.Dense(1)

    def build(self, input_shape):
        self.W1 = self.add_weight(
            name="W1",
            shape=(input_shape[-1], self.units),
            initializer=self.init_func,
            trainable=True,
        )
        self.b1 = self.add_weight(
            name="b1",
            shape=(self.units,),
            initializer=self.init_func,
            trainable=True,
        )

        self.W = self.add_weight(
            name="W",
            shape=(self.units, input_shape[-1]),
            initializer=self.init_func,
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(input_shape[-1],),
            initializer=self.init_func,
            trainable=True,
        )

        self.built = True

    def call(self, inputs):
        x = inputs

        S = self.activation_func(tf.matmul(x, self.W1) + self.b1)
        for layer in self.dgm_layers:
            S = layer((x, S))

        y = self.activation_func(tf.matmul(S, self.W) + self.b)

        # Final dense layer
        y = self.dense_final(y)
        return y
