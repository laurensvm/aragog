import tensorflow as tf
from aragog.networks.layers.dgm import DGMLayer


class DGM(tf.keras.layers.Layer):
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
        super(DGM, self).__init__(*args, **kwargs)
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


class DGMSpaceTime(tf.keras.layers.Layer):
    def __init__(
        self,
        #                  input_shape: int,
        units: int = 32,
        n_layers: int = 4,
        init_func: tf.keras.initializers.Initializer = tf.keras.initializers.GlorotNormal(),
        activation_func: tf.keras.activations = tf.keras.activations.tanh,
        layer_instance: tf.keras.layers.Layer = DGMLayer,
        depth: int = 1,
        *args,
        **kwargs
    ):
        super(DGMSpaceTime, self).__init__(*args, **kwargs)
        self.units = units
        self.init_func = init_func
        self.activation_func = activation_func
        self.dgm_layers = [
            layer_instance(
                units=units,
                init_func=self.init_func,
                activation_func=self.activation_func,
                depth=depth,
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

    def call(self, t, x, *args):
        inputs = tf.concat([t, x, *args], axis=1)

        S = self.activation_func(tf.matmul(inputs, self.W1) + self.b1)
        for layer in self.dgm_layers:
            S = layer((inputs, S))

        y = self.activation_func(tf.matmul(S, self.W) + self.b)

        # Final dense layer
        y = self.dense_final(y)
        return y


class DGMParametric(DGMSpaceTime):
    def __init__(self, *args, **kwargs):
        super(DGMParametric, self).__init__(*args, **kwargs)

    def call(self, t, x, params):
        inputs = tf.concat([t, x, params], axis=1)

        S = self.activation_func(tf.matmul(inputs, self.W1) + self.b1)
        for layer in self.dgm_layers:
            S = layer((inputs, S))

        y = self.activation_func(tf.matmul(S, self.W) + self.b)

        # Final dense layer
        y = self.dense_final(y)
        return y
