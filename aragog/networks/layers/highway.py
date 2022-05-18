import tensorflow as tf


class GenericHighwayLayer(tf.keras.layers.Layer):
    """
    Highway Network as described in https://arxiv.org/pdf/1505.00387.pdf,
    but instead of taking C = 1 - T, we take C to be a nonlinear transformation independent of T
    """

    def __init__(
        self,
        units: int = 32,
        init_func=tf.keras.initializers.HeNormal(),
        transform_activation_func=tf.nn.tanh,
        activation_func=tf.nn.relu,
        transform_bias_value: float = -3.0,
        carry_bias_value: float = 3.0,
    ):
        super(GenericHighwayLayer, self).__init__()
        self.units = units
        self.init_func = init_func
        self.activation_func = activation_func
        self.transform_activation_func = transform_activation_func
        self.transform_bias_value = transform_bias_value
        self.carry_bias_value = carry_bias_value

    def build(self, input_shape):
        self.W_T = self.add_weight(
            name="WT",
            shape=(input_shape[-1], self.units),
            initializer=self.init_func,
            trainable=True,
        )
        self.b_T = self.add_weight(
            name="bT",
            shape=(self.units,),
            initializer=self.init_func,
            trainable=True,
        )

        self.W_C = self.add_weight(
            name="WC",
            shape=(input_shape[-1], self.units),
            initializer=self.init_func,
            trainable=True,
        )
        self.b_C = self.add_weight(
            name="bC",
            shape=(self.units,),
            initializer=self.init_func,
            trainable=True,
        )

        self.W = self.add_weight(
            name="W",
            shape=(input_shape[-1], self.units),
            initializer=self.init_func,
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(self.units,),
            initializer=self.init_func,
            trainable=True,
        )

        self.built = True

    def call(self, inputs, *args, **kwargs):
        x = inputs

        H = self.activation_func(tf.matmul(x, self.W) + self.b)
        T = self.transform_activation_func(tf.matmul(x, self.W_T) + self.b_T)
        C = self.transform_activation_func(tf.matmul(x, self.W_C) + self.b_C)

        # y = tf.add(tf.math.multiply(tf.subtract(1.0, T), H), tf.math.multiply(C, x))
        y = tf.add(tf.math.multiply(T, H), tf.math.multiply(C, x))
        return y


class HighwayLayer(GenericHighwayLayer):
    """
    Highway Network as described in https://arxiv.org/pdf/1505.00387.pdf, with C = 1 - T
    """

    def __init__(self, *args, **kwargs):
        super(HighwayLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.W_T = self.add_weight(
            name="WT",
            shape=(input_shape[-1], self.units),
            initializer=self.init_func,
            trainable=True,
        )
        self.b_T = self.add_weight(
            name="bT",
            shape=(self.units,),
            initializer=self.init_func,
            trainable=True,
        )

        self.W = self.add_weight(
            name="W",
            shape=(input_shape[-1], self.units),
            initializer=self.init_func,
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(self.units,),
            initializer=self.init_func,
            trainable=True,
        )

        self.built = True

    def call(self, inputs, *args, **kwargs):
        x = inputs

        H = self.activation_func(tf.matmul(x, self.W) + self.b)
        T = self.transform_activation_func(tf.matmul(x, self.W_T) + self.b_T)
        C = tf.subtract(1.0, T)

        y = tf.add(H * T, x * C)
        return y


class ResidualLayer(tf.keras.layers.Layer):
    """
    Simple Residual Network. This network is a special case of the
    Highway Network, where T = C = 1
    """

    def __init__(
        self,
        units: int = 32,
        init_func=tf.keras.initializers.HeNormal(),
        activation_func=tf.nn.tanh,
    ):
        super(ResidualLayer, self).__init__()
        self.units = units
        self.init_func = init_func
        self.activation_func = activation_func

    def build(self, input_shape):
        self.W = self.add_weight(
            name="W",
            shape=(input_shape[-1], self.units),
            initializer=self.init_func,
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(self.units,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        self.built = True

    def call(self, inputs, *args, **kwargs):
        x = inputs
        H = self.activation_func(tf.matmul(x, self.W) + self.b)
        y = tf.add(H, x)
        return y
