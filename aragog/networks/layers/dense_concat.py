import tensorflow as tf

"""
Due to errors when backpropagating gradients, we cannot simply write

def call(self, *args):
    inputs = tf.concat([*args], axis=1)
    return self.activation(tf.matmul(inputs, self.W1) + self.b1)

Hence, we need to create a DenseConcat class for each input scenario
"""


class DenseConcat(tf.keras.layers.Layer):
    def __init__(self, units, activation=tf.nn.relu, *args, **kwargs):
        super(DenseConcat, self).__init__(*args, **kwargs)
        self.activation = activation
        self.units = units

    def build(self, input_shape):
        self.W1 = self.add_weight(
            name="W1",
            shape=(input_shape[-1], self.units),
            initializer="glorot_normal",
            trainable=True,
        )
        self.b1 = self.add_weight(
            name="b1",
            shape=(self.units,),
            initializer="glorot_normal",
            trainable=True,
        )
        self.built = True

    def call(self, *args):
        raise NotImplementedError(
            "Call a subclass with the appropriate amount of inputs."
        )


class DenseConcatTwoInputs(DenseConcat):
    def call(self, t, x):
        inputs = tf.concat([t, x], axis=1)
        return self.activation(tf.matmul(inputs, self.W1) + self.b1)


class DenseConcatThreeInputs(DenseConcat):
    def call(self, t, x, v):
        inputs = tf.concat([t, x, v], axis=1)
        return self.activation(tf.matmul(inputs, self.W1) + self.b1)
