import tensorflow as tf


@tf.function
def g_arithmetic(x: tf.Tensor, K: tf.constant) -> tf.Tensor:
    # g(x) = max(0, 1 / d * \sum_{i=1}^d x_i - K)
    return tf.math.maximum(tf.math.reduce_mean(x, axis=1) - K, 0)


@tf.function
def g_minimum(x: tf.Tensor, K: tf.constant) -> tf.Tensor:
    # g(x) = max(0, min(x_1, ..., x_k) - K)
    return tf.math.maximum(tf.math.reduce_min(x, axis=1) - K, 0)


@tf.function
def g_geometric(x: tf.Tensor, K: tf.constant) -> tf.Tensor:
    # g(x) = max(\prod_{i=1}^d x_i)^{\frac{1}{d}} - K)
    d_inv = tf.constant(1 / x.shape[1], dtype=tf.float32)
    return tf.math.maximum(
        tf.math.pow(tf.math.reduce_prod(x, axis=1), d_inv) - K, 0
    )
