from typing import List
import tensorflow as tf


@tf.function
def sample_interior(
    batch_size: tf.int32,
    dimension_x: tf.int32,
    x_min: tf.float32,
    x_max: tf.float32,
    t_min: tf.float32,
    t_max: tf.float32,
) -> List[tf.Tensor]:
    # Sample from the interior of [t,x]. This is sampled as [t_min, t_max), [x_min, x_max)
    x_interior_normalized = tf.random.uniform(
        [batch_size, dimension_x],
        minval=x_min,
        maxval=x_max,
        dtype=tf.float32,
    )
    t_interior_normalized = tf.random.uniform(
        [batch_size, 1], minval=t_min, maxval=t_max, dtype=tf.float32
    )
    return [x_interior_normalized, t_interior_normalized]


@tf.function
def sample_terminal(
    batch_size: tf.int32,
    dimension_x: tf.int32,
    x_min: tf.float32,
    x_max: tf.float32,
    t_max: tf.float32,
) -> List[tf.Tensor]:
    # Sample for terminal condition. E.g. t = T
    x_terminal_normalized = tf.random.uniform(
        [batch_size, dimension_x],
        minval=x_min,
        maxval=x_max,
        dtype=tf.float32,
    )
    t_terminal_normalized = t_max * tf.ones([batch_size, 1], dtype=tf.float32)
    return [x_terminal_normalized, t_terminal_normalized]


@tf.function
def sample_boundary(
    batch_size: tf.int32,
    dimension_x: tf.int32,
    x_min: tf.float32,
    x_max: tf.float32,
    t_min: tf.float32,
    t_max: tf.float32,
) -> List[tf.Tensor]:
    # Sample from the boundary of x.
    # To do this, we randomly fix x_i and sample x = (X_1, X_2, ..., x_i, ..., X_n)
    x_boundary_normalized = tf.random.uniform(
        [batch_size, dimension_x],
        minval=x_min,
        maxval=x_max,
        dtype=tf.float32,
    )
    t_boundary_normalized = tf.random.uniform(
        [batch_size, 1], minval=t_min, maxval=t_max, dtype=tf.float32
    )

    # Step 1 - Randomly sample i for each x_i
    boundary_dimension_idx = tf.random.uniform(
        shape=[batch_size, 1], minval=0, maxval=dimension_x, dtype=tf.int32
    )

    # Step 2 - Create multi index to access the values that must be changed to the boundary values
    multi_idx = tf.stack(
        [
            tf.range(start=0, limit=batch_size, delta=1, dtype=tf.int32),
            tf.squeeze(boundary_dimension_idx),
        ],
        axis=1,
    )

    # Step 3 - Randomly select whether to take the lower boundary or the upper boundary
    boundary_values = tf.map_fn(
        lambda x: (1 - x) * x_min + x * x_max,
        tf.cast(
            tf.random.uniform(
                shape=[batch_size], minval=0, maxval=2, dtype=tf.int32
            ),
            dtype=tf.float32,
        ),
        fn_output_signature=tf.float32,
    )

    # Step 4 - Change the sampled points to the boundary points at the 'boundary_dimension_idx'
    x_boundary_normalized = tf.tensor_scatter_nd_update(
        x_boundary_normalized, multi_idx, boundary_values
    )

    return [x_boundary_normalized, t_boundary_normalized]


@tf.function
def sample_volatilities(
    batch_size: tf.int32,
    dimension_x: tf.int32,
    vol_min: tf.float32 = 0.0,
    vol_max: tf.float32 = 1.0,
) -> List[tf.Tensor]:
    return [
        tf.random.uniform(
            [batch_size, dimension_x], minval=vol_min, maxval=vol_max
        )
    ]


@tf.function
def sample_riskfree_rates(
    batch_size: tf.int32, rfr_min: tf.float32 = 0.0, rfr_max: tf.float32 = 1.0
) -> List[tf.Tensor]:
    return [tf.random.uniform([batch_size, 1], minval=rfr_min, maxval=rfr_max)]


@tf.function
def sample_correlation_matrix(
    batch_size: tf.int32,
    dimension_x: tf.int32,
    corr_min: tf.float32 = 0.0,
    corr_max: tf.float32 = 1.0,
) -> List[tf.Tensor]:
    correlation_matrix = tf.random.uniform(
        shape=[dimension_x, dimension_x],
        minval=corr_min,
        maxval=corr_max,
        dtype=tf.float32,
    )
    # Set diagonal to 1
    correlation_matrix = tf.linalg.set_diag(
        correlation_matrix, tf.ones(shape=[dimension_x])
    )

    # Transform correlation matrix to upper triangular matrix
    correlation_matrix = tf.linalg.band_part(correlation_matrix, 0, -1)

    # Transpose and divide by two
    correlation_matrix = 0.5 * (
        correlation_matrix + tf.transpose(correlation_matrix)
    )

    # Expand dimension and tile this correlation matrix * batch_size
    return [tf.tile(tf.expand_dims(correlation_matrix, 0), [batch_size, 1, 1])]


@tf.function
def sample_correlations(
    batch_size: tf.int32,
    dimension_x: tf.int32,
    corr_min: tf.float32 = 0.0,
    corr_max: tf.float32 = 1.0,
) -> List[tf.Tensor]:
    """
    Samples independent pairwise correlations
    """
    correlations = tf.random.uniform(
        shape=[batch_size, dimension_x - 1],
        minval=corr_min,
        maxval=corr_max,
        dtype=tf.float32,
    )
    return [correlations]
