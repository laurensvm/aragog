from typing import List
import tensorflow as tf
from aragog.generators.samplers import (
    sample_riskfree_rates,
    sample_volatilities,
    sample_correlations,
)
from aragog.generators.space_time import SpaceTimeGenerator


class BlackScholesParametricSpaceTimeGenerator(SpaceTimeGenerator):
    def __init__(
        self,
        riskfree_rate_range: List[float],
        volatility_range: List[float],
        correlation_range: List[float],
        *args,
        **kwargs
    ):
        super(BlackScholesParametricSpaceTimeGenerator, self).__init__(
            *args, **kwargs
        )
        self.rfr_min = tf.constant(riskfree_rate_range[0], dtype=tf.float32)
        self.rfr_max = tf.constant(riskfree_rate_range[1], dtype=tf.float32)
        self.vol_min = tf.constant(volatility_range[0], dtype=tf.float32)
        self.vol_max = tf.constant(volatility_range[1], dtype=tf.float32)
        self.corr_min = tf.constant(correlation_range[0], dtype=tf.float32)
        self.corr_max = tf.constant(correlation_range[1], dtype=tf.float32)

    def __getitem__(self, *args) -> List[tf.Tensor]:
        space_time = super(
            BlackScholesParametricSpaceTimeGenerator, self
        ).__getitem__(*args)
        riskfree_rates = sample_riskfree_rates(
            self.batch_size, self.rfr_min, self.rfr_max
        )
        volatilities = sample_volatilities(
            self.batch_size, self.dimension_x, self.vol_min, self.vol_max
        )
        pairwise_correlations = sample_correlations(
            self.batch_size, self.dimension_x, self.corr_min, self.corr_max
        )
        return (
            space_time + riskfree_rates + volatilities + pairwise_correlations
        )
