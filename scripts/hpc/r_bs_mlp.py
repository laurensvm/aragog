"""
This Python script trains multiple simple MLP's with the same configuration
on the Black-Scholes and Implied Volatility datasets. It should be run with Slurm
as a batch job on DHPC.
"""


import logging
from aragog.logger import configure_logger

LOGGER = logging.getLogger(__name__)


def runner():
    configure_logger()


if __name__ == "__main__":
    runner()
