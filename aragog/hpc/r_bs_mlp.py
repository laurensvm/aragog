import logging
from aragog.logger import configure_logger

LOGGER = logging.getLogger(__name__)


def runner():
    configure_logger()


if __name__ == "__main__":
    runner()
