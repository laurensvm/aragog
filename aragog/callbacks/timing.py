import logging
import os
from typing import Dict, Optional
from timeit import default_timer

import pandas as pd
import tensorflow as tf


LOGGER = logging.getLogger(__name__)


class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.history: Dict[int, float] = {}
        self.epoch_start_time: Optional[float] = None
        self.start_training_time: Optional[float] = None
        self.end_training_time: Optional[float] = None
        self.training_time: Optional[float] = None

    def _save(self):
        os.makedirs(self.save_path, exist_ok=True)
        csv_path = os.path.join(self.save_path, "timings.csv")
        df = pd.DataFrame({"epoch_time": self.history})

        LOGGER.info(f"Saving Timings in {csv_path}")
        df.to_csv(csv_path)

    def on_train_begin(self, logs=None):
        self.start_training_time = default_timer()

    def on_train_end(self, logs=None):
        self.end_training_time = default_timer()
        self.training_time = self.end_training_time - self.start_training_time
        LOGGER.info(f"Training Time: {self.training_time} sec")
        self._save()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = default_timer()

    def on_epoch_end(self, epoch, logs=None):
        self.history[epoch] = default_timer() - self.epoch_start_time
        LOGGER.info(f"Epoch Training Time: {self.history[epoch]}")
