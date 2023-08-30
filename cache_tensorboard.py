from configuration import config 
import os
from typing import List
import tensorflow as tf

def log_scalar(tb_writer: tf.summary.SummaryWriter,
               name: str,
               data: float,
               step: int) -> None:
    """Log a scalar value to tensorboard.
    
    Args:
        name: The tensorboard name.
        value: The value.
        step: The step number.
    """
    with tb_writer.as_default():
        tf.summary.scalar(name, data, step=step)

def log_hit_rates(tb_writter: tf.summary.SummaryWriter,
                  name: str,
                  hit_rates: List[float],
                  step: int) -> None:
    """Log the hit rates to tensorboard.
    
    Args:
        name: The tensorboard name.
        hit_rates: The list of hit rates. Assumed that hit_rates[i] is
          the cumulative hitr rate on the first i / len(hit_rates) 
          portion of the dataset.
        step: The step number.
    """
    for i, hit_rate in enumerate(hit_rates[:-1]):
        log_scalar(
            tb_writter,
            name + "_{:.2f}".format((i + 1) / len(hit_rates)),
            hit_rate,
            step
        )
    log_scalar(tb_writter, name, hit_rates[-1], step)