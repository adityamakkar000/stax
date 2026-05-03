import jax
from absl import logging


class Logger:

    def info(self, *args, log_for_all=False):
        if log_for_all or self.should_log:
            logging.info(*args, stacklevel=2)
    
    def warning(self, *args, log_for_all=False):
        if log_for_all or self.should_log:
            logging.warning(*args, stacklevel=2)

    @property
    def should_log(self):
        return jax.process_index() == 0

staxLogger = Logger()

__all__ = ["staxLogger"]
