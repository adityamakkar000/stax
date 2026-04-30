import jax
from loguru import logger

if jax.process_index() != 0:
    logger.remove()

staxLogger = logger

__all__ = ["staxLogger"]
