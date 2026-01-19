import os

from loguru import logger

RANK = os.environ.get("RANK", None)


class fakeLogger:
    def debug(self, msg): ...
    def info(self, msg): ...
    def warning(self, msg): ...
    def error(self, msg): ...
    def critical(self, msg): ...


staxLogger = logger if RANK == "0" else fakeLogger()

__all__ = ["staxLogger"]
