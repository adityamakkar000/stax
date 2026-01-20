import os

from loguru import logger

RANK = os.environ.get("RANK", None)


class FakeLogger:
    def debug(self, msg, *args, **kwargs): ...
    def info(self, msg, *args, **kwargs): ...
    def warning(self, msg, *args, **kwargs): ...
    def error(self, msg, *args, **kwargs): ...
    def critical(self, msg, *args, **kwargs): ...


staxLogger = logger if RANK == "0" else FakeLogger()

__all__ = ["staxLogger"]
