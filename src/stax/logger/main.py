import jax
import wandb
from typing import Any, Mapping, Optional
from jaxtyping import Array 
import os 

from loguru import logger
import abc
from torch.utils.tensorboard import SummaryWriter

class BaseLogger(abc.ABC):

    def __init__(self):
        self.metrics = []

    @abc.abstractmethod
    def async_log(self, step: int, data):
        raise NotImplementedError("base class ")
    
    @abc.abstractmethod
    def finish(self):
        raise NotImplementedError("base class ")
    
    def __call__(self, step: int, data: dict[str, any]): 
        self.metrics.append(
            {'step': step, 'data': data}
        )

    #TODO: find some way to log a string 
    # like maybe take a template function in? 
    # that way we can print strings as well    
    def flush(self): 
        for metric in self.metrics: 
            self.async_log(**metric)
        self.metrics = []


class WandBLogger(BaseLogger):
    def __init__(self, entity: str, project: str, config: Optional[Mapping[str, Any]] = None, run_id: Optional[str] = None):
        super().__init__()
        assert (config is not None) or (run_id is not None), "Either config or run_id must be provided"
        init_args = {
            'entity': entity,
            'project': project,
            'resume': "allow"
        }
        if run_id is not None: 
            init_args['id'] = run_id
            init_args['resume'] = "must"
        else: 
            init_args["config"] = config

        self._run = wandb.init(
            **init_args,
        )

        logger.info(
            f"Initialized WandB Logger with run id {self.id}"
        )

    def async_log(self, step: int, data: dict[str, Any]):

        data = jax.tree.map(
            lambda x: x.item() if isinstance(x, Array) else x, 
            data
        )
        self._run.log(
            data,
            step=step
        )

    def finish(self) -> None:
        self._run.finish()

    @property
    def id(self) -> Optional[str]: 
        return self._run.id


class TensorboardLogger(BaseLogger):

    def __init__(self, name: str, *, log_dir: str = "./tensorboard_logs"):
        super().__init__()

        log_dir = os.path.join(log_dir, name) 
        self.writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"Initialized Tensorboard Logger at {log_dir}")

    def async_log(self, step: int, data: dict[str, Any]):
        data = jax.tree.map(
            lambda x: x.item() if isinstance(x, Array) else x, 
            data
        )
        for key, value in data.items():
            self.writer.add_scalar(key, value, step)

    def finish(self) -> None:
        self.writer.flush()
        self.writer.close()