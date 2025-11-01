import jax
import wandb
from typing import Any, Mapping, Optional
from jaxtyping import Array 
import os 

from loguru import logger
import abc

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


class WandBLogger(BaseLogger):
    def __init__(self, project: str, config: Optional[Mapping[str, Any]], run_id: Optional[str] = None):
        super().__init__()
        init_args = {
            'project': project,
            'resume': "allow"
        }
        if run_id is not None: 
            init_args['id'] = run_id
            init_args['resume'] = "must"

        self._run = wandb.init(
            **init_args,
            config=config
        )

        wandb.config.update(config)
        logger.info(
            f"Initialized WandB Logger with run id {self.id}"
        )

    def async_log(self, step: int, data: dict[str, Any]):

        data = jax.tree.map(
            lambda x: x.item() if isinstance(x, Array) else x, 
            data
        )
        wandb.log(
            data,
            step=step
        )

    def finish(self) -> None:
        wandb.finish()

    @property
    def id(self) -> Optional[str]: 
        return wandb.run.id

