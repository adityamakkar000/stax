import jax
import jax.numpy as jnp
import neptune 
from typing import Any, Mapping, Optional
import os 

from loguru import logger
import abc

class BaseLogger(abc.ABC):

    def __init__(self, keys_to_print: list[str]):
        self.keys_to_print = keys_to_print
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

class NeptuneLogger(BaseLogger):
    def __init__(self, name: str, config: Optional[dict[str, any]] = None, run_id: Optional[str]  = None ):
        assert not (config is None and run_id is None), f"config or run id must be provided"
        project_name = os.environ.get("NEPTUNE_PROJECT")
        api_key = os.environ.get("NEPTUNE_API_KEY")

        init_args = {
            'project': project_name, 
            'api_token': api_key,
            'name': name,
        }

        if run_id is not None: 
            init_args['with_id'] = run_id

        self._run  = neptune.Run(
            **init_args
        )

        if run_id is None: 
            self._run['parameters'] = config

        logger.info(
            f"Initialized Neptune Logger with run id {self._run._custom_run_id}"
        )

    def async_log(self, step : int, data : dict[str, Any]): 

        def convert_to_float(x): 
            if isinstance(x, jnp.Array): 
                return x.item() 
            return x

        data = jax.tree.map(
            convert_to_float, data
        )

        for key in data.keys():
            self._run[key].append(data[key], step=step)

    def finish(self) -> None:
        self._run.stop()

    @property
    def id(self) -> Optional[str]: 
        return self._run._sys_id
