import jax 
import numpy as np
from jaxtyping import PyTree
from loguru import logger

from jax.sharding import (
    NamedSharding, 
    PartitionSpec as P, 
    Mesh
)

from typing import Union, Callable

"""
TODO: 
add multihost sharding 
FSDP sharding
integrate with training loop to add Zero 1,2,3 
"""

def setup_dp(devices : np.ndarray | None = None):

    if not jax.distributed.is_initialized():
        raise ValueError("jax distributed has not been initalizated")

    if devices is None:
        devices = np.array(jax.devices())

    # if jax cannot create optimal mesh layout, make a manual mesh
    try:
        mesh = jax.make_mesh((len(devices), ), ('dp',), devices=devices)
    except:
        logger.warning("Failed to create mesh with make_mesh, falling back to sharding.Mesh")
        mesh = Mesh(devices, ('dp',))

    logger.info(f"setup DP mesh with {mesh}")
    return mesh

def get_dp_sharding(mesh : Mesh, data_axis : int = 0) -> dict[str, Union[PyTree, Callable]]:

    """inspired by https://github.com/kvfrans/jaxtransformer"""
    assert len(mesh.axis_names) == 1, f"dp mesh should only have one mesh"

    replicate_sharding = NamedSharding(mesh, P())

    data_tuple = [None for _ in range(data_axis - 1)] + [mesh.axis_names[0]]
    data_sharding = NamedSharding(mesh, P(*(data_tuple))) 

    param_sharding = replicate_sharding
    opt_state_sharding = replicate_sharding 
    
    #TODO: make this different for multicontroller jax 
    def shard_data(batch):
        # special function to handle keys in batch
        def put_batch_fn(x):
            is_key = lambda x:  (x.ndim == 2 and x.shape[0] == 1 and x.shape[1] == 2)
            if is_key(x): 
                x = jax.array(jax.random.split(x, jax.device_count()))
            return jax.device_put(x, data_sharding)
        return jax.tree.map(put_batch_fn, batch)

    return shard_data, (param_sharding, opt_state_sharding)

SHARDING_TYPES = {
    'dp': get_dp_sharding
}

if __name__ == '__main__':
    mesh = setup_dp()
    breakpoint()