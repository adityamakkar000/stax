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
        logger.warning("jax distributed has not been initalizated, initalizing now but add the call to your code to control process, adresses, etc.")
        try: 
            jax.distributed.initialize()
        except Exception as e: 
            logger.critical(f"could not initalize add to client side code error: {e}")

    if devices is None:
        devices = np.array(jax.devices())

    # sometimes jax cannot create optimal mesh layout
    # in that case manually make it
    try:
        mesh = jax.make_mesh((len(devices), ) ('dp',), devices=devices)
    except:
        logger.warning("Failed to create mesh with make_mesh, falling back to sharding.Mesh")
        mesh = Mesh(devices, ('dp',))

    logger.info(f"setup DP mesh with {mesh}")
    return mesh

def get_dp_sharding(mesh : Mesh, params: PyTree, opt_state: PyTree, data_axis : int = 0) -> dict[str, Union[PyTree, Callable]]:

    """inspired by https://github.com/kvfrans/jaxtransformer"""
    assert len(mesh.axis_names) == 1, f"dp mesh should only have one mesh"

    replicate_sharding = NamedSharding(mesh, P())
    data_sharding = NamedSharding(mesh, P(*(None for _ in range(data_axis - 1)), mesh.axis_names[0])) 

    param_sharding = replicate_sharding
    opt_state_sharding = replicate_sharding 

    def shard_data(*batch):
        #TODO: make this different for multicontroller jax 
        batch = jax.tree.map(lambda x: jax.device_put(x, data_sharding))
        return batch if len(batch) > 1 else batch[0]

    return {"param_sharding": param_sharding, "opt_state_sharding": opt_state_sharding, "shard_data": shard_data}

if __name__ == '__main__':
    mesh = setup_dp()
    breakpoint()