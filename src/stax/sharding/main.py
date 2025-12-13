import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree
from loguru import logger

from jax.sharding import NamedSharding, PartitionSpec as P, Mesh

from typing import Union, Callable, Optional
from jaxtyping import Array
import enum

from dataclasses import dataclass

from stax.utils import is_key

"""
TODO: 
- add multihost sharding 
- integrate with training loop to add Zero 1,2,3 
- switch everything to a ShardingConfig
"""


class ShardingType(enum.Enum):
    SINGLE = None
    DP = "dp"
    FSDP = "fsdp"


@dataclass
class ShardingConfig:
    sharding_type: ShardingType = None
    # dp options
    data_axis: int = 0
    # fsdp options
    min_bytes_for_fsdp: int = 1e6  # 1e6/(1024*1024) = 1MB
    params_shape: Optional[PyTree] = None


def setup_mesh(devices: np.ndarray | None = None):
    if not jax.distributed.is_initialized():
        raise ValueError("jax distributed has not been initalizated")

    if devices is None:
        devices = np.array(jax.devices())

    try:
        mesh = jax.make_mesh((len(devices),), ("dp",), devices=devices)
    except:
        # if jax cannot create optimal mesh layout, make a manual mesh
        logger.warning(
            "Failed to create mesh with make_mesh, falling back to `jax.sharding.Mesh`"
        )
        mesh = Mesh(devices, ("dp",))

    logger.info(f"setup DP mesh with {mesh}")
    return mesh


def get_sharding(
    mesh: Mesh, config: ShardingConfig
) -> dict[str, Union[PyTree, Callable]]:
    """adapted from https://github.com/kvfrans/jaxtransformer"""
    assert len(mesh.axis_names) == 1, f"dp mesh should only have one mesh"

    replicate_sharding = NamedSharding(mesh, P())

    data_tuple = [None for _ in range(config.data_axis)] + [mesh.axis_names[0]]
    data_sharding = NamedSharding(mesh, P(*(data_tuple)))

    if config.sharding_type == ShardingType.DP:
        param_sharding = replicate_sharding
        opt_state_sharding = replicate_sharding
    elif config.sharding_type == ShardingType.FSDP:
        if config.params_shape is None:
            raise ValueError("params_shape must be provided for FSDP sharding")
        dp_shard = NamedSharding(
            mesh,
            P(
                mesh.axis_names[0],
            ),
        )

        def shard_param(param):
            if param.ndim < 2 or param.nbytes < config.min_bytes_for_fsdp:
                return replicate_sharding
            return dp_shard

        param_sharding = jax.tree.map(shard_param, config.params_shape)
        opt_state_sharding = jax.tree.map(shard_param, config.params_shape)

    # TODO: make this different for multicontroller jax
    # using jax.make_array_from_local_devices
    def shard_data(batch: PyTree) -> PyTree:
        def put_batch_fn(x: Array):
            return jax.device_put(x, replicate_sharding if is_key(x) else data_sharding)

        return jax.tree.map(put_batch_fn, batch)

    return shard_data, (param_sharding, opt_state_sharding)
