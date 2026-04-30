import enum
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, SingleDeviceSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, PyTree

from stax.logger import staxLogger as logger
from stax.utils import is_key

"""
TODO: 
- add multihost sharding 
- integrate with training loop to add Zero 1,2,3 
- switch everything to a ShardingConfig
"""


class ShardingType(enum.StrEnum):
    SINGLE = "single"
    DP = "dp"
    FSDP = "fsdp"


@dataclass
class ShardingConfig:
    params_shape: PyTree[jax.ShapeDtypeStruct] = jax.ShapeDtypeStruct((1,), jnp.float32)
    opt_state_shape: PyTree[jax.ShapeDtypeStruct] = jax.ShapeDtypeStruct((1,), jnp.float32)
    # general options
    sharding_type: ShardingType = ShardingType.SINGLE
    # TODO: actually fix this
    # params_offload: bool = False
    opt_state_offload: bool = False
    # dp options
    data_shard_dim: int = 0
    # fsdp options
    min_bytes_for_fsdp: int = int(1e6)  # 1e6/(1024*1024) = 1MB
    weight_shard_dim: int = 0

    def __post_init__(self):
        if self.sharding_type == ShardingType.FSDP:
            logger.info(
                "Using FSDP make sure to set `xla_tpu_enable_latency_hiding_scheduler=false` for better comms-compute overlap"
            )


@dataclass
class Shardings:
    param_sharding: PyTree[NamedSharding]
    opt_state_sharding: PyTree[NamedSharding]
    metrics_sharding: NamedSharding
    shard_data: Callable[[PyTree], PyTree]
    mesh: jax.sharding.Mesh

def setup_mesh(devices: np.ndarray | None = None):
    if not jax.distributed.is_initialized():
        raise ValueError("jax distributed has not been initialized")

    if devices is None:
        devices = np.array(jax.devices())

    axis_names = ("dp",)
    axis_type = (jax.sharding.AxisType.Auto,)
    try:
        mesh = jax.make_mesh((len(devices),), axis_names, axis_type, devices=list(devices))
    except Exception as _:
        # if jax cannot create optimal mesh layout, make a manual mesh
        logger.warning("Failed to create mesh with make_mesh, falling back to `jax.sharding.Mesh`")
        mesh = Mesh(devices, axis_names, axis_type)
    jax.set_mesh(mesh)
    logger.info(f"setup mesh : {mesh}")
    return mesh


def get_sharding(mesh: Mesh, config: ShardingConfig) -> Shardings:
    """Adapted from https://github.com/kvfrans/jaxtransformer"""
    assert len(mesh.axis_names) == 1, "dp mesh should only have one mesh"

    replicate_sharding = NamedSharding(mesh, P())

    def shard_param(param):
        match config.sharding_type:
            case ShardingType.DP:
                shard = replicate_sharding
            case ShardingType.FSDP:
                if param.ndim < 2 or jnp.dtype(param.dtype).itemsize * param.size < config.min_bytes_for_fsdp:
                    shard = replicate_sharding
                else:
                    param_tuple = [None for _ in range(config.weight_shard_dim)] + [mesh.axis_names[0]]
                    shard = NamedSharding(mesh, P(*(param_tuple)))
            case ShardingType.SINGLE:
                shard = SingleDeviceSharding(mesh.devices[0])
            case _:
                raise ValueError(f"Unknown sharding type {config.sharding_type}")
        return shard

    param_sharding = jax.tree.map(shard_param, config.params_shape)
    opt_state_sharding = jax.tree.map(shard_param, config.opt_state_shape)
    metrics_sharding = replicate_sharding

    data_tuple = [None for _ in range(config.data_shard_dim)] + [mesh.axis_names[0]]
    data_sharding = NamedSharding(mesh, P(*(data_tuple)))

    def shard_data(batch: PyTree) -> PyTree:
        def put_batch_fn(x: Array):
            if is_key(x):
                return jax.device_put(x, replicate_sharding)

            num_hosts = jax.process_count()
            x_shape = (
                *x.shape[: config.data_shard_dim],
                x.shape[config.data_shard_dim] * num_hosts,
                *x.shape[config.data_shard_dim + 1 :],
            )
            return jax.make_array_from_process_local_data(data_sharding, x, x_shape)

        return jax.tree.map(put_batch_fn, batch)

    if config.opt_state_offload:
        opt_state_sharding = jax.tree.map(lambda x: x.with_memory_kind("pinned_host"), opt_state_sharding)

    return Shardings(param_sharding, opt_state_sharding, metrics_sharding, shard_data, mesh)
