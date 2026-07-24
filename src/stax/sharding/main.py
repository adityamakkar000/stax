import enum
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, PyTree

from stax.logger import staxLogger as logger
from stax.utils import is_key


@dataclass
class ShardingConfig:
    params_shape: PyTree[jax.ShapeDtypeStruct] = jax.ShapeDtypeStruct((1,), jnp.float32)
    opt_state_shape: PyTree[jax.ShapeDtypeStruct] = jax.ShapeDtypeStruct((1,), jnp.float32)

    # TODO: fix this
    # params_offload: bool = False
    opt_state_offload: bool = False
    # dp options
    data_shard_dim: int = 0
    # fsdp options
    min_bytes_for_fsdp: int = int(1e6)  # 1e6/(1024*1024) = 1MB
    weight_shard_dim: int = 0
    # cp options
    cp_shard_dim: int = 0

    dp_group_size: int = 1
    cp_group_size: int = 1
    fsdp_group_size: int = -1


@dataclass
class Shardings:
    param_sharding: PyTree[NamedSharding]
    opt_state_sharding: PyTree[NamedSharding]
    metrics_sharding: NamedSharding
    shard_data: Callable[[PyTree], PyTree]
    mesh: jax.sharding.Mesh


class AXIS_NAMES_ENUM(enum.Enum):
    DP = "dp"
    FSDP = "fsdp"
    CP = "cp"

    @classmethod
    def batch_mesh(cls):
        return ()

    @classmethod
    def full_mesh(cls):
        return (cls.DP.value, cls.FSDP.value, cls.CP.value)


def resolve_axis_sizes(axis_sizes: tuple[int, ...], n_devices: int) -> tuple[int, ...]:
    axis_sizes_list = list(axis_sizes)
    less_than_zero = [i for i, s in enumerate(axis_sizes) if s < 0]
    if len(less_than_zero) > 1:
        raise ValueError(f"axis_sizes contains {len(less_than_zero)} negative values, which is not allowed")
    if len(less_than_zero) == 1:
        if (n_devices % (-1 * (product := np.prod(axis_sizes)))) != 0:
            raise ValueError(f"Cannot resolve axis_sizes with one negative value: {axis_sizes} for {n_devices} devices")
        axis_sizes_list[less_than_zero[0]] = int(n_devices // (-1 * product))
    if np.prod(axis_sizes_list) != n_devices:
        raise ValueError(f"Resolved axis_sizes {axis_sizes_list} do not match number of devices {n_devices}")
    return tuple(axis_sizes_list)


def setup_mesh(axis_sizes: tuple[int, ...], devices: np.ndarray | None = None):
    if not jax.distributed.is_initialized():
        raise ValueError("jax distributed has not been initialized")

    if devices is None:
        devices = np.array(jax.devices())
    n_devices = np.prod(devices.shape)

    axis_type = (jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto)
    axis_sizes = resolve_axis_sizes(axis_sizes, n_devices)
    axis_names = AXIS_NAMES_ENUM.full_mesh()

    try:
        mesh = jax.make_mesh(axis_sizes, axis_names, axis_type, devices=list(devices))
    except Exception as _:
        # if jax cannot create optimal mesh layout, make a manual mesh
        logger.warning("Failed to create mesh with make_mesh, falling back to `jax.sharding.Mesh`")
        mesh = Mesh(devices, axis_names, axis_type)

    jax.set_mesh(mesh)
    logger.info(f"setup mesh : {mesh}")
    logger.info("Set `xla_tpu_enable_latency_hiding_scheduler=false` for better comms-compute overlap")
    return mesh


def get_sharding(mesh: Mesh, config: ShardingConfig) -> Shardings:
    """Adapted from https://github.com/kvfrans/jaxtransformer"""
    assert len(mesh.axis_names) == 3, "mesh should have three axes: dp, fsdp, cp"

    replicate_sharding = NamedSharding(mesh, P())

    def shard_param(param):
        if param.ndim < 2 or jnp.dtype(param.dtype).itemsize * param.size < config.min_bytes_for_fsdp:
            shard = replicate_sharding
        else:
            param_tuple = [None for _ in range(config.weight_shard_dim)] + [AXIS_NAMES_ENUM.FSDP.value]
            shard = NamedSharding(mesh, P(*(param_tuple)))
        return shard

    param_sharding = jax.tree.map(shard_param, config.params_shape)
    opt_state_sharding = jax.tree.map(shard_param, config.opt_state_shape)
    metrics_sharding = replicate_sharding

    def shard_data(batch: PyTree) -> PyTree:
        def put_batch_fn(x: Array):
            if is_key(x):
                return jax.device_put(x, replicate_sharding)

            data_tuple = [None for _ in range(x.ndim)]
            if x.ndim == 1:
                data_tuple[0] = (AXIS_NAMES_ENUM.DP.value, AXIS_NAMES_ENUM.FSDP.value)
            else:
                if config.cp_shard_dim == config.data_shard_dim:
                    data_tuple[config.data_shard_dim] = (
                        AXIS_NAMES_ENUM.DP.value,
                        AXIS_NAMES_ENUM.FSDP.value,
                        AXIS_NAMES_ENUM.CP.value,
                    )
                else:
                    data_tuple[config.data_shard_dim] = (AXIS_NAMES_ENUM.DP.value, AXIS_NAMES_ENUM.FSDP.value)
                    data_tuple[config.cp_shard_dim] = AXIS_NAMES_ENUM.CP.value
                data_sharding = NamedSharding(mesh, P(*data_tuple,))

            num_hosts = mesh.devices.size // jax.local_device_count()
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
