import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, PyTree

from stax.logger import staxLogger as logger
from stax.sharding.config import AXIS_NAMES, MeshConfig, ShardingConfig, Shardings
from stax.utils import is_key


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


def setup_mesh(mesh_config: MeshConfig, devices: np.ndarray | None = None):
    if not jax.distributed.is_initialized():
        raise ValueError("jax distributed has not been initialized")

    if devices is None:
        devices = np.array(jax.devices())
    n_devices = np.prod(devices.shape)

    axis_sizes = mesh_config.axes

    axis_type = (jax.sharding.AxisType.Auto,) * len(axis_sizes)
    axis_sizes = resolve_axis_sizes(axis_sizes, n_devices)
    axis_names = AXIS_NAMES.full_mesh()

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
    assert tuple(mesh.axis_names) == AXIS_NAMES.full_mesh(), f"mesh axes {mesh.axis_names} != {AXIS_NAMES.full_mesh()}"

    replicate_sharding = NamedSharding(mesh, P())

    def shard_param(path, param: Array):
        if (param.size * param.dtype.itemsize) < config.min_bytes_for_fsdp:
            return replicate_sharding
        path_str = jax.tree_util.keystr(path)
        pspec = config.weight_shard_fn(path_str, param).pspec
        if len(pspec) > param.ndim:
            raise ValueError(f"{path_str}: sharding {pspec} has more dims than param shape {param.shape}")
        for dim, axis in enumerate(pspec):
            if axis is not None and param.shape[dim] % mesh.shape[axis] != 0:
                raise ValueError(
                    f"{path_str}: dim {dim} (size {param.shape[dim]}) not divisible by mesh axis '{axis}' "
                    f"(size {mesh.shape[axis]})"
                )
        return NamedSharding(mesh, pspec)

    param_sharding = jax.tree.map_with_path(shard_param, config.params_shape)
    opt_state_sharding = jax.tree.map_with_path(shard_param, config.opt_state_shape)
    metrics_sharding = replicate_sharding

    def shard_data(batch: PyTree) -> PyTree:
        def put_batch_fn(x: Array):
            if is_key(x):
                return jax.device_put(x, replicate_sharding)

            data_tuple = [None for _ in range(x.ndim)]
            if x.ndim == 1:
                data_tuple[0] = AXIS_NAMES.data_parallel_mesh()
            else:
                if config.cp_shard_dim == config.data_shard_dim:
                    data_tuple[config.data_shard_dim] = AXIS_NAMES.data_mesh()
                else:
                    data_tuple[config.data_shard_dim] = AXIS_NAMES.data_parallel_mesh()
                    data_tuple[config.cp_shard_dim] = AXIS_NAMES.context_parallel_mesh()

            target_data_sharding = NamedSharding(mesh, P(*data_tuple))
            global_sharding = [None] * config.data_shard_dim + [AXIS_NAMES.full_mesh()]
            global_data_sharding = NamedSharding(mesh, P(*global_sharding))

            num_hosts = mesh.devices.size // jax.local_device_count()
            global_x_shape = (
                *x.shape[: config.data_shard_dim],
                x.shape[config.data_shard_dim] * num_hosts,
                *x.shape[config.data_shard_dim + 1 :],
            )

            global_x = jax.make_array_from_process_local_data(global_data_sharding, x, global_x_shape)
            target_x = jax.device_put(global_x, target_data_sharding)
            return target_x

        return jax.tree.map(put_batch_fn, batch)

    if config.opt_state_offload:
        opt_state_sharding = jax.tree.map(lambda x: x.with_memory_kind("pinned_host"), opt_state_sharding)

    return Shardings(param_sharding, opt_state_sharding, metrics_sharding, shard_data, mesh)
