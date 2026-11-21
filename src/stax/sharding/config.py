import enum
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, PyTree


class AXIS_NAMES(enum.StrEnum):
    DP = "dp"
    FSDP = "fsdp"
    CP_ULYSSES = "cp_ulysses"
    CP_RING = "cp_ring"
    TP = "tp"

    @classmethod
    def model_parallel_mesh(cls):
        return (cls.FSDP, cls.TP)

    @classmethod
    def data_parallel_mesh(cls):
        return (cls.DP, cls.FSDP)

    @classmethod
    def context_parallel_mesh(cls):
        return (cls.CP_RING, cls.CP_ULYSSES)

    @classmethod
    def data_mesh(cls):
        return (*cls.data_parallel_mesh(), *cls.context_parallel_mesh())

    @classmethod
    def full_mesh(cls):
        return (*cls.data_mesh(), cls.TP)


class ModelSharding(NamedTuple):
    fsdp: int | None = None
    tp: int | None = None

    @property
    def pspec(self) -> P:
        dims = [d for d in (self.fsdp, self.tp) if d is not None]
        if not dims:
            return P()
        if self.fsdp == self.tp:
            raise ValueError(f"fsdp and tp can't shard the same dim: {self}")
        partition = [None] * (max(dims) + 1)
        if self.fsdp is not None:
            partition[self.fsdp] = AXIS_NAMES.FSDP
        if self.tp is not None:
            partition[self.tp] = AXIS_NAMES.TP
        return P(*partition)


@dataclass
class MeshConfig:
    dp: int = 1
    fsdp: int = -1
    cp_ulysses: int = 1
    cp_ring: int = 1
    tp: int = 1

    @property
    def axes(self) -> tuple[int, ...]:
        return tuple(getattr(self, axis_name) for axis_name in AXIS_NAMES.full_mesh())

    def __post_init__(self):
        if self.cp_ring != 1:
            raise NotImplementedError("cp ring isn't implemented yet, set cp_ring=1")


@dataclass
class ShardingConfig:
    params_shape: PyTree[jax.ShapeDtypeStruct]
    opt_state_shape: PyTree[jax.ShapeDtypeStruct]

    mesh_config: MeshConfig

    # TODO: fix this
    # params_offload: bool = False
    opt_state_offload: bool = False
    # dp options
    data_shard_dim: int = 0
    # fsdp options
    min_bytes_for_fsdp: int = int(1e7)  # 1e7/(1024*1024) = 10MB
    # cp options
    cp_shard_dim: int = 0
    # model parallelism options
    # NOTE: also applied to opt_state_shape, where paths look like "[0].mu['layer']['kernel']",
    # so match on suffix/substring rather than exact paths to keep opt state aligned with params.
    weight_shard_fn: Callable[[str, Array], ModelSharding] = lambda _path, param: (
        ModelSharding(0, None) if param.ndim >= 2 else ModelSharding()
    )


@dataclass
class Shardings:
    param_sharding: PyTree[NamedSharding]
    opt_state_sharding: PyTree[NamedSharding]
    metrics_sharding: NamedSharding
    shard_data: Callable[[PyTree], PyTree]
    mesh: jax.sharding.Mesh
