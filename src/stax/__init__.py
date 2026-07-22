from stax.checkpointer import Checkpointer, OldCheckpointer
from stax.fn import (
    SingleStepFn,
    StepFn,
    TrainFn,
    ValFn,
    get_steps_fn,
)
from stax.logger import staxLogger
from stax.model_module import (
    HFModelBase,
    modelBase,
)
from stax.multihost_utils import process_allgather_over_mesh, sync_over_mesh
from stax.sharding import (
    ShardingConfig,
    Shardings,
    get_sharding,
    setup_mesh,
)
from stax.utils import (
    Tracker,
    estimate_compile_stats,
    get_memory,
    get_perf_func,
    get_rank,
    init_distributed_jax,
    is_key,
    reshape_batch_key,
    reshape_key_into_array,
)
from stax.writer import (
    BaseMetricWriter,
    TextWriter,
    WandBWriter,
)

__all__ = [
    # checkpointer
    "Checkpointer",
    "OldCheckpointer",
    # fn
    "StepFn",
    "SingleStepFn",
    "TrainFn",
    "ValFn",
    "get_steps_fn",
    # utils
    "reshape_key_into_array",
    "Tracker",
    "get_perf_func",
    "is_key",
    "estimate_compile_stats",
    "reshape_batch_key",
    "init_distributed_jax",
    "get_rank",
    "get_memory",
    # writer
    "BaseMetricWriter",
    "TextWriter",
    "WandBWriter",
    # logger
    "staxLogger",
    # sharding
    "setup_mesh",
    "get_sharding",
    "ShardingConfig",
    "Shardings",
    # model_module
    "HFModelBase",
    "modelBase",
    # multihost_utils
    "sync_over_mesh",
    "process_allgather_over_mesh",
]

__version__ = "0.1.0"


def __getattr__(name):
    raise AttributeError(f"module {__name__} has no attribute {name}")
