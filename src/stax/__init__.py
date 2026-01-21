from stax.checkpointer import (
    Checkpointer,
)
from stax.fn import (
    get_steps_fn,
)
from stax.logger import staxLogger
from stax.model_module import (
    HFModelBase,
    modelBase,
)
from stax.sharding import (
    ShardingConfig,
    Shardings,
    ShardingType,
    get_sharding,
    setup_mesh,
)
from stax.utils import (
    Tracker,
    estimate_compile_stats,
    get_perf_func,
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
    # fn
    "get_steps_fn",
    # utils
    "reshape_key_into_array",
    "Tracker",
    "get_perf_func",
    "is_key",
    "estimate_compile_stats",
    "reshape_batch_key",
    "init_distributed_jax",
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
    "ShardingType",
    "Shardings",
    # model_module
    "HFModelBase",
    "modelBase",
]

__version__ = "0.1.0"


def __getattr__(name):
    raise AttributeError(f"module {__name__} has no attribute {name}")
