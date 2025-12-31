from stax.checkpointer import (
    Checkpointer,
)
from stax.fn import (
    get_steps_fn,
)
from stax.logger import (
    BaseLogger,
    WandBLogger,
)
from stax.model_module import (
    HFMixing,
    mainMixin,
)
from stax.sharding import (
    ShardingConfig,
    ShardingType,
    get_sharding,
    setup_mesh,
)
from stax.utils import (
    Tracker,
    estimate_compile_stats,
    get_perf_func,
    is_key,
    reshape_batch_key,
    reshape_key_into_array,
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
    # logger
    "BaseLogger",
    "WandBLogger",
    # sharding
    "setup_mesh",
    "get_sharding",
    "ShardingConfig",
    "ShardingType",
    # model_module
    "mainMixin",
    "HFMixing",
]

__version__ = "0.1.0"


def __getattr__(name):
    raise AttributeError(f"module {__name__} has no attribute {name}")
