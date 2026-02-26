<picture>
  <source srcset="public/staxBannerLight.png" media="(prefers-color-scheme: dark)">
  <source srcset="public/staxBannerDark.png" media="(prefers-color-scheme: light)">
  <img src="public/staxBannerDark.png" alt="STAX banner">
</picture>

<br>

# STAX

A JAX deep learning stack built on Flax, Optax, and Orbax. Provides utilities for checkpointing, sharding, logging, and training loops.

## Installation

Requires Python >= 3.11 and JAX >= 0.6.2.

```bash
uv pip install -e .
```

## Project Structure

```
stax/
├── checkpointer/     # Orbax checkpoint management with GCS support and best-checkpoint tracking
│   └── main.py       # Checkpointer class
├── fn/               # Training loop utilities with gradient accumulation
│   └── main.py       # train_step, val_step, get_steps_fn
├── logger/           # WandB logging with run resumption
│   └── main.py       # BaseLogger, WandBLogger, TextLogger
├── sharding/         # Distributed training (DP, FSDP, single-device)
│   └── main.py       # setup_mesh, get_sharding, ShardingConfig, ShardingType
└── utils.py          # Tracker, get_perf_func, PRNG key helpers
```

## Exports

```python
from stax import (
    # checkpointer
    Checkpointer,
    # fn
    get_steps_fn,
    # logger
    BaseLogger,
    WandBLogger,
    # sharding
    setup_mesh,
    get_sharding,
    ShardingConfig,
    ShardingType,
    # utils
    Tracker,
    get_perf_func,
    estimate_compile_stats,
    is_key,
    reshape_key_into_array,
    reshape_batch_key,
)
```

## Development

```bash
ruff check src/
ruff format src/
```

