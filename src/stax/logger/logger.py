import wandb
from typing import Any, Mapping, Optional


class WandBLogger:
    def __init__(
        self,
        *,
        project: str,
        entity: str,
        name: str,
        config: dict[str, Any],
        run_id: Optional[str] = None,
        **init_kwargs: Any,
    ) -> None:
        self._run = None
        resume = "must" if run_id else "allow"

        if not isinstance(config, dict):
            config = dict(config)

        self._run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            id=run_id,
            resume=resume,
            **init_kwargs,
        )

    def log(
        self,
        step: int,
        data: dict[str, Any],
    ) -> None:
        self._run.log(step=step, data=data)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
        self._run = None

    @property
    def id(self) -> Optional[str]:
        if self._run is None:
            return None
        return self._run.id
