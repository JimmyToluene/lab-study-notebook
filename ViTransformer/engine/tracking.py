"""Weights & Biases logging, wrapped so it degrades gracefully.

The rest of the code calls ``tracker.log(...)`` unconditionally; this wrapper
turns that into a no-op when W&B is disabled in the config, not installed, or
the user isn't logged in. That keeps train.py free of `if wandb_enabled`
branches and means a machine without wandb can still train.
"""

from __future__ import annotations

from .config import Config


class Tracker:
    def __init__(self, cfg: Config):
        self.run = None
        self._wandb = None
        wcfg = cfg.wandb
        if not wcfg.enabled:
            return
        try:
            import wandb
        except ImportError:
            print("[wandb] enabled in config but not installed; skipping logging. "
                  "Run `pip install wandb` to enable.")
            return
        try:
            self.run = wandb.init(
                project=wcfg.project,
                entity=wcfg.entity,
                name=wcfg.name or cfg.name,
                mode=wcfg.mode,
                config=cfg.to_dict(),
            )
            self._wandb = wandb
            # url is None for offline runs
            print(f"[wandb] logging to {self.run.url or f'offline run {self.run.id}'}")
        except Exception as e:  # e.g. not logged in; don't kill training over logging
            print(f"[wandb] init failed ({e}); continuing without logging.")
            self.run = None

    def log(self, metrics: dict, step: int | None = None) -> None:
        if self.run is not None:
            self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self._wandb.finish()
            self.run = None

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, *exc) -> None:
        self.finish()
