"""Run logging and resumable checkpoint storage for experiments."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


class ExperimentLogger:
    """Persist immutable run config, append-only metrics, logs, and checkpoints."""

    def __init__(self, run_dir: str | Path, config: dict[str, Any], *, resume: bool = False) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "run.log"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.config_path = self.run_dir / "config.json"
        normalized = json.loads(json.dumps(config, ensure_ascii=False, default=str))
        if self.config_path.exists():
            stored = json.loads(self.config_path.read_text(encoding="utf-8"))
            if not resume:
                raise FileExistsError(f"Run directory already exists; pass --resume to continue: {self.run_dir}")
            if resume and stored != normalized:
                raise ValueError("Resume config does not match the saved run config")
        else:
            self.config_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def log(self, message: str, *, level: str = "INFO") -> None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = f"{timestamp} {level.upper()} {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def log_metrics(self, record: dict[str, Any]) -> None:
        row = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def save_checkpoint(
        self,
        name: str,
        *,
        model_state: dict[str, Any],
        optimizer: torch.optim.Optimizer,
        epoch: int,
        global_step: int,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        path = self.run_dir / "checkpoints" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": model_state,
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "extra": extra or {},
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)
        return path

    @staticmethod
    def load_checkpoint(
        path: str | Path,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        restore_rng: bool = True,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        try:
            state = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            state = torch.load(path, map_location=map_location)
        if optimizer is not None and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if restore_rng:
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"])
            if torch.cuda.is_available() and state.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
        return state
