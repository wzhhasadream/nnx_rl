import atexit
import csv
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional, Union

import numpy as np

_current_run = None
run_dir = None


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, np.generic):
        value = value.item()
        if isinstance(value, float):
            return round(value, 4)
        return value
    if hasattr(value, "shape"):
        arr = np.asarray(value)
        if arr.ndim == 0:
            scalar = arr.item()
            if isinstance(scalar, float):
                return round(scalar, 4)
            return scalar
    return value


class Run:
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        dir: str = "Results",
        flush_every: int = 100,
        flush_interval: float = 10.0,
    ):
        self.project = project
        self.name = name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.step_count = 0
        self.metrics_by_step: dict[int, dict[str, Any]] = {}
        self.columns: set[str] = {"step"}
        self.flush_every = flush_every
        self.flush_interval = flush_interval
        self.pending_updates = 0
        self.last_flush_time = time.monotonic()
        self.dirty = False

        cfg = config or {}
        self.seed = cfg.get("seed")

        os.makedirs(dir, exist_ok=True)
        self.run_dir = os.path.join(dir, project, self.name)
        os.makedirs(self.run_dir, exist_ok=True)

        if self.seed is not None:
            self.run_dir = os.path.join(self.run_dir, f"seed_{self.seed}")
            global run_dir
            run_dir = self.run_dir
            os.makedirs(self.run_dir, exist_ok=True)

        self.metrics_file = os.path.join(self.run_dir, "metrics.csv")
        self.config_file = os.path.join(self.run_dir, "config.json")

        self.config = cfg
        if self.config:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=2)

    def log(
        self,
        data: Dict[str, Union[float, int]],
        step: Optional[int] = None,
        commit: bool = True,
    ):
        if step is None:
            step = self.step_count

        row = self.metrics_by_step.get(step, {"step": step})
        for key, value in data.items():
            row[key] = _to_python_scalar(value)
            self.columns.add(key)

        self.metrics_by_step[step] = row
        self.pending_updates += 1
        self.dirty = True

        if commit:
            self.step_count = max(self.step_count, step + 1)

        self._maybe_flush(commit=commit)

    def _maybe_flush(self, commit: bool):
        if not self.dirty:
            return

        now = time.monotonic()
        should_flush = False

        if commit and self.pending_updates >= self.flush_every:
            should_flush = True
        elif now - self.last_flush_time >= self.flush_interval:
            should_flush = True

        if should_flush:
            self.flush()

    def flush(self):
        if not self.metrics_by_step:
            return

        columns = ["step"] + sorted(k for k in self.columns if k != "step")
        tmp_file = f"{self.metrics_file}.tmp"

        with open(tmp_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for step in sorted(self.metrics_by_step.keys()):
                row = self.metrics_by_step[step]
                csv_row = {col: row.get(col, "") for col in columns}
                writer.writerow(csv_row)

        os.replace(tmp_file, self.metrics_file)
        self.pending_updates = 0
        self.last_flush_time = time.monotonic()
        self.dirty = False

    def finish(self):
        self.flush()
        global _current_run
        _current_run = None
        print(f"Run finished. Data saved to: {self.run_dir}")


def init(
    project: str,
    name: Optional[str] = None,
    config: Optional[Dict] = None,
    dir: str = "Results",
    flush_every: int = 100,
    flush_interval: float = 10.0,
) -> Run:
    """
    Create a new run and return it.
    run_path = dir/project/name
    """

    global _current_run
    _current_run = Run(
        project=project,
        name=name,
        config=config,
        dir=dir,
        flush_every=flush_every,
        flush_interval=flush_interval,
    )
    return _current_run


def log(data: Dict[str, Union[float, int]], step: Optional[int] = None, commit: bool = True):
    if _current_run is None:
        raise RuntimeError("No active run. Call init() first.")
    _current_run.log(data, step, commit)


def finish():
    if _current_run is not None:
        _current_run.finish()


atexit.register(finish)
