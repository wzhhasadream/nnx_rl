import atexit
import csv
import json
import os
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
        return arr.tolist()
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
        self.flush_every = flush_every
        self.flush_interval = flush_interval
        self._finished = False

        cfg = config or {}
        self.seed = cfg.get("seed")

        os.makedirs(dir, exist_ok=True)
        self.run_dir = os.path.join(dir, project, self.name)
        os.makedirs(self.run_dir, exist_ok=True)
        global run_dir
        run_dir = self.run_dir

        if self.seed is not None:
            self.run_dir = os.path.join(self.run_dir, f"seed_{self.seed}")
            run_dir = self.run_dir
            os.makedirs(self.run_dir, exist_ok=True)

        self.metrics_jsonl_file = os.path.join(self.run_dir, "metrics.jsonl")
        self.metrics_file = os.path.join(self.run_dir, "metrics.csv")
        self.config_file = os.path.join(self.run_dir, "config.json")

        self.config = cfg
        if self.config:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)

        self._metrics_fp = open(
            self.metrics_jsonl_file,
            "w",
            encoding="utf-8",
            buffering=1,
        )

    def log(
        self,
        data: Dict[str, Union[float, int]],
        step: Optional[int] = None,
        commit: bool = True,
    ):
        if self._finished:
            raise RuntimeError("Run has already been finished.")

        if step is None:
            step = self.step_count

        row = {"step": step}
        for key, value in data.items():
            row[key] = _to_python_scalar(value)

        self._metrics_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._metrics_fp.flush()

        if commit:
            self.step_count = max(self.step_count, step + 1)

    def flush(self):
        if self._finished:
            return

        self._metrics_fp.flush()

    def _convert_jsonl_to_csv(self):
        rows_by_step: dict[int, dict[str, Any]] = {}
        all_keys: set[str] = set()

        with open(self.metrics_jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)
                step = row["step"]
                merged_row = rows_by_step.setdefault(step, {"step": step})
                for key, value in row.items():
                    if key == "step":
                        continue
                    merged_row[key] = value
                    all_keys.add(key)

        columns = ["step"] + sorted(all_keys)

        with open(self.metrics_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for step in sorted(rows_by_step):
                row = rows_by_step[step]
                writer.writerow({col: row.get(col, "") for col in columns})

    def finish(self):
        if self._finished:
            return

        self.flush()
        self._metrics_fp.close()
        self._convert_jsonl_to_csv()
        self._finished = True
        global _current_run
        if _current_run is self:
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
