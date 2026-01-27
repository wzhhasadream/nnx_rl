import csv
import json
import os
from datetime import datetime
from typing import Dict, Optional, Union

_current_run = None
run_dir = None

class Run:

    def __init__(self, project: str, name: Optional[str] = None, config: Optional[Dict] = None, dir: str = "Results"):
        self.project = project
        self.name = name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.step_count = 0
        self.metrics_data = []  
        self.metrics_by_step = {}  

        cfg = config or {}
        self.seed = cfg.get('seed')


        os.makedirs(dir, exist_ok=True)
        self.run_dir = os.path.join(os.path.join(dir, f"{project}"), f"{self.name}")
        os.makedirs(self.run_dir, exist_ok=True)

        if self.seed is not None:
            self.run_dir = os.path.join(self.run_dir, f"seed_{self.seed}")  
            global run_dir
            run_dir = self.run_dir
            os.makedirs(self.run_dir, exist_ok=True)  

        self.metrics_file = os.path.join(self.run_dir, "metrics.csv")
        self.config_file = os.path.join(self.run_dir, "config.json")


        self.config = config or {}
        if self.config:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)

    def log(self, data: Dict[str, Union[float, int]], step: Optional[int] = None, commit: bool = True):
        if step is None:
            step = self.step_count

        row = self.metrics_by_step.get(step, {"step": step})

        for key, value in data.items():
            if isinstance(value, float):
                row[key] = round(value, 4)
            else:
                row[key] = value

        self.metrics_by_step[step] = row

        if commit:
            self.step_count = max(self.step_count, step + 1)
            
        self._write_csv()

    def _write_csv(self):

        if not self.metrics_by_step:
            return


        all_keys = set()
        for row in self.metrics_by_step.values():
            all_keys.update(row.keys())

        columns = ["step"] + sorted([k for k in all_keys if k != "step"])


        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for step in sorted(self.metrics_by_step.keys()):
                row = self.metrics_by_step[step]
                csv_row = {col: row.get(col, "") for col in columns}
                writer.writerow(csv_row)

    def finish(self):

        global _current_run
        _current_run = None
        print(f"Run finished. Data saved to: {self.run_dir}")

def init(project: str, name: Optional[str] = None, config: Optional[Dict] = None, dir: str = "Results") -> Run:
    '''
    create a new run and return it 
    run_path = dir//project//name
    '''

    global _current_run

    _current_run = Run(project=project, name=name, config=config, dir=dir)
    return _current_run

def log(data: Dict[str, Union[float, int]], step: Optional[int] = None, commit: bool = True):

    if _current_run is None:
        raise RuntimeError("No active run. Call init() first.")
    _current_run.log(data, step, commit)

def finish():

    if _current_run is not None:
        _current_run.finish()
