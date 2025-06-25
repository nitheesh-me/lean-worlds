import json
import time
import subprocess
import select
import sys
from pathlib import Path
from dataclasses import dataclass
import shutil

import numpy as np

from common import Logger, CellRule
from gym_cellular.agent.planner import WorldModel

class FormalizedWorldModel(WorldModel):
    def __init__(self, path: Path, oracle_model: WorldModel | None = None):
        assert (path / "lean-toolchain").exists()
        self.oracle_model = oracle_model
        self.path = path
        self.process = None
        self.total_runtime = 0.0

    @staticmethod
    def _get_rules_path(base_path: Path) -> Path:
        return base_path / "Server" / "Rules.lean"

    @staticmethod
    def _get_exe_path(base_path: Path) -> Path:
        return base_path / ".lake" / "build" / "bin" / "server"

    @staticmethod
    def setup_server(source_path: Path, path: Path, rules: list[CellRule]):
        assert not path.exists()
        shutil.copytree(source_path, path)
        if FormalizedWorldModel._get_rules_path(path).exists():
            FormalizedWorldModel._get_rules_path(path).unlink()
        with open(FormalizedWorldModel._get_rules_path(path), "w") as f:
            f.write("import Server.Common\n\n")
            for rule in rules:
                f.write(f"abbrev {rule.id} (g : Grid) (a : Coords) (c : Coords) : Bool := {rule.body}\n\n")
            f.write("\n\n")
            for cell_type, cell_type_name in zip(range(6), ["EMPTY", "TREE", "FIRE_1", "FIRE_2", "FIRE_3", "ROCK"]):
                subrules = [r for r in rules if r.cell_type == cell_type]
                if len(subrules) == 0:
                    body = "False"
                else:
                    body = " ∨ ".join([f"({r.id} s.grid s.agent c)" for r in subrules])
                f.write(f"abbrev {cell_type_name.lower().replace('_', '')} (s : State) (c : Coords) : Bool := {body}\n\n")

        build_result = subprocess.run(["lake", "build"], cwd=path, capture_output=True, text=True)
        if build_result.returncode != 0:
            raise RuntimeError(f"Failed to build the server: {build_result.stderr}")

    @staticmethod
    def destroy_server(path: Path):
        assert (path / "lean-toolchain").exists()
        try:
            shutil.rmtree(path)
        except OSError as e:
            print(f"Failed to destroy the server: {e}")

    def start(self):
        assert self.process is None
        self.process = subprocess.Popen(
            [str(self._get_exe_path(self.path).absolute())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.path,
            text=True,
            bufsize=1,
        )
    
    def stop(self):
        assert self.process is not None
        self.process.kill()
        self.process.wait()
        self.process = None

    def _predict_multiple(self, states: list[dict]) -> list[np.ndarray]:
        input_data = [{
            "grid": {
                "data" : s["grid"].tolist(),
            },
            "agent": {
                "row": int(s["position"][0]),
                "col": int(s["position"][1]),
            },
        } for s in states]

        try:
            self.process.stdin.write(json.dumps(input_data) + "\n")
            self.process.stdin.flush()

            ouput_lines = []
            while True:
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError("Server ended unexpectedly.")
                if line.strip() == "":
                    break
                ouput_lines.append(line)
            output = "".join(ouput_lines)
            output_data = json.loads(output)

            stderr_fd = self.process.stderr.fileno()
            while True:
                ready, _, _ = select.select([stderr_fd], [], [], 0)
                if not ready:
                    break
                error_output = self.process.stderr.readline()
                if not error_output:
                    break
                raise RuntimeError(f"Server returned error: {error_output}")
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError("Could not communicate with Lean REPL.") from e

        if "error" in output_data:
            raise RuntimeError(f"Server returned error: {output_data['error']}")
        return [np.array(g["data"]) for g in output_data["next_grids"]]

    def predict_multiple(self, states: list[dict]) -> list[np.ndarray]:
        next_grids = self._predict_multiple(states)
        if self.oracle_model is not None:
            confusion_matrix = np.zeros((6, 6), dtype=int)
            for state, next_grid in zip(states, next_grids):
                oracle_grid = self.oracle_model.predict(state)
                for i in range(6):
                    for j in range(6):
                        confusion_matrix[i, j] += np.sum((next_grid == i) * (oracle_grid == j))
            # Logger.Instance().log_world_predictions(confusion_matrix)
        return next_grids

    def predict(self, state: dict) -> np.ndarray:
        start_time = time.time()
        result = self.predict_multiple([state])[0]
        self.total_runtime += time.time() - start_time
        return result
