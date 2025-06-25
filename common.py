import argparse
from dataclasses import dataclass
from collections import defaultdict
import json
import re
import copy
import time
import multiprocessing
from pathlib import Path
import concurrent.futures
from enum import Enum
import asyncio
from typing import cast, Self
import traceback
import math
import random
import datetime
import os
import sys
import numpy as np
import torch

from filelock import FileLock

@dataclass(frozen=True)
class CellRule:
    id: str
    body: str
    cell_type: int
    parent_id: str | None = None

    @classmethod
    def create(cls, body: str, cell_type: int, parent_id: str | None = None):
        return cls(
            id="rule_" + "".join(np.random.choice(list("0123456789"), size=8)),
            body=body,
            cell_type=cell_type,
            parent_id=parent_id,
        )

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            id=data["id"],
            body=data["body"],
            cell_type=data["cell_type"],
            parent_id=data["parent_id"],
        )

    def to_dict(self):
        return {
            "id": self.id,
            "body": self.body,
            "cell_type": self.cell_type,
            "parent_id": self.parent_id,
        }

    def clone(self) -> Self:
        return self.create(self.body, self.cell_type, self.parent_id)

    def __str__(self):
        return f"{self.cell_type} ({self.id}): {self.body}"

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def semantic_eq(self, other: Self) -> bool:
        return self.body == other.body and self.cell_type == other.cell_type

@dataclass
class Individual:
    id: str
    rules: list[CellRule]
    parent_ids: list[str]
    fitness: float | None = None

    f_score: float | None = None
    accuracy: float | None = None
    reward: float | None = None

    @classmethod
    def create(cls, rules: list[CellRule], parent_ids: list[str] | None = None):
        return cls(
            id="individual_" + "".join(np.random.choice(list("0123456789"), size=8)),
            rules=rules,
            parent_ids=parent_ids if parent_ids is not None else [],
        )

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            id=data["id"],
            rules=[CellRule.from_dict(rule) for rule in data["rules"]],
            fitness=data["fitness"],
            parent_ids=data["parent_ids"],
        )


    def to_dict(self):
        return {
            "id": self.id,
            "rules": [rule.to_dict() for rule in self.rules],
            "fitness": self.fitness,
            "parent_ids": self.parent_ids,
        }

    def clone(self) -> Self:
        return self.create([rule.clone() for rule in self.rules], self.parent_ids)

    def __str__(self):
        s = f"individual {self.id}\n"
        s += f"fitness: {self.fitness}\n"
        for rule in self.rules:
            s += f"{rule}\n"
        return s.rstrip()

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def semantic_eq(self, other: Self) -> bool:
        return all(rule.semantic_eq(other_rule) for rule, other_rule in zip(self.rules, other.rules))

class Logger:
    _instance: Self | None = None
    _log_dir: Path | None = None

    @classmethod
    def configure(cls, log_dir: Path) -> None:
        cls._log_dir = log_dir

    @classmethod
    def Instance(cls) -> Self:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        log_dir = Logger._log_dir
        self._paths = None
        self._locks = None

        if log_dir is not None:
            filenames = {
                "model_outputs":     "model_outputs.txt",
                "mutations":         "mutations.txt",
                "world_predictions": "world_predictions.jsonl",
                "invalid_mutations": "invalid_mutations.txt",
                "evolution":         "evolution.txt",
                "population":        "population.jsonl",
                "stats":             "stats.json",
            }

            self._paths: dict[str, Path] = {
                name: (log_dir / fname)
                for name, fname in filenames.items()
            }

            self._locks: dict[str, FileLock] = {
                name: FileLock(str(path) + ".lock")
                for name, path in self._paths.items()
            }

        self._start_time = time.time()
        self.total_mutations = 0
        self.invalid_mutations = 0

    def log_model_outputs(self, outputs: list[str]) -> None:
        if self._paths is None:
            return
        path = self._paths["model_outputs"]
        lock = self._locks["model_outputs"]
        with lock:
            with open(path, "a") as f:
                for output in outputs:
                    quotes = '"""\n'
                    f.write(f"{quotes}{output}\n{quotes}\n\n")

    def log_mutations(self, source: "CellRule", mutations: list["CellRule"]) -> None:
        if self._paths is None:
            return
        path = self._paths["mutations"]
        lock = self._locks["mutations"]
        with lock:
            with open(path, "a") as f:
                f.write(f"{source.body}\n->\n")
                if not mutations:
                    f.write("[NOTHING]\n")
                else:
                    for mutated in mutations:
                        f.write(f"{mutated.body}\n")
                f.write("-" * 80 + "\n")

    def log_invalid_mutations(
        self,
        source: "CellRule",
        invalid: list[tuple["CellRule", str]],
    ) -> None:
        if self._paths is None:
            return
        path = self._paths["invalid_mutations"]
        lock = self._locks["invalid_mutations"]
        with lock:
            with open(path, "a") as f:
                f.write(f"{source.body}\n->\n")
                for mutation, error in invalid:
                    f.write(f"{mutation.body}\nerror: {error}\n\n")
                f.write("\n")

    def log_world_predictions(self, confusion_matrix: np.ndarray) -> None:
        if self._paths is None:
            return
        path = self._paths["world_predictions"]
        lock = self._locks["world_predictions"]
        # if np.any(confusion_matrix - np.diag(np.diag(confusion_matrix))):
        #     print("Confusion matrix:")
        #     labels = ["empty", "tree", "fire1", "fire2", "fire3", "rock"]
        #     header = "      " + " ".join(f"{label:>5}" for label in labels)
        #     print(header)
        #     for i in range(confusion_matrix.shape[0]):
        #         row = " ".join(f"{confusion_matrix[i, j]:5d}" for j in range(confusion_matrix.shape[1]))
        #         print(f"{labels[i]:>5} {row}")
        with lock:
            with open(path, "a") as f:
                data = {"confusion_matrix": confusion_matrix.tolist()}
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def log_evolution_step(
        self,
        generation: int,
        population: list[Individual],
        new_population: list[Individual],
    ) -> None:
        if self._paths is None:
            return
        path = self._paths["evolution"]
        lock = self._locks["evolution"]
        with lock:
            with open(path, "a") as f:
                f.write(f"=== Generation {generation} ===\n")
                new_remaining = set(new_population)

                for ind in population:
                    f.write(f"{ind}\n")
                    if ind in new_population:
                        f.write("-> SURVIVED\n")
                        new_remaining.discard(ind)
                    else:
                        f.write(f"-> DIED\n")

                    children = [c for c in new_population if ind.id in c.parent_ids]
                    if children:
                        f.write("-> CHILDREN:\n")
                        for child in children:
                            f.write(f"{child}\n")
                            new_remaining.discard(child)
                    f.write("-" * 80 + "\n")

                if new_remaining:
                    f.write(f"= NEW ({len(new_remaining)}) =\n")
                    for child in new_remaining:
                        f.write(f"{child}\n")
                        f.write("-" * 80 + "\n")
                f.write("\n")

    def log_population(self, population: list["Individual"], best_fitness: float | None) -> None:
        if self._paths is None:
            return
        path = self._paths["population"]
        lock = self._locks["population"]
        with lock:
            with open(path, "a") as f:
                data = {
                    "population":   [ind.to_dict() for ind in population],
                    "best_fitness": best_fitness,
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def log_stats(self) -> None:
        if self._paths is None:
            return
        path = self._paths["stats"]
        lock = self._locks["stats"]
        with lock:
            with open(path, "a") as f:
                total = self.total_mutations
                invalid = self.invalid_mutations
                data = {
                    "total_mutations":        total,
                    "invalid_mutations":      invalid,
                    "invalid_mutations_rate": (invalid / total) if total > 0 else 0.0,
                    "runtime":                time.time() - self._start_time,
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
    

class SimpleTimer:
    def __init__(self):
        self._start_time = time.time()
        self._curr_section = None
        self._total_durations = defaultdict(float)

    def start_section(self, name: str, do_print: bool = True):
        if self._curr_section is not None:
            self.end_section(do_print)
        self._curr_section = (name, time.time())
        if do_print:
            print(f"[START] {name}")

    def end_section(self, do_print: bool = True):
        if self._curr_section is None:
            raise ValueError("No section to end")
        name, start_time = self._curr_section
        self._curr_section = None
        duration = time.time() - start_time
        self._total_durations[name] += duration
        if do_print:
            print(f"[END] {name} (took {duration:.2f}s)")

    def print_stats(self, stats: dict | None = None):
        if stats is None:
            stats = self.get_stats()
        total_runtime = stats["total_runtime"]
        for name, duration in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            if name in ("total_runtime",):
                continue
            percent = (duration / total_runtime) if total_runtime > 0 else 0
            print(f"{name}: {duration:.2f}s ({percent:.2%})")
        print(f"total runtime: {total_runtime:.2f}s")

    def get_stats(self):
        runtime = time.time() - self._start_time
        stats = {}
        for name, duration in sorted(self._total_durations.items(), key=lambda x: x[1], reverse=True):
            stats[name] = duration
        unaccounted = runtime - sum(self._total_durations.values())
        stats["unaccounted"] = unaccounted
        stats["total_runtime"] = runtime
        return stats
        

def combine_stats(stats_list: list[dict]) -> dict:
    combined = {}
    for stats in stats_list:
        for name, value in stats.items():
            if name not in combined:
                combined[name] = 0.0
            combined[name] += value
    return combined

def setup_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_args_descriptor(
        args_ns: argparse.Namespace,
        param_whitelist: list[str] | None = None,
        include_slurm_id=True,
        include_time=True,
) -> str:
    args = vars(args_ns)
    if include_time:
        descriptor = datetime.datetime.now().strftime("%y-%m-%d_%H%M%S")
    else:
        descriptor = ""

    if include_slurm_id and "SLURM_JOB_ID" in os.environ:
        if len(descriptor) > 0:
            descriptor += "-"
        descriptor += f"id={os.environ['SLURM_JOB_ID']}"

    visible_args = {k: v for k, v in sorted(args.items())}
    if param_whitelist is not None:
        visible_args = {k: v for k, v in visible_args.items() if k in param_whitelist}

    def format_value(v: str) -> str:
        if isinstance(v, Path) or "/" in str(v):
            v = str(v)
            if v.endswith("/"):
                v = v[:-1]
            parts = [p for p in v.split("/") if len(p) != 0]
            return "_".join([v[:50] for v in parts[-2:]])
        if isinstance(v, str):
            return v.replace("<", "").replace(">", "")
        return str(v)

    if len(visible_args) > 0:
        if len(descriptor) > 0:
            descriptor += "-"
        descriptor += ",".join((
            "{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), format_value(v))
            for k, v in visible_args.items()
        ))

    assert len(descriptor) > 0
    return descriptor

def dump_args(args, logdir):
    path = os.path.join(logdir, "args.json")
    with open(path, "w") as f:
        data = {k: str(v) for k, v in args.__dict__.items()}
        json.dump(data, f, indent=4, sort_keys=True)
        f.write("\n")

def boxed(s: str) -> str:
    lines = s.split("\n")
    max_width = max(len(line) for line in lines)

    return "\n".join([
        "⎡" + "-" * max_width + "⎤",
        *["|" + line + " " * (max_width - len(line)) + "|" for line in lines],
        "⎣" + "-" * max_width + "⎦",
    ])