#!/usr/bin/env python3
"""
lean-worlds: Symbolic World Models in Lean 4 for Reinforcement Learning

Subcommands
-----------
  eval   Evaluate the built-in perfect world model (no GPU required).
         This is the quickest way to verify your setup and see the agent act.

  train  Run the evolutionary algorithm with an LLM-guided mutation operator
         to discover symbolic world models (requires a CUDA-capable GPU).

Quick-start
-----------
  # 1. Evaluate the perfect world model (measures fitness over 5 episodes)
  python run.py eval

  # 2. Same, but render the environment in a live pygame window
  python run.py eval --render

  # 3. Train with the evolutionary algorithm (pragmatic / reward-based fitness)
  python run.py train --objective pragmatic

ARC-AGI-3 integration
---------------------
  To run lean-worlds as an ARC-AGI-3 agent, see arc_agent.py and the
  "ARC-AGI-3 integration" section in README.md.

Prerequisites
-------------
  1. Python dependencies:
       pip install -r requirements.txt

  2. gym_cellular package (not on PyPI — install from its source directory):
       pip install -e <path/to/gym_cellular>

  3. Lean 4 and Lake build tool:
       https://leanprover.github.io/lean4/doc/setup.html

  4. Build the Lean server (only needed once, or after Lean source changes):
       cd lean-server && lake build
"""

import argparse
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Subcommand handlers — delegate to the existing scripts via subprocess so
# that the entry point itself stays importable without any heavy dependencies.
# ---------------------------------------------------------------------------

def cmd_eval(args: argparse.Namespace) -> None:
    cmd = [sys.executable, "eval.py"]
    cmd += ["--objective", args.objective]
    cmd += ["--seed", str(args.seed)]
    cmd += ["--planning_depth", str(args.planning_depth)]
    cmd += ["--num_episodes", str(args.num_episodes)]
    cmd += ["--num_workers", str(args.num_workers)]
    cmd += ["--server_path", str(args.server_path)]
    cmd += ["--tmp_dir", str(args.tmp_dir)]
    if args.render:
        cmd += ["--render"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def cmd_train(args: argparse.Namespace) -> None:
    cmd = [sys.executable, "gp.py"]
    cmd += ["--objective", args.objective]
    cmd += ["--seed", str(args.seed)]
    cmd += ["--checkpoint", args.checkpoint]
    cmd += ["--num_generations", str(args.num_generations)]
    cmd += ["--population_size", str(args.population_size)]
    cmd += ["--num_workers", str(args.num_workers)]
    cmd += ["--output_dir", str(args.output_dir)]
    cmd += ["--server_path", str(args.server_path)]
    cmd += ["--tmp_dir", str(args.tmp_dir)]
    cmd += ["--planning_depth", str(args.planning_depth)]
    cmd += ["--algorithm", args.algorithm]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="{eval,train}")

    # ── eval ──────────────────────────────────────────────────────────────
    ep = sub.add_parser(
        "eval",
        help="Evaluate the built-in perfect world model (no GPU required)",
        description=(
            "Compiles the hardcoded 'perfect' Lean world model and measures its\n"
            "fitness in the Forest-Fire environment.\n\n"
            "Examples:\n"
            "  python run.py eval                         # compute fitness (5 episodes)\n"
            "  python run.py eval --render                # watch the agent in a live window\n"
            "  python run.py eval --objective descriptive # use descriptive fitness"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ep.add_argument(
        "--objective",
        choices=["pragmatic", "descriptive"],
        default="pragmatic",
        help=(
            "Fitness function to use.\n"
            "  pragmatic   – total reward collected by the planning agent (default)\n"
            "  descriptive – how accurately the model predicts the next grid state"
        ),
    )
    ep.add_argument(
        "--render",
        action="store_true",
        help="Open a pygame window and render the episode in real time",
    )
    ep.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    ep.add_argument("--planning_depth", type=int, default=4, help="Look-ahead depth for the planning agent (default: 4)")
    ep.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to evaluate (default: 5)")
    ep.add_argument("--num_workers", type=int, default=1, help="Parallel worker processes (default: 1)")
    ep.add_argument("--server_path", type=Path, default="lean-server", help="Path to the lean-server directory (default: lean-server)")
    ep.add_argument("--tmp_dir", type=Path, default="tmp", help="Temporary directory for compiled servers (default: tmp)")
    ep.set_defaults(func=cmd_eval)

    # ── train ─────────────────────────────────────────────────────────────
    tp = sub.add_parser(
        "train",
        help="Run the evolutionary algorithm to discover world models (requires GPU)",
        description=(
            "Runs a genetic algorithm guided by an LLM mutation operator to evolve\n"
            "symbolic Lean world models for the Forest-Fire environment.\n\n"
            "Requirements:\n"
            "  - At least one CUDA-capable GPU\n"
            "  - lean-server already built (cd lean-server && lake build)\n\n"
            "Examples:\n"
            "  python run.py train --objective pragmatic\n"
            "  python run.py train --objective pragmatic --checkpoint Qwen/Qwen3-4B --num_generations 100"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    tp.add_argument(
        "--objective",
        choices=["pragmatic", "descriptive"],
        required=True,
        help="Fitness function: reward-based (pragmatic) or prediction-accuracy (descriptive)",
    )
    tp.add_argument(
        "--checkpoint",
        default="Qwen/Qwen3-4B",
        help="HuggingFace model checkpoint used for mutation (default: Qwen/Qwen3-4B)",
    )
    tp.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    tp.add_argument("--num_generations", type=int, default=500, help="Number of evolutionary generations (default: 500)")
    tp.add_argument("--population_size", type=int, default=16, help="Number of individuals in the population (default: 16)")
    tp.add_argument("--num_workers", type=int, default=64, help="Parallel worker processes for fitness evaluation (default: 64)")
    tp.add_argument("--output_dir", type=Path, default="out", help="Directory for logs and results (default: out)")
    tp.add_argument("--server_path", type=Path, default="lean-server", help="Path to the lean-server directory (default: lean-server)")
    tp.add_argument("--tmp_dir", type=Path, default="tmp", help="Temporary directory for compiled servers (default: tmp)")
    tp.add_argument("--planning_depth", type=int, default=4, help="Look-ahead depth for the planning agent (default: 4)")
    tp.add_argument(
        "--algorithm",
        choices=["classic", "simple"],
        default="classic",
        help="EA variant to use: classic (SUS selection + crossover) or simple (default: classic)",
    )
    tp.set_defaults(func=cmd_train)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
