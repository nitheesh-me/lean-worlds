This repository accompanies the paper _Symbolic World Models in Lean 4 for Reinforcement Learning_
accepted to the RLC 2025 Workshop on Programmatic Reinforcement Learning.

## Quick start

### 1. Install prerequisites

```bash
# Python dependencies
pip install -r requirements.txt

# gym_cellular (not on PyPI — install from its source directory)
pip install -e <path/to/gym_cellular>

# Lean 4 and the Lake build tool
# See: https://leanprover.github.io/lean4/doc/setup.html
```

### 2. Build the Lean server (once)

```bash
cd lean-server && lake build
```

### 3. Run

All entry points are accessible through the single script `run.py`:

```bash
# Evaluate the built-in perfect world model (no GPU needed)
python run.py eval

# Same, but render the environment in a live pygame window
python run.py eval --render

# Use descriptive (prediction-accuracy) fitness instead of reward
python run.py eval --objective descriptive

# Train with the evolutionary algorithm (requires a CUDA GPU)
python run.py train --objective pragmatic
python run.py train --objective pragmatic --checkpoint Qwen/Qwen3-4B --num_generations 100
```

Run `python run.py --help` or `python run.py eval --help` / `python run.py train --help` for the
full list of options.

---

## File overview

```
.
├── run.py - Unified entry point (start here).
├── eval.py - Evaluates a specific world model.
├── gp.py - Evolutionary / genetic algorithm implementation.
├── llm.py - LLM-guided mutation operator.
├── world_model.py - Python wrapper around lean-server.
├── common.py - Shared data structures.
├── mutation_prompt.txt - Mutation prompt (Appendix A in the paper).
├── plot.py - Generates plots from training runs.
├── compute_series.py - Computes the time series shown in Figure 4.
├── requirements.txt
└── lean-server - Lean server for executing synthesized world models.
    ├── lakefile.toml
    ├── lake-manifest.json
    ├── lean-toolchain
    ├── Server
    │   ├── Chess
    │   │   ├── Common.lean
    │   │   └── Fitness.lean
    │   ├── Common.lean
    │   ├── OracleRules.lean
    │   ├── REPL.lean
    │   └── Rules.lean
    └── Server.lean
```
