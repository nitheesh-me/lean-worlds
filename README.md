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

## ARC-AGI-3 integration

`arc_agent.py` is a drop-in `Agent` subclass for the
[ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents) framework.
It routes every frame through lean-worlds' symbolic Lean 4 world model and uses
look-ahead planning (`PlanningAgent`) to choose actions.

### Step-by-step setup

```bash
# 1. Clone both repos side-by-side.
git clone https://github.com/nitheesh-me/lean-worlds.git
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git

# 2. Install lean-worlds dependencies (inside lean-worlds/).
cd lean-worlds
pip install -r requirements.txt
pip install -r requirements-arc.txt   # adds arc-agi / arcengine
pip install -e <path/to/gym_cellular>

# 3. Build the Lean server (once).
cd lean-server && lake build && cd ..

# 4. Copy the agent template into ARC-AGI-3-Agents.
cp arc_agent.py ../ARC-AGI-3-Agents/agents/templates/lean_worlds_agent.py

# 5. Register it — add ONE line to ARC-AGI-3-Agents/agents/__init__.py:
#    from .templates.lean_worlds_agent import LeanWorldsAgent  # noqa: F401

# 6. Get your API key from https://three.arcprize.org/ and set it.
export ARC_API_KEY="your_key_here"
# or put it in ARC-AGI-3-Agents/.env

# 7. Run the lean-worlds agent against a game.
cd ../ARC-AGI-3-Agents
python main.py --agent=leanworldsagent --game=ls20
```

### Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `LEAN_WORLDS_PATH` | auto-detected | Absolute path to this lean-worlds checkout |
| `LEAN_SERVER_PATH` | `<lean-worlds>/lean-server` | Path to the compiled Lean server |
| `LEAN_TMP_DIR` | `<lean-worlds>/tmp` | Temporary directory for Lean server copies |
| `LEAN_PLANNING_DEPTH` | `4` | Look-ahead depth for the planning agent |

### Using an evolved rule set

After running `python run.py train --objective pragmatic`, use
`python plot.py show_best <output_dir>` to inspect the best individual, then
load its rules in a subclass:

```python
# my_lean_agent.py
from arc_agent import LeanWorldsAgent
from common import CellRule

class MyLeanAgent(LeanWorldsAgent):
    rules = [
        CellRule(id="r0", body="g.get c = FIRE_3", cell_type=0),
        # … paste rules from plot.py show_best output …
    ]
```

### Game compatibility

The default configuration targets **cellular-automaton grid games** where:
- `frame[0]` is a 2-D integer grid (cell type per cell),
- `frame[1]` (optional) marks the agent's position.
- ACTION1–4 map to UP / DOWN / LEFT / RIGHT, ACTION5 = STAY.

Override `CELL_MAP` and `ACTION_MAP` class attributes in a subclass to adapt to
a different game encoding.

---

## File overview

```
.
├── run.py              - Unified entry point for eval/train (start here).
├── arc_agent.py        - ARC-AGI-3 compatible agent using lean-worlds planning.
├── eval.py             - Evaluates a specific world model.
├── gp.py               - Evolutionary / genetic algorithm implementation.
├── llm.py              - LLM-guided mutation operator.
├── world_model.py      - Python wrapper around lean-server.
├── common.py           - Shared data structures.
├── mutation_prompt.txt - Mutation prompt (Appendix A in the paper).
├── plot.py             - Generates plots from training runs.
├── compute_series.py   - Computes the time series shown in Figure 4.
├── requirements.txt    - Core Python dependencies.
├── requirements-arc.txt - Extra dependencies for ARC-AGI-3 integration.
└── lean-server         - Lean server for executing synthesized world models.
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
