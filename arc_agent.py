"""
lean-worlds × ARC-AGI-3 integration
=====================================
An ARC-AGI-3 compatible ``Agent`` subclass that uses lean-worlds' symbolic Lean 4
world model for look-ahead planning.

Setup
-----
1. Clone both repos side-by-side (or adjust LEAN_WORLDS_PATH below):

       git clone https://github.com/nitheesh-me/lean-worlds.git
       git clone https://github.com/arcprize/ARC-AGI-3-Agents.git

2. Build the Lean server (once):

       cd lean-worlds/lean-server && lake build

3. Copy *this* file into the ARC-AGI-3-Agents agent templates:

       cp lean-worlds/arc_agent.py ARC-AGI-3-Agents/agents/templates/lean_worlds_agent.py

4. Register the agent in ``ARC-AGI-3-Agents/agents/__init__.py`` by adding:

       from .templates.lean_worlds_agent import LeanWorldsAgent  # noqa: F401

5. Run:

       cd ARC-AGI-3-Agents
       python main.py --agent=leanworldsagent --game=<game_id>

Environment variables
---------------------
LEAN_WORLDS_PATH  Absolute path to the lean-worlds checkout.
                  Defaults to ``../lean-worlds`` relative to this file.
LEAN_SERVER_PATH  Path to the lean-server directory inside lean-worlds.
                  Defaults to ``<LEAN_WORLDS_PATH>/lean-server``.
LEAN_TMP_DIR      Temporary directory for compiled Lean servers.
                  Defaults to ``<LEAN_WORLDS_PATH>/tmp``.
LEAN_PLANNING_DEPTH  Look-ahead depth for the planning agent (default: 4).

Game compatibility
------------------
This agent is designed for **cellular-automaton grid games** whose observation
is a 2-D integer grid where:
  • ``frame[0]`` is a 2-D grid of cell-type integers,
  • ``frame[1]`` (optional) is a 2-D layer whose non-zero value indicates the
    agent's (row, col) position.

The default cell mapping assumes the game's values match the ForestFire
encoding: 0=EMPTY, 1=TREE, 2=FIRE_1, 3=FIRE_2, 4=FIRE_3, 5=ROCK.
Override ``CELL_MAP`` in a subclass to adapt to your game.

The default action mapping:
  lean action 0 → ACTION5 (STAY / no-op)
  lean action 1 → ACTION1 (UP)
  lean action 2 → ACTION2 (DOWN)
  lean action 3 → ACTION3 (LEFT)
  lean action 4 → ACTION4 (RIGHT)

Override ``ACTION_MAP`` in a subclass to adjust.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Locate the lean-worlds repo and add it to sys.path so that lean-worlds
# modules (common, world_model, gp, …) are importable from here.
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent

def _find_lean_worlds() -> Path:
    env_path = os.environ.get("LEAN_WORLDS_PATH")
    if env_path:
        p = Path(env_path).resolve()
        if p.is_dir():
            return p
        raise RuntimeError(f"LEAN_WORLDS_PATH={env_path!r} does not exist.")
    # Try common relative locations.
    candidates = [
        _THIS_DIR,                   # this file already lives inside lean-worlds
        _THIS_DIR / ".." / "lean-worlds",
        _THIS_DIR / ".." / ".." / "lean-worlds",
    ]
    for c in candidates:
        if (c / "world_model.py").exists():
            return c.resolve()
    raise RuntimeError(
        "Cannot locate lean-worlds directory.  "
        "Set the LEAN_WORLDS_PATH environment variable to its absolute path."
    )

_LEAN_WORLDS = _find_lean_worlds()
if str(_LEAN_WORLDS) not in sys.path:
    sys.path.insert(0, str(_LEAN_WORLDS))

# ---------------------------------------------------------------------------
# Now import lean-worlds modules (available after the sys.path tweak above).
# ---------------------------------------------------------------------------
from common import CellRule, Individual  # noqa: E402
from gp import Runner                    # noqa: E402
from world_model import FormalizedWorldModel  # noqa: E402

# ---------------------------------------------------------------------------
# Import ARC-AGI-3 / arcengine types.  These come from the arc-agi / arcengine
# packages which must be installed (pip install arc-agi).
# ---------------------------------------------------------------------------
from arcengine import FrameData, GameAction, GameState  # noqa: E402

# The Agent base class lives in the ARC-AGI-3-Agents repo, not a package.
# It will be importable when this file is used *inside* that repo.
try:
    from agents.agent import Agent  # type: ignore[import]
except ImportError as exc:
    raise ImportError(
        "Could not import 'agents.agent'.  Make sure this file is placed inside "
        "the ARC-AGI-3-Agents repository under agents/templates/ and that the "
        "repository root is on sys.path (it is when you run main.py from there)."
    ) from exc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# The "perfect" ForestFire rules shipped with lean-worlds.
PERFECT_RULES: list[CellRule] = [
    CellRule(id="empty_1",  body="g.get c = FIRE_3",                                          cell_type=0),
    CellRule(id="empty_2",  body="g.get c = FIRE_2 ∧ c = a",                                  cell_type=0),
    CellRule(id="empty_3",  body="g.get c = FIRE_1 ∧ c = a",                                  cell_type=0),
    CellRule(id="tree_1",   body="g.get c = EMPTY ∧ exists_foo c (fun c => g.get c = TREE)",  cell_type=1),
    CellRule(id="fire1_1",  body="g.get c = TREE ∧ exists_foo c (fun c => g.get c = FIRE_3)", cell_type=2),
    CellRule(id="fire2_1",  body="g.get c = FIRE_1 ∧ c ≠ a",                                  cell_type=3),
    CellRule(id="fire3_1",  body="g.get c = FIRE_2 ∧ c ≠ a",                                  cell_type=4),
]


class LeanWorldsAgent(Agent):
    """
    ARC-AGI-3 agent that uses a symbolic Lean 4 world model for look-ahead planning.

    The agent compiles the Lean world model the first time ``choose_action`` is
    called (or on ``__init__`` if ``eager=True``), then uses ``PlanningAgent``
    from *gym_cellular* to pick actions via look-ahead search.

    Subclass and override ``CELL_MAP`` / ``ACTION_MAP`` / ``rules`` to adapt
    to a different game or a differently-discovered rule set.
    """

    # Overrides Agent.MAX_ACTIONS (default 80).  The Lean planning agent is
    # slower than a random agent, so we allow more steps before a forced stop.
    MAX_ACTIONS: int = 200

    # ── configurable class-level attributes ──────────────────────────────────

    # ARC-AGI-3 cell value → lean-worlds cell type (0-5).
    # Identity mapping works out-of-the-box for ForestFire-compatible games.
    CELL_MAP: dict[int, int] = {i: i for i in range(16)}

    # lean-worlds action index → ARC-AGI-3 GameAction.
    # Adjust if your game uses different action semantics.
    ACTION_MAP: dict[int, GameAction] = {
        0: GameAction.ACTION5,  # STAY / no-op
        1: GameAction.ACTION1,  # UP
        2: GameAction.ACTION2,  # DOWN
        3: GameAction.ACTION3,  # LEFT
        4: GameAction.ACTION4,  # RIGHT
    }

    # World model rules.  Swap in evolved rules from gp.py to improve play.
    rules: list[CellRule] = PERFECT_RULES

    # Grid dimensions (must match the Lean server configuration).
    GRID_HEIGHT: int = 10
    GRID_WIDTH: int = 10
    PLANNING_DEPTH: int = int(os.environ.get("LEAN_PLANNING_DEPTH", "4"))

    # ── init ─────────────────────────────────────────────────────────────────

    def __init__(
        self,
        *args: Any,
        eager: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._server_path = Path(
            os.environ.get("LEAN_SERVER_PATH", str(_LEAN_WORLDS / "lean-server"))
        )
        self._tmp_dir = Path(
            os.environ.get("LEAN_TMP_DIR", str(_LEAN_WORLDS / "tmp"))
        )
        self._world_model: Optional[FormalizedWorldModel] = None
        self._planning_agent: Any = None
        self._compiled_path: Optional[Path] = None
        self._agent_pos: np.ndarray = np.array([0, 0])
        if eager:
            self._setup()

    # ── private helpers ───────────────────────────────────────────────────────

    def _setup(self) -> None:
        """Compile the Lean server and initialise the planning agent (once)."""
        if self._planning_agent is not None:
            return

        try:
            from gym_cellular.agent.planner import OracleWorldModel, PlanningAgent
            from gym_cellular.cellular.forest_fire import ForestFire
        except ImportError as exc:
            raise ImportError(
                "gym_cellular is not installed.  "
                "Install it from its source directory: pip install -e <path/to/gym_cellular>"
            ) from exc

        import argparse as _ap
        fake_args = _ap.Namespace(
            server_path=self._server_path,
            tmp_dir=self._tmp_dir,
        )
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

        individual = Individual.create(rules=self.rules)

        # Reuse an already-compiled server if the rules haven't changed.
        rule_hash = abs(hash(tuple(r.body + str(r.cell_type) for r in self.rules)))
        cached = self._tmp_dir / f"arc_lean_{rule_hash}"
        if cached.exists():
            logger.info("Reusing cached Lean server at %s", cached)
            self._compiled_path = cached
        else:
            logger.info("Compiling Lean server (this may take a minute)…")
            self._compiled_path = Runner.setup_world_model(fake_args, individual, "arc_lean_")
            # Rename to the deterministic hash path for future reuse.
            try:
                self._compiled_path.rename(cached)
                self._compiled_path = cached
            except OSError:
                pass  # rename failed (e.g., cross-device) — keep original path

        oracle = OracleWorldModel(ForestFire(self.GRID_HEIGHT, self.GRID_WIDTH))
        self._world_model = FormalizedWorldModel(
            path=self._compiled_path,
            oracle_model=oracle,
        )
        self._world_model.start()

        self._planning_agent = PlanningAgent(
            depth=self.PLANNING_DEPTH,
            world_model=self._world_model,
            height=self.GRID_HEIGHT,
            width=self.GRID_WIDTH,
        )
        logger.info("LeanWorldsAgent ready (planning depth=%d).", self.PLANNING_DEPTH)

    def _extract_grid_and_pos(
        self, latest_frame: FrameData
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract a (H, W) integer grid and agent position from ``FrameData``.

        Frame layout expectations:
          frame[0] — 2-D grid of cell-type integers.
          frame[1] — (optional) 2-D layer; first non-zero cell gives agent pos.

        Falls back to the last known agent position if detection fails.
        """
        if not latest_frame.frame:
            return (
                np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.int32),
                self._agent_pos,
            )

        raw = np.array(latest_frame.frame[0], dtype=np.int32)
        # Remap cell values through CELL_MAP (vectorised).
        # Build a lookup array sized to cover all keys; values outside CELL_MAP
        # are clamped to the highest key and then mapped accordingly.
        max_key = max(self.CELL_MAP.keys())
        cell_map_arr = np.zeros(max_key + 1, dtype=np.int32)
        for src, dst in self.CELL_MAP.items():
            cell_map_arr[src] = dst
        clipped = np.clip(raw, 0, max_key)
        grid = cell_map_arr[clipped]

        # Agent position from a dedicated second layer, if present.
        if len(latest_frame.frame) > 1:
            agent_layer = np.array(latest_frame.frame[1], dtype=np.int32)
            positions = np.argwhere(agent_layer > 0)
            if len(positions) > 0:
                self._agent_pos = positions[0]  # [row, col]
        else:
            # Fall back: look for a cell value that is NOT a valid terrain type.
            # (Some games encode the agent as a distinct integer in the main grid.)
            agent_val = max(self.CELL_MAP.keys()) + 1
            positions = np.argwhere(raw == agent_val)
            if len(positions) > 0:
                self._agent_pos = positions[0]

        return grid, self._agent_pos

    # ── Agent interface ───────────────────────────────────────────────────────

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """The agent is done when the game is won."""
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """
        Use lean-worlds planning to select the next action.

        On the first call the Lean server is compiled and the planning agent is
        initialised, which takes ~30–60 s.  Subsequent calls are fast.
        """
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        self._setup()  # no-op after the first call

        grid, agent_pos = self._extract_grid_and_pos(latest_frame)
        lean_action: int = self._planning_agent.select_action(grid, agent_pos)
        game_action: GameAction = self.ACTION_MAP.get(lean_action, GameAction.ACTION1)
        game_action.reasoning = (
            f"lean-worlds PlanningAgent (depth={self.PLANNING_DEPTH}) "
            f"selected lean action {lean_action} → {game_action.name}"
        )
        logger.debug(
            "Agent pos %s → lean action %d → %s",
            agent_pos.tolist(),
            lean_action,
            game_action.name,
        )
        return game_action

    def cleanup(self, scorecard: Any = None) -> None:
        """Stop the Lean server process before exiting."""
        if self._world_model is not None:
            try:
                self._world_model.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error stopping Lean world model: %s", exc)
            self._world_model = None
            self._planning_agent = None
        super().cleanup(scorecard)
