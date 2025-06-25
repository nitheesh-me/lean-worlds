This repository accompanies the paper _Symbolic World Models in Lean 4 for Reinforcement Learning_
accepted to the RLC 2025 Workshop on Programmatic Reinforcement Learning.

Notable files and folders are described below:

```
.
├── common.py - Data structures for the evolutionary algorithm.
├── gp.py - Implementation of the evolutionary / genetic algorithm.
├── llm.py - The mutation function guided by a LLM.
├── world_model.py - Wrapper around lean-server to enable Python interaction.
├── mutation_prompt.txt - The mutation prompt listed in Appendix A in the paper.
├── plot.py - Script for generating various plots.
├── compute_series.py - Script for computing the time series shown in Figure 4 in the paper.
├── eval.py - Script for evaluating a specific world model.
├── requirements.txt
└── lean-server - The Lean server for executing the synthesized world models.
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
