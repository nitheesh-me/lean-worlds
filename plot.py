import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import CellRule


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="action", required=True)
    fitness = subparsers.add_parser("fitness")
    algorithms = subparsers.add_parser("algorithms")
    show_best = subparsers.add_parser("show_best")
    show_perfect = subparsers.add_parser("show_perfect")
    plot_series = subparsers.add_parser("plot_series")

    for subparser in [fitness, algorithms]:
        subparser.add_argument("dirs", type=Path, nargs="+")
        subparser.add_argument("--base_dir", type=Path, default=".")

    show_best.add_argument("dir", type=Path)
    plot_series.add_argument("dir", type=Path)

    return parser

def extract_args(dir: Path) -> dict:
    args_file = dir / "args.json"
    if not args_file.exists():
        return None
    with open(args_file, "r") as f:
        return json.load(f)

def extract_fitness(dir: Path) -> list[float] | None:
    fitness_values = []
    population_file = dir / "population.jsonl"
    if not population_file.exists():
        return None
    with open(population_file, "r") as f:
        for line in f:
            data = json.loads(line)
            fitness_values.append(data["best_fitness"])
    assert fitness_values[0] is None
    return fitness_values[1:]

def plot_fitness(data: list[float], out_file: Path):
    plt.figure(figsize=(4, 4))
    plt.plot(data)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.tight_layout()
    print(f"Saving to {out_file}")
    plt.savefig(out_file)
    plt.close()

def run_fitness(args: argparse.Namespace):
    dirs = [args.base_dir / dir for dir in args.dirs]

    for dir in dirs:
        print(f"Processing {dir}")
        fitness = extract_fitness(dir)
        if fitness is None:
            print(f"WARNING: No fitness data found.")
            continue
        print(", ".join(map(str, fitness)))
        plot_fitness(fitness, dir / "fitness.pdf")

def run_algorithms(args: argparse.Namespace):
    dirs = [args.base_dir / dir for dir in args.dirs]
    data = {
        "classic": [],
        "simple": [],
        "unknown": [],
    }
    for dir in dirs:
        print(f"Processing {dir}")
        run_args = extract_args(dir)
        if run_args is None:
            print(f"WARNING: No args found.")
            continue
        fitness = extract_fitness(dir)
        if fitness is None:
            print(f"WARNING: No fitness data found.")
            continue
        if len(fitness) == 0:
            print(f"WARNING: Empty fitness data.")
            continue
        print(f"  Fitness: {fitness[-1]}")
        algo = run_args.get("algorithm", "unknown")
        data[algo].append(fitness[-1])
    
    for algo in ["classic", "simple"]:
        print(f"\nAlgorithm: {algo.capitalize()}")
        if len(data[algo]) == 0:
            print("  No data available.")
            continue
        samples = np.array(data[algo])
        print("  Samples:")
        for i, val in enumerate(samples):
            print(f"    Run {i+1}: {val}")
        mean = np.mean(samples)
        std = np.std(samples)
        print(f"  Mean: {mean}")
        print(f"  Std:  {std}")

cell_type_to_name = {
    0: "E",
    1: "T",
    2: "F1",
    3: "F2",
    4: "F3",
    5: "R",
}

def format_rule(rule: dict):
    body, cell_type = rule["body"], rule["cell_type"]
    for key, value in cell_type_to_name.items():
        body = body.replace(str(key), value)
    body = body.replace("g.get", "grid")
    body = body.replace(" c ", " p ")
    body = body.replace("exists_foo", "neighbor")
    return f"{cell_type_to_name[cell_type]} ← {body}"

def format_rule_latex(rule: dict):
    s = format_rule(rule)
    s = s.replace("←", "&\\leftarrow")
    s = s.replace("∧", "\\land")
    s = s.replace("∨", "\\lor")
    s = s.replace("¬", "\\neg")
    s = s.replace("F1", "\\text{F}_1")
    s = s.replace("F2", "\\text{F}_2")
    s = s.replace("F3", "\\text{F}_3")
    s = s.replace("T", "\\text{T}")
    s = s.replace("E", "\\text{E}")
    s = s.replace("R", "\\text{R}")
    s = s.replace("grid", "\\text{grid }")
    s = s.replace("neighbor", "\\text{neighbor }")
    s = s.replace("fun", "\\lambda")
    s = s.replace("=>", "\\mapsto")
    s = s.replace("true", "\\text{true }")
    s = s.replace("≠", "\\neq")
    return s

def run_show_best(args: argparse.Namespace):
    population_file = args.dir / "population.jsonl"
    if not population_file.exists():
        print(f"ERROR: No population file found.")
        return
    # Read the last line of the population file
    with open(population_file, "r") as f:
        lines = f.readlines()
        if not lines:
            print("ERROR: Population file is empty.")
            return
        last_line = lines[-1]
    data = json.loads(last_line)
    best = max(data["population"], key=lambda ind: ind["fitness"])
    print("Best individual in the last generation:")
    print(f"Fitness: {best['fitness']}")
    print("Raw rules:")
    for rule in best["rules"]:
        print(f"{rule['cell_type']}: {rule['body']}")

    print(f"Rules:")
    for rule in best["rules"]:
        print(format_rule(rule))
    
    print("\nLaTeX:")
    print("\\\\\n".join(map(format_rule_latex, best["rules"])))

PERFECT_RULES = [
    CellRule(
        id="empty_1",
        body="g.get c = FIRE_3",
        cell_type=0,
    ),
    CellRule(
        id="empty_2",
        body="g.get c = FIRE_2 ∧ c = a",
        cell_type=0,
    ),
    CellRule(
        id="empty_3",
        body="g.get c = FIRE_1 ∧ c = a",
        cell_type=0,
    ),
    CellRule(
        id="tree_1",
        body="g.get c = EMPTY ∧ exists_foo c (fun c => g.get c = TREE)",
        cell_type=1,
    ),
    CellRule(
        id="fire1_1",
        body="g.get c = TREE ∧ exists_foo c (fun c => g.get c = FIRE_3)",
        cell_type=2,
    ),
    CellRule(
        id="fire2_1",
        body="g.get c = FIRE_1 ∧ c ≠ a",
        cell_type=3,
    ),
    CellRule(
        id="fire3_1",
        body="g.get c = FIRE_2 ∧ c ≠ a",
        cell_type=4,
    ),
]

def run_show_perfect(args: argparse.Namespace):
    rules = [rule.to_dict() for rule in PERFECT_RULES]
    for rule in rules:
        rule["body"] = rule["body"].replace("EMPTY", "0")
        rule["body"] = rule["body"].replace("TREE", "1")
        rule["body"] = rule["body"].replace("FIRE_1", "2")
        rule["body"] = rule["body"].replace("FIRE_2", "3")
        rule["body"] = rule["body"].replace("FIRE_3", "4")


    for rule in rules:
        print(format_rule(rule))

    print("\nLaTeX:")
    print("\\\\\n".join(map(format_rule_latex, rules)))

def run_plot_series(args: argparse.Namespace):
    series_file = args.dir / "series.json"
    with open(series_file, "r") as f:
        data = json.load(f)
    
    reward = data["reward"]
    f_score = data["f1"]
    
    fig, ax1 = plt.subplots(figsize=(4, 4))
    
    # Plot reward on left y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Reward', color=color1)
    ax1.plot(reward, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Create second y-axis for f_score
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('F Score', color=color2)
    ax2.plot(f_score, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.tight_layout()
    out_file = args.dir / "series.pdf"
    print(f"Saving to {out_file}")
    plt.savefig(out_file)
    plt.close()

def main(args: argparse.Namespace):
    if args.action == "fitness":
        run_fitness(args)
    elif args.action == "algorithms":
        run_algorithms(args)
    elif args.action == "show_best":
        run_show_best(args)
    elif args.action == "show_perfect":
        run_show_perfect(args)
    elif args.action == "plot_series":
        run_plot_series(args)
    else:
        raise ValueError(f"Unknown action: {args.action}")

if __name__ == "__main__":
    main(get_parser().parse_args())