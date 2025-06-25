import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import CellRule, Individual, SimpleTimer, combine_stats
from gp import FitnessEvaluator, Runner

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", type=Path, nargs="+")
    parser.add_argument("--base_dir", type=Path)
    parser.add_argument("--human_trajectories_path", type=Path, default="human_trajectories.jsonl")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=62)

    parser.add_argument("--server_path", type=Path, default="lean-server")
    parser.add_argument("--tmp_dir", type=Path, default="tmp")
    parser.add_argument("--planning_depth", type=int, default=4)
    return parser

def extract_history(dir: Path) -> list[dict] | None:
    history = []
    population_file = dir / "population.jsonl"
    if not population_file.exists():
        return None
    with open(population_file, "r") as f:
        for line in f:
            data = json.loads(line)
            history.append(data)
    return history

def extract_human_experience(path: Path) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def evaluate_reward(
        args: argparse.Namespace,
        fitness_evaluator: FitnessEvaluator,
        leaders: list[Individual],
    ) -> list[float]:
    reward = []
    for i in tqdm(range(0, len(leaders), args.num_workers)):
        individuals = leaders[i:i + args.num_workers]
        fitness_evaluator.calculate_reward_fitness(individuals, args.num_episodes, args.num_workers)
        reward.extend([float(ind.reward) for ind in individuals])
    return reward

def evaluate_accuracy(
        args: argparse.Namespace,
        fitness_evaluator: FitnessEvaluator,
        leaders: list[Individual],
        human_experience: list[list[dict]],
    ) -> tuple[list[float], list[float]]:
    def get_state(obs: dict) -> str:
        return {
            "grid": np.array(obs["grid"]),
            "position": np.array(obs["position"]),
        }

    experience = []
    for human_trajectory in human_experience:
        for i in range(len(human_trajectory) - 1):
            experience.append({
                "state": get_state(human_trajectory[i]["obs"]),
                "next_state": get_state(human_trajectory[i + 1]["obs"]),
            })

    accuracy, f1_score = [], []
    for i in tqdm(range(0, len(leaders), args.num_workers)):
        individuals = leaders[i:i + args.num_workers]
        fitness_evaluator.calculate_observation_fitness(
            [],
            individuals,
            args.num_episodes,
            args.num_workers,
            experience,
        )
        accuracy.extend([float(ind.accuracy) for ind in individuals])
        f1_score.extend([float(ind.f_score) for ind in individuals])
    return accuracy, f1_score

def main(args: argparse.Namespace):
    args.render = False
    args.length_penalty_coeff = 0.0

    fitness_evaluator = FitnessEvaluator(args)
    human_experience = extract_human_experience(args.human_trajectories_path)

    dirs = [args.base_dir / dir if args.base_dir else dir for dir in args.dirs]
    for dir in dirs:
        print(f"Processing {dir}")
        history = extract_history(dir)
        if history is None:
            print(f"WARNING: No history data found.")
            continue

        leaders = [
            Individual.from_dict(history[i]["population"][0])
            for i in range(len(history))
        ]
        fitness = [ind.fitness for ind in leaders]
        print(f"Final fitness: {fitness[-1]}")

        print("Evaluating reward...")
        reward = evaluate_reward(args, fitness_evaluator, leaders)
        print(f"Final reward: {reward[-1]}")

        print("Evaluating accuracy...")
        accuracy, f1_score = evaluate_accuracy(args, fitness_evaluator, leaders, human_experience)
        print(f"Final accuracy: {accuracy[-1]}")
        print(f"Final F1 score: {f1_score[-1]}")

        with open(dir / "series.json", "w") as f:
            json.dump({
                "reward": reward,
                "accuracy": accuracy,
                "f1": f1_score,
            }, f)

if __name__ == "__main__":
    main(get_parser().parse_args())