import argparse
from pathlib import Path
import time

from common import CellRule, Individual, SimpleTimer, combine_stats
from gp import FitnessEvaluator, Runner

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--server_path", type=Path, default="lean-server")
    parser.add_argument("--tmp_dir", type=Path, default="tmp")

    parser.add_argument("--planning_depth", type=int, default=4)
    parser.add_argument("--render", action="store_true")

    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--objective", type=str, choices=["pragmatic", "descriptive"])

    return parser

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

def main():
    args = get_parser().parse_args()
    assert not (args.render and args.num_workers > 1)

    fitness_evaluator = FitnessEvaluator(args)
    if args.render:
        individual = Individual.create(
            rules=PERFECT_RULES,
        )
        server_path = Runner.setup_world_model(args, individual, "test")
        result = Runner.run_episode_contained(args, args.seed, server_path)
        print(f"Total reward: {result.total_reward}")
        Runner.destroy_world_model(server_path)
    else:
        individuals = [
            Individual.create(
                rules=PERFECT_RULES,
            )
            for _ in range(8)
        ]
        start_time = time.time()

        if args.objective == "pragmatic":
            fitness_evaluator.calculate_reward_fitness(
                individuals, args.num_episodes, args.num_workers
            )
        else:
            assert args.objective == "descriptive"
            fitness_evaluator.calculate_observation_fitness(
                individuals, individuals, args.num_episodes, args.num_workers
            )

        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        for individual in individuals:
            print(f"Fitness: {individual.fitness}")

if __name__ == "__main__":
    main()