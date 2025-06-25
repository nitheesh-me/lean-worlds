import argparse
import json
import multiprocessing
import select
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import itertools

import gymnasium as gym
import numpy as np
from gym_cellular.agent.planner import PlanningAgent, OracleWorldModel
from gym_cellular.cellular.forest_fire import ForestFire

from common import Logger, setup_seeds, get_args_descriptor, dump_args, CellRule, Individual, SimpleTimer
from llm import ModelProvider
from world_model import FormalizedWorldModel

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--max_gpus", type=int, default=0)
    parser.add_argument("--objective", type=str, choices=["pragmatic", "descriptive"], required=True)

    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--min_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)

    parser.add_argument("--output_dir", type=Path, default="out")

    parser.add_argument("--server_path", type=Path, default="lean-server")
    parser.add_argument("--tmp_dir", type=Path, default="tmp")

    parser.add_argument("--planning_depth", type=int, default=4)
    parser.add_argument("--render", action="store_true")

    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument("--num_generations", type=int, default=500)
    parser.add_argument("--population_size", type=int, default=16)
    parser.add_argument("--ruleset_size", type=int, default=5)
    parser.add_argument("--num_fitness_episodes", type=int, default=5)
    parser.add_argument("--replace_with_random_rule_prob", type=float, default=0.2)
    parser.add_argument("--change_cell_type_prob", type=float, default=0.2)
    parser.add_argument("--crossover_prob", type=float, default=0.5)
    parser.add_argument("--length_penalty_coeff", type=float, default=0.0001)
    parser.add_argument("--elitism", type=int, default=4)

    parser.add_argument("--algorithm", type=str, choices=["classic", "simple"], default="classic")

    return parser


ARGS_WHITELIST = [
    "seed", "checkpoint", "objective", "max_new_tokens", "min_new_tokens", "temperature", "top_p", "top_k",
    "num_generations", "population_size", "ruleset_size", "num_fitness_episodes", "elitism",
    "replace_with_random_rule_prob", "change_cell_type_prob", "crossover_prob",
    "planning_depth", "algorithm",
]


@dataclass
class EpisodeResult:
    rewards: list[float]
    timer_data: dict
    history: list[dict] | None

    @property
    def total_reward(self):
        return sum(self.rewards)


class Runner:
    @staticmethod
    def setup_world_model(args: argparse.Namespace, individual: Individual, folder_prefix: str) -> Path:
        path = args.tmp_dir / (folder_prefix + individual.id)
        FormalizedWorldModel.setup_server(args.server_path, path, individual.rules)
        return path

    @staticmethod
    def destroy_world_model(path: Path):
        FormalizedWorldModel.destroy_server(path)

    @staticmethod
    def run_episode_contained(args: argparse.Namespace, seed: int, server_path: Path,
                              store_history: bool = False) -> EpisodeResult:
        # Create and initialize world model.
        oracle_model = OracleWorldModel(ForestFire(10, 10))
        world_model = FormalizedWorldModel(
            path=server_path,
            oracle_model=oracle_model,
        )
        world_model.start()

        try:
            agent = PlanningAgent(
                depth=args.planning_depth,
                world_model=world_model,
                height=10,
                width=10,
            )
            env = gym.make(
                "HelicopterCellularAutomaton-v0",
                seed=seed,
                render_mode="human" if args.render else None,
            )
            return Runner.run_episode(env, agent, args.render, store_history)
        finally:
            world_model.stop()

    @staticmethod
    def run_episode(env: gym.Env, agent: PlanningAgent, render: bool, store_history: bool = False) -> EpisodeResult:
        terminated, truncated = False, False
        rewards = []
        history = []

        obs, _ = env.reset()

        if render:
            # Force an initial render so that pygame is initialized.
            env.render()

        timer = SimpleTimer()
        while not (terminated or truncated):
            current_grid = obs["grid"]
            agent_pos = obs["position"]

            timer.start_section("agent", do_print=False)
            action = agent.select_action(current_grid, agent_pos)
            timer.end_section(do_print=False)

            timer.start_section("environment", do_print=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            timer.end_section(do_print=False)
            rewards.append(reward)

            if store_history:
                history.append(obs)

            if render:
                import pygame
                env.render()
                # Essential to call `pygame.event.pump()` so the window stays responsive.
                pygame.event.pump()

        return EpisodeResult(rewards, timer.get_stats(), history if store_history else None)

    @staticmethod
    def evaluate_world_model(server_path: Path, experience: list[dict]) -> tuple[float, float]:
        world_model = FormalizedWorldModel(
            path=server_path,
            oracle_model=OracleWorldModel(ForestFire(10, 10)),
        )
        world_model.start()

        try:
            predictions = world_model.predict_multiple([sample["state"] for sample in experience])
        finally:
            world_model.stop()

        tp, fp, fn = 0, 0, 0
        tn = 0
        for sample, predicted_grid in zip(experience, predictions):
            old_grid = sample["state"]["grid"]
            next_grid = sample["next_state"]["grid"]

            correct = predicted_grid == next_grid
            pred_changed = predicted_grid != old_grid

            tp += np.sum(correct & pred_changed)
            fp += np.sum(~correct & pred_changed)
            fn += np.sum(~correct & ~pred_changed)
            tn += np.sum(correct & ~pred_changed)

        f_score = 2 * tp / (2 * tp + fp + fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return f_score, accuracy


class FitnessEvaluator:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def _setup_world_models(self, pool: multiprocessing.Pool, individuals: list[Individual]) -> list[Path]:
        return pool.starmap(Runner.setup_world_model, [
            (
                self.args,
                individual,
                # This is generated in the main process because otherwise all the processes would generate the same prefix.
                "server_" + "".join(np.random.choice(list("0123456789"), size=8)) + "-",
            )
            for individual in individuals
        ])

    def _run_episodes(self, pool: multiprocessing.Pool, server_paths: list[Path], num_episodes: int,
                      store_history: bool = False) -> list[list[EpisodeResult]]:
        parameters = []
        for server_path in server_paths:
            # Use the same sequence of seeds for each evaluation.
            local_rng = np.random.RandomState(0)
            for _ in range(num_episodes):
                parameters.append((self.args, local_rng.randint(0, 1000000), server_path, store_history))

        all_results = pool.starmap(Runner.run_episode_contained, parameters)
        all_results = [
            all_results[i:i + num_episodes]
            for i in range(0, len(all_results), num_episodes)
        ]
        return all_results

    def _destroy_world_models(self, pool: multiprocessing.Pool, server_paths: list[Path]):
        pool.map(Runner.destroy_world_model, server_paths)

    def calculate_reward_fitness(self, individuals: list[Individual], num_episodes: int, num_workers: int):
        with multiprocessing.Pool(num_workers) as pool:
            server_paths = self._setup_world_models(pool, individuals)
            all_results = self._run_episodes(pool, server_paths, num_episodes)
            self._destroy_world_models(pool, server_paths)

        assert len(all_results) == len(individuals)
        for individual, samples in zip(individuals, all_results):
            reward = sum(sample.total_reward for sample in samples) / len(samples)
            length_penalty = 1 - (self.args.length_penalty_coeff * sum(len(rule.body) for rule in individual.rules))
            individual.fitness = reward * length_penalty

            individual.reward = reward

    def calculate_observation_fitness(
            self,
            population: list[Individual],
            to_evaluate: list[Individual],
            num_episodes: int,
            num_workers: int,
            experience: list[dict] | None = None,
    ):
        with multiprocessing.Pool(num_workers) as pool:
            if experience is None:
                server_paths = self._setup_world_models(pool, population)
                all_results = self._run_episodes(pool, server_paths, num_episodes, store_history=True)
                self._destroy_world_models(pool, server_paths)

                experience = []
                for results in all_results:
                    for result in results:
                        for i in range(len(result.history) - 1):
                            experience.append({
                                "state": result.history[i],
                                "next_state": result.history[i + 1],
                            })

            server_paths = self._setup_world_models(pool, to_evaluate)
            all_results = [
                Runner.evaluate_world_model(server_path, experience)
                for server_path in server_paths
            ]
            self._destroy_world_models(pool, server_paths)

            for individual, (f_score, accuracy) in zip(to_evaluate, all_results):
                length_penalty = 1 - (self.args.length_penalty_coeff * sum(len(rule.body) for rule in individual.rules))
                individual.fitness = f_score * length_penalty

                individual.f_score = f_score
                individual.accuracy = accuracy


class EvolutionaryAlgorithm:
    def __init__(
            self,
            args: argparse.Namespace,
            fitness_evaluator: FitnessEvaluator,
            model_provider: ModelProvider,
    ):
        self.args = args
        self.fitness_evaluator = fitness_evaluator
        self.model_provider = model_provider
        self.rng = np.random.RandomState(self.args.seed)
        self.timer = SimpleTimer()

        self.lean_repl = LeanRepl()
        self.lean_repl.start()

        self.population = self._random_population()
        Logger.Instance().log_population(self.population, best_fitness=None)

    def _eval_all(self, population: list[Individual], to_evaluate: list[Individual]):
        self.timer.start_section("fitness_eval")

        if self.args.objective == "pragmatic":
            self.fitness_evaluator.calculate_reward_fitness(
                to_evaluate, self.args.num_fitness_episodes, self.args.num_workers
            )
        else:
            assert self.args.objective == "descriptive"
            self.fitness_evaluator.calculate_observation_fitness(
                population, to_evaluate, self.args.num_fitness_episodes, self.args.num_workers
            )

        self.timer.end_section()

    def _random_population(self):
        population = [self._random_individual() for _ in range(self.args.population_size)]

        self._eval_all(population, to_evaluate=population)
        return population

    def _random_individual(self):
        return Individual.create(rules=[self._random_rule() for _ in range(self.args.ruleset_size)])

    def _random_rule(self):
        return CellRule.create(f"g.get c = {self.rng.randint(0, 5)}", self.rng.randint(0, 5), parent_id=None)

    def step(self):
        population = self.population

        assert all(individual.fitness is not None for individual in population)
        population = list(sorted(population, key=lambda individual: individual.fitness, reverse=True))

        # Selection: The worst individual is discarded. A random one is born instead.
        parents = population[:-1]
        random_child = self._random_individual()

        mutations_indices = [self.rng.randint(0, len(par.rules)) for par in parents]
        self.timer.start_section("mutating")
        all_mutations = self.model_provider.mutate([par.rules[idx] for par, idx in zip(parents, mutations_indices)])
        self.timer.end_section()

        self.timer.start_section("validating")
        all_mutations = self._filter_valid_mutations(
            [par.rules[idx] for par, idx in zip(parents, mutations_indices)],
            all_mutations,
        )
        self.timer.end_section()

        offspring = []
        for i, par in enumerate(parents):
            if len(all_mutations[i]) == 0:
                child = par.clone()
            else:
                mutated = self.rng.choice(all_mutations[i])
                child = Individual.create(
                    rules=par.rules[:mutations_indices[i]] + [mutated] + par.rules[mutations_indices[i] + 1:],
                    parent_ids=[par.id],
                )

            if self.rng.rand() < self.args.replace_with_random_rule_prob:
                to_replace = self.rng.randint(0, len(child.rules))
                child.rules[to_replace] = self._random_rule()
            if self.rng.rand() < self.args.change_cell_type_prob:
                to_change = self.rng.randint(0, len(child.rules))
                child.rules[to_change] = CellRule.create(
                    child.rules[to_change].body,
                    self.rng.randint(0, 5),
                    parent_id=child.rules[to_change].id,
                )
            # Kind of one-way crossover.
            if self.rng.rand() < self.args.crossover_prob:
                to_crossover = self.rng.randint(0, len(child.rules))
                donor = parents[self.rng.randint(0, len(parents))]
                child.rules[to_crossover] = donor.rules[to_crossover]

            offspring.append(child)

        assert len(offspring) == len(parents)
        self._eval_all(population, to_evaluate=offspring + [random_child])
        new_population = [
            offspring[i] if offspring[i].fitness > parents[i].fitness else parents[i]
            for i in range(len(parents))
        ]
        new_population.append(random_child)

        assert all(individual.fitness is not None for individual in new_population)
        new_population = list(sorted(new_population, key=lambda individual: individual.fitness, reverse=True))
        Logger.Instance().log_population(new_population, best_fitness=new_population[0].fitness)

        self.population = new_population

    def _get_selection_probs(self, population: list[Individual], temperature: float) -> list[float]:
        values = [ind.fitness ** (1 / temperature) for ind in population]
        total_value = sum(values)
        return [v / total_value for v in values]

    def _sus_selection(
        self,
        population: list[Individual],
        n_select: int,
        selection_probs: list[float]
    ) -> list[Individual]:
        assert len(population) == len(selection_probs)
        total_fitness = sum(selection_probs)
        assert total_fitness > 0

        pointer_spacing = total_fitness / n_select

        start = self.rng.uniform(0.0, pointer_spacing)
        pointers = [start + i * pointer_spacing for i in range(n_select)]

        cum_fitness_iter = itertools.accumulate(selection_probs)
        selected = []
        current_cum = next(cum_fitness_iter)
        idx = 0

        for pointer in pointers:
            # Advance until cumulative fitness >= pointer
            while current_cum < pointer:
                idx += 1
                current_cum = next(cum_fitness_iter)
            selected.append(population[idx])

        return selected

    def step2(self):
        population = self.population

        assert all(individual.fitness is not None for individual in population)
        population = list(sorted(population, key=lambda individual: individual.fitness, reverse=True))
        
        # Apply elitism - keep the best individuals.
        elites = population[:self.args.elitism]
        
        selection_probs = self._get_selection_probs(population, temperature=2.0)
        n_select = self.args.population_size - len(elites)
        parents = self._sus_selection(population, n_select, selection_probs)
        
        # Create offspring through crossover and mutation.
        offspring = []
        
        # Create pairs for crossover.
        self.rng.shuffle(parents)
        parent_pairs = [(parents[i], parents[i+1]) for i in range(0, len(parents)-1, 2)]
        if len(parents) % 2 == 1:
            parent_pairs.append((parents[-1], parents[self.rng.randint(0, len(parents)-1)]))
        
        for parent1, parent2 in parent_pairs:
            # One-point crossover
            if self.rng.random() < self.args.crossover_prob:
                # Uniform crossover: for each rule position, randomly select from parent1 or parent2
                child1_rules = []
                child2_rules = []
                for r1, r2 in zip(parent1.rules, parent2.rules):
                    if self.rng.rand() < 0.5:
                        child1_rules.append(r1)
                        child2_rules.append(r2)
                    else:
                        child1_rules.append(r2)
                        child2_rules.append(r1)
                child1 = Individual.create(rules=child1_rules, parent_ids=[parent1.id, parent2.id])
                child2 = Individual.create(rules=child2_rules, parent_ids=[parent1.id, parent2.id])
            else:
                child1 = parent1.clone()
                child2 = parent2.clone()
            
            offspring.extend([child1, child2])
        
        assert len(offspring) == n_select
        
        # Apply mutation to offspring.
        # First, collect rules to mutate (one randomly selected rule from each offspring).
        mutation_indices = [self.rng.randint(0, len(child.rules)) for child in offspring]
        
        self.timer.start_section("mutating")
        all_mutations = self.model_provider.mutate([child.rules[idx] for child, idx in zip(offspring, mutation_indices)])
        self.timer.end_section()
        
        self.timer.start_section("validating")
        all_mutations = self._filter_valid_mutations(
            [child.rules[idx] for child, idx in zip(offspring, mutation_indices)],
            all_mutations
        )
        self.timer.end_section()
        
        # Apply mutations.
        for i, child in enumerate(offspring):
            # If at least one valid mutation was generated, apply one randomly
            if len(all_mutations[i]) > 0:
                mutated = self.rng.choice(all_mutations[i])
                child.rules[mutation_indices[i]] = mutated
            
            # Random rule replacement with small probability
            if self.rng.rand() < self.args.replace_with_random_rule_prob:
                to_replace = self.rng.randint(0, len(child.rules))
                child.rules[to_replace] = self._random_rule()
                
            # Cell type change with small probability
            if self.rng.rand() < self.args.change_cell_type_prob:
                to_change = self.rng.randint(0, len(child.rules))
                child.rules[to_change] = CellRule.create(
                    child.rules[to_change].body,
                    self.rng.randint(0, 5),
                    parent_id=child.rules[to_change].id,
                )
        
        self._eval_all(population, to_evaluate=offspring)
        
        new_population = elites + offspring
        assert all(individual.fitness is not None for individual in new_population)
        new_population = list(sorted(new_population, key=lambda individual: individual.fitness, reverse=True))
        Logger.Instance().log_population(new_population, best_fitness=new_population[0].fitness)
        
        self.population = new_population

    def _filter_valid_mutations(
            self, sources: list[CellRule], all_mutations: list[list[CellRule]]
    ) -> list[list[CellRule]]:
        result = []
        for source, mutations in zip(sources, all_mutations):
            valid = []
            invalid = []
            for mutation in mutations:
                error = self.lean_repl.check_rule_valid(mutation)
                if error is None:
                    valid.append(mutation)
                else:
                    invalid.append((mutation, error))

            Logger.Instance().log_mutations(source, valid)
            Logger.Instance().log_invalid_mutations(source, invalid)
            Logger.Instance().total_mutations += len(mutations)
            Logger.Instance().invalid_mutations += len(invalid)

            result.append(valid)
        return result

    def run(self):
        for i in range(self.args.num_generations):
            print(f"=== Generation {i} ===")
            old_population = self.population
            if self.args.algorithm == "classic":
                self.step2()
            else:
                assert self.args.algorithm == "simple"
                self.step()
            print("== done ==")
            # print(f"Best individual:\n{self.population[0]}")
            print("Population:")
            for individual in reversed(self.population):
                print(f"{individual}")
                print("-")
            print("----")
            Logger.Instance().log_stats()
            Logger.Instance().log_evolution_step(i, old_population, self.population)
            print()
            self.timer.print_stats()
            print("-" * 80)


class LeanRepl:
    def __init__(self, suppress_output: bool = True):
        self.process = None
        self.suppress_output = suppress_output
        self.prelude = Path("lean-server/Server/Common.lean").read_text()
        self.env_id = None

    @property
    def started(self):
        return self.process is not None

    def start(self):
        if self.started:
            raise RuntimeError("REPL already started.")
        cmd = ["lake", "env", str(self.get_repl_path() / ".lake/build/bin/repl")]
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.get_repl_path(),
            text=True,
            bufsize=1,  # Line buffering.
        )

        response = self.send_command(self.prelude)
        self.env_id = response["env"]

    def get_repl_path(self):
        return Path("lean-repl").absolute()

    def check_rule_valid(self, rule: CellRule) -> str | None:
        try:
            self.send_command(
                f"abbrev {rule.id} (g : Grid) (a : Coords) (c : Coords) : Bool := {rule.body}"
            )
            return None
        except LeanInteractionException as e:
            return str(e)

    def send_command(self, content: str) -> dict:
        self._assert_started()

        try:
            inp = {"cmd": content}
            inp["env"] = self.env_id
            inp_str = json.dumps(inp, ensure_ascii=False) + "\n\n"
            self.process.stdin.write(inp_str)
            self.process.stdin.flush()

            output = []
            while True:
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError("Lean REPL ended unexpectedly")
                line = line.strip()
                if line == "":
                    # Empty line marks end of response.
                    break
                output.append(line)

            if not self.suppress_output:
                print("REPL OUT:", "\n".join(output))

            # Only read from stderr if there is data available, to avoid blocking.
            stderr_fd = self.process.stderr.fileno()
            while True:
                ready, _, _ = select.select([stderr_fd], [], [], 0)
                if not ready:
                    break
                error_output = self.process.stderr.readline()
                if not error_output:
                    break
                if not self.suppress_output:
                    print("REPL ERR:", error_output, file=sys.stderr)
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError("Could not communicate with Lean REPL.") from e

        response = json.loads("\n".join(output))
        messages = response.get("messages", [])
        errors = [m for m in messages if m["severity"] == "error"]
        warnings = [m for m in messages if m["severity"] == "warning" and m["data"] != "declaration uses 'sorry'"]
        if len(errors) > 0:
            raise LeanInteractionException(f"REPL returned error messages: {json.dumps(errors, ensure_ascii=False)}")
        # if len(warnings) > 0:
        #     print(f"REPL returned warning messages: {json.dumps(warnings, ensure_ascii=False)}", file=sys.stderr)
        return response

    def _assert_started(self):
        if not self.started:
            raise RuntimeError("REPL not started. Call start() first.")


class LeanInteractionException(Exception):
    pass


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
    setup_seeds(args.seed)

    descriptor = get_args_descriptor(args, param_whitelist=ARGS_WHITELIST)
    log_dir = args.output_dir / descriptor
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logging to {log_dir}")
    Logger.configure(log_dir)
    dump_args(args, log_dir)

    fitness_evaluator = FitnessEvaluator(args)
    model_provider = ModelProvider(args)

    with model_provider:
        alg = EvolutionaryAlgorithm(args, fitness_evaluator, model_provider)
        alg.run()


if __name__ == "__main__":
    main()
