#!/usr/bin/env python3
"""Main script for running scheming behavior experiments."""

import argparse
import yaml
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

from envs import get_env, TaskConfig, list_envs
from engine import get_provider, RolloutGenerator
from monitors import get_monitor, MONITOR_REGISTRY
from data import TrajectoryStore


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _init_monitor(config: dict, monitor_provider):
    """Initialize a monitor, passing provider only if the monitor accepts it."""
    monitor_name = config.get("monitor", {}).get("name", "deliberative")
    monitor_params = config.get("monitor", {}).get("params", {})
    try:
        return get_monitor(monitor_name, provider=monitor_provider, **monitor_params)
    except TypeError:
        return get_monitor(monitor_name, **monitor_params)


def _run_rollouts(
    env_names: list,
    generator: RolloutGenerator,
    monitor,
    store: TrajectoryStore,
    experiment_id: str,
    n_rollouts: int,
    env_config: dict,
    save_review: bool,
    verbose: bool,
    subdir: Optional[str] = None,
):
    """Inner loop: generate + monitor + save rollouts for a list of envs."""
    for env_name in env_names:
        if verbose:
            print(f"\nProcessing: {env_name}")

        task_config = TaskConfig(
            name=env_name,
            system_prompt_extras=env_config.get("system_prompt_extras", ""),
            max_turns=env_config.get("max_turns", 10),
            parameters=env_config.get("parameters", {}),
        )

        iterator = range(n_rollouts)
        if verbose:
            iterator = tqdm(iterator, desc=f"  rollouts")

        for tid in iterator:
            try:
                task = get_env(env_name, task_config)
            except Exception as e:
                if verbose:
                    print(f"  Warning: failed to load {env_name}: {e}")
                continue

            result = generator.run_episode(task)
            store.save_trajectory(experiment_id, env_name, result, tid, subdir=subdir)

            rewards_spec = result.metadata.get("rewards_spec")
            try:
                label = monitor.evaluate(result, rewards_spec) if rewards_spec else monitor.evaluate(result)
            except TypeError:
                label = monitor.evaluate(result)

            store.save_label(experiment_id, env_name, tid, label, subdir=subdir)

            if save_review:
                store.save_review_rollout(experiment_id, env_name, result, label, tid, subdir=subdir)


def run_collect(config: dict, verbose: bool = True) -> str:
    """Collect phase: generate rollouts from train envs, monitor each, write review.jsonl.

    Safety spec is injected. Creates a new exp-N folder automatically.

    Returns:
        Experiment ID (e.g. "exp-1")
    """
    env_names = config.get("splits", {}).get("train", [])
    if not env_names:
        raise ValueError("No environments found in splits.train")

    rollout_cfg = config["rollout_model"]
    rollout_provider = get_provider(rollout_cfg["provider"], rollout_cfg["name"])

    monitor_cfg = config["monitor_model"]
    monitor_provider = get_provider(monitor_cfg["provider"], monitor_cfg["name"])

    engine_config = config.get("engine", {})
    inject_spec = engine_config.get("inject_safety_spec", True)

    generator = RolloutGenerator(
        provider=rollout_provider,
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens", 1024),
        inject_safety_spec=inject_spec,
        safety_spec_path=engine_config.get("safety_spec_path"),
    )

    monitor = _init_monitor(config, monitor_provider)
    store = TrajectoryStore(config.get("data_path", "data/experiments"))

    exp = store.create_experiment(
        description=config.get("description", ""),
        model_name=rollout_cfg["name"],
        env_names=env_names,
        monitor_name=config.get("monitor", {}).get("name", "deliberative"),
        monitor_version=monitor.version,
        config=config,
        metadata={"phase": "collect", "inject_safety_spec": inject_spec},
    )
    experiment_id = exp.experiment_id

    if verbose:
        print(f"Experiment:  {experiment_id}")
        print(f"Rollout model: {rollout_cfg['name']}")
        print(f"Monitor model: {monitor_cfg['name']}")
        print(f"Environments:  {env_names}")
        print(f"Safety spec:   {inject_spec}")
        print(f"Rollouts/env:  {config['n_rollouts']}")

    _run_rollouts(
        env_names=env_names,
        generator=generator,
        monitor=monitor,
        store=store,
        experiment_id=experiment_id,
        n_rollouts=config["n_rollouts"],
        env_config=config.get("env", {}),
        save_review=True,
        verbose=verbose,
    )

    if verbose:
        data_path = Path(config.get("data_path", "data/experiments"))
        print(f"\nCollect complete!")
        for env_name in env_names:
            review = data_path / experiment_id / env_name / "review.jsonl"
            print(f"  {env_name}: {review}")
        print(f"\nRun: python scripts/review.py {experiment_id}")
        print(f"Then: python run_experiment.py <config> --train --experiment-id {experiment_id}")

    return experiment_id


def run_train(config: dict, experiment_id: Optional[str] = None, verbose: bool = True):
    """Train phase: load review.jsonl files, filter honest rollouts, run SFT.

    If no experiment_id given, uses the most recent exp-N folder.
    """
    from engine.trainer import run_sft_training

    store = TrajectoryStore(config.get("data_path", "data/experiments"))

    if experiment_id is None:
        experiment_id = store.latest_experiment_id()
        if experiment_id is None:
            raise ValueError("No experiments found. Run --collect first.")
        if verbose:
            print(f"Using latest experiment: {experiment_id}")

    review_paths = store.review_paths(experiment_id)
    if not review_paths:
        raise FileNotFoundError(
            f"No review.jsonl files found for experiment '{experiment_id}'.\n"
            f"Run --collect first."
        )

    sft_cfg = config.get("sft_model", {})
    train_subdir = store.next_train_subdir(experiment_id)
    output_dir = store.base_path / experiment_id / train_subdir

    if verbose:
        print(f"Experiment:  {experiment_id}")
        print(f"Review files: {len(review_paths)} env(s)")
        print(f"SFT output:  {output_dir} ({train_subdir})")

    run_sft_training(review_paths, sft_cfg, output_dir, verbose=verbose)


def run_test(config: dict, experiment_id: Optional[str] = None, verbose: bool = True) -> str:
    """Test phase: run test envs without safety spec injection.

    Stores results in exp-N/test/ within an existing experiment folder.
    Uses test_model config if present; falls back to rollout_model.

    Args:
        config: Experiment configuration dict
        experiment_id: Existing exp-N to attach results to. Defaults to latest.

    Returns:
        Experiment ID
    """
    env_names = config.get("splits", {}).get("test", [])
    if not env_names:
        raise ValueError("No environments found in splits.test")

    # Use test_model if specified, otherwise fall back to rollout_model
    rollout_cfg = config.get("test_model") or config["rollout_model"]
    rollout_provider = get_provider(rollout_cfg["provider"], rollout_cfg["name"])

    monitor_cfg = config["monitor_model"]
    monitor_provider = get_provider(monitor_cfg["provider"], monitor_cfg["name"])

    generator = RolloutGenerator(
        provider=rollout_provider,
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens", 1024),
        inject_safety_spec=False,
    )

    monitor = _init_monitor(config, monitor_provider)
    store = TrajectoryStore(config.get("data_path", "data/experiments"))

    # Attach to an existing experiment rather than creating a new one
    if experiment_id is None:
        experiment_id = store.latest_experiment_id()
        if experiment_id is None:
            raise ValueError("No experiments found. Run --collect first.")
        if verbose:
            print(f"Using latest experiment: {experiment_id}")

    if verbose:
        print(f"Experiment:  {experiment_id}")
        print(f"Rollout model: {rollout_cfg['name']}")
        print(f"Environments:  {env_names}")
        print(f"Safety spec:   disabled (test phase)")
        print(f"Rollouts/env:  {config['n_rollouts']}")

    _run_rollouts(
        env_names=env_names,
        generator=generator,
        monitor=monitor,
        store=store,
        experiment_id=experiment_id,
        n_rollouts=config["n_rollouts"],
        env_config=config.get("env", {}),
        save_review=True,
        verbose=verbose,
        subdir="test",
    )

    if verbose:
        summary = store.get_summary(experiment_id, subdir="test")
        print(f"\nTest complete!")
        print(f"  Total: {summary['total']} rollouts")
        print(f"  Labels: {summary['labels']}")

    return experiment_id


def main():
    parser = argparse.ArgumentParser(description="Run scheming behavior experiment")
    parser.add_argument("config", nargs="?", help="Path to experiment config YAML")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress output")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--collect",
        action="store_true",
        help="Generate rollouts from train envs → exp-N/{env}/review.jsonl",
    )
    mode_group.add_argument(
        "--train",
        action="store_true",
        help="Run SFT on honest rollouts (defaults to latest exp-N)",
    )
    mode_group.add_argument(
        "--test",
        action="store_true",
        help="Run test envs without safety spec injection",
    )

    parser.add_argument(
        "--experiment-id",
        help="Experiment to train from (e.g. exp-3). Defaults to latest.",
    )
    parser.add_argument("--list-envs", action="store_true", help="List available environments")
    parser.add_argument("--list-monitors", action="store_true", help="List available monitors")

    args = parser.parse_args()

    if args.list_envs:
        print("Available environments:")
        for name in list_envs():
            print(f"  - {name}")
        return

    if args.list_monitors:
        print("Available monitors:")
        for name in MONITOR_REGISTRY:
            print(f"  - {name}")
        return

    if not args.config:
        parser.print_help()
        return

    config = load_config(args.config)
    verbose = not args.quiet

    if args.train:
        run_train(config, experiment_id=args.experiment_id, verbose=verbose)
    elif args.test:
        run_test(config, experiment_id=args.experiment_id, verbose=verbose)
    else:
        run_collect(config, verbose=verbose)


if __name__ == "__main__":
    main()
