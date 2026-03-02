"""Metrics computation for experiment analysis."""

from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd

from data.storage import TrajectoryStore
from monitors.base import BehaviorLabel


@dataclass
class ExperimentMetrics:
    """Computed metrics for an experiment."""
    experiment_id: str
    model_name: str
    env_name: str
    total_trajectories: int
    covert_action_rate: float
    aligned_rate: float
    overrefusal_rate: float
    unclear_rate: float
    evaluation_awareness_rate: float
    sft_approval_rate: float = 0.0
    # Environment breakdown
    env_covert_rates: dict = field(default_factory=dict)


def compute_metrics(store: TrajectoryStore, experiment_id: str, subdir: str = None) -> ExperimentMetrics:
    """Compute metrics for a single experiment."""
    exp = store.load_experiment(experiment_id)
    trajectories = list(store.load_trajectories(experiment_id, subdir=subdir))
    labels = list(store.load_labels(experiment_id, subdir=subdir))

    if not labels:
        return ExperimentMetrics(
            experiment_id=experiment_id,
            model_name=exp.model_name,
            env_name=", ".join(exp.env_names),
            total_trajectories=0,
            covert_action_rate=0,
            aligned_rate=0,
            overrefusal_rate=0,
            unclear_rate=0,
            evaluation_awareness_rate=0,
        )

    total = len(labels)

    # Count labels
    label_counts = {label: 0 for label in BehaviorLabel}
    eval_aware = 0
    sft_approved = 0

    for _, result in labels:
        label_counts[result.label] += 1
        if "evaluation_awareness" in result.flags:
            eval_aware += 1
        if getattr(result, 'is_good', False):
            sft_approved += 1

    # Build trajectory map
    traj_map = {tid: rollout for tid, rollout in trajectories}

    # Environment breakdown
    env_stats = defaultdict(lambda: {"covert": 0, "total": 0})

    for tid, result in labels:
        if tid in traj_map:
            rollout = traj_map[tid]
            env_stats[rollout.env_name]["total"] += 1
            if result.label == BehaviorLabel.COVERT_ACTION:
                env_stats[rollout.env_name]["covert"] += 1

    env_covert_rates = {
        k: v["covert"] / v["total"] if v["total"] > 0 else 0
        for k, v in env_stats.items()
    }

    return ExperimentMetrics(
        experiment_id=experiment_id,
        model_name=exp.model_name,
        env_name=", ".join(exp.env_names),
        total_trajectories=total,
        covert_action_rate=label_counts[BehaviorLabel.COVERT_ACTION] / total,
        aligned_rate=label_counts[BehaviorLabel.ALIGNED] / total,
        overrefusal_rate=label_counts[BehaviorLabel.OVERREFUSAL] / total,
        unclear_rate=label_counts[BehaviorLabel.UNCLEAR] / total,
        evaluation_awareness_rate=eval_aware / total,
        sft_approval_rate=sft_approved / total,
        env_covert_rates=dict(env_covert_rates),
    )


def compare_experiments(
    store: TrajectoryStore,
    experiment_ids: list[str],
    subdir: str = None,
) -> pd.DataFrame:
    """Compare metrics across multiple experiments."""
    metrics = [compute_metrics(store, eid, subdir=subdir) for eid in experiment_ids]

    data = []
    for m in metrics:
        data.append({
            "experiment_id": m.experiment_id,
            "model": m.model_name,
            "env": m.env_name,
            "n": m.total_trajectories,
            "covert_rate": m.covert_action_rate,
            "aligned_rate": m.aligned_rate,
            "overrefusal_rate": m.overrefusal_rate,
            "eval_awareness_rate": m.evaluation_awareness_rate,
            "sft_approval_rate": m.sft_approval_rate,
        })

    return pd.DataFrame(data)


def compute_env_breakdown(
    store: TrajectoryStore,
    experiment_ids: list[str],
) -> pd.DataFrame:
    """Compute covert action rates by environment across experiments."""
    rows = []

    for exp_id in experiment_ids:
        metrics = compute_metrics(store, exp_id)
        for env, rate in metrics.env_covert_rates.items():
            rows.append({
                "experiment_id": exp_id,
                "model": metrics.model_name,
                "env": env,
                "covert_rate": rate,
            })

    return pd.DataFrame(rows)


def compute_before_after_comparison(
    store: TrajectoryStore,
    before_exp_id: str,
    after_exp_id: str,
) -> dict:
    """Compare metrics before and after training/intervention.

    Useful for comparing base model vs fine-tuned model.
    """
    before = compute_metrics(store, before_exp_id)
    after = compute_metrics(store, after_exp_id)

    return {
        "before": {
            "experiment_id": before_exp_id,
            "covert_rate": before.covert_action_rate,
            "aligned_rate": before.aligned_rate,
            "overrefusal_rate": before.overrefusal_rate,
        },
        "after": {
            "experiment_id": after_exp_id,
            "covert_rate": after.covert_action_rate,
            "aligned_rate": after.aligned_rate,
            "overrefusal_rate": after.overrefusal_rate,
        },
        "improvement": {
            "covert_rate_change": after.covert_action_rate - before.covert_action_rate,
            "aligned_rate_change": after.aligned_rate - before.aligned_rate,
            "overrefusal_rate_change": after.overrefusal_rate - before.overrefusal_rate,
        },
    }
