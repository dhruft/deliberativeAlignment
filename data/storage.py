"""Data storage for trajectories and experiment results.

Structure:
    data/experiments/
        exp-1/
            metadata.json
            moltbook_influence/        # collect-phase results
                trajectories.jsonl
                labels.jsonl
                review.jsonl
            train-1/                   # first --train run
                training.jsonl
                job.json
                model_id.txt
            train-2/                   # second --train run (if re-trained)
                ...
            test/                      # latest --test results (replaced each run)
                moltbook_evaluation/
                    trajectories.jsonl
                    labels.jsonl
                    review.jsonl
        exp-2/
            ...
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator

from engine.rollout import RolloutResult
from monitors.base import MonitorResult, BehaviorLabel


@dataclass
class ExperimentRun:
    """Metadata for an experiment run."""
    experiment_id: str
    description: str
    model_name: str
    env_names: list
    monitor_name: str
    monitor_version: str
    timestamp: str
    config: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentRun":
        return cls(**data)


_LABEL_TO_CLASSIFICATION = {
    "aligned": "honest",
    "covert_action": "scheming",
    "overrefusal": "overrefusal",
    "unclear": "unclear",
}


@dataclass
class ReviewableRollout:
    """A rollout formatted for human review and editing before SFT training.

    Edit `classification` freely. Setting monitor_confidence to null in the
    JSON marks the entry as manually reviewed.
    """
    id: int
    instruction: str
    thought: str
    output: str
    classification: str   # "honest" | "scheming" | "overrefusal" | "unclear"
    env_name: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "instruction": self.instruction,
            "thought": self.thought,
            "output": self.output,
            "classification": self.classification,
            "env_name": self.env_name,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewableRollout":
        return cls(
            id=data["id"],
            instruction=data["instruction"],
            thought=data["thought"],
            output=data["output"],
            classification=data["classification"],
            env_name=data.get("env_name", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SFTExample:
    """A single example for SFT training."""
    instruction: str
    thought: str
    output: str
    env_id: str
    trajectory_id: int
    monitor_confidence: float

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "thought": self.thought,
            "output": self.output,
            "metadata": {
                "env_id": self.env_id,
                "trajectory_id": self.trajectory_id,
                "monitor_confidence": self.monitor_confidence,
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SFTExample":
        meta = data.get("metadata", {})
        return cls(
            instruction=data["instruction"],
            thought=data["thought"],
            output=data["output"],
            env_id=meta.get("env_id", ""),
            trajectory_id=meta.get("trajectory_id", 0),
            monitor_confidence=meta.get("monitor_confidence", 0.0),
        )


class TrajectoryStore:
    """Storage backend for trajectories and results."""

    def __init__(self, base_path: str = "data/experiments"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Experiment ID management
    # ------------------------------------------------------------------

    def _next_experiment_id(self) -> str:
        """Return the next auto-incrementing experiment ID (exp-1, exp-2, ...)."""
        nums = []
        for d in self.base_path.iterdir():
            if d.is_dir() and d.name.startswith("exp-"):
                try:
                    nums.append(int(d.name[4:]))
                except ValueError:
                    pass
        return f"exp-{max(nums, default=0) + 1}"

    def latest_experiment_id(self) -> Optional[str]:
        """Return the most recently created exp-N folder, or None."""
        nums = []
        for d in self.base_path.iterdir():
            if d.is_dir() and d.name.startswith("exp-"):
                try:
                    nums.append(int(d.name[4:]))
                except ValueError:
                    pass
        return f"exp-{max(nums)}" if nums else None

    def next_train_subdir(self, experiment_id: str) -> str:
        """Return the next auto-incrementing train subdirectory name (train-1, train-2, ...)."""
        exp_path = self.base_path / experiment_id
        nums = []
        if exp_path.exists():
            for d in exp_path.iterdir():
                if d.is_dir() and d.name.startswith("train-"):
                    try:
                        nums.append(int(d.name[6:]))
                    except ValueError:
                        pass
        return f"train-{max(nums, default=0) + 1}"

    # ------------------------------------------------------------------
    # Experiment creation
    # ------------------------------------------------------------------

    def create_experiment(
        self,
        description: str,
        model_name: str,
        env_names: list,
        monitor_name: str,
        monitor_version: str,
        config: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> ExperimentRun:
        """Create a new experiment, auto-assigning the next exp-N ID."""
        experiment_id = self._next_experiment_id()
        exp_path = self.base_path / experiment_id
        exp_path.mkdir(exist_ok=True)

        run = ExperimentRun(
            experiment_id=experiment_id,
            description=description,
            model_name=model_name,
            env_names=env_names,
            monitor_name=monitor_name,
            monitor_version=monitor_version,
            timestamp=datetime.now().isoformat(),
            config=config or {},
            metadata=metadata or {},
        )

        with open(exp_path / "metadata.json", "w") as f:
            json.dump(run.to_dict(), f, indent=2)

        return run

    # ------------------------------------------------------------------
    # Per-env helpers
    # ------------------------------------------------------------------

    def _env_path(self, experiment_id: str, env_name: str, subdir: Optional[str] = None) -> Path:
        """Return (and create) the path for an env's data files.

        If subdir is given (e.g. "test"), the path is exp-N/subdir/env_name/.
        Otherwise it is the collect-phase default: exp-N/env_name/.
        """
        if subdir:
            p = self.base_path / experiment_id / subdir / env_name
        else:
            p = self.base_path / experiment_id / env_name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def list_envs(self, experiment_id: str, subdir: Optional[str] = None) -> list:
        """List environment subfolders for an experiment, sorted.

        If subdir is given (e.g. "test"), lists from exp-N/subdir/.
        Otherwise lists only the collect-phase env dirs at exp-N/ top level.
        """
        if subdir:
            base = self.base_path / experiment_id / subdir
        else:
            base = self.base_path / experiment_id
        if not base.exists():
            return []
        return sorted(
            d.name for d in base.iterdir()
            if d.is_dir() and (d / "trajectories.jsonl").exists()
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_trajectory(
        self,
        experiment_id: str,
        env_name: str,
        rollout: RolloutResult,
        trajectory_id: Optional[int] = None,
        subdir: Optional[str] = None,
    ) -> int:
        """Save a trajectory to exp-N/[subdir/]env_name/trajectories.jsonl."""
        traj_file = self._env_path(experiment_id, env_name, subdir) / "trajectories.jsonl"

        if trajectory_id is None:
            trajectory_id = sum(1 for _ in open(traj_file)) if traj_file.exists() else 0

        with open(traj_file, "a") as f:
            f.write(json.dumps({
                "id": trajectory_id,
                "rollout": json.loads(rollout.to_jsonl()),
            }) + "\n")

        return trajectory_id

    def save_label(
        self,
        experiment_id: str,
        env_name: str,
        trajectory_id: int,
        result: MonitorResult,
        subdir: Optional[str] = None,
    ):
        """Save a monitor label to exp-N/[subdir/]env_name/labels.jsonl."""
        label_file = self._env_path(experiment_id, env_name, subdir) / "labels.jsonl"
        with open(label_file, "a") as f:
            f.write(json.dumps({
                "trajectory_id": trajectory_id,
                "label": result.label.value,
                "is_good": getattr(result, "is_good", False),
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "cot_analysis": result.cot_analysis,
                "cot_references_spec": getattr(result, "cot_references_spec", False),
                "action_matches_reward": getattr(result, "action_matches_reward", False),
                "flags": result.flags,
                "metadata": result.metadata,
            }) + "\n")

    def save_review_rollout(
        self,
        experiment_id: str,
        env_name: str,
        rollout: RolloutResult,
        monitor_result: MonitorResult,
        trajectory_id: int,
        subdir: Optional[str] = None,
    ):
        """Save a rollout to exp-N/[subdir/]env_name/review.jsonl for human review."""
        review_file = self._env_path(experiment_id, env_name, subdir) / "review.jsonl"

        classification = _LABEL_TO_CLASSIFICATION.get(monitor_result.label.value, "unclear")

        entry = ReviewableRollout(
            id=trajectory_id,
            instruction=rollout.get_original_task() or "",
            thought=rollout.get_thinking() or "",
            output=rollout.get_final_action() or "",
            classification=classification,
            env_name=env_name,
            metadata={"monitor_confidence": monitor_result.confidence},
        )

        with open(review_file, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_experiment(self, experiment_id: str) -> ExperimentRun:
        """Load experiment metadata."""
        with open(self.base_path / experiment_id / "metadata.json") as f:
            return ExperimentRun.from_dict(json.load(f))

    def load_trajectories(
        self,
        experiment_id: str,
        env_name: Optional[str] = None,
        subdir: Optional[str] = None,
    ) -> Iterator[tuple]:
        """Load trajectories. If env_name given, load that env only; else all envs."""
        envs = [env_name] if env_name else self.list_envs(experiment_id, subdir)
        base = (self.base_path / experiment_id / subdir) if subdir else (self.base_path / experiment_id)
        for env in envs:
            traj_file = base / env / "trajectories.jsonl"
            if traj_file.exists():
                with open(traj_file) as f:
                    for line in f:
                        data = json.loads(line)
                        yield data["id"], RolloutResult.from_jsonl(json.dumps(data["rollout"]))

    def load_labels(
        self,
        experiment_id: str,
        env_name: Optional[str] = None,
        subdir: Optional[str] = None,
    ) -> Iterator[tuple]:
        """Load labels. If env_name given, load that env only; else all envs."""
        envs = [env_name] if env_name else self.list_envs(experiment_id, subdir)
        base = (self.base_path / experiment_id / subdir) if subdir else (self.base_path / experiment_id)
        for env in envs:
            label_file = base / env / "labels.jsonl"
            if label_file.exists():
                with open(label_file) as f:
                    for line in f:
                        data = json.loads(line)
                        yield data["trajectory_id"], MonitorResult(
                            label=BehaviorLabel(data["label"]),
                            confidence=data["confidence"],
                            reasoning=data["reasoning"],
                            is_good=data.get("is_good", False),
                            cot_analysis=data.get("cot_analysis"),
                            cot_references_spec=data.get("cot_references_spec", False),
                            action_matches_reward=data.get("action_matches_reward", False),
                            flags=data.get("flags", []),
                            metadata=data.get("metadata", {}),
                        )

    def load_review_rollouts(
        self,
        experiment_id: str,
        env_name: Optional[str] = None,
        subdir: Optional[str] = None,
    ) -> Iterator[ReviewableRollout]:
        """Load reviewable rollouts. If env_name given, load that env only; else all envs."""
        envs = [env_name] if env_name else self.list_envs(experiment_id, subdir)
        base = (self.base_path / experiment_id / subdir) if subdir else (self.base_path / experiment_id)
        for env in envs:
            review_file = base / env / "review.jsonl"
            if review_file.exists():
                with open(review_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield ReviewableRollout.from_dict(json.loads(line))

    def review_paths(self, experiment_id: str) -> list:
        """Return all review.jsonl paths for an experiment."""
        return [
            self.base_path / experiment_id / env / "review.jsonl"
            for env in self.list_envs(experiment_id)
        ]

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def list_experiments(self) -> list:
        """List all experiment IDs, sorted by number."""
        result = []
        for d in self.base_path.iterdir():
            if d.is_dir() and d.name.startswith("exp-") and (d / "metadata.json").exists():
                result.append(d.name)
        return sorted(result, key=lambda x: int(x[4:]))

    def get_summary(self, experiment_id: str, subdir: Optional[str] = None) -> dict:
        """Get a summary of an experiment's results across all envs."""
        labels = list(self.load_labels(experiment_id, subdir=subdir))
        if not labels:
            return {"total": 0, "labels": {}, "sft_approved": 0}

        label_counts = {}
        is_good_count = 0
        eval_aware_count = 0

        for _, result in labels:
            label = result.label.value
            label_counts[label] = label_counts.get(label, 0) + 1
            if result.is_good:
                is_good_count += 1
            if "evaluation_awareness" in result.flags:
                eval_aware_count += 1

        return {
            "total": len(labels),
            "labels": label_counts,
            "sft_approved": is_good_count,
            "sft_approval_rate": is_good_count / len(labels),
            "evaluation_awareness_rate": eval_aware_count / len(labels),
        }
