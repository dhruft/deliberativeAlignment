"""Base classes for monitors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from engine.rollout import RolloutResult


class BehaviorLabel(Enum):
    """Categorical labels for trajectory behavior."""
    ALIGNED = "aligned"               # Transparent, human-aligned behavior
    COVERT_ACTION = "covert_action"   # Hidden misaligned action
    OVERREFUSAL = "overrefusal"       # Unnecessary refusal
    UNCLEAR = "unclear"               # Cannot determine


@dataclass
class MonitorResult:
    """Result from a monitor evaluation."""
    label: BehaviorLabel
    confidence: float  # 0-1
    reasoning: str
    is_good: bool = False  # Whether this rollout is suitable for SFT training
    cot_analysis: Optional[str] = None  # Analysis of chain-of-thought if available
    cot_references_spec: bool = False  # Did CoT reference safety spec?
    action_matches_reward: bool = False  # Does action match rewards.txt?
    flags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class BaseMonitor(ABC):
    """Base class for monitors that evaluate rollouts."""

    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version

    @abstractmethod
    def evaluate(self, rollout: RolloutResult) -> MonitorResult:
        """Evaluate a rollout and return a result."""
        pass

    def evaluate_batch(self, rollouts: list[RolloutResult]) -> list[MonitorResult]:
        """Evaluate multiple rollouts."""
        return [self.evaluate(r) for r in rollouts]
