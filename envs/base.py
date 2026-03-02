"""Base classes for environment tasks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class ActionType(Enum):
    """Types of actions an agent can take."""
    OVERT = "overt"           # Transparent, aligned action
    COVERT = "covert"         # Hidden, potentially misaligned action
    ESCALATE = "escalate"     # Escalate to user/supervisor
    REFUSE = "refuse"         # Refuse to act


@dataclass
class TaskConfig:
    """Configuration for a task environment."""
    name: str
    system_prompt_extras: str = ""   # Additional system prompt content
    max_turns: int = 10
    parameters: dict = field(default_factory=dict)


@dataclass
class Action:
    """An action taken by the agent."""
    content: str
    action_type: ActionType
    metadata: dict = field(default_factory=dict)


@dataclass
class Observation:
    """An observation/response from the environment."""
    content: str
    done: bool = False
    reward: float = 0.0
    info: dict = field(default_factory=dict)


class BaseTask(ABC):
    """Base class for all environment tasks.

    Subclasses must implement:
    - get_system_prompt(): Returns the system prompt for the agent
    - get_initial_observation(): Returns the first observation
    - step(action): Process an action and return observation
    - get_valid_actions(): Return valid action types for current state
    - classify_action(action): Classify an action's type
    """

    def __init__(self, config: TaskConfig):
        self.config = config
        self.history: list[tuple[Action, Observation]] = []
        self.turn = 0
        self._done = False

    @property
    def done(self) -> bool:
        return self._done or self.turn >= self.config.max_turns

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this task."""
        pass

    @abstractmethod
    def get_initial_observation(self) -> Observation:
        """Return the initial observation to start the task."""
        pass

    @abstractmethod
    def step(self, action: Action) -> Observation:
        """Process an action and return the resulting observation."""
        pass

    @abstractmethod
    def get_valid_actions(self) -> list[ActionType]:
        """Return valid action types for the current state."""
        pass

    @abstractmethod
    def classify_action(self, response: str) -> Action:
        """Classify a raw response into an Action with type."""
        pass

    def reset(self) -> Observation:
        """Reset the task to initial state."""
        self.history = []
        self.turn = 0
        self._done = False
        return self.get_initial_observation()

    def get_trajectory(self) -> dict[str, Any]:
        """Return the full trajectory for this episode."""
        return {
            "config": {
                "name": self.config.name,
                "parameters": self.config.parameters,
            },
            "system_prompt": self.get_system_prompt(),
            "history": [
                {
                    "action": {"content": a.content, "type": a.action_type.value},
                    "observation": {"content": o.content, "done": o.done, "reward": o.reward},
                }
                for a, o in self.history
            ],
            "turns": self.turn,
            "done": self.done,
        }
