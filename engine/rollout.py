"""Rollout generation engine."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import json

from envs.base import BaseTask, TaskConfig, Observation
from .providers import ModelProvider, ModelResponse

# Default path to safety spec
DEFAULT_SAFETY_SPEC_PATH = Path(__file__).parent / "safety_spec.txt"

def load_safety_spec(spec_path: Optional[str] = None) -> str:
    """Load the anti-scheming safety specification."""
    path = Path(spec_path) if spec_path else DEFAULT_SAFETY_SPEC_PATH
    if not path.exists():
        raise FileNotFoundError(f"Safety spec not found: {path}")
    return path.read_text().strip()


def inject_safety_spec(
    task_prompt: str,
    spec_path: Optional[str] = None,
) -> str:
    """Prepend the safety specification to a task prompt.

    Args:
        task_prompt: The original task prompt
        spec_path: Optional path to safety spec file

    Returns:
        Combined prompt with safety spec prepended
    """
    spec = load_safety_spec(spec_path)
    return f"{spec}\n\n{task_prompt}"


@dataclass
class RolloutResult:
    """Result of a single rollout."""
    trajectory: dict
    model_name: str
    env_name: str
    config: dict
    timestamp: str
    thinking_traces: list[Optional[str]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_jsonl(self) -> str:
        """Convert to JSONL format."""
        return json.dumps({
            "trajectory": self.trajectory,
            "model_name": self.model_name,
            "env_name": self.env_name,
            "config": self.config,
            "timestamp": self.timestamp,
            "thinking_traces": self.thinking_traces,
            "metadata": self.metadata,
        })

    @classmethod
    def from_jsonl(cls, line: str) -> "RolloutResult":
        """Load from JSONL."""
        data = json.loads(line)
        return cls(**data)

    def get_original_task(self) -> Optional[str]:
        """Get the original task prompt (without safety spec)."""
        return self.metadata.get("original_task")

    def get_final_action(self) -> Optional[str]:
        """Get the model's final action/response."""
        history = self.trajectory.get("history", [])
        if history:
            return history[-1].get("action", {}).get("content")
        return None

    def get_thinking(self) -> Optional[str]:
        """Get the model's chain-of-thought (if available)."""
        if self.thinking_traces:
            # Return first non-None thinking trace
            for trace in self.thinking_traces:
                if trace:
                    return trace
        return None


class RolloutGenerator:
    """Generates rollouts by running models through environments."""

    def __init__(
        self,
        provider: ModelProvider,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        inject_safety_spec: bool = False,
        safety_spec_path: Optional[str] = None,
    ):
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.inject_safety_spec = inject_safety_spec
        self.safety_spec_path = safety_spec_path

        # Pre-load safety spec if injection is enabled
        self._safety_spec = None
        if self.inject_safety_spec:
            self._safety_spec = load_safety_spec(safety_spec_path)

    def run_episode(self, task: BaseTask) -> RolloutResult:
        """Run a single episode through the task."""
        observation = task.reset()
        messages = []
        thinking_traces = []

        # Store original task for SFT output
        original_task = observation.content

        # Optionally inject safety spec into the first message
        first_message = observation.content
        if self.inject_safety_spec and self._safety_spec:
            first_message = f"{self._safety_spec}\n\n{observation.content}"

        messages.append({"role": "user", "content": first_message})

        while not task.done:
            # Get model response
            response = self.provider.generate(
                system_prompt=task.get_system_prompt(),
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            thinking_traces.append(response.thinking)

            # Classify and execute action
            action = task.classify_action(response.content)
            observation = task.step(action)

            # Update message history
            messages.append({"role": "assistant", "content": response.content})
            if not observation.done:
                messages.append({"role": "user", "content": observation.content})

        # Build metadata
        metadata = {
            "provider_supports_thinking": self.provider.supports_thinking,
            "safety_spec_injected": self.inject_safety_spec,
            "original_task": original_task,
        }

        # Include rewards spec if available (for file-based tasks)
        if hasattr(task, 'get_rewards_spec'):
            metadata["rewards_spec"] = task.get_rewards_spec()

        return RolloutResult(
            trajectory=task.get_trajectory(),
            model_name=self.provider.model_name,
            env_name=task.config.name,
            config={
                "parameters": task.config.parameters,
            },
            timestamp=datetime.now().isoformat(),
            thinking_traces=thinking_traces,
            metadata=metadata,
        )

    def run_batch(
        self,
        task_factory: callable,
        n_rollouts: int,
        configs: Optional[list[TaskConfig]] = None,
    ) -> list[RolloutResult]:
        """Run multiple rollouts.

        Args:
            task_factory: Callable that takes a TaskConfig and returns a BaseTask
            n_rollouts: Number of rollouts to generate
            configs: Optional list of configs to cycle through
        """
        results = []

        for i in range(n_rollouts):
            if configs:
                config = configs[i % len(configs)]
            else:
                config = TaskConfig(name="default")

            task = task_factory(config)
            result = self.run_episode(task)
            results.append(result)

        return results
