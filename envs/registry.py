"""Environment registry for dynamic task loading."""

from pathlib import Path
from typing import Type, Union
from .base import BaseTask, TaskConfig

# Registry for code-based environments (using @register_env decorator)
ENV_REGISTRY: dict[str, Type[BaseTask]] = {}

# Path to file-based environments
ENVIRONMENTS_DIR = Path(__file__).parent / "environments"


def register_env(name: str):
    """Decorator to register a code-based environment class."""
    def decorator(cls: Type[BaseTask]):
        ENV_REGISTRY[name] = cls
        return cls
    return decorator


def discover_file_envs() -> list[str]:
    """Discover all file-based environments in the environments directory."""
    if not ENVIRONMENTS_DIR.exists():
        return []

    envs = []
    for path in ENVIRONMENTS_DIR.iterdir():
        if path.is_dir() and (path / "task.txt").exists():
            envs.append(path.name)
    return envs


def get_env(name: str, config: TaskConfig) -> BaseTask:
    """Get an environment instance by name.

    First checks code-based registry, then looks for file-based environments.
    """
    # Check code-based registry first
    if name in ENV_REGISTRY:
        return ENV_REGISTRY[name](config)

    # Check file-based environments
    env_path = ENVIRONMENTS_DIR / name
    if env_path.exists() and (env_path / "task.txt").exists():
        from .file_loader import FileBasedTask
        return FileBasedTask(env_path, config)

    # Not found
    available = list_envs()
    raise ValueError(f"Unknown environment: {name}. Available: {available}")


def list_envs() -> list[str]:
    """List all available environments (code-based and file-based)."""
    code_envs = list(ENV_REGISTRY.keys())
    file_envs = discover_file_envs()
    return sorted(set(code_envs + file_envs))


def get_env_path(name: str) -> Path:
    """Get the path to a file-based environment."""
    return ENVIRONMENTS_DIR / name


def env_exists(name: str) -> bool:
    """Check if an environment exists."""
    if name in ENV_REGISTRY:
        return True
    env_path = ENVIRONMENTS_DIR / name
    return env_path.exists() and (env_path / "task.txt").exists()
