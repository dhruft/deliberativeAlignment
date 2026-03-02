"""Monitor registry for dynamic loading."""

from typing import Type
from .base import BaseMonitor

MONITOR_REGISTRY: dict[str, Type[BaseMonitor]] = {}


def register_monitor(name: str):
    """Decorator to register a monitor class."""
    def decorator(cls: Type[BaseMonitor]):
        MONITOR_REGISTRY[name] = cls
        return cls
    return decorator


def get_monitor(name: str, **kwargs) -> BaseMonitor:
    """Get a monitor instance by name."""
    if name not in MONITOR_REGISTRY:
        raise ValueError(f"Unknown monitor: {name}. Available: {list(MONITOR_REGISTRY.keys())}")
    return MONITOR_REGISTRY[name](**kwargs)


def list_monitors() -> list[str]:
    """List all registered monitors."""
    return list(MONITOR_REGISTRY.keys())
