"""Atlas terminal chat package."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, List

__all__ = ["AtlasAgent"]

if TYPE_CHECKING:  # pragma: no cover - type checkers need the real symbol
    from .agent import AtlasAgent


def __getattr__(name: str) -> Any:
    if name == "AtlasAgent":
        module = import_module(".agent", __name__)
        return module.AtlasAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals()) | {"AtlasAgent"})
