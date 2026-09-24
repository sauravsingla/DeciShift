from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class CallableModelAdapter:
    """Wrap a generic callable so it exposes the minimal predict interface."""

    fn: Callable[[Any], Any]
    version: str = "callable"

    def predict(self, X: Any) -> Any:
        return self.fn(X)
