# models/__init__.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class TranslationAdapter(ABC):
    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name = name
        self.config = config or {}
        self._is_ready = False

    @abstractmethod
    def setup(self) -> None: ...
    @abstractmethod
    def translate(self, text: str, *, params: Optional[Dict[str, Any]] = None) -> str: ...

    def is_ready(self) -> bool: return self._is_ready
    def _mark_ready(self) -> None: self._is_ready = True

def merged_params(defaults: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(defaults or {})
    if overrides: merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged

