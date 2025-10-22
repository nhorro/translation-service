# models/dummy.py
from typing import Any, Dict, Optional
from . import TranslationAdapter  # <-- note the relative import

class DummyAdapter(TranslationAdapter):
    def setup(self) -> None:
        self._mark_ready()

    def translate(self, text: str, *, params: Optional[Dict[str, Any]] = None) -> str:
        fixed = (self.config.get("params") or {}).get("fixed_response", "OK: dummy translation")
        return fixed

