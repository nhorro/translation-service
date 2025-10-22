# factory.py
from typing import Dict, Any, Callable
from models import TranslationAdapter
from models.dummy import DummyAdapter
from models.pytorch_hf import PytorchHFAdapter

_ADAPTERS: Dict[str, Callable[..., TranslationAdapter]] = {
    "dummy": DummyAdapter,
    "pytorch_hf": PytorchHFAdapter,
}

def build_adapter(name: str, adapter_key: str, merged_config: Dict[str, Any]) -> TranslationAdapter:
    try:
        cls = _ADAPTERS[adapter_key]
    except KeyError:
        raise ValueError(f"Unknown adapter '{adapter_key}'. Available: {list(_ADAPTERS)}")
    return cls(name=name, config=merged_config)

