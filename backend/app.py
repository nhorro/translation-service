from __future__ import annotations
import os
import yaml
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()  # loads .env into process env

from factory import build_adapter
from models import TranslationAdapter

# ---------- Load config ----------
CONFIG_PATH = os.environ.get("CONFIG_YML", "config.yml")
with open(CONFIG_PATH, "r") as f:
    RAW_CFG: Dict[str, Any] = yaml.safe_load(f) or {}

SERVER_CFG = RAW_CFG.get("server", {})
DEFAULTS = RAW_CFG.get("defaults", {}) or {}
HF_CFG = RAW_CFG.get("huggingface", {}) or {}
MODEL_REGISTRY = RAW_CFG.get("models", {}) or {}

# ---------- State (lazy cache of loaded adapters) ----------
adapter_cache: Dict[str, TranslationAdapter] = {}


# ---------- Request/Response models ----------
class TranslateRequest(BaseModel):
    text: str = Field(..., description="Plain text to translate")
    model: Optional[str] = Field(None, description="Logical model key in config.yml:models (e.g., 'opus-es-en')")
    params: Optional[Dict[str, Any]] = Field(None, description="Optional overrides (e.g., generation params)")


class TranslateResponse(BaseModel):
    model: str
    adapter: str
    output: str


# ---------- FastAPI ----------
app = FastAPI(title="Unified Translation API", version="0.1.0")


def get_or_create_adapter(model_key: str) -> TranslationAdapter:
    """
    Returns a cached adapter instance for the given logical model key.
    Loads and caches on first use.
    """
    if model_key in adapter_cache:
        return adapter_cache[model_key]

    if model_key not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found in config.yml")

    entry = MODEL_REGISTRY[model_key] or {}
    adapter_key = entry.get("adapter")
    params = entry.get("params", {})

    # Merge a small config bundle for the adapter
    merged_cfg = {
        "params": params,
        "defaults": DEFAULTS,
        "huggingface": HF_CFG,
    }

    adapter = build_adapter(name=model_key, adapter_key=adapter_key, merged_config=merged_cfg)
    # Lazy setup: done inside adapter.translate() if not ready
    adapter_cache[model_key] = adapter
    return adapter


@app.get("/health")
def health():
    return {"status": "ok", "loaded_adapters": list(adapter_cache.keys())}


@app.get("/models")
def list_models():
    return {
        "models": [
            {
                "name": k,
                "adapter": (v or {}).get("adapter"),
                "params_keys": sorted((v or {}).get("params", {}).keys()),
            }
            for k, v in MODEL_REGISTRY.items()
        ],
        "default": DEFAULTS.get("adapter", None),
    }


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    model_key = req.model or DEFAULTS.get("adapter")
    if not model_key:
        raise HTTPException(status_code=400, detail="No model provided and no default adapter configured.")

    adapter = get_or_create_adapter(model_key)

    try:
        output = adapter.translate(req.text, params=req.params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {e}")

    # Report *which adapter* actually backed the model
    adapter_kind = MODEL_REGISTRY.get(model_key, {}).get("adapter", "<unknown>")

    return TranslateResponse(model=model_key, adapter=adapter_kind, output=output)
