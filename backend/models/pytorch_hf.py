# models/pytorch_hf.py
from __future__ import annotations
from typing import Any, Dict, Optional
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub.utils import HfHubHTTPError
from . import TranslationAdapter, merged_params

class PytorchHFAdapter(TranslationAdapter):
    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        super().__init__(name, config)
        self.tokenizer = None
        self.model = None
        self.device = "cpu"

    def setup(self) -> None:
        params = (self.config or {}).get("params", {})
        model_id = params.get("model_id")          # e.g. "Helsinki-NLP/opus-mt-en-es"
        model_path = params.get("model_path")      # e.g. "./hf_models/opus-mt-en-es"
        revision = params.get("revision")          # optional
        local_only_param = bool(params.get("local_files_only", False))

        if not model_id and not model_path:
            raise ValueError(f"[{self.name}] Provide either 'model_id' (Hub) or 'model_path' (local).")

        hf_cfg = (self.config or {}).get("huggingface", {})
        hf_token = hf_cfg.get("token") or os.getenv("HF_TOKEN") or None
        cache_dir = hf_cfg.get("cache_dir", None)
        strict_auth = bool(hf_cfg.get("strict_auth", True))

        # Device & dtype
        dev_cfg = (params.get("device") or "auto").lower()
        if dev_cfg == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif dev_cfg in ("cpu", "cuda"):
            self.device = "cuda" if (dev_cfg == "cuda" and torch.cuda.is_available()) else "cpu"
        else:
            raise ValueError(f"[{self.name}] Invalid device '{dev_cfg}'.")

        dtype_cfg = (params.get("dtype") or "auto").lower()
        if dtype_cfg == "auto":
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            m = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
            torch_dtype = m.get(dtype_cfg)
            if torch_dtype is None:
                raise ValueError(f"[{self.name}] Invalid dtype '{dtype_cfg}'.")
            if self.device == "cpu" and torch_dtype in (torch.float16, torch.bfloat16):
                torch_dtype = torch.float32

        # Loader helpers
        def _load_kwargs(base_auth: bool) -> Dict[str, Any]:
            kw = {
                "cache_dir": cache_dir,
                "revision": revision,
                "local_files_only": local_only_param,
            }
            # If loading from local path, force offline
            if model_path:
                kw["local_files_only"] = True
            # Hub auth only if using model_id AND a token is provided
            if base_auth and (model_id and hf_token):
                kw["token"] = hf_token
            return kw

        def _load_any(auth: bool):
            if model_path:
                # Local directory only; ignore token
                tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                mdl = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch_dtype, local_files_only=True)
                return tok, mdl
            else:
                # From Hub (may use auth)
                kw = _load_kwargs(base_auth=auth)
                tok = AutoTokenizer.from_pretrained(model_id, **kw)
                mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch_dtype, **kw)
                return tok, mdl

        # Main load
        try:
            # If local path is given: single path; otherwise Hub first with configured auth policy
            if model_path:
                self.tokenizer, self.model = _load_any(auth=False)
            else:
                try:
                    self.tokenizer, self.model = _load_any(auth=True)
                except HfHubHTTPError as e:
                    status = getattr(e.response, "status_code", None)
                    if status == 401 and hf_token:
                        if strict_auth:
                            raise RuntimeError(
                                f"[{self.name}] Auth failed for '{model_id}' with provided token. "
                                f"Public models: remove token. Private/gated: ensure access."
                            ) from e
                        # non-strict: try anonymous
                        self.tokenizer, self.model = _load_any(auth=False)
                    elif status == 401 and not hf_token:
                        raise RuntimeError(
                            f"[{self.name}] Model '{model_id}' requires authentication."
                        ) from e
                    elif status == 404:
                        raise RuntimeError(f"[{self.name}] Model '{model_id}' not found (404).") from e
                    else:
                        raise

        except Exception as e:
            raise RuntimeError(f"[{self.name}] Failed to load model: {e}") from e

        self.model = self.model.to(self.device)
        self._mark_ready()

    def translate(self, text: str, *, params: Optional[Dict[str, Any]] = None) -> str:
        if not self.is_ready():
            self.setup()

        gen_defaults = (self.config.get("defaults") or {}).get("generation", {})
        gen_params = merged_params(gen_defaults, params)

        # Optional: language hints for MBART/NLLB
        forced_bos_token_id = None
        cfg_params = (self.config.get("params") or {})
        tgt_lang = (params or {}).get("tgt_lang") or cfg_params.get("tgt_lang")
        src_lang = (params or {}).get("src_lang") or cfg_params.get("src_lang")

        if src_lang and hasattr(self.tokenizer, "src_lang"):
            try: self.tokenizer.src_lang = src_lang
            except Exception: pass
        if tgt_lang and hasattr(self.tokenizer, "tgt_lang"):
            try: self.tokenizer.tgt_lang = tgt_lang
            except Exception: pass
        if tgt_lang and hasattr(self.tokenizer, "lang_code_to_id"):
            forced_bos_token_id = getattr(self.tokenizer, "lang_code_to_id", {}).get(tgt_lang)

        inputs = self.tokenizer([text], return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            generate_kwargs = dict(
                max_new_tokens=int(gen_params.get("max_new_tokens", 256)),
                num_beams=int(gen_params.get("num_beams", 4)),
                do_sample=bool(gen_params.get("do_sample", False)),
            )
            if forced_bos_token_id is not None:
                generate_kwargs["forced_bos_token_id"] = forced_bos_token_id

            output_ids = self.model.generate(**inputs, **generate_kwargs)

        out = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return out[0] if out else ""
