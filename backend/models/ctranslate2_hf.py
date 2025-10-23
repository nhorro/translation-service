# models/ctranslate2_hf.py
from __future__ import annotations
from typing import Any, Dict, Optional, List
import os
import subprocess
import shutil
from pathlib import Path

import ctranslate2
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

from . import TranslationAdapter, merged_params


class CTranslate2HFAdapter(TranslationAdapter):
    """
    Load a pre-converted CTranslate2 model from Hugging Face,
    OR auto-convert a Transformers model on first use.

    Supported params:
      # Option A: direct CT2 repo
      - model_id: str                 # HF repo id WITH CT2 files
      - model_subdir: str (optional)  # subfolder within the repo (if any)

      # Option B: auto-convert from Transformers model
      - transformers_model_id: str    # e.g. "Helsinki-NLP/opus-mt-es-en"
      - auto_convert_if_missing: bool # default False; if True, will convert once and cache

      # Shared
      - revision: str (optional)
      - tokenizer_id: str (optional; default: transformers_model_id or model_id)
      - local_files_only: bool (default False)
      - device: "auto"|"cpu"|"cuda" (default "auto")
      - compute_type: str (e.g., "int8_float16" default, "int8", "int16", "float16", "float32")
      - num_threads: int (optional)
      - ct2_cache_subdir: str (optional; where to store auto-converted CT2 dir)
      - src_lang / tgt_lang: optional (for multilingual models)
    """

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        super().__init__(name, config)
        self.translator: Optional[ctranslate2.Translator] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device = "cpu"
        self.model_dir: Optional[str] = None

    def _pick_device(self, dev: str) -> str:
        if dev == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        if dev in ("cpu", "cuda"):
            return dev
        raise ValueError(f"[{self.name}] Invalid device '{dev}'")

    def _download_snapshot(self, repo_id: str, revision: Optional[str], cache_dir: Optional[str],
                           local_only: bool, token: Optional[str], strict_auth: bool) -> str:
        def _snap(use_auth: bool) -> str:
            kw = dict(repo_id=repo_id, revision=revision, cache_dir=cache_dir, local_files_only=local_only)
            if use_auth and token:
                kw["token"] = token
            return snapshot_download(**kw)

        try:
            try:
                return _snap(use_auth=True)
            except HfHubHTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status == 401 and token:
                    if strict_auth:
                        raise RuntimeError(f"[{self.name}] Auth failed for '{repo_id}' with provided token.") from e
                    return _snap(use_auth=False)  # fallback anon if allowed
                elif status == 401 and not token:
                    raise RuntimeError(f"[{self.name}] '{repo_id}' requires authentication.") from e
                elif status == 404:
                    raise RuntimeError(f"[{self.name}] Repo '{repo_id}' not found (404).") from e
                else:
                    raise
        except Exception as e:
            raise RuntimeError(f"[{self.name}] Failed to download '{repo_id}': {e}") from e

    def _ensure_ct2_from_transformers(self, t_repo: str, revision: Optional[str], cache_dir: Optional[str],
                                      local_only: bool, token: Optional[str], strict_auth: bool,
                                      compute_type: str, out_subdir: Optional[str]) -> str:
        """
        Download a Transformers model snapshot, convert it to CT2 once, and return CT2 dir path.
        """
        # 1) Make a stable output folder inside cache_dir (or under ~/.cache if None)
        base_cache = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "ct2"
        out_dir = base_cache / (out_subdir or (t_repo.replace("/", "__") + f"__{compute_type}"))
        out_dir.mkdir(parents=True, exist_ok=True)

        # If directory already contains a CT2 model (model.bin present), reuse it
        if (out_dir / "model.bin").exists():
            return str(out_dir)

        # 2) Download the original Transformers model
        snap_dir = self._download_snapshot(t_repo, revision, cache_dir, local_only, token, strict_auth)

        # 3) Convert with the ct2 CLI
        cmd = [
            "ct2-transformers-converter",
            "--model", snap_dir,
            "--output_dir", str(out_dir),
            "--quantization", compute_type,
            "--force"
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            raise RuntimeError(
                f"[{self.name}] ct2-transformers-converter not found. Is ctranslate2 installed?"
            )
        except subprocess.CalledProcessError as e:
            # Clean up partial conversion
            try:
                shutil.rmtree(out_dir)
            except Exception:
                pass
            raise RuntimeError(f"[{self.name}] CT2 conversion failed: {e.stderr.decode('utf-8', 'ignore')}")

        return str(out_dir)

    def setup(self) -> None:
        params = (self.config or {}).get("params", {}) or {}
        model_id = params.get("model_id")  # CT2 repo
        model_subdir = params.get("model_subdir")
        t_repo = params.get("transformers_model_id")  # vanilla Transformers repo for auto-convert
        auto_convert = bool(params.get("auto_convert_if_missing", False))
        revision = params.get("revision")
        local_only = bool(params.get("local_files_only", False))
        compute_type = (params.get("compute_type") or "int8_float16").lower()
        ct2_cache_subdir = params.get("ct2_cache_subdir")

        hf_cfg = (self.config or {}).get("huggingface", {})
        token = hf_cfg.get("token") or os.getenv("HF_TOKEN") or None
        cache_dir = hf_cfg.get("cache_dir", None)
        strict_auth = bool(hf_cfg.get("strict_auth", True))

        # Device selection
        self.device = self._pick_device((params.get("device") or "auto").lower())

        # Resolve model_dir: Either from CT2 repo or via auto-conversion
        if model_id:
            snap = self._download_snapshot(model_id, revision, cache_dir, local_only, token, strict_auth)
            self.model_dir = os.path.join(snap, model_subdir) if model_subdir else snap
        elif t_repo and auto_convert:
            self.model_dir = self._ensure_ct2_from_transformers(
                t_repo, revision, cache_dir, local_only, token, strict_auth, compute_type, ct2_cache_subdir
            )
        else:
            raise ValueError(
                f"[{self.name}] Provide either 'model_id' (CT2 repo) OR "
                f"'transformers_model_id' with 'auto_convert_if_missing: true'."
            )

        if not os.path.isdir(self.model_dir):
            raise RuntimeError(f"[{self.name}] CT2 model directory not found: {self.model_dir}")

        # Init CTranslate2 Translator
        tkwargs: Dict[str, Any] = {"device": self.device, "compute_type": compute_type}
        num_threads = params.get("num_threads")
        if isinstance(num_threads, int) and num_threads > 0:
            tkwargs["inter_threads"] = num_threads
            tkwargs["intra_threads"] = num_threads
        self.translator = ctranslate2.Translator(self.model_dir, **tkwargs)

        # Load tokenizer
        tok_id = params.get("tokenizer_id") or t_repo or model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_id,
            cache_dir=cache_dir,
            local_files_only=local_only,
            token=(token if token else None),
            revision=revision,
        )

        self._mark_ready()

    def translate(self, text: str, *, params: Optional[Dict[str, Any]] = None) -> str:
        if not self.is_ready():
            self.setup()

        cfg = (self.config.get("params") or {})
        gen_defaults = (self.config.get("defaults") or {}).get("generation", {})
        gen = merged_params(gen_defaults, params)

        src_lang = (params or {}).get("src_lang") or cfg.get("src_lang")
        tgt_lang = (params or {}).get("tgt_lang") or cfg.get("tgt_lang")

        enc = self.tokenizer(text, add_special_tokens=True)
        ids: List[int] = enc["input_ids"]
        src_tokens: List[str] = self.tokenizer.convert_ids_to_tokens(ids)

        beam_size = max(1, int(gen.get("num_beams", 4)))
        max_out = int(gen.get("max_new_tokens", 128))
        sample = bool(gen.get("do_sample", False))

        results = self.translator.translate_batch(
            [src_tokens],
            beam_size=1 if sample else beam_size,
            sampling_topk=1 if not sample else 50,
            sampling_temperature=1.0 if not sample else float(gen.get("temperature", 1.0)),
            max_decoding_length=max_out,
        )

        tgt_tokens: List[str] = results[0].hypotheses[0]
        if tgt_lang and tgt_tokens and tgt_tokens[0] == tgt_lang:
            tgt_tokens = tgt_tokens[1:]

        out_text = self.tokenizer.convert_tokens_to_string(tgt_tokens).strip()
        return out_text
