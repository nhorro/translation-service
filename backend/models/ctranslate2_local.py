# models/ctranslate2_local.py
from __future__ import annotations
from typing import Any, Dict, Optional, List
import os
import ctranslate2
from transformers import AutoTokenizer
from . import TranslationAdapter, merged_params


class CTranslate2LocalAdapter(TranslationAdapter):
    """
    CTranslate2 adapter that loads a pre-converted model from a local directory.

    Expected params:
      - model_path: str  (path to CT2 converted model dir)
      - tokenizer_id: str (HF tokenizer ID, e.g., "Helsinki-NLP/opus-mt-es-en")
      - device: "auto" | "cpu" | "cuda"        (default: "auto")
      - compute_type: one of:
          float32, float16, bfloat16, int8, int8_float16, int8_bfloat16, int16 (default: int8_float16)
      - num_threads: int (optional; CPU threading)
      - src_lang / tgt_lang: optional for multilingual models (NLLB/MBART); ignored for Marian.
    """

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        super().__init__(name, config)
        self.translator: Optional[ctranslate2.Translator] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device: str = "cpu"

    def setup(self) -> None:
        params = (self.config or {}).get("params", {}) or {}
        model_path = params.get("model_path")
        if not model_path or not os.path.isdir(model_path):
            raise ValueError(f"[{self.name}] 'model_path' is required and must be a directory: {model_path}")

        # Device selection
        dev = (params.get("device") or "auto").lower()
        if dev == "auto":
            try:
                import torch  # only to check CUDA availability if present
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        elif dev in ("cpu", "cuda"):
            self.device = dev
        else:
            raise ValueError(f"[{self.name}] Invalid device '{dev}'")

        # Compute type
        compute_type = (params.get("compute_type") or "int8_float16").lower()
        # CT2 will validate allowed values; keep as string

        # (Optional) threads for CPU
        num_threads = params.get("num_threads", None)
        kwargs = {
            "device": self.device,
            "compute_type": compute_type,
        }
        if isinstance(num_threads, int) and num_threads > 0:
            kwargs["inter_threads"] = num_threads
            kwargs["intra_threads"] = num_threads

        self.translator = ctranslate2.Translator(model_path, **kwargs)

        # Load tokenizer (HF) for detok/tokenization roundtrip
        tok_id = params.get("tokenizer_id") or params.get("hf_model_id") or "Helsinki-NLP/opus-mt-es-en"
        self.tokenizer = AutoTokenizer.from_pretrained(tok_id)

        self._mark_ready()

    def translate(self, text: str, *, params: Optional[Dict[str, Any]] = None) -> str:
        if not self.is_ready():
            self.setup()

        cfg = (self.config.get("params") or {})
        gen_defaults = (self.config.get("defaults") or {}).get("generation", {})
        gen = merged_params(gen_defaults, params)

        # Language hints (used for MBART/NLLB; Marian ignores)
        src_lang = (params or {}).get("src_lang") or cfg.get("src_lang")
        tgt_lang = (params or {}).get("tgt_lang") or cfg.get("tgt_lang")

        # Tokenize to token *strings* for CT2
        enc = self.tokenizer(text, add_special_tokens=True)
        ids: List[int] = enc["input_ids"]
        src_tokens: List[str] = self.tokenizer.convert_ids_to_tokens(ids)

        # Optional lang prefix for some multilingual models
        if src_lang and hasattr(self.tokenizer, "lang_code_to_id"):
            # For MBART/NLLB, lang is usually handled via tokenizer fields; but as a fallback:
            # prepend a language token string if your tokenizer uses explicit tokens.
            pass

        beam_size = max(1, int(gen.get("num_beams", 4)))
        max_out = int(gen.get("max_new_tokens", 128))
        sample = bool(gen.get("do_sample", False))

        # CT2 generation settings
        # For deterministic translation we use beam search (sampling off).
        results = self.translator.translate_batch(
            [src_tokens],
            beam_size=1 if sample else beam_size,
            sampling_topk=1 if not sample else 50,  # default top-k if sampling
            sampling_temperature=1.0 if not sample else float(gen.get("temperature", 1.0)),
            max_decoding_length=max_out,
        )

        tgt_tokens: List[str] = results[0].hypotheses[0]

        # If tgt_lang is a leading token in some multilingual models, strip it (not needed for Marian)
        if tgt_lang and tgt_tokens and tgt_tokens[0] == tgt_lang:
            tgt_tokens = tgt_tokens[1:]

        out_text = self.tokenizer.convert_tokens_to_string(tgt_tokens).strip()
        return out_text
