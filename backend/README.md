# Translation service (backend)

The backend is a FastAPI powered standard HTTP endpoint.

## API specification

Awesome—here’s a clean, copy-pasteable spec for your backend.

### Base

* **Base URL:** `http://<host>:<port>` (e.g., `http://localhost:8080`)
* **Content type:** `application/json`
* **Auth:** None (model auth handled internally via HF token/config)

### Endpoints

#### GET `/health`

Simple liveness probe.

**Response 200**

```json
{
  "status": "ok",
  "loaded_adapters": ["es-en-tiny", "dummy"]
}
```

---

#### GET `/models`

Lists the logical models exposed by `config.yml`.

**Response 200**

```json
{
  "models": [
    {
      "name": "dummy",
      "adapter": "dummy",
      "params_keys": ["fixed_response"]
    },
    {
      "name": "es-en-tiny",
      "adapter": "pytorch_hf",
      "params_keys": ["model_id", "device", "dtype"]
    },
    {
      "name": "es-en-nllb600m",
      "adapter": "pytorch_hf",
      "params_keys": ["model_id","device","dtype","src_lang","tgt_lang"]
    }
  ],
  "default": "dummy"
}
```

Notes

* `name` = logical key from `config.yml`.
* `adapter` = which backend will be used (e.g., `dummy`, `pytorch_hf`, etc.).
* `params_keys` = names of adapter-specific params available in config (for UI hints).

---

#### POST `/translate`

Unified translation call.

**Request**

```json
{
  "text": "hola mundo",
  "model": "es-en-tiny",          // optional; if omitted uses config.defaults.adapter
  "params": {                     // optional overrides (per-request)
    "max_new_tokens": 128,        // generation overrides (common)
    "num_beams": 4,
    "do_sample": false,

    "src_lang": "spa_Latn",       // adapter-specific (e.g., NLLB/MBART)
    "tgt_lang": "eng_Latn",

    "token": "hf_...",            // (optional) per-request HF token override
    "local_files_only": true      // (optional) force offline for warm cached/HF dir
  }
}
```

**Response 200**

```json
{
  "model": "es-en-tiny",
  "adapter": "pytorch_hf",
  "output": "Hello, world."
}
```

**Common status codes**

* **200** OK
* **400** Bad request (e.g., missing text, no default model configured)

  ```json
  {"detail":"No model provided and no default adapter configured."}
  ```
* **404** Unknown model key

  ```json
  {"detail":"Model 'foo' not found in config.yml"}
  ```
* **500** Adapter error (clear message)

  * HF auth failed (strict mode):

    ```json
    {"detail":"Translation failed: [es-en-nllb600m] Hugging Face authentication failed for 'facebook/nllb-200-distilled-600M': invalid token or no access. Remove the token for public models or provide a valid one."}
    ```
  * Model not found:

    ```json
    {"detail":"Translation failed: [es-en] Model 'Helsinki-NLP/opus-mt-es-en' not found (404)."}
    ```
  * CPU half-precision guard, etc. (message explains the issue)

### Behavior details

* **Model selection:**
  `model` picks a logical entry from `config.yml > models`. If omitted, `defaults.adapter` is used.

* **Generation params:**
  `params` can override `defaults.generation` (e.g., `max_new_tokens`, `num_beams`, `do_sample`). Unknown keys are ignored by generic code but may be used by specific adapters (e.g., `src_lang`, `tgt_lang`, `local_files_only`).

* **Auth & tokens (HF):**

  * Config precedence: `config.huggingface.token` → `HF_TOKEN` env → none.
  * If `strict_auth: true` (recommended):

    * Token provided but invalid → **fail** (no silent fallback).
    * No token and model is gated → **fail**.
  * Optional **per-request override**: `params.token` (useful for testing/rotating tokens without restarting).

* **Local models:**
  If a model entry has `params.model_path`, it is loaded **from disk only** (offline); any token is ignored.
  If a model entry has `params.model_id`, the Hub is used (can be forced offline with `local_files_only: true` + warm cache).

* **Lazy load & caching:**
  Adapters are instantiated on first use and cached in-process.

### Example calls

List models:

```bash
curl -s http://localhost:8080/models | jq
```

Translate (deterministic, beam=4):

```bash
curl -s -X POST http://localhost:8080/translate \
  -H 'Content-Type: application/json' \
  -d '{"model":"es-en-tiny","text":"Necesito una traducción confiable.","params":{"max_new_tokens":128,"num_beams":4,"do_sample":false}}' \
| jq
```

Translate with per-request token (testing a gated model):

```bash
curl -s -X POST http://localhost:8080/translate \
  -H 'Content-Type: application/json' \
  -d '{"model":"es-en-nllb600m","text":"¿Cómo estás?","params":{"src_lang":"spa_Latn","tgt_lang":"eng_Latn","token":"hf_xxx"}}' \
| jq
```

Force offline (warm cache / local dir):

```bash
curl -s -X POST http://localhost:8080/translate \
  -H 'Content-Type: application/json' \
  -d '{"model":"en-es-local","text":"Hello world","params":{"local_files_only":true}}' \
| jq
```

## Instructions

### Configuration

A `.env` file is used to store secret variables. Optionally, it can be used to define variables to keep the YAML generic.

~~~sh
# Hugging Face
HF_TOKEN=hf_******

# Optional: override server or cache
PORT=8080
HF_HOME=$(pwd)/hf_cache
~~~

Generation and model-specific parameters are passed in `config.yml`.

### Generation parameters

Parameters that live under “generation” control how the decoder produces the translation. 

### `max_new_tokens: 256`

* **What it is:** Hard cap on how many tokens the model can *generate* (not counting the input).
* **Why it matters:** Prevents runaway outputs and limits latency/VRAM.
* **Trade-off:** Too small → truncated translations; too large → slower responses.
* **Typical:** 64–256 for sentence/short paragraph translation. For single sentences, 64–128 is usually plenty.

### `num_beams: 4`

* **What it is:** Beam search width. The model explores multiple candidate continuations in parallel and picks the best.
* **Why it matters:** Improves adequacy/fluency vs greedy (beam=1), especially on tricky sentences.
* **Trade-off:** Higher beams = better quality (sometimes) but **slower**. Diminishing returns past 4–6.
* **Typical:** 1 (fast, greedy) to 4 (quality-oriented). For low latency, use 1–2; for quality, 4.

### `do_sample: false`

* **What it is:** Enables *stochastic sampling* (randomness) if `true`. With beam search, this is usually kept **false** for translation.
* **Why it matters:** Sampling adds variability/creativity (good for story text), but translation prefers **determinism**.
* **Typical:** `false` for translation. If you set `true`, also use `temperature`/`top_p` and keep `num_beams: 1`.

#### Good presets (pick one)

* **Deterministic (recommended for translation):**

  ```yaml
  max_new_tokens: 128
  num_beams: 4
  do_sample: false
  ```

  Quality-oriented, stable outputs.

* **Low-latency:**

  ```yaml
  max_new_tokens: 96
  num_beams: 1
  do_sample: false
  ```

  Fast, still solid for short sentences.

### Running the service

```bash
# 1) Create venv
python -m venv .venv
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) (Optional) Export HF token if needed
export HF_TOKEN=hf_xxx

# 4) Start server
python -m uvicorn app:app --host 0.0.0.0 --port 8080
```

> Set `CONFIG_YML=/path/to/config.yml` if multiple configs are needed.

## Obtaining models

Models are downloaded by default from HuggingFace, but they can also be loaded from disk. Some models can be downloaded straightforward, while others require an authorization process.

#### Public models (no token needed)

~~~bash
# Choose a directory to keep the local models:
mkdir -p ./hf_models/opus-mt-en-es

# Materialize real files (no symlinks) into that folder:
huggingface-cli download Helsinki-NLP/opus-mt-en-es \
  --local-dir ./hf_models/opus-mt-en-es \
  --local-dir-use-symlinks False
~~~

#### Gated NLLB (token + license acceptance required)

~~~bash
# Login & ensure you accepted the model terms on the model page first
huggingface-cli login

mkdir -p ./hf_models/nllb-200-distilled-600M
huggingface-cli download facebook/nllb-200-distilled-600M \
  --local-dir ./hf_models/nllb-200-distilled-600M \
  --local-dir-use-symlinks False
~~~

### Try it (examples)

```bash
# Health
curl -s http://YOUR_PUBLIC_HOST:8080/health | jq

# List models
curl -s http://YOUR_PUBLIC_HOST:8080/models | jq

# Dummy (uses default if config.yml defaults.adapter=dummy)
curl -s -X POST http://YOUR_PUBLIC_HOST:8080/translate \
  -H 'Content-Type: application/json' \
  -d '{ "text": "hola mundo" }' | jq

# Explicit HF translation (logical key from config.yml)
curl -s -X POST http://YOUR_PUBLIC_HOST:8080/translate \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "opus-es-en",
        "text": "Necesito una traducción confiable.",
        "params": { "max_new_tokens": 120, "num_beams": 5 }
      }' | jq
```


### How to add a new adapter

1. Create a new file in `models/` (e.g., `ctranslate2_local.py`) implementing:

```python
from models import TranslationAdapter

class MyAdapter(TranslationAdapter):
    def setup(self): ...
    def translate(self, text: str, *, params=None) -> str: ...
```

2. Register it in `factory.py`:

```python
from models.my_adapter import MyAdapter
_ADAPTERS["my_adapter"] = MyAdapter
```

3. Add a logical model in `config.yml`:

```yaml
models:
  my-nllb-local:
    adapter: my_adapter
    params:
      # whatever your adapter needs
```

Now clients can call `{"model": "my-nllb-local", "text": "..."}` with the **same** `/translate` API.