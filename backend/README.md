# Translation service (backend)

## Instructions

### Run it

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