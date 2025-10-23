import json
import time
from typing import Dict, List, Any

import requests
import streamlit as st
from streamlit.components.v1 import html


# ---- Helpers ----------------------------------------------------------------
def filter_es_to_en(model_items: List[Dict[str, Any]]) -> List[str]:
    """
    Given payload["models"] from /models, return names that look like ES->EN.
    Heuristics:
      - name contains 'es-en'
      - OR params include src_lang=spa* AND tgt_lang=eng* (if backend returns params keys)
    """
    out: List[str] = []
    for m in model_items:
        name = (m.get("name") or "").lower()
        if "es-en" in name:
            out.append(m["name"])
            continue
        # Some backends might expose src/tgt in params via /models. If not, skip.
        p = m.get("params", {}) or {}
        src = (p.get("src_lang") or "").lower()
        tgt = (p.get("tgt_lang") or "").lower()
        if src.startswith("spa") and tgt.startswith("eng"):
            out.append(m["name"])
    return sorted(set(out))


def copy_to_clipboard_js(text: str):
    safe = json.dumps(text)
    html(
        f"""
        <button id="copybtn" style="padding:0.4rem 0.7rem; border-radius:8px;">Copy to clipboard</button>
        <span id="copied" style="margin-left:8px; color:gray;"></span>
        <script>
          const txt = {safe};
          const btn = document.getElementById('copybtn');
          const info = document.getElementById('copied');
          btn.addEventListener('click', async () => {{
            try {{
              await navigator.clipboard.writeText(txt);
              info.textContent = "Copied!";
              setTimeout(() => info.textContent = "", 1200);
            }} catch(e) {{
              info.textContent = "Copy failed";
            }}
          }});
        </script>
        """,
        height=38,
    )


# ---- App --------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Demostración de traductor (esp-ing)", page_icon="🌐", layout="wide")
    st.title("Traductor español a inglés")
    st.caption("Traductor de español a inglés usando un servicio de backend *on-premise*.")

    # Sidebar: backend URL + controls
    st.sidebar.header("Backend")
    api_base = st.sidebar.text_input("API base URL", value="http://localhost:8080").rstrip("/")
    refresh = st.sidebar.button("Refrescar lista de modelos", use_container_width=True)

    # Fetch /models
    models_payload: Dict[str, Any] = {}
    try:
        r = requests.get(f"{api_base}/models", timeout=5)
        r.raise_for_status()
        models_payload = r.json()
    except Exception as e:
        st.error(f"Falla al descargar {api_base}/models: {e}")
        st.stop()

    # Derive ES->EN list
    # Your backend /models returns:
    # { "models": [ { "name": "...", "adapter": "...", "params_keys": [...] }, ...], "default": "..." }
    model_items = models_payload.get("models", [])
    # Try to preserve any params we may need (some backends include more fields; we pass through if present)
    es_en_names = filter_es_to_en(model_items)

    if not es_en_names:
        st.warning("No se encontraron modelos de traducción español a inglés en /models. "
                   "Tip: nombrar los modelos con 'es-en' o proveer src_lang=tgt_lang en configuración.")
        st.json(models_payload)  # show what we got for debugging
        st.stop()

    model_name = st.selectbox("Modelo", options=es_en_names, index=0)

    in_text = st.text_area(
        "Spanish text",
        height=160,
        placeholder="Escribe aquí el texto en español…",
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        run_btn = st.button("Traducir", type="primary", use_container_width=True)
    with col2:
        t_label = st.empty()  # latency label

    out_status = st.empty()
    out_text_area = st.empty()
    copy_zone = st.empty()

    if run_btn:
        if not in_text.strip():
            st.warning("Por favor ingresar algún texto en español.")
        else:
            body = {"model": model_name, "text": in_text}
            started = time.perf_counter()
            try:
                resp = requests.post(f"{api_base}/translate", json=body, timeout=60)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                t_label.caption(f"⏱️ {elapsed_ms:.0f} ms")

                resp.raise_for_status()
                payload = resp.json()
                output = payload.get("output", "")

                out_status.success("Done")
                out_text_area.text_area("English (read-only)", value=output, height=160, disabled=True)
                copy_to_clipboard_js(output)

            except requests.HTTPError as e:
                # Try to surface backend error detail
                try:
                    detail = resp.json().get("detail")
                except Exception:
                    detail = str(e)
                st.error(f"HTTP {resp.status_code}: {detail}")
            except Exception as e:
                st.error(f"Request failed: {e}")


if __name__ == "__main__":
    main()
