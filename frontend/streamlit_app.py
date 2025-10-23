import json
import time
from typing import Dict, List, Any
from pathlib import Path

import requests
import streamlit as st
from streamlit.components.v1 import html


# ----------------- Helpers -----------------
def filter_es_to_en(model_items: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for m in model_items:
        name = (m.get("name") or "").lower()
        if "es-en" in name:
            out.append(m["name"])
            continue
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


def load_examples(dir_path: Path, max_bytes: int = 20_000) -> Dict[str, str]:
    """
    Load *.txt files from dir_path (non-recursive).
    Returns {filename: text}. Truncates very large files to max_bytes.
    """
    examples: Dict[str, str] = {}
    if not dir_path.exists():
        return examples
    for p in sorted(dir_path.glob("*.txt")):
        try:
            data = p.read_bytes()
            if len(data) > max_bytes:
                data = data[:max_bytes]
            text = data.decode("utf-8", errors="replace").strip()
            examples[p.name] = text
        except Exception as e:
            examples[p.name] = f"[Error reading file: {e}]"
    return examples


# ----------------- App -----------------
def main():
    st.set_page_config(page_title="Translator UI (es→en)", page_icon="🌐", layout="wide")
    st.title("Translator UI (es → en)")
    st.caption("Using FastAPI backend `/models` and `/translate`")

    # Keep input text in session so the examples selector can set it
    if "in_text" not in st.session_state:
        st.session_state.in_text = ""

    # Sidebar: backend + examples
    st.sidebar.header("Backend")
    api_base = st.sidebar.text_input("API base URL", value="http://localhost:8080").rstrip("/")

    st.sidebar.header("Examples")
    ex_dir_str = st.sidebar.text_input("Examples directory", value="examples")
    ex_dir = Path(ex_dir_str).expanduser().resolve()
    reload_examples = st.sidebar.button("Reload examples")

    # Fetch model list from backend
    try:
        r = requests.get(f"{api_base}/models", timeout=5)
        r.raise_for_status()
        models_payload = r.json()
    except Exception as e:
        st.error(f"Failed to fetch {api_base}/models: {e}")
        st.stop()

    model_items = models_payload.get("models", [])
    es_en_names = filter_es_to_en(model_items)
    if not es_en_names:
        st.warning("No es→en models found in /models.")
        st.json(models_payload)
        st.stop()

    model_name = st.selectbox("Model", options=es_en_names, index=0)

    # Load example files
    examples = load_examples(ex_dir)
    ex_names = list(examples.keys())

    # Examples selector (only if there are files)
    if ex_names:
        st.markdown("**Examples**")
        # Keyed selectbox so we can detect changes
        selected_example = st.selectbox(
            "Pick an example (.txt)",
            options=["— (none) —"] + ex_names,
            index=0,
            key="example_select",
            help=f"Loaded from: {ex_dir}",
        )
        # Auto-fill input when a file is selected
        if selected_example != "— (none) —":
            st.session_state.in_text = examples[selected_example]
    else:
        st.info(f"No .txt files found in {ex_dir}")

    # Input / output areas
    in_text = st.text_area(
        "Spanish text",
        height=160,
        placeholder="Escribe aquí el texto en español…",
        key="in_text",  # bound to session_state
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        run_btn = st.button("Translate", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("Clear input", use_container_width=True)
    with col3:
        t_label = st.empty()  # latency label

    # --- Clear button handling (place after buttons, BEFORE using st.session_state["in_text"] again) ---
    if clear_btn:
        # Remove the widget state, then rerun so the text_area is recreated empty
        st.session_state.pop("in_text", None)
        # Optional: also reset the example picker
        st.session_state.pop("example_select", None)
        st.rerun()


    out_status = st.empty()
    out_text_area = st.empty()
    copy_zone = st.empty()

    if run_btn:
        text_to_send = (st.session_state.in_text or "").strip()
        if not text_to_send:
            st.warning("Please enter some Spanish text (or pick an example).")
        else:
            body = {"model": model_name, "text": text_to_send}
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
                try:
                    detail = resp.json().get("detail")
                except Exception:
                    detail = str(e)
                st.error(f"HTTP {resp.status_code}: {detail}")
            except Exception as e:
                st.error(f"Request failed: {e}")


if __name__ == "__main__":
    main()
