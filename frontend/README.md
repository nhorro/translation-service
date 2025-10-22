# Translator service (frontend)

~~~bash
# in your Streamlit client folder
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# run the FastAPI backend in your other terminal
# python -m uvicorn app:app --host 0.0.0.0 --port 8080

# start the UI
streamlit run streamlit_app.py
~~~