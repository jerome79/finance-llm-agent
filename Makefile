init:
	python -m venv .venv && . .venv/Scripts/activate && pip install -r requirements.txt

ui:
	streamlit run app/ui_streamlit.py

ingest:
	.venv/Scripts/python scripts/ingest.py --source_dir data/seed_docs --persist_dir .chroma --reset

test:
	pytest -q
