.PHONY: install data index train test lint api app docker-build docker-run clean

PYTHON = python
PIP = pip

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

data:
	$(PYTHON) -m src.intent.data

index:
	$(PYTHON) -m src.rag.store

train:
	$(PYTHON) -m src.intent.models --train

test:
	pytest tests/ -v

lint:
	ruff check src tests
	black --check src tests

api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

app:
	streamlit run app/streamlit_app.py

docker-build:
	docker build -t customer-intent-api .

docker-run:
	docker run -p 8000:8000 --env-file .env customer-intent-api

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
