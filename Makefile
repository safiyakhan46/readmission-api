PY ?= python
export PYTHONPATH := src:.

.PHONY: install data train test lint serve docker drift mlflow

install:
	pip install -r requirements-dev.txt

data:            ## synthetic data, for trying things without the real CSV
	$(PY) scripts/make_synthetic_data.py --n 8000 --out data/synthetic.csv

train:           ## real data: download diabetic_data.csv from UCI into data/ first
	$(PY) -m readmission.train --data data/diabetic_data.csv
	$(PY) scripts/render_model_card.py

test:
	$(PY) -m pytest -q

lint:
	ruff check src app tests scripts

serve:
	uvicorn app.main:app --reload --port 7860

docker:
	docker build -t readmission-api . && docker run --rm -p 7860:7860 readmission-api

drift:           ## e.g. make drift CURRENT=data/new_month.csv
	$(PY) -m readmission.drift --current $(CURRENT)

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db
