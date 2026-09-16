.PHONY: setup run schedule train test down status mlflow
PYTHON ?= .venv/bin/python
setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements-dev.txt
run:
	$(PYTHON) pipeline.py run
schedule:
	$(PYTHON) pipeline.py schedule --interval 300
train:
	$(PYTHON) pipeline.py run --train-only
test:
	$(PYTHON) -m pytest -q
status:
	$(PYTHON) pipeline.py status
down:
	$(PYTHON) pipeline.py down
mlflow:
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
