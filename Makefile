PY = .venv/bin/python
export PYTHONPATH = src

setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

data:
	$(PY) scripts/download_data.py

eda:
	$(PY) -m fraud.eda

train:
	$(PY) -W ignore -m fraud.train

drift:
	$(PY) -m fraud.monitor

serve:
	.venv/bin/uvicorn fraud.serve:app --port 8000

bench:
	$(PY) scripts/bench_latency.py

test:
	.venv/bin/pytest -q

all: data eda train drift bench
