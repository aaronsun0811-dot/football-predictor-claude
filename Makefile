PYTHON ?= python3

.PHONY: install test init update serve

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

init:
	$(PYTHON) predict.py init-db

update:
	$(PYTHON) predict.py update

serve:
	$(PYTHON) predict.py serve --port 8000
