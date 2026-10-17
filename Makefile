PYTHON ?= python

.PHONY: setup contracts check dev
setup:
	$(PYTHON) -m pip install -e "backend[dev,service]"

contracts:
	$(PYTHON) -m navidu.contracts.generate

check:
	$(PYTHON) -m ruff check backend/navidu backend/tests
	$(PYTHON) -m pytest backend/tests

dev:
	$(PYTHON) -m navidu.service
