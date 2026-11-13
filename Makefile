.PHONY: all setup install download test train evaluate report notebooks pdf clean help

PYTHON := ./.venv/bin/python
PYTEST := ./.venv/bin/pytest

help:
	@echo "Available commands:"
	@echo "  make setup      - Full environment setup, dataset download, and execution"
	@echo "  make install    - Install requirements in virtual environment"
	@echo "  make download   - Download Kaggle credit card fraud datasets"
	@echo "  make test       - Run unit and integration tests"
	@echo "  make train      - Train all models (Autoencoder, Supervised, Hybrid, Baselines)"
	@echo "  make evaluate   - Evaluate all models and generate PR/ROC curves and tables"
	@echo "  make report     - Generate SHAP/LIME plots and banking analyst alert cards"
	@echo "  make notebooks  - Generate/refresh all 5 Jupyter notebooks"
	@echo "  make pdf        - Compile Markdown project report to PDF using pandoc"
	@echo "  make all        - Run end-to-end pipeline and generate all deliverables"
	@echo "  make clean      - Clean temporary cache files"

setup:
	bash setup.sh

install:
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e . --no-deps

download:
	$(PYTHON) scripts/download_data.py

test:
	$(PYTEST) -v tests/

train:
	$(PYTHON) scripts/train.py

evaluate:
	$(PYTHON) scripts/evaluate.py

report:
	$(PYTHON) scripts/generate_report.py

notebooks:
	$(PYTHON) scripts/build_notebooks.py

pdf:
	pandoc reports/project_report.md -o reports/project_report.pdf --pdf-engine=pdflatex -V geometry:margin=1in

all: test train evaluate report notebooks pdf

clean:
	rm -rf __pycache__ src/**/__pycache__ tests/__pycache__ .pytest_cache

