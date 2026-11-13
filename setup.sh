#!/usr/bin/env bash
# ==============================================================================
# Setup & Execution Automation Script
# Project: XAI System in Banking: AI Anomaly-Driven Credit Card Fraud Detection
# ==============================================================================

set -e

echo "======================================================================"
echo " Starting Full Setup & Pipeline Execution"
echo "======================================================================"

# 1. Virtual Environment
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

echo "[INFO] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# 2. Upgrade pip and install dependencies
echo "[INFO] Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e . --no-deps

# 3. Create required directories
mkdir -p data/raw data/processed results/models results/figures results/tables results/explanations reports

# 4. Download Dataset via Kaggle API
echo "[INFO] Checking and downloading dataset..."
python scripts/download_data.py

# 5. Run Unit Tests
echo "[INFO] Running test suite..."
pytest -v tests/

# 6. Train Models
echo "[INFO] Executing training pipeline..."
python scripts/train.py

# 7. Evaluate Models & Generate Benchmark Tables
echo "[INFO] Evaluating benchmark models..."
python scripts/evaluate.py

# 8. Generate XAI Explanations & Alert Cards
echo "[INFO] Generating Explainable AI (SHAP & LIME) explanations..."
python scripts/generate_report.py

# 9. Build Jupyter Notebooks
echo "[INFO] Verifying Jupyter notebooks..."
python scripts/build_notebooks.py

# 10. Compile PDF Report if pandoc & pdflatex available
if command -v pandoc &> /dev/null && command -v pdflatex &> /dev/null; then
    if [ -f "reports/project_report.md" ]; then
        echo "[INFO] Compiling Project Report to PDF..."
        pandoc reports/project_report.md -o reports/project_report.pdf --pdf-engine=pdflatex -V geometry:margin=1in || true
    fi
fi

echo "======================================================================"
echo " [SUCCESS] Pipeline execution complete! All deliverables generated."
echo "======================================================================"

