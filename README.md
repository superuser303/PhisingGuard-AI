# PhishingGuardAI
AI-powered phishing email detector using NLP and BERT.

## Overview
Detects phishing emails with ML/DL. Built with Hugging Face, Transformers.

## Setup
1. Clone repo: `git clone https://github.com/superuser303/PhishingGuardAI`
2. Install: `pip install -r requirements.txt`
3. Run preprocessing: `python scripts/preprocess.py`

## Usage
- Train: Run `notebooks/02_model_training.ipynb`
- Evaluate: Run `notebooks/03_evaluation.ipynb`
- Predict: `python scripts/predict.py`

## Results
- Accuracy: ~95% (see results/metrics.txt)

## Demo
Run `streamlit run app.py`

MIT License.
