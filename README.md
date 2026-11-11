# MarketMind AI

Explainable Financial Market Intelligence & Volatility Analytics.

## Features
- Yahoo Finance historical market data ingestion
- Return, volatility and descriptive statistical analysis
- K-Means market regime clustering
- Welch t-test for exploratory hypothesis testing
- Random Forest five-day realized-volatility forecasting with chronological holdout
- Financial news sentiment with FinBERT when installed; keyword fallback otherwise
- Feature importance and optional SHAP explanations
- Streamlit interactive dashboard

## Run locally (Windows PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If activation is blocked, use:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Notes
FinBERT downloads pretrained model weights on first use and requires internet access. SHAP is optional at runtime. PyTorch and LangChain are included as extension dependencies; the current dashboard's trained baseline is Scikit-learn Random Forest. Do not claim a PyTorch model or LangChain RAG pipeline has been implemented until those modules are added and tested.

## Methodology and limitations
- Uses adjusted daily market prices.
- Forecast target is a five-day-ahead realized-volatility proxy.
- Uses chronological 80/20 train/test split.
- K-Means cluster IDs are not inherently bull/bear labels.
- Results are educational research outputs, not investment advice.
