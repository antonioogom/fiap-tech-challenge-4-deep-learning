import pickle
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Iniciar: uvicorn app:app --reload

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "modelo"))
from modelo.train import LSTMForecaster

PKL_PATH = ROOT / "modelo" / "lstm_petr4.pkl"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model  = None
_scaler = None
_meta   = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _scaler, _meta

    if not PKL_PATH.exists():
        raise RuntimeError(f"Modelo não encontrado em {PKL_PATH}. Execute 'python modelo/train.py' primeiro.")

    with open(PKL_PATH, "rb") as f:
        payload = pickle.load(f)

    cfg    = payload["hparams"]
    _model = LSTMForecaster(hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"], dropout=cfg["dropout"]).to(DEVICE)
    _model.load_state_dict(payload["model_state_dict"])
    _model.eval()
    _scaler = payload["scaler"]
    _meta   = {"ticker": payload["ticker"], "lookback": payload["lookback"]}

    yield



class PredictResponse(BaseModel):
    ticker:          str
    data_referencia: str
    ultimo_close:    float
    previsto:        float
    variacao_pct:    float

app = FastAPI(title="PETR4 LSTM Forecaster", version="1.0.0", lifespan=lifespan)
@app.get("/predict", response_model=PredictResponse, summary="Previsão do próximo fechamento")
def predict():
    ticker   = "PETR4.SA"
    lookback = 120

    end   = datetime.today()
    start = end - timedelta(days=lookback * 2)  # margem para feriados / gaps

    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
    )
    if df.empty:
        raise HTTPException(status_code=503, detail="Não foi possível obter dados do yfinance.")

    df.columns = df.columns.get_level_values(0)
    close = df["Close"].dropna().values

    if len(close) < lookback:
        raise HTTPException(
            status_code=422,
            detail=f"Dados insuficientes: {len(close)} dias disponíveis, mínimo {lookback}.",
        )

    last_close    = float(close[-1])
    window_scaled = _scaler.transform(close[-lookback:].reshape(-1, 1)).reshape(1, lookback, 1)
    window_t      = torch.tensor(window_scaled, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        pred_scaled = _model(window_t).cpu().numpy()

    predicted    = float(_scaler.inverse_transform(pred_scaled)[0, 0])
    variacao_pct = round((predicted - last_close) / last_close * 100, 4)

    return PredictResponse(
        ticker          = ticker,
        data_referencia = df.index[-1].strftime("%Y-%m-%d"),
        ultimo_close    = round(last_close, 2),
        previsto        = round(predicted, 2),
        variacao_pct    = variacao_pct,
    )


@app.get("/health", summary="Status da API")
def health():
    return {"status": "ok", "modelo_carregado": _model is not None, "ticker": _meta.get("ticker")}
