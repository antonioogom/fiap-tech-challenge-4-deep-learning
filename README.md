# Tech Challenge Fase 4 - Deep Learning

Previsão do próximo fechamento da ação **PETR4** (Petrobras) usando uma rede LSTM treinada com dados históricos do Yahoo Finance.

## Estrutura

```
├── modelo/
│   ├── train.py          # Pipeline completo: extração, pré-processamento e treinamento
│   └── lstm_petr4.pkl    # Modelo treinado (gerado após rodar train.py)
├── dados/
│   └── PETR4_clean.csv   # Dataset processado (gerado após rodar train.py)
├── notebooks/
│   └── train_test.ipynb  # Notebook exploratório
├── app.py                # API FastAPI para inferência
└── mlflow.db             # Registro de experimentos MLflow
```

## Pipeline de treinamento

1. **Extração** — dados dos últimos 2 anos via `yfinance`
2. **Tratamento** — reindex em dias úteis, interpolação de nulos, winsorização do volume
3. **Feature engineering** — retornos, log-retornos, médias móveis (7 e 21 dias), volatilidade
4. **Modelo** — LSTM com 2 camadas, hidden size 64, dropout 0.2, lookback de 120 dias
5. **Treinamento** — 100 épocas, otimizador Adam, scheduler `ReduceLROnPlateau`
6. **Tracking** — métricas (MAE, MAPE, RMSE) e artefatos logados via MLflow

## Como usar

**1. Treinar o modelo**
```bash
python modelo/train.py
```

**2. Subir a API**
```bash
uvicorn app:app --reload
```

**3. Endpoints**

| Método | Rota       | Descrição                                      |
|--------|------------|------------------------------------------------|
| GET    | `/predict` | Previsão do próximo fechamento da PETR4        |
| GET    | `/health`  | Status da API e confirmação do modelo carregado |

**Exemplo de resposta `/predict`:**
```json
{
  "ticker": "PETR4.SA",
  "data_referencia": "2025-05-19",
  "ultimo_close": 38.50,
  "previsto": 39.12,
  "variacao_pct": 1.6104
}
```

## Dependências principais

- `torch` — modelo LSTM
- `fastapi` / `uvicorn` — API REST
- `yfinance` — dados históricos
- `mlflow` — rastreamento de experimentos
- `scikit-learn` — normalização (MinMaxScaler)
