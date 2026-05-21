# PETR4 — Pipeline completo: extração, pré-processamento e treinamento LSTM
# Uso: python modelo/train.py  (executar da raiz do projeto)

import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# Diretórios
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "dados"
MODEL_DIR = ROOT / "modelo"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

TICKER = "PETR4.SA"
LOOKBACK = 120
TEST_RATIO = 0.20
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HPARAMS = {
    "ticker"      : TICKER,
    "lookback"    : LOOKBACK,
    "hidden_size" : 64,
    "num_layers"  : 2,
    "dropout"     : 0.2,
    "epochs"      : 100,
    "batch_size"  : BATCH_SIZE,
    "lr"          : 1e-3,
    "optimizer"   : "Adam",
    "loss_fn"     : "MSE",
}


class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,   # dropout entre as camadas LSTM (inter-layer)
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def iqr_bounds(series: pd.Series, factor: float = 1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr


def build_sequences(data: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(preds)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

  

if __name__ == "__main__":
    pd.set_option("display.float_format", "{:.4f}".format)

    # 1 - Extração de dados 
    END_DATE   = datetime.today().strftime("%Y-%m-%d")
    START_DATE = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")
    print(f"\nPeríodo: {START_DATE} → {END_DATE}")

    raw_df = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
    raw_df.columns = raw_df.columns.get_level_values(0)
    raw_df.index.name = "Date"
    print(raw_df.dtypes)
    print(raw_df.describe())

    # 2 - Tratamento de valores nulos 
    df = raw_df.copy()
    print("\n=== Valores nulos (antes do tratamento) ===")
    print(df.isnull().sum())

    reindex_end = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    business_days = pd.bdate_range(start=START_DATE, end=reindex_end)
    df = df.reindex(business_days)
    df.index.name = "Date"
    print(f"\nDias úteis esperados : {len(business_days)}")
    print(f"Linhas após reindex  : {len(df)}")

    price_cols = ["Open", "High", "Low", "Close"]
    volume_col = ["Volume"]

    df[price_cols] = df[price_cols].interpolate(method="time")
    df[volume_col] = df[volume_col].ffill().bfill()

    status = "Nenhum nulo restante." if df.isnull().sum().sum() == 0 else "ATENÇÃO: ainda há nulos."
    print(f"Valores nulos após tratamento: {df.isnull().sum().sum()}  — {status}")

    # 3 - Detecção e tratamento de outliers
    outlier_report = {}
    for col in price_cols + volume_col:
        lo, hi = iqr_bounds(df[col])
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        outlier_report[col] = {"lower": lo, "upper": hi, "outliers": n_out}

    print("\n=== Relatório de outliers (IQR) ===")
    print(pd.DataFrame(outlier_report).T)

    df_clean = df.copy()
    print("\n=== Pontos com |Z-score| > 3 por coluna ===")
    for col in price_cols + volume_col:
        z = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
        print(f"  {col:8s}: {(z.abs() > 3.0).sum()} ponto(s)")

    # Winsorização apenas no Volume — preços extremos são sinal real de mercado
    lo_vol, hi_vol = iqr_bounds(df_clean["Volume"])
    df_clean["Volume"] = df_clean["Volume"].clip(lower=lo_vol, upper=hi_vol)

    print("\nWinsorização aplicada apenas ao Volume.")

    # 4 - Feature Engineering
    df_feat = df_clean.copy()
    df_feat["Return"]        = df_feat["Close"].pct_change()
    df_feat["Log_Return"]    = np.log(df_feat["Close"] / df_feat["Close"].shift(1))
    df_feat["MA_7"]          = df_feat["Close"].rolling(7).mean()
    df_feat["MA_21"]         = df_feat["Close"].rolling(21).mean()
    df_feat["Volatility_21"] = df_feat["Log_Return"].rolling(21).std()
    df_feat.dropna(inplace=True)

    print(f"\nShape final do dataset: {df_feat.shape}")

    csv_path = DATA_DIR / "PETR4_clean.csv"
    df_feat.to_csv(csv_path)
    print(f"Dataset salvo em: {csv_path}")

    # 5 - Preparação LSTM
    close_vals = df_feat[["Close"]].values
    split_raw  = int(len(close_vals) * (1 - TEST_RATIO))

    # Scaler ajustado em todo o dataset: preços financeiros são não-estacionários e o
    # período de teste tem preços acima do máximo do treino. Escalar só no treino geraria
    # inputs > 1 no LSTM. Leve leakage nos parâmetros de escala — o modelo treina apenas
    # nas sequências de treino, sem ver o futuro.
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaled = scaler.fit_transform(close_vals)

    X, y   = build_sequences(close_scaled, LOOKBACK)
    split  = int(len(X) * (1 - TEST_RATIO))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    X_train = X_train.reshape(*X_train.shape, 1)
    X_test  = X_test.reshape(*X_test.shape,  1)

    print(f"X_train : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"X_test  : {X_test.shape}   |  y_test  : {y_test.shape}")

    # 6 - DataLoaders
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=BATCH_SIZE, shuffle=False)

    # 7 -Modelo
    model = LSTMForecaster(
        hidden_size=HPARAMS["hidden_size"],
        num_layers=HPARAMS["num_layers"],
        dropout=HPARAMS["dropout"],
    ).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{model}\nParâmetros treináveis: {total_params:,}")

    # 8 - Treinamento com mlflow
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=HPARAMS["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("PETR4_LSTM")
    with mlflow.start_run(run_name="lstm_baseline"):
        mlflow.log_params(HPARAMS)

        for epoch in range(1, HPARAMS["epochs"] + 1):
            train_loss = run_epoch(model, train_loader, criterion, optimizer)
            val_loss   = run_epoch(model, test_loader,  criterion)
            scheduler.step(val_loss)
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            if epoch % 10 == 0 or epoch == 1:
                print(f"Época {epoch:3d}/{HPARAMS['epochs']}  "
                      f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        mlflow.pytorch.log_model(model, name="lstm_model")
        run_id = mlflow.active_run().info.run_id
        print(f"\nRun finalizada — ID: {run_id}")

    # Salva PKL
    payload = {
        "model_state_dict" : model.state_dict(),
        "hparams"          : HPARAMS,
        "scaler"           : scaler,
        "lookback"         : LOOKBACK,
        "ticker"           : TICKER,
    }
    pkl_path = MODEL_DIR / "lstm_petr4.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Modelo salvo em: {pkl_path}")

    # 9 - Predições e métricas
    y_train_pred = scaler.inverse_transform(predict(model, train_loader))
    y_test_pred  = scaler.inverse_transform(predict(model, test_loader))
    y_train_real = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_real  = scaler.inverse_transform(y_test.reshape(-1,  1))

    metrics = {
        "set" : ["Treino",          "Teste"],
        "MAE" : [mae(y_train_real,  y_train_pred),  mae(y_test_real,  y_test_pred)],
        "MAPE%" : [mape(y_train_real, y_train_pred),  mape(y_test_real, y_test_pred)],
        "RMSE" : [rmse(y_train_real, y_train_pred),  rmse(y_test_real, y_test_pred)],
    }
    print("\n Métricas de Avaliação")
    print(pd.DataFrame(metrics).set_index("set").to_string())

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({
            "test_MAE"  : metrics["MAE"][1],
            "test_MAPE" : metrics["MAPE%"][1],
            "test_RMSE" : metrics["RMSE"][1],
        })

    # 10 - Previsão do próximo pregão
    last_window   = close_scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)
    last_window_t = torch.tensor(last_window, dtype=torch.float32).to(DEVICE)

    model.eval()
    with torch.no_grad():
        next_scaled = model(last_window_t).cpu().numpy()

    next_price = scaler.inverse_transform(next_scaled)[0, 0]
    last_close = df_feat["Close"].iloc[-1]
    variation = (next_price - last_close) / last_close * 100

    print(f"Último Close : R$ {last_close:.2f}  ({df_feat.index[-1].date()})")
    print(f"Previsão LSTM : R$ {next_price:.2f}")
    print(f"Variação : {variation:+.4f}%")
