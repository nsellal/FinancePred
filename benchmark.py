"""Benchmark : un LSTM bat-il des baselines simples sur les rendements journaliers ?

Reprend exactement la méthode du notebook financePred.ipynb :
log-rendements, scaler ajusté sur le train seul, fenêtres de 30 jours,
early stopping sur un jeu de validation, comparaison à 3 baselines.
"""
import copy

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error


class PredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)          # h0 et c0 valent zéro par défaut
        return self.fc(out[:, -1, :])


def load_returns(ticker, start, end):
    """Télécharge les cours et renvoie les log-rendements journaliers (Series)."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"aucune donnée pour {ticker}")
    close = df["Close"].squeeze()      # MultiIndex yfinance -> Series simple
    return np.log(close).diff().dropna()


def make_windows(values, seq_length):
    """Fenêtres glissantes de seq_length lignes (pas de 1 jour)."""
    return np.array([values[i:i + seq_length] for i in range(len(values) - seq_length)])


def direction_acc(y_true, y_pred):
    return np.mean((y_true > 0) == (y_pred > 0))


def train_lstm(x_tr, y_tr, x_val, y_val, hidden_dim=32, num_layers=2,
               lr=0.01, num_epochs=200, patience=20):
    """Entraîne le LSTM avec early stopping ; renvoie le meilleur modèle et l'époque d'arrêt."""
    model = PredictionModel(1, hidden_dim, num_layers, 1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val, best_state, wait, stop_epoch = float("inf"), None, 0, num_epochs
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x_tr), y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val), y_val).item()

        if val_loss < best_val:
            best_val, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                stop_epoch = epoch
                break

    model.load_state_dict(best_state)
    return model, stop_epoch


def run_benchmark(ticker, start="2020-01-01", end="2026-11-01",
                  seq_length=30, train_ratio=0.8, seed=42):
    """Lance tout le benchmark sur un ticker et renvoie une ligne de résultats (dict)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Données
    returns = load_returns(ticker, start, end)

    # 2. Split, puis scaler ajusté sur les lignes du train seulement
    train_size = int(train_ratio * (len(returns) - seq_length))
    n_train_rows = train_size + seq_length - 1
    values = returns.to_numpy().reshape(-1, 1)
    scaler = StandardScaler().fit(values[:n_train_rows])
    scaled = scaler.transform(values)

    # 3. Fenêtres : 29 jours en entrée, le 30e en cible
    data = make_windows(scaled, seq_length)
    X, y = data[:, :-1, :], data[:, -1, :]
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # 4. LSTM avec validation = 20 % les plus récents du train
    val_size = int(0.2 * len(X_train))
    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    model, stop_epoch = train_lstm(
        to_t(X_train[:-val_size]), to_t(y_train[:-val_size]),
        to_t(X_train[-val_size:]), to_t(y_train[-val_size:]),
    )
    model.eval()
    with torch.no_grad():
        lstm_pred = scaler.inverse_transform(model(to_t(X_test)).numpy())

    # 5. Baselines (tout repasse en vrais rendements)
    y_test_true = scaler.inverse_transform(y_test)
    y_train_true = scaler.inverse_transform(y_train)

    lin = LinearRegression().fit(X_train[:, :, 0], y_train)
    lin_pred = scaler.inverse_transform(lin.predict(X_test[:, :, 0]))
    mean_pred = np.full_like(y_test_true, y_train_true.mean())
    zero_pred = np.zeros_like(y_test_true)

    def rmse(pred):
        return root_mean_squared_error(y_test_true[:, 0], pred[:, 0])

    rmse_zero, rmse_lstm = rmse(zero_pred), rmse(lstm_pred)
    return {
        "Ticker": ticker,
        "Jours test": len(y_test_true),
        "RMSE zéro": rmse_zero,
        "RMSE moyenne": rmse(mean_pred),
        "RMSE linéaire": rmse(lin_pred),
        "RMSE LSTM": rmse_lstm,
        "LSTM vs zéro (%)": 100 * (rmse_lstm / rmse_zero - 1),
        "Dir. toujours hausse": np.mean(y_test_true > 0),
        "Dir. linéaire": direction_acc(y_test_true, lin_pred),
        "Dir. LSTM": direction_acc(y_test_true, lstm_pred),
        "Époque d'arrêt": stop_epoch,
    }
