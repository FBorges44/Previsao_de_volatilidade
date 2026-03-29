"""
================================================================================
MÓDULO DE DEEP LEARNING — LSTM para Previsão de Volatilidade
================================================================================

Por que LSTM para volatilidade?
────────────────────────────────
LSTMs (Long Short-Term Memory, Hochreiter & Schmidhuber, 1997) resolvem o
problema do vanishing gradient de RNNs simples, capturando dependências
de longa distância em séries temporais.

Arquitetura da célula LSTM:
  f_t = σ(W_f·[h_{t-1}, x_t] + b_f)   ← forget gate
  i_t = σ(W_i·[h_{t-1}, x_t] + b_i)   ← input gate
  C̃_t = tanh(W_C·[h_{t-1}, x_t] + b_C) ← candidate cell
  C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t     ← cell state update
  o_t = σ(W_o·[h_{t-1}, x_t] + b_o)   ← output gate
  h_t = o_t ⊙ tanh(C_t)               ← hidden state

Para volatilidade, isso significa: o modelo pode "lembrar" que estamos
num regime de alta vol por 30+ dias e "esquecer" spikes antigos irrelevantes.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# SKLEARN-STYLE SCALER (sem dependência de sklearn)
# ─────────────────────────────────────────────────────────────────────────────
class MinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        denom = self.max_ - self.min_
        denom[denom == 0] = 1.0
        return (X - self.min_) / denom

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * (self.max_ - self.min_) + self.min_


# ─────────────────────────────────────────────────────────────────────────────
# PYTORCH LSTM ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:

    class LSTMNet(nn.Module):
        """
        Rede LSTM multi-camada com Dropout para previsão de volatilidade.

        Arquitetura:
          [Input(seq, features)] → LSTM(hidden) × num_layers → Dropout → FC → [Output(1)]

        Parameters
        ----------
        input_size   : número de features por timestep
        hidden_size  : dimensão do estado oculto h_t
        num_layers   : profundidade do LSTM (stacked)
        dropout      : taxa de Dropout entre camadas LSTM (regularização)
        output_size  : 1 (previsão escalar de RV)
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.3,
            output_size: int = 1,
        ):
            super(LSTMNet, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,          # (batch, seq, features)
            )
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden_size, 32)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, output_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch_size, seq_len, input_size)
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

            out, _ = self.lstm(x, (h0, c0))
            out = self.dropout(out[:, -1, :])   # último timestep apenas
            out = self.relu(self.fc1(out))
            out = self.fc2(out)
            return out.squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# LSTM MODEL WRAPPER (Interface Unificada)
# ─────────────────────────────────────────────────────────────────────────────
class LSTMVolatilityModel:
    """
    Wrapper de alto nível para o modelo LSTM com:
      - Preparação de janelas temporais (lookback windows)
      - Normalização de features
      - Treino com early stopping
      - Previsão de T+1

    Parameters
    ----------
    lookback     : janela temporal (30–60 dias recomendado)
    hidden_size  : neurônios na camada LSTM
    num_layers   : camadas LSTM empilhadas
    dropout      : taxa de regularização Dropout
    epochs       : épocas de treino máximo
    batch_size   : tamanho do mini-batch
    lr           : learning rate (Adam optimizer)
    patience     : early stopping patience
    backend      : 'pytorch' ou 'tensorflow'
    """

    def __init__(
        self,
        lookback: int = 30,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        patience: int = 15,
        backend: str = "pytorch",
    ):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.backend = backend if (backend == "tensorflow" and TF_AVAILABLE) else "pytorch"

        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.model = None
        self.train_history: List[float] = []
        self.val_history: List[float] = []
        self.feature_cols: List[str] = []

    # ─────────────────────────────────────────────────────────────────────────
    # WINDOW CREATION
    # ─────────────────────────────────────────────────────────────────────────
    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converte arrays 2D em sequências 3D para o LSTM.
        Input:  X(T, F), y(T,)
        Output: Xs(T-L, L, F), ys(T-L,)
        onde L = lookback
        """
        Xs, ys = [], []
        for i in range(self.lookback, len(X)):
            Xs.append(X[i - self.lookback : i])
            ys.append(y[i])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # FIT
    # ─────────────────────────────────────────────────────────────────────────
    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "rv_target",
        val_split: float = 0.15,
    ) -> "LSTMVolatilityModel":
        """
        Treina o modelo LSTM com early stopping.

        Parameters
        ----------
        df           : DataFrame com features e target
        feature_cols : lista de colunas usadas como input
        target_col   : coluna target (RV do próximo período)
        val_split    : fração para validação
        """
        self.feature_cols = feature_cols
        data = df[feature_cols + [target_col]].dropna()

        X_raw = data[feature_cols].values
        y_raw = data[[target_col]].values

        # Normalização
        X_scaled = self.feature_scaler.fit_transform(X_raw)
        y_scaled = self.target_scaler.fit_transform(y_raw).ravel()

        # Sequências temporais
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)

        # Split temporal (sem shuffle — dados de séries temporais!)
        n_val = int(len(X_seq) * val_split)
        X_train, X_val = X_seq[:-n_val], X_seq[-n_val:]
        y_train, y_val = y_seq[:-n_val], y_seq[-n_val:]

        print(f"[LSTM] Treino: {X_train.shape} | Validação: {X_val.shape}")
        print(f"[LSTM] Lookback: {self.lookback} | Features: {len(feature_cols)} | Backend: {self.backend}")

        if self.backend == "pytorch" and TORCH_AVAILABLE:
            self._fit_pytorch(X_train, y_train, X_val, y_val)
        elif TF_AVAILABLE:
            self._fit_tensorflow(X_train, y_train, X_val, y_val)
        else:
            print("[LSTM] Nenhum framework disponível. Usando modelo dummy.")
            self._fit_dummy(X_train, y_train)

        return self

    def _fit_pytorch(self, X_tr, y_tr, X_val, y_val):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LSTM/PyTorch] Device: {device}")

        self.model = LSTMNet(
            input_size=X_tr.shape[2],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()

        # DataLoaders
        train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)
        val_dl = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            # — Treino —
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # gradient clipping
                optimizer.step()
                train_loss += loss.item()

            # — Validação —
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = self.model(xb)
                    val_loss += criterion(pred, yb).item()

            train_loss /= len(train_dl)
            val_loss /= len(val_dl)
            self.train_history.append(train_loss)
            self.val_history.append(val_loss)

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

            if patience_counter >= self.patience:
                print(f"  [EarlyStopping] Parado na época {epoch+1}. Melhor val loss: {best_val_loss:.6f}")
                break

        # Restaura melhor modelo
        if best_state:
            self.model.load_state_dict(best_state)
        self._device = device

    def _fit_tensorflow(self, X_tr, y_tr, X_val, y_val):
        inputs = keras.Input(shape=(X_tr.shape[1], X_tr.shape[2]))
        x = keras.layers.LSTM(self.hidden_size, return_sequences=True, dropout=self.dropout)(inputs)
        x = keras.layers.LSTM(self.hidden_size // 2, dropout=self.dropout)(x)
        x = keras.layers.Dense(32, activation="relu")(x)
        x = keras.layers.Dropout(self.dropout)(x)
        output = keras.layers.Dense(1)(x)

        self.model = keras.Model(inputs, output)
        self.model.compile(optimizer=keras.optimizers.Adam(self.lr), loss="mse")

        cb = [keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True)]
        hist = self.model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                              epochs=self.epochs, batch_size=self.batch_size,
                              callbacks=cb, verbose=0)
        self.train_history = hist.history["loss"]
        self.val_history = hist.history["val_loss"]

    def _fit_dummy(self, X_tr, y_tr):
        """Fallback: média móvel simples como proxy quando nenhum framework disponível."""
        self._dummy_mean = float(y_tr.mean())

    # ─────────────────────────────────────────────────────────────────────────
    # PREDICT
    # ─────────────────────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Gera previsões de RV para todo o DataFrame (modo inference).

        Returns
        -------
        np.ndarray com volatilidade prevista na escala original
        """
        data = df[self.feature_cols].dropna()
        X_scaled = self.feature_scaler.transform(data.values.astype(np.float32))
        X_seq = np.array([X_scaled[i - self.lookback:i]
                          for i in range(self.lookback, len(X_scaled))], dtype=np.float32)

        if self.backend == "pytorch" and TORCH_AVAILABLE and isinstance(self.model, nn.Module):
            self.model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X_seq).to(self._device)
                preds_scaled = self.model(X_t).cpu().numpy()
        elif TF_AVAILABLE and hasattr(self.model, 'predict'):
            preds_scaled = self.model.predict(X_seq, verbose=0).ravel()
        else:
            preds_scaled = np.full(len(X_seq), self._dummy_mean)

        # Desnormalizar
        preds_scaled = preds_scaled.reshape(-1, 1)
        preds = self.target_scaler.inverse_transform(preds_scaled).ravel()
        return preds

    def predict_series(self, df: pd.DataFrame) -> pd.Series:
        """Retorna previsões como pd.Series com DatetimeIndex correto."""
        data = df[self.feature_cols].dropna()
        valid_idx = data.index[self.lookback:]
        preds = self.predict(df)
        n = min(len(preds), len(valid_idx))
        return pd.Series(preds[:n], index=valid_idx[:n], name="lstm_forecast")

    def save(self, path: str):
        if self.backend == "pytorch" and TORCH_AVAILABLE:
            torch.save(self.model.state_dict(), path)
        print(f"[LSTM] Modelo salvo em: {path}")

    def load(self, path: str, input_size: int):
        if self.backend == "pytorch" and TORCH_AVAILABLE:
            self.model = LSTMNet(input_size, self.hidden_size, self.num_layers, self.dropout)
            self.model.load_state_dict(torch.load(path))
            self.model.eval()