"""
================================================================================
MÓDULO XGBOOST — Gradient Boosting para Previsão de Volatilidade
================================================================================

Por que XGBoost para volatilidade?
────────────────────────────────────
XGBoost (Chen & Guestrin, 2016) é um ensemble de árvores de decisão com
gradient boosting. Para position trading, tem vantagens sobre LSTM:

  1. INTERPRETABILIDADE: Feature importance mostra o que realmente importa
  2. VELOCIDADE: Treina em segundos vs minutos do LSTM
  3. ROBUSTEZ: Não precisa de normalização, lida bem com outliers
  4. TABULAR DATA: Superior ao LSTM quando features são bem construídas

A Renaissance Technologies usa variantes de gradient boosting extensivamente
em seus modelos de curto e médio prazo.

Pipeline:
  Features_t → XGBoost(n_estimators, max_depth) → RV_{t+1}
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[WARNING] XGBoost não instalado. Rode: pip install xgboost")


class XGBoostVolatilityModel:
    """
    Modelo XGBoost para previsão de volatilidade com:
      - Feature importance automática
      - Walk-forward validation
      - Early stopping
      - Hyperparameter grid search simples

    Parameters
    ----------
    n_estimators  : número de árvores
    max_depth     : profundidade máxima das árvores
    learning_rate : taxa de aprendizado (eta)
    subsample     : fração de amostras por árvore (regularização)
    colsample     : fração de features por árvore (regularização)
    lookback      : janela de lag features (alinha com LSTM)
    """

    def __init__(
        self,
        n_estimators:  int   = 500,
        max_depth:     int   = 4,
        learning_rate: float = 0.05,
        subsample:     float = 0.8,
        colsample:     float = 0.8,
        min_child_weight: int = 5,
        lookback:      int   = 1,
    ):
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.learning_rate    = learning_rate
        self.subsample        = subsample
        self.colsample        = colsample
        self.min_child_weight = min_child_weight
        self.lookback         = lookback
        self.model            = None
        self.feature_cols_:   List[str] = []
        self.feature_importance_: Optional[pd.Series] = None

    # ─────────────────────────────────────────────────────────────────────────
    # FIT
    # ─────────────────────────────────────────────────────────────────────────
    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "rv_target",
        val_split: float = 0.15,
    ) -> "XGBoostVolatilityModel":
        """
        Treina o XGBoost com early stopping na validação.

        O XGBoost usa o target diretamente (sem janelas temporais),
        mas inclui lag features que já estão no feature_cols.
        """
        if not XGB_AVAILABLE:
            print("[XGBoost] Biblioteca não disponível. Usando fallback.")
            self._fit_fallback(df, feature_cols, target_col)
            return self

        self.feature_cols_ = feature_cols
        data = df[feature_cols + [target_col]].dropna()

        X = data[feature_cols].values
        y = data[target_col].values

        # Split temporal
        n_val   = int(len(X) * val_split)
        X_train = X[:-n_val]
        y_train = y[:-n_val]
        X_val   = X[-n_val:]
        y_val   = y[-n_val:]

        print(f"[XGBoost] Treino: {X_train.shape} | Validação: {X_val.shape}")
        print(f"[XGBoost] Features: {len(feature_cols)} | Estimadores: {self.n_estimators}")

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
        dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=feature_cols)

        params = {
            "objective":        "reg:squarederror",
            "eval_metric":      "rmse",
            "max_depth":        self.max_depth,
            "learning_rate":    self.learning_rate,
            "subsample":        self.subsample,
            "colsample_bytree": self.colsample,
            "min_child_weight": self.min_child_weight,
            "tree_method":      "hist",
            "seed":             42,
        }

        evals_result = {}
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=50,
            evals_result=evals_result,
            verbose_eval=False,
        )

        best_round = self.model.best_iteration
        val_rmse   = evals_result["val"]["rmse"][best_round]
        print(f"[XGBoost] Convergiu na rodada {best_round} | Val RMSE: {val_rmse:.6f}")

        # Feature importance
        scores = self.model.get_score(importance_type="gain")
        self.feature_importance_ = pd.Series(scores).sort_values(ascending=False)
        print(f"[XGBoost] Top 5 features: {list(self.feature_importance_.head(5).index)}")

        return self

    def _fit_fallback(self, df, feature_cols, target_col):
        """Fallback simples: média dos últimos valores de RV."""
        data = df[feature_cols + [target_col]].dropna()
        self.feature_cols_ = feature_cols
        self._fallback_mean = float(data[target_col].mean())
        self._fallback_std  = float(data[target_col].std())

    # ─────────────────────────────────────────────────────────────────────────
    # PREDICT
    # ─────────────────────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Gera previsões para o DataFrame inteiro."""
        if not XGB_AVAILABLE or self.model is None:
            n = len(df.dropna(subset=self.feature_cols_))
            return np.full(n, getattr(self, "_fallback_mean", 0.3))

        data = df[self.feature_cols_].dropna()
        dtest = xgb.DMatrix(data.values, feature_names=self.feature_cols_)
        return self.model.predict(dtest)

    def predict_series(self, df: pd.DataFrame) -> pd.Series:
        """Retorna previsões como pd.Series com DatetimeIndex."""
        data  = df[self.feature_cols_].dropna()
        preds = self.predict(df)
        n     = min(len(preds), len(data))
        return pd.Series(preds[:n], index=data.index[:n], name="xgb_forecast")

    def get_feature_importance(self, top_n: int = 15) -> pd.Series:
        """Retorna as N features mais importantes."""
        if self.feature_importance_ is None:
            return pd.Series(dtype=float)
        return self.feature_importance_.head(top_n)

    # ─────────────────────────────────────────────────────────────────────────
    # WALK-FORWARD BACKTEST
    # ─────────────────────────────────────────────────────────────────────────
    def forecast_rolling(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "rv_target",
        train_size: int = 400,
        retrain_every: int = 21,
    ) -> pd.Series:
        """
        Walk-forward com re-treino periódico.
        Re-treina a cada `retrain_every` dias (simula uso real).

        Parameters
        ----------
        retrain_every : re-treina o modelo a cada N passos (21 = mensal)
        """
        data = df[feature_cols + [target_col]].dropna()
        X    = data[feature_cols].values
        y    = data[target_col].values
        idx  = data.index

        preds, pred_idx = [], []
        model = None

        print(f"[XGBoost] Walk-forward: {len(data) - train_size} steps | Re-treino a cada {retrain_every} dias...")

        for t in range(train_size, len(data)):
            # Re-treina periodicamente
            if (t - train_size) % retrain_every == 0 or model is None:
                X_tr, y_tr = X[:t], y[:t]
                if XGB_AVAILABLE:
                    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
                    params = {
                        "objective":        "reg:squarederror",
                        "max_depth":        self.max_depth,
                        "learning_rate":    self.learning_rate,
                        "subsample":        self.subsample,
                        "colsample_bytree": self.colsample,
                        "tree_method":      "hist",
                        "seed":             42,
                    }
                    model = xgb.train(params, dtrain,
                                      num_boost_round=200, verbose_eval=False)

            # Previsão
            if model and XGB_AVAILABLE:
                dpred = xgb.DMatrix(X[t:t+1], feature_names=feature_cols)
                pred  = float(model.predict(dpred)[0])
            else:
                pred = float(np.mean(y[:t]))

            preds.append(max(pred, 0.0))
            pred_idx.append(idx[t])

        print(f"[XGBoost] Walk-forward concluído. {len(preds)} previsões.")
        return pd.Series(preds, index=pred_idx, name="xgb_forecast")