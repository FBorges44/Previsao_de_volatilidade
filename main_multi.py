"""
================================================================================
MAIN MULTI-ASSET — Pipeline com GARCH + LSTM + XGBoost
================================================================================
Uso:
  python main_multi.py --symbol AAPL
  python main_multi.py --symbol NVDA --period 5y
  python main_multi.py --symbol PETR4.SA
  python main_multi.py --symbol BTC/USDT --period 3y
  python main_multi.py --symbol GC=F     # Ouro
"""

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.multi_asset_engine import MultiAssetLoader, ASSET_REGISTRY
from models.garch_model import GARCHModel
from models.lstm_model import LSTMVolatilityModel
from models.xgboost_model import XGBoostVolatilityModel
from backtesting.backtester import Backtester

# ─────────────────────────────────────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "log_return", "rv", "rv_5", "rv_10", "rv_ratio",
    "sma_5_ratio", "sma_10_ratio", "sma_21_ratio", "sma_50_ratio",
    "ema_9", "ema_21",
    "atr_ratio", "rsi_14", "rsi_21",
    "macd_hist", "bb_width", "bb_pos",
    "rv_lag_1", "rv_lag_2", "rv_lag_3", "rv_lag_5", "rv_lag_10", "rv_lag_21",
    "volume_ratio", "volume_std",
    "hl_spread", "abs_return", "ret_5d", "ret_21d",
]

DEFAULT_CONFIG = {
    "period":       "3y",
    "interval":     "1d",
    "rv_window":    22,
    "train_split":  0.70,
    "val_split":    0.15,
    "lookback":     30,
    "lstm_hidden":  64,
    "lstm_layers":  2,
    "lstm_dropout": 0.3,
    "lstm_epochs":  80,
    "lstm_batch":   32,
    "lstm_lr":      1e-3,
    "xgb_estimators":  500,
    "xgb_depth":       4,
    "xgb_lr":          0.05,
    "retrain_every":   21,
    "output_dir":      "outputs",
}


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_multi_pipeline(symbol: str, cfg: dict):
    out_dir = os.path.join(cfg["output_dir"], symbol.replace("/", "_").replace("=", ""))
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 68)
    print(f"  Multi-Asset Volatility Forecasting — {symbol}")
    print(f"  GARCH(1,1)  vs  LSTM  vs  XGBoost")
    print("=" * 68 + "\n")

    # ── ETAPA 1: DATA
    print("--- ETAPA 1/5 — Data Engine ---")
    loader = MultiAssetLoader(
        symbol=symbol,
        period=cfg["period"],
        interval=cfg["interval"],
        rv_window=cfg["rv_window"],
    )
    df = loader.run()
    info = loader.get_asset_info()
    print(f"[Main] Ativo: {info['name']} | Moeda: {info['currency']} | Fator anualização: {info['ann_factor']:.2f}")

    # Filtra features disponíveis
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    print(f"[Main] Features disponíveis: {len(feature_cols)}")

    # ── SPLIT
    df_clean = df[feature_cols + ["rv", "rv_target", "log_return"]].dropna()
    n        = len(df_clean)
    n_train  = int(n * cfg["train_split"])
    n_val    = int(n * cfg["val_split"])
    n_test   = n - n_train - n_val

    train_df     = df_clean.iloc[:n_train]
    val_df       = df_clean.iloc[n_train:n_train + n_val]
    test_df      = df_clean.iloc[n_train + n_val:]
    train_val_df = df_clean.iloc[:n_train + n_val]

    print(f"\n[Main] Split: Treino={len(train_df)} | Val={len(val_df)} | Teste={len(test_df)} dias")
    print(f"[Main] Teste: {test_df.index[0].date()} → {test_df.index[-1].date()}")

    # ── ETAPA 2: GARCH
    print("\n--- ETAPA 2/5 — GARCH(1,1) ---")
    garch = GARCHModel(p=1, q=1, dist="t")
    garch.fit(train_df["log_return"])
    garch_preds   = garch.forecast_rolling(df_clean["log_return"], train_size=n_train + n_val)
    garch_aligned = garch_preds.reindex(test_df.index).ffill().dropna()
    y_true_garch  = test_df["rv_target"].reindex(garch_aligned.index).dropna()
    garch_final   = garch_aligned.reindex(y_true_garch.index)

    # ── ETAPA 3: LSTM
    print("\n--- ETAPA 3/5 — LSTM ---")
    lstm = LSTMVolatilityModel(
        lookback=cfg["lookback"],
        hidden_size=cfg["lstm_hidden"],
        num_layers=cfg["lstm_layers"],
        dropout=cfg["lstm_dropout"],
        epochs=cfg["lstm_epochs"],
        batch_size=cfg["lstm_batch"],
        lr=cfg["lstm_lr"],
    )
    lstm.fit(train_val_df, feature_cols=feature_cols, target_col="rv_target")
    lstm_series  = lstm.predict_series(test_df)
    y_true_lstm  = test_df["rv_target"].reindex(lstm_series.index).dropna()
    lstm_final   = lstm_series.reindex(y_true_lstm.index)

    # ── ETAPA 4: XGBOOST
    print("\n--- ETAPA 4/5 — XGBoost ---")
    xgb_model = XGBoostVolatilityModel(
        n_estimators=cfg["xgb_estimators"],
        max_depth=cfg["xgb_depth"],
        learning_rate=cfg["xgb_lr"],
    )
    xgb_preds   = xgb_model.forecast_rolling(
        df_clean, feature_cols=feature_cols,
        train_size=n_train + n_val,
        retrain_every=cfg["retrain_every"],
    )
    xgb_aligned = xgb_preds.reindex(test_df.index).ffill().dropna()
    y_true_xgb  = test_df["rv_target"].reindex(xgb_aligned.index).dropna()
    xgb_final   = xgb_aligned.reindex(y_true_xgb.index)

    # ── ETAPA 5: BACKTEST
    print("\n--- ETAPA 5/5 — Backtesting ---")
    bt = Backtester()
    bt.add_forecast("GARCH(1,1)", y_true_garch.index, y_true_garch.values, garch_final.values)
    bt.add_forecast("LSTM",       y_true_lstm.index,  y_true_lstm.values,  lstm_final.values)
    bt.add_forecast("XGBoost",    y_true_xgb.index,   y_true_xgb.values,   xgb_final.values)

    metrics = bt.evaluate()
    report  = bt.generate_report()
    print(report)

    # Feature importance XGBoost
    fi = xgb_model.get_feature_importance(10)
    if not fi.empty:
        print("\n[XGBoost] Feature Importance (Top 10):")
        for feat, score in fi.items():
            print(f"  {feat:25s} {score:.2f}")

    # Salva outputs
    metrics.to_csv(os.path.join(out_dir, "metrics.csv"))
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    if not fi.empty:
        fi.to_csv(os.path.join(out_dir, "feature_importance.csv"))

    plot_path = os.path.join(out_dir, "forecast_comparison.png")
    bt.plot_all(save_path=plot_path)

    print(f"\n[Main] Outputs salvos em: {out_dir}/")
    return bt, metrics, xgb_model


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Asset Volatility Forecasting")
    parser.add_argument("--symbol",  default="AAPL",
                        help="Símbolo do ativo. Ex: AAPL, NVDA, PETR4.SA, BTC/USDT, GC=F")
    parser.add_argument("--period",  default="3y",
                        help="Período de dados: 1y, 2y, 3y, 5y, max")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--epochs",   type=int, default=80)
    parser.add_argument("--retrain",  type=int, default=21,
                        help="Re-treina XGBoost a cada N dias (padrão: 21 = mensal)")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg["period"]        = args.period
    cfg["lookback"]      = args.lookback
    cfg["lstm_epochs"]   = args.epochs
    cfg["retrain_every"] = args.retrain

    run_multi_pipeline(args.symbol, cfg)