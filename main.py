"""
================================================================================
MAIN — Pipeline Orquestrador
================================================================================
Uso:
  python main.py --mode mock       # Sem conexão com Binance (testes)
  python main.py --mode live       # Dados reais da Binance
  python main.py --mode mock --lookback 45 --epochs 80
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_engine import DataLoader, MockDataLoader
from models.garch_model import GARCHModel
from models.lstm_model import LSTMVolatilityModel
from pipeline.sentiment_pipeline import SentimentFeatureBuilder
from backtesting.backtester import Backtester

DEFAULT_CONFIG = {
    "symbol":        "BTC/USDT",
    "timeframe":     "1d",
    "limit":         800,
    "rv_window":     22,
    "lookback":      30,
    "train_split":   0.70,
    "val_split":     0.15,
    "lstm_hidden":   64,
    "lstm_layers":   2,
    "lstm_dropout":  0.3,
    "lstm_epochs":   80,
    "lstm_batch":    32,
    "lstm_lr":       1e-3,
    "garch_p":       1,
    "garch_q":       1,
    "garch_dist":    "t",
    "use_sentiment": True,
    "output_dir":    "outputs",
}

FEATURE_COLS = [
    "log_return", "rv",
    "sma_7_ratio", "sma_21_ratio", "sma_50_ratio",
    "ema_9", "ema_21",
    "atr_ratio", "rsi_14",
    "rv_lag_1", "rv_lag_2", "rv_lag_3", "rv_lag_5",
    "volume_ratio", "hl_spread", "abs_return",
]

SENTIMENT_COLS = ["sentiment_mean", "sentiment_ema", "sentiment_std"]


def run_pipeline(cfg: dict, mode: str = "mock"):
    os.makedirs(cfg["output_dir"], exist_ok=True)

    print("\n" + "=" * 68)
    print("  BTC/USDT — Realized Volatility Forecasting System")
    print("  GARCH(1,1)  vs  LSTM + Sentiment (FinBERT)")
    print("=" * 68 + "\n")

    # ETAPA 1: DATA ENGINE
    print("--- ETAPA 1/5 — Data Engine ---")
    Loader = MockDataLoader if mode == "mock" else DataLoader
    loader = Loader(
        symbol=cfg["symbol"],
        timeframe=cfg["timeframe"],
        limit=cfg["limit"],
        rv_window=cfg["rv_window"],
    )
    df = loader.run()

    # ETAPA 2: SENTIMENTO
    feature_cols = FEATURE_COLS.copy()

    if cfg["use_sentiment"]:
        print("\n--- ETAPA 2/5 — Sentiment Analysis Pipeline ---")
        start = str(df.index[0].date())
        end   = str(df.index[-1].date())
        sent_builder = SentimentFeatureBuilder()
        df = sent_builder.inject_into_dataframe(df, start, end)
        feature_cols += SENTIMENT_COLS
        print(f"[Main] Features de sentimento adicionadas.")
    else:
        print("\n[Main] Sentimento desabilitado.")

    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"[Main] Total de features: {len(feature_cols)}")
    print(f"[Main] Shape do DataFrame: {df.shape}")

    # SPLIT TEMPORAL
    df_clean = df[feature_cols + ["rv", "rv_target", "log_return"]].dropna()
    n        = len(df_clean)
    n_train  = int(n * cfg["train_split"])
    n_val    = int(n * cfg["val_split"])

    train_df = df_clean.iloc[:n_train]
    val_df   = df_clean.iloc[n_train:n_train + n_val]
    test_df  = df_clean.iloc[n_train + n_val:]

    print(f"\n[Main] Split temporal:")
    print(f"  Treino:    {len(train_df)} dias")
    print(f"  Validacao: {len(val_df)} dias")
    print(f"  Teste:     {len(test_df)} dias")

    # ETAPA 3: GARCH
    print("\n--- ETAPA 3/5 — GARCH(1,1) Benchmark ---")
    garch = GARCHModel(p=cfg["garch_p"], q=cfg["garch_q"], dist=cfg["garch_dist"])
    garch.fit(train_df["log_return"])

    full_returns  = df_clean["log_return"]
    garch_preds   = garch.forecast_rolling(full_returns, train_size=n_train + n_val, horizon=1)
    garch_aligned = garch_preds.reindex(test_df.index).ffill().dropna()
    y_true_garch  = test_df["rv_target"].reindex(garch_aligned.index).dropna()
    garch_final   = garch_aligned.reindex(y_true_garch.index)

    # ETAPA 4: LSTM
    print("\n--- ETAPA 4/5 — LSTM Deep Learning ---")
    lstm = LSTMVolatilityModel(
        lookback=cfg["lookback"],
        hidden_size=cfg["lstm_hidden"],
        num_layers=cfg["lstm_layers"],
        dropout=cfg["lstm_dropout"],
        epochs=cfg["lstm_epochs"],
        batch_size=cfg["lstm_batch"],
        lr=cfg["lstm_lr"],
    )
    train_val_df     = df_clean.iloc[:n_train + n_val]
    lstm.fit(train_val_df, feature_cols=feature_cols, target_col="rv_target")
    lstm_pred_series = lstm.predict_series(test_df)
    y_true_lstm      = test_df["rv_target"].reindex(lstm_pred_series.index).dropna()
    lstm_final       = lstm_pred_series.reindex(y_true_lstm.index)

    # ETAPA 5: BACKTESTING
    print("\n--- ETAPA 5/5 — Backtesting e Avaliacao ---")
    bt = Backtester()
    bt.add_forecast("GARCH(1,1)", dates=y_true_garch.index,
                    y_true=y_true_garch.values, y_pred=garch_final.values)
    bt.add_forecast("LSTM", dates=y_true_lstm.index,
                    y_true=y_true_lstm.values, y_pred=lstm_final.values)

    metrics = bt.evaluate()
    report  = bt.generate_report()
    print(report)

    report_path = os.path.join(cfg["output_dir"], "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    metrics.to_csv(os.path.join(cfg["output_dir"], "metrics.csv"))

    plot_path = os.path.join(cfg["output_dir"], "volatility_forecast_comparison.png")
    bt.plot_all(save_path=plot_path)

    print(f"\n[Main] Grafico salvo em: {plot_path}")
    print(f"[Main] Relatorio salvo em: {report_path}")
    print(f"[Main] Pipeline concluido!\n")

    return bt, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTC Volatility Forecasting System")
    parser.add_argument("--mode",         default="mock", choices=["mock", "live"])
    parser.add_argument("--lookback",     type=int, default=30)
    parser.add_argument("--epochs",       type=int, default=80)
    parser.add_argument("--no-sentiment", action="store_true")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg["lookback"]      = args.lookback
    cfg["lstm_epochs"]   = args.epochs
    cfg["use_sentiment"] = not args.no_sentiment

    run_pipeline(cfg, mode=args.mode)