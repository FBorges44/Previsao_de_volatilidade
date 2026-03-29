"""
================================================================================
DATA ENGINE — BTC/USDT Realized Volatility Forecasting System
================================================================================

Por que prever VOLATILIDADE e não PREÇO?
────────────────────────────────────────
Prever preço bruto (P_t) é um problema de regressão sobre uma série não-estacionária,
com tendência estocástica (random walk). A volatilidade realizada (RV), derivada dos
log-retornos ao quadrado, é:
  (1) Estacionária no longo prazo → mais tratável por modelos ML/estatísticos
  (2) Mean-reverting → possui clusters identificáveis (volatility clustering)
  (3) Diretamente ligada a risco financeiro: opções, VaR, gestão de portfólio
  (4) Tem propriedades regulares de autocorrelação que LSTMs/GARCH exploram bem

Fórmula da Volatilidade Realizada (janela h):
  RV_t = Σ_{i=1}^{h} r_{t-i+1}²   onde   r_t = ln(P_t / P_{t-1})
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


class DataLoader:
    """
    Coleta dados OHLCV do par BTC/USDT via API da Binance (ccxt),
    computa log-retornos e a Volatilidade Realizada como target.

    Parameters
    ----------
    symbol     : Par de trading, ex: 'BTC/USDT'
    timeframe  : Granularidade: '1h', '4h', '1d'
    limit      : Número de candles a buscar
    rv_window  : Janela de rolling para calcular RV (em períodos)
    """

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1d",
        limit: int = 1000,
        rv_window: int = 22,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
        self.rv_window = rv_window
        self.exchange = ccxt.binance({"enableRateLimit": True})
        self.raw_df: Optional[pd.DataFrame] = None
        self.feature_df: Optional[pd.DataFrame] = None

    # ─────────────────────────────────────────────────────────────────────────
    # 1. FETCH OHLCV
    # ─────────────────────────────────────────────────────────────────────────
    def fetch_ohlcv(self) -> pd.DataFrame:
        """
        Busca candles OHLCV da Binance e retorna DataFrame estruturado.

        Returns
        -------
        DataFrame com colunas: [open, high, low, close, volume]
        Index: DatetimeIndex em UTC
        """
        print(f"[DataLoader] Buscando {self.limit} candles de {self.symbol} ({self.timeframe})...")
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=self.limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)
        self.raw_df = df
        print(f"[DataLoader] {len(df)} candles carregados. Período: {df.index[0]} → {df.index[-1]}")
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 2. LOG-RETURNS E REALIZED VOLATILITY
    # ─────────────────────────────────────────────────────────────────────────
    def compute_realized_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computa log-retornos e Volatilidade Realizada (target do modelo).

        Matemática:
          r_t = ln(Close_t / Close_{t-1})
          RV_t = sqrt( Σ_{i=0}^{h-1} r_{t-i}² )   ← desvio padrão anualizado

        A raiz quadrada da soma dos retornos ao quadrado é análoga ao desvio
        padrão amostral (sem ajuste de média, pois E[r]≈0 em HF data).
        """
        df = df.copy()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # Realized Volatility: rolling std of log returns (× sqrt(252) para anualizar)
        df["rv"] = df["log_return"].rolling(self.rv_window).std() * np.sqrt(252)

        # Target: RV do próximo período (previsão T+1)
        df["rv_target"] = df["rv"].shift(-1)

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 3. FEATURE ENGINEERING
    # ─────────────────────────────────────────────────────────────────────────
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features técnicas e estatísticas para o modelo LSTM.

        Features geradas:
          • Médias Móveis (SMA 7, 21, 50)
          • EMA (9, 21)
          • ATR — Average True Range (proxy de volatilidade intra-período)
          • RSI — Relative Strength Index (momentum / oversold/overbought)
          • Lag features de RV (autocorrelação)
          • Volume normalizado
          • High-Low spread (range relativo)
        """
        df = df.copy()

        # ── Médias Móveis Simples
        for w in [7, 21, 50]:
            df[f"sma_{w}"] = df["close"].rolling(w).mean()
            df[f"sma_{w}_ratio"] = df["close"] / df[f"sma_{w}"]  # preço relativo à SMA

        # ── EMAs
        for w in [9, 21]:
            df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()

        # ── ATR — Average True Range
        # TR = max(H-L, |H-C_prev|, |L-C_prev|)
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1))
            )
        )
        df["atr_14"] = df["tr"].rolling(14).mean()
        df["atr_ratio"] = df["atr_14"] / df["close"]  # ATR normalizado pelo preço

        # ── RSI — Relative Strength Index (Wilder, 1978)
        df["rsi_14"] = self._compute_rsi(df["close"], window=14)

        # ── Lag Features da Volatilidade (autocorrelação temporal)
        for lag in [1, 2, 3, 5, 10]:
            df[f"rv_lag_{lag}"] = df["rv"].shift(lag)

        # ── Volume features
        df["volume_ma20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma20"]

        # ── Range relativo (High-Low / Close)
        df["hl_spread"] = (df["high"] - df["low"]) / df["close"]

        # ── Retorno absoluto (proxy simples de vol)
        df["abs_return"] = df["log_return"].abs()

        self.feature_df = df.dropna()
        print(f"[DataLoader] Feature engineering concluído. Shape final: {self.feature_df.shape}")
        return self.feature_df

    @staticmethod
    def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """
        RSI de Wilder:
          RS = média(ganhos) / média(perdas) em janela exponencial
          RSI = 100 - (100 / (1 + RS))
        """
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
        avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    # ─────────────────────────────────────────────────────────────────────────
    # 4. PIPELINE COMPLETO
    # ─────────────────────────────────────────────────────────────────────────
    def run(self) -> pd.DataFrame:
        """Executa o pipeline completo: fetch → RV → features."""
        df = self.fetch_ohlcv()
        df = self.compute_realized_volatility(df)
        df = self.engineer_features(df)
        return df

    def save(self, path: str = "data/btc_features.parquet"):
        """Salva o DataFrame processado em Parquet (eficiente para séries temporais)."""
        if self.feature_df is not None:
            self.feature_df.to_parquet(path)
            print(f"[DataLoader] Dados salvos em: {path}")

    def load(self, path: str = "data/btc_features.parquet") -> pd.DataFrame:
        """Carrega dados pré-processados do disco."""
        self.feature_df = pd.read_parquet(path)
        return self.feature_df


# ─────────────────────────────────────────────────────────────────────────────
# MOCK DATA GENERATOR (para testes sem conexão com a Binance)
# ─────────────────────────────────────────────────────────────────────────────
class MockDataLoader(DataLoader):
    """
    Gera dados sintéticos realistas de BTC/USDT para desenvolvimento e testes.
    Usa um modelo GBM (Geometric Brownian Motion) com volatility clustering.
    """

    def fetch_ohlcv(self) -> pd.DataFrame:
        print(f"[MockDataLoader] Gerando {self.limit} candles sintéticos de BTC/USDT...")
        np.random.seed(42)
        n = self.limit

        # GBM com regime-switching para volatility clustering (proxy GARCH)
        dates = pd.date_range("2021-01-01", periods=n, freq="1D", tz="UTC")
        price = 30000.0
        prices = [price]

        # Simulação com dois regimes de volatilidade
        vol_high, vol_low = 0.05, 0.015
        regime = 0
        for _ in range(n - 1):
            if np.random.rand() < (0.05 if regime == 0 else 0.1):
                regime = 1 - regime
            vol = vol_high if regime == 1 else vol_low
            ret = np.random.normal(0.0002, vol)
            price *= np.exp(ret)
            prices.append(price)

        prices = np.array(prices)
        spread = np.random.uniform(0.005, 0.025, n)
        df = pd.DataFrame({
            "open":   prices * (1 - spread / 2),
            "high":   prices * (1 + spread),
            "low":    prices * (1 - spread),
            "close":  prices,
            "volume": np.random.lognormal(15, 1, n),
        }, index=dates)
        self.raw_df = df
        return df