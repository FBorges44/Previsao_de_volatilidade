"""
================================================================================
DATA ENGINE MULTI-ASSET — Stocks + Crypto
================================================================================
Suporta:
  - Ações: AAPL, NVDA, PETR4.SA via yfinance
  - Cripto: BTC/USDT, ETH/USDT via ccxt (Binance)
  - Forex: EUR/USD via yfinance
  - Commodities: GC=F (Ouro), CL=F (Petróleo) via yfinance
"""

import pandas as pd
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("[WARNING] yfinance não instalado. Rode: pip install yfinance")

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# ASSET REGISTRY — mapeamento de símbolos
# ─────────────────────────────────────────────────────────────────────────────
ASSET_REGISTRY = {
    # Ações US
    "AAPL":   {"source": "yfinance", "name": "Apple Inc.",         "currency": "USD"},
    "NVDA":   {"source": "yfinance", "name": "NVIDIA Corp.",       "currency": "USD"},
    "MSFT":   {"source": "yfinance", "name": "Microsoft Corp.",    "currency": "USD"},
    "TSLA":   {"source": "yfinance", "name": "Tesla Inc.",         "currency": "USD"},
    "GOOGL":  {"source": "yfinance", "name": "Alphabet Inc.",      "currency": "USD"},
    "SPY":    {"source": "yfinance", "name": "S&P 500 ETF",        "currency": "USD"},
    "QQQ":    {"source": "yfinance", "name": "Nasdaq 100 ETF",     "currency": "USD"},
    # Ações BR
    "PETR4.SA": {"source": "yfinance", "name": "Petrobras PN",     "currency": "BRL"},
    "VALE3.SA": {"source": "yfinance", "name": "Vale ON",          "currency": "BRL"},
    "ITUB4.SA": {"source": "yfinance", "name": "Itaú PN",          "currency": "BRL"},
    "BVSP":     {"source": "yfinance", "name": "Ibovespa",         "currency": "BRL"},
    # Commodities
    "GC=F":   {"source": "yfinance", "name": "Ouro Futuro",        "currency": "USD"},
    "CL=F":   {"source": "yfinance", "name": "Petróleo WTI",       "currency": "USD"},
    "SI=F":   {"source": "yfinance", "name": "Prata Futuro",       "currency": "USD"},
    # Forex
    "EURUSD=X": {"source": "yfinance", "name": "EUR/USD",          "currency": "USD"},
    "USDBRL=X": {"source": "yfinance", "name": "USD/BRL",          "currency": "BRL"},
    # Cripto
    "BTC/USDT": {"source": "ccxt",    "name": "Bitcoin",           "currency": "USDT"},
    "ETH/USDT": {"source": "ccxt",    "name": "Ethereum",          "currency": "USDT"},
    "SOL/USDT": {"source": "ccxt",    "name": "Solana",            "currency": "USDT"},
}

# Dias de negociação por ano por tipo de ativo
TRADING_DAYS = {
    "stocks":    252,   # mercado de ações
    "crypto":    365,   # cripto 24/7
    "forex":     252,
    "commodity": 252,
}


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-ASSET DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────
class MultiAssetLoader:
    """
    Carrega dados OHLCV de qualquer ativo suportado e calcula
    Volatilidade Realizada + features técnicas.

    Parameters
    ----------
    symbol    : Símbolo do ativo (ex: 'AAPL', 'BTC/USDT', 'PETR4.SA')
    period    : Período para yfinance ('2y', '5y', 'max') ou número de candles para ccxt
    interval  : Granularidade ('1d', '1wk' para yfinance | '1d', '4h' para ccxt)
    rv_window : Janela para cálculo de Volatilidade Realizada
    """

    def __init__(
        self,
        symbol: str = "AAPL",
        period: str = "3y",
        interval: str = "1d",
        rv_window: int = 22,
    ):
        self.symbol    = symbol
        self.period    = period
        self.interval  = interval
        self.rv_window = rv_window
        self.asset_info = ASSET_REGISTRY.get(symbol, {"source": "yfinance", "name": symbol, "currency": "USD"})
        self.source    = self.asset_info["source"]
        self.ann_factor = np.sqrt(TRADING_DAYS.get("crypto" if self.source == "ccxt" else "stocks", 252))
        self.raw_df:     Optional[pd.DataFrame] = None
        self.feature_df: Optional[pd.DataFrame] = None

    def fetch(self) -> pd.DataFrame:
        """Busca dados da fonte correta."""
        print(f"[MultiAssetLoader] Buscando {self.symbol} ({self.asset_info.get('name', '')}) via {self.source}...")
        if self.source == "yfinance":
            return self._fetch_yfinance()
        elif self.source == "ccxt":
            return self._fetch_ccxt()
        else:
            raise ValueError(f"Fonte desconhecida: {self.source}")

    def _fetch_yfinance(self) -> pd.DataFrame:
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance não instalado. Rode: pip install yfinance")

        ticker = yf.Ticker(self.symbol)
        df = ticker.history(period=self.period, interval=self.interval, auto_adjust=True)

        if df.empty:
            raise ValueError(f"Nenhum dado encontrado para {self.symbol}")

        df.index = pd.to_datetime(df.index, utc=True)
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        df = df.dropna()

        self.raw_df = df
        print(f"[MultiAssetLoader] {len(df)} candles. Período: {df.index[0].date()} → {df.index[-1].date()}")
        return df

    def _fetch_ccxt(self) -> pd.DataFrame:
        if not CCXT_AVAILABLE:
            raise ImportError("ccxt não instalado. Rode: pip install ccxt")

        exchange = ccxt.binance({"enableRateLimit": True})
        limit    = int(self.period.replace("y", "")) * 365 if "y" in self.period else 800
        ohlcv    = exchange.fetch_ohlcv(self.symbol, self.interval, limit=limit)

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").astype(float)

        self.raw_df = df
        print(f"[MultiAssetLoader] {len(df)} candles. Período: {df.index[0].date()} → {df.index[-1].date()}")
        return df

    def compute_realized_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula log-retornos e RV anualizada."""
        df = df.copy()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["rv"]         = df["log_return"].rolling(self.rv_window).std() * self.ann_factor
        df["rv_target"]  = df["rv"].shift(-1)
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features técnicas + estatísticas para o modelo."""
        df = df.copy()

        # ── Médias Móveis
        for w in [5, 10, 21, 50, 200]:
            df[f"sma_{w}"]       = df["close"].rolling(w).mean()
            df[f"sma_{w}_ratio"] = df["close"] / df[f"sma_{w}"]

        # ── EMAs
        for w in [9, 21, 50]:
            df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()

        # ── ATR
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(abs(df["high"] - df["close"].shift(1)),
                       abs(df["low"]  - df["close"].shift(1)))
        )
        df["atr_14"]    = df["tr"].rolling(14).mean()
        df["atr_21"]    = df["tr"].rolling(21).mean()
        df["atr_ratio"] = df["atr_14"] / df["close"]

        # ── RSI
        df["rsi_14"] = self._rsi(df["close"], 14)
        df["rsi_21"] = self._rsi(df["close"], 21)

        # ── MACD
        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        df["macd"]        = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"]   = df["macd"] - df["macd_signal"]

        # ── Bollinger Bands
        sma20           = df["close"].rolling(20).mean()
        std20           = df["close"].rolling(20).std()
        df["bb_upper"]  = sma20 + 2 * std20
        df["bb_lower"]  = sma20 - 2 * std20
        df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / sma20
        df["bb_pos"]    = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

        # ── Lag features de RV
        for lag in [1, 2, 3, 5, 10, 21]:
            df[f"rv_lag_{lag}"] = df["rv"].shift(lag)

        # ── Volume features
        df["volume_ma20"]  = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / (df["volume_ma20"] + 1e-10)
        df["volume_std"]   = df["volume"].rolling(20).std() / (df["volume_ma20"] + 1e-10)

        # ── Price features
        df["hl_spread"]   = (df["high"] - df["low"]) / df["close"]
        df["abs_return"]  = df["log_return"].abs()
        df["ret_5d"]      = df["close"].pct_change(5)
        df["ret_21d"]     = df["close"].pct_change(21)

        # ── Volatility features
        df["rv_5"]  = df["log_return"].rolling(5).std()  * self.ann_factor
        df["rv_10"] = df["log_return"].rolling(10).std() * self.ann_factor
        df["rv_ratio"] = df["rv_5"] / (df["rv"] + 1e-10)   # vol de curto/longo prazo

        self.feature_df = df.dropna()
        print(f"[MultiAssetLoader] Features calculadas. Shape: {self.feature_df.shape} | Colunas: {len(self.feature_df.columns)}")
        return self.feature_df

    @staticmethod
    def _rsi(series: pd.Series, window: int) -> pd.Series:
        delta    = series.diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
        avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
        rs       = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def run(self) -> pd.DataFrame:
        """Pipeline completo: fetch → RV → features."""
        df = self.fetch()
        df = self.compute_realized_volatility(df)
        df = self.engineer_features(df)
        return df

    def get_asset_info(self) -> dict:
        return {
            "symbol":   self.symbol,
            "name":     self.asset_info.get("name", self.symbol),
            "source":   self.source,
            "currency": self.asset_info.get("currency", "USD"),
            "ann_factor": float(self.ann_factor),
        }