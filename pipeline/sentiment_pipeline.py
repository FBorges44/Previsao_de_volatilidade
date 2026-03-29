"""
================================================================================
PIPELINE DE NLP — Sentiment Analysis com FinBERT
================================================================================

Racional de incorporar sentimento como feature:
─────────────────────────────────────────────────
A hipótese de eficiência de mercado fraco sugere que preços incorporam toda
informação histórica. Porém, evidências empíricas mostram que o SENTIMENTO
do mercado (fear/greed) é preditivo da volatilidade futura, especialmente
em mercados de cripto onde o varejo domina.

Pipeline de NLP:
  Texto bruto → Tokenização → FinBERT → Logits → Softmax → Score [-1, 1]
    Onde: score = P(positivo) - P(negativo)    ∈ [-1, 1]

FinBERT (Huang et al., 2023) é um BERT fine-tunado em textos financeiros,
superior ao BERT genérico para classificação de sentimento em finanças.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# Imports condicionais — modelos grandes
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# SCRAPERS / DATA SOURCES
# ─────────────────────────────────────────────────────────────────────────────
class NewsCollector:
    """
    Coleta notícias de cripto de múltiplas fontes.

    Fontes suportadas:
      - CryptoPanic API (free tier disponível)
      - NewsAPI (requer API key)
      - CoinTelegraph RSS (scraping)
      - Placeholder para Twitter/X API v2

    Parameters
    ----------
    cryptopanic_key : API key do CryptoPanic (grátis em cryptopanic.com)
    newsapi_key     : API key do NewsAPI
    """

    def __init__(
        self,
        cryptopanic_key: Optional[str] = None,
        newsapi_key: Optional[str] = None,
    ):
        self.cryptopanic_key = cryptopanic_key
        self.newsapi_key = newsapi_key

    def fetch_cryptopanic(self, currencies: str = "BTC", limit: int = 100) -> List[Dict]:
        """
        CryptoPanic API — notícias + votos da comunidade (bullish/bearish).
        Endpoint: https://cryptopanic.com/api/v1/posts/
        Documentação: https://cryptopanic.com/developers/api/

        Returns
        -------
        Lista de dicts com campos: title, published_at, votes
        """
        if not self.cryptopanic_key:
            print("[NewsCollector] CryptoPanic key não fornecida. Retornando mock data.")
            return self._mock_news(limit)

        if not REQUESTS_AVAILABLE:
            return self._mock_news(limit)

        url = "https://cryptopanic.com/api/v1/posts/"
        params = {
            "auth_token": self.cryptopanic_key,
            "currencies": currencies,
            "public": "true",
            "kind": "news",
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("results", [])[:limit]:
                results.append({
                    "title": item.get("title", ""),
                    "published_at": item.get("published_at", ""),
                    "source": item.get("source", {}).get("title", ""),
                    "votes_positive": item.get("votes", {}).get("positive", 0),
                    "votes_negative": item.get("votes", {}).get("negative", 0),
                })
            return results
        except Exception as e:
            print(f"[NewsCollector] Erro ao buscar CryptoPanic: {e}")
            return self._mock_news(limit)

    def fetch_newsapi(self, query: str = "Bitcoin BTC", days_back: int = 7) -> List[Dict]:
        """
        NewsAPI — manchetes de grandes veículos de mídia.
        Documentação: https://newsapi.org/docs

        NOTA: Requer plano pago para histórico > 1 mês.
        """
        if not self.newsapi_key:
            print("[NewsCollector] NewsAPI key não fornecida.")
            return []

        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": from_date,
            "language": "en",
            "sortBy": "publishedAt",
            "apiKey": self.newsapi_key,
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            return [{"title": a["title"], "published_at": a["publishedAt"]}
                    for a in data.get("articles", [])]
        except Exception as e:
            print(f"[NewsCollector] Erro NewsAPI: {e}")
            return []

    def fetch_twitter_placeholder(self, query: str = "#Bitcoin OR $BTC", limit: int = 200) -> List[Dict]:
        """
        PLACEHOLDER — Twitter/X API v2 Bearer Token.

        Para usar:
          1. Criar app em developer.twitter.com
          2. Obter Bearer Token
          3. Implementar com biblioteca 'tweepy'

        Exemplo de implementação com tweepy:
          import tweepy
          client = tweepy.Client(bearer_token=BEARER_TOKEN)
          tweets = client.search_recent_tweets(query=query, max_results=limit)
        """
        print("[NewsCollector] Twitter API: placeholder. Retornando mock tweets.")
        return self._mock_tweets(limit)

    def _mock_news(self, n: int) -> List[Dict]:
        """Dados sintéticos para testes sem API keys."""
        headlines = [
            "Bitcoin rallies as institutional demand surges",
            "Crypto market faces regulatory headwinds in Asia",
            "BTC breaks key resistance, bulls target new highs",
            "SEC postpones decision on Bitcoin ETF applications",
            "Whale wallets accumulate Bitcoin at current levels",
            "Federal Reserve hints at rate pause, crypto markets react positively",
            "Bitcoin hash rate reaches all-time high, network security strengthens",
            "Major exchange reports record trading volumes amid volatility spike",
            "El Salvador increases Bitcoin reserves amid price consolidation",
            "Crypto fear and greed index enters extreme greed territory",
        ]
        base = datetime.now()
        return [
            {
                "title": np.random.choice(headlines),
                "published_at": (base - timedelta(hours=i * 4)).isoformat(),
                "source": "mock",
            }
            for i in range(n)
        ]

    def _mock_tweets(self, n: int) -> List[Dict]:
        tweets = [
            "BTC to the moon! 🚀 Just bought the dip",
            "Crypto is dead, selling everything",
            "Bitcoin looks technically strong, accumulating here",
            "Regulatory FUD is overblown, this is a buying opportunity",
            "Market looks bearish short-term, caution advised",
        ]
        base = datetime.now()
        return [
            {"title": np.random.choice(tweets),
             "published_at": (base - timedelta(minutes=i * 30)).isoformat()}
            for i in range(n)
        ]


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT ANALYZER — FinBERT
# ─────────────────────────────────────────────────────────────────────────────
class SentimentAnalyzer:
    """
    Analisa sentimento de textos financeiros usando FinBERT.

    Modelo: ProsusAI/finbert (HuggingFace Hub)
    Classes: positive, neutral, negative
    Score: P(positive) - P(negative) ∈ [-1, 1]

    Fallback: VADER lexicon (sem GPU) ou score aleatório (mock)
    """

    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = "auto"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._loaded = False

        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def load_model(self) -> bool:
        """Carrega o FinBERT. Download automático via HuggingFace Hub (~400MB)."""
        if not FINBERT_AVAILABLE:
            print("[SentimentAnalyzer] transformers não disponível. Usando fallback.")
            return False
        try:
            print(f"[SentimentAnalyzer] Carregando {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            print(f"[SentimentAnalyzer] Modelo carregado em {self.device}.")
            return True
        except Exception as e:
            print(f"[SentimentAnalyzer] Falha ao carregar modelo: {e}")
            return False

    def score_text(self, text: str) -> float:
        """
        Retorna score de sentimento para um texto.

        Matemática:
          logits = BERT(text)
          probs = softmax(logits)   → [P_neg, P_neu, P_pos]
          score = P_pos - P_neg      ∈ [-1, 1]

        Returns
        -------
        float em [-1, 1]: -1 = muito negativo, 0 = neutro, 1 = muito positivo
        """
        if not self._loaded:
            return self._fallback_score(text)

        import torch
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=512, padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        # FinBERT labels: [positive, negative, neutral] — verificar ordem do modelo
        # ProsusAI/finbert: 0=positive, 1=negative, 2=neutral
        score = float(probs[0] - probs[1])
        return np.clip(score, -1.0, 1.0)

    def _fallback_score(self, text: str) -> float:
        """
        Fallback lexicon simples para sentimento financeiro.
        Não substitui FinBERT, mas é útil para testes.
        """
        positive_words = {"bull", "bullish", "surge", "rally", "moon", "gain",
                          "buy", "accumulate", "high", "strong", "positive", "rise"}
        negative_words = {"bear", "bearish", "crash", "dump", "sell", "fear",
                          "loss", "weak", "down", "negative", "fud", "drop"}

        words = text.lower().split()
        pos = sum(1 for w in words if w in positive_words)
        neg = sum(1 for w in words if w in negative_words)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    def score_batch(self, texts: List[str], batch_size: int = 16) -> List[float]:
        """Processa lista de textos em batches (eficiência para GPU)."""
        if not self._loaded:
            return [self._fallback_score(t) for t in texts]

        import torch
        scores = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True,
                                    max_length=512, padding=True).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            scores.extend((probs[:, 0] - probs[:, 1]).tolist())
        return [np.clip(s, -1.0, 1.0) for s in scores]


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────
class SentimentFeatureBuilder:
    """
    Constrói features de sentimento alinhadas à série temporal de preços.

    Método:
      1. Coleta notícias/tweets por período
      2. Calcula score FinBERT para cada texto
      3. Agrega por dia (média, desvio, máx absoluto)
      4. Suaviza com EMA para reduzir ruído
      5. Alinha ao índice do DataFrame de preços via merge_asof

    Parameters
    ----------
    collector : NewsCollector
    analyzer  : SentimentAnalyzer
    ema_span  : suavização exponencial do score diário
    """

    def __init__(
        self,
        collector: Optional[NewsCollector] = None,
        analyzer: Optional[SentimentAnalyzer] = None,
        ema_span: int = 3,
    ):
        self.collector = collector or NewsCollector()
        self.analyzer = analyzer or SentimentAnalyzer()
        self.ema_span = ema_span

    def build_daily_sentiment(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Gera DataFrame diário com scores de sentimento agregados.

        Returns
        -------
        DataFrame com colunas: [sentiment_mean, sentiment_std, sentiment_ema]
        """
        print("[SentimentFeatureBuilder] Coletando notícias...")

        # Tenta carregar FinBERT (pode falhar se sem GPU/internet)
        self.analyzer.load_model()

        # Coleta notícias (usa mock se sem API key)
        news_items = self.collector.fetch_cryptopanic(limit=500)

        records = []
        for item in news_items:
            title = item.get("title", "")
            published = item.get("published_at", "")
            if not title or not published:
                continue
            try:
                dt = pd.to_datetime(published, utc=True).normalize()
                score = self.analyzer.score_text(title)
                records.append({"date": dt, "score": score})
            except Exception:
                continue

        if not records:
            return self._mock_sentiment_series(start_date, end_date)

        sent_df = pd.DataFrame(records)
        sent_df = sent_df.set_index("date")

        # Agregação diária
        daily = sent_df["score"].resample("1D").agg(
            sentiment_mean="mean",
            sentiment_std="std",
            sentiment_count="count",
        )
        daily["sentiment_std"] = daily["sentiment_std"].fillna(0)
        daily["sentiment_ema"] = daily["sentiment_mean"].ewm(span=self.ema_span).mean()

        return daily.loc[start_date:end_date]

    def _mock_sentiment_series(self, start: str, end: str) -> pd.DataFrame:
        """Gera série sintética de sentimento com padrão realista."""
        dates = pd.date_range(start, end, freq="1D", tz="UTC")
        np.random.seed(123)
        # Score com autocorrelação (AR(1))
        n = len(dates)
        scores = [0.0]
        for _ in range(n - 1):
            scores.append(0.7 * scores[-1] + np.random.normal(0, 0.2))
        scores = np.clip(scores, -1, 1)

        df = pd.DataFrame({
            "sentiment_mean": scores,
            "sentiment_std": np.random.uniform(0.05, 0.3, n),
            "sentiment_count": np.random.randint(5, 50, n),
            "sentiment_ema": pd.Series(scores).ewm(span=3).mean().values,
        }, index=dates)
        print("[SentimentFeatureBuilder] Usando sentimento mock (sem API key).")
        return df

    def inject_into_dataframe(
        self, price_df: pd.DataFrame, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Injeta features de sentimento no DataFrame de preços.
        Usa merge_asof para alinhar datas sem forward-looking.
        """
        sent_df = self.build_daily_sentiment(start_date, end_date)
        sent_df = sent_df.reset_index().rename(columns={"date": "timestamp"})

        price_reset = price_df.reset_index()
        merged = pd.merge_asof(
            price_reset.sort_values("timestamp"),
            sent_df.sort_values("timestamp"),
            on="timestamp",
            direction="backward",   # sem lookahead: usa sentimento mais recente ≤ t
        )
        merged = merged.set_index("timestamp")
        # Preenche NaN iniciais com neutro
        for col in ["sentiment_mean", "sentiment_ema", "sentiment_std"]:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0)
        return merged