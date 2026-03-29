"""
================================================================================
MÓDULO ESTATÍSTICO — GARCH(1,1) Benchmark
================================================================================

Teoria do GARCH(1,1) (Bollerslev, 1986):
──────────────────────────────────────────
O modelo GARCH (Generalized AutoRegressive Conditional Heteroskedasticity)
captura dois fenômenos empíricos chave em retornos financeiros:

  1. VOLATILITY CLUSTERING: períodos de alta vol seguidos de alta vol
  2. FAT TAILS: distribuição dos retornos tem caudas mais pesadas que a Normal

Especificação matemática:
  r_t = μ + ε_t,        ε_t = σ_t · z_t,    z_t ~ N(0,1)

  Equação da variância condicional:
  σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

  onde:
    ω > 0             : constante (variância de longo prazo ponderada)
    α ≥ 0             : coeficiente ARCH — reação a choques recentes
    β ≥ 0             : coeficiente GARCH — persistência da volatilidade
    α + β < 1         : condição de estacionariedade (covariância)

  Variância incondicional (média de longo prazo):
  σ̄² = ω / (1 - α - β)

  Half-life do choque de volatilidade:
  t_{1/2} = ln(0.5) / ln(α + β)

O GARCH(1,1) é o "SPY" dos modelos de volatilidade — simples, robusto,
difícil de bater em horizontes curtos. Serve como benchmark sólido.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
import warnings
warnings.filterwarnings("ignore")

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("[WARNING] Biblioteca 'arch' não instalada. Usando implementação manual do GARCH.")


class GARCHModel:
    """
    Wrapper para o modelo GARCH(1,1) com interface unificada para o Backtester.

    Parameters
    ----------
    p : int — Ordem GARCH (lags de variância condicional)
    q : int — Ordem ARCH (lags de inovações ao quadrado)
    dist : str — Distribuição dos resíduos: 'normal', 't', 'skewt'
    """

    def __init__(self, p: int = 1, q: int = 1, dist: str = "t"):
        self.p = p
        self.q = q
        self.dist = dist
        self.model = None
        self.result = None
        self.fitted = False
        self.train_returns: pd.Series = None

    # ─────────────────────────────────────────────────────────────────────────
    # FIT
    # ─────────────────────────────────────────────────────────────────────────
    def fit(self, returns: pd.Series) -> "GARCHModel":
        """
        Estima os parâmetros GARCH via MLE (Maximum Likelihood Estimation).

        Parameters
        ----------
        returns : pd.Series de log-retornos (NÃO multiplicados por 100 aqui —
                  a biblioteca arch escala internamente)
        """
        self.train_returns = returns.dropna()
        scaled = self.train_returns * 100  # arch trabalha melhor em escala %

        if ARCH_AVAILABLE:
            self.model = arch_model(
                scaled,
                vol="Garch",
                p=self.p,
                q=self.q,
                dist=self.dist,
                mean="Constant",
            )
            self.result = self.model.fit(disp="off", show_warning=False)
            self.fitted = True
            self._print_summary()
        else:
            # Fallback: GARCH manual via estimação numérica simples
            self._fit_manual(self.train_returns)

        return self

    def _print_summary(self):
        if self.result is not None:
            params = self.result.params
            print("\n[GARCH] Parâmetros estimados:")
            print(f"  ω (omega)  = {params.get('omega', 'N/A'):.6f}")
            print(f"  α (alpha1) = {params.get('alpha[1]', 'N/A'):.4f}  ← sensibilidade a choques")
            print(f"  β (beta1)  = {params.get('beta[1]', 'N/A'):.4f}  ← persistência")
            a = params.get("alpha[1]", 0)
            b = params.get("beta[1]", 0)
            print(f"  α+β        = {a+b:.4f}  (< 1 → estacionário)")
            if a + b < 1:
                hl = np.log(0.5) / np.log(a + b)
                print(f"  Half-life  ≈ {hl:.1f} períodos")

    # ─────────────────────────────────────────────────────────────────────────
    # FORECAST (rolling / expanding window)
    # ─────────────────────────────────────────────────────────────────────────
    def forecast_rolling(
        self,
        full_returns: pd.Series,
        train_size: int,
        horizon: int = 1,
    ) -> pd.Series:
        """
        Previsão rolling walk-forward: re-estima o modelo a cada passo.
        Simula condições reais de trading — sem lookahead bias.

        Parameters
        ----------
        full_returns : série completa de log-retornos
        train_size   : tamanho da janela de treino inicial
        horizon      : passos à frente para prever (default 1 = T+1)

        Returns
        -------
        pd.Series com volatilidade prevista (anualizada) alinhada ao índice
        """
        preds = []
        idx = []
        returns = full_returns.dropna()
        scaled = returns * 100

        print(f"[GARCH] Iniciando rolling forecast ({len(returns) - train_size} steps)...")

        for t in range(train_size, len(returns) - horizon + 1):
            window = scaled.iloc[:t]
            try:
                if ARCH_AVAILABLE:
                    m = arch_model(window, vol="Garch", p=self.p, q=self.q,
                                   dist=self.dist, mean="Constant")
                    res = m.fit(disp="off", show_warning=False, options={"maxiter": 200})
                    fcast = res.forecast(horizon=horizon, reindex=False)
                    # variância em % → desvio padrão anualizado
                    var_pct = fcast.variance.iloc[-1, -1]
                    sigma_daily = np.sqrt(var_pct) / 100
                    sigma_annual = sigma_daily * np.sqrt(252)
                else:
                    # Fallback: vol histórica como aproximação
                    sigma_annual = window.std() / 100 * np.sqrt(252)

                preds.append(sigma_annual)
                idx.append(returns.index[t + horizon - 1])

            except Exception as e:
                # Em caso de falha de convergência, usa última previsão
                preds.append(preds[-1] if preds else 0.0)
                idx.append(returns.index[t + horizon - 1])

        forecast_series = pd.Series(preds, index=idx, name="garch_forecast")
        print(f"[GARCH] Rolling forecast concluído. {len(forecast_series)} previsões geradas.")
        return forecast_series

    def _fit_manual(self, returns: pd.Series):
        """Implementação GARCH(1,1) manual via variância recursiva (fallback)."""
        print("[GARCH] Usando implementação manual (fallback).")
        r = returns.values
        n = len(r)
        omega = np.var(r) * 0.1
        alpha, beta = 0.1, 0.8
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(r)
        for t in range(1, n):
            sigma2[t] = omega + alpha * r[t-1]**2 + beta * sigma2[t-1]
        self._manual_sigma2 = sigma2
        self.fitted = True

    def get_conditional_volatility(self) -> pd.Series:
        """Retorna a volatilidade condicional do período de treino (anualizada)."""
        if ARCH_AVAILABLE and self.result is not None:
            cv = self.result.conditional_volatility / 100 * np.sqrt(252)
            cv.name = "garch_cond_vol"
            return cv
        return pd.Series(dtype=float)

    def get_params(self) -> Dict:
        """Retorna dicionário com parâmetros estimados."""
        if self.result is not None:
            return dict(self.result.params)
        return {}