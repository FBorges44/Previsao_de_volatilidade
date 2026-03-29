import numpy as np
import pandas as pd
from typing import Dict
import warnings
warnings.filterwarnings("ignore")

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("[WARNING] Biblioteca 'arch' nao instalada.")


class GARCHModel:
    def __init__(self, p: int = 1, q: int = 1, dist: str = "t"):
        self.p = p
        self.q = q
        self.dist = dist
        self.model = None
        self.result = None
        self.fitted = False

    def fit(self, returns) -> "GARCHModel":
        # Garante que eh uma Series 1D
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]
        returns = pd.Series(returns.values.ravel(), name="log_return")
        returns = returns.dropna()

        self.train_returns = returns
        scaled = returns * 100

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
            self._fit_manual(returns)

        return self

    def _print_summary(self):
        if self.result is not None:
            params = self.result.params
            print("\n[GARCH] Parametros estimados:")
            print(f"  omega = {params.get('omega', 0):.6f}")
            print(f"  alpha = {params.get('alpha[1]', 0):.4f}")
            print(f"  beta  = {params.get('beta[1]', 0):.4f}")
            a = params.get("alpha[1]", 0)
            b = params.get("beta[1]", 0)
            print(f"  alpha+beta = {a+b:.4f} (< 1 estacionario)")
            if a + b < 1 and a + b > 0:
                hl = np.log(0.5) / np.log(a + b)
                print(f"  Half-life  = {hl:.1f} periodos")

    def forecast_rolling(self, full_returns, train_size: int, horizon: int = 1) -> pd.Series:
        if isinstance(full_returns, pd.DataFrame):
            full_returns = full_returns.iloc[:, 0]
        returns = pd.Series(full_returns.values.ravel(), index=full_returns.index).dropna()
        scaled = returns * 100

        preds, idx = [], []
        total = len(returns) - train_size
        print(f"[GARCH] Iniciando rolling forecast ({total} steps)...")

        for t in range(train_size, len(returns) - horizon + 1):
            window = scaled.iloc[:t]
            try:
                if ARCH_AVAILABLE:
                    m = arch_model(window, vol="Garch", p=self.p, q=self.q,
                                   dist=self.dist, mean="Constant")
                    res = m.fit(disp="off", show_warning=False, options={"maxiter": 200})
                    fcast = res.forecast(horizon=horizon, reindex=False)
                    var_pct = fcast.variance.iloc[-1, -1]
                    sigma_annual = np.sqrt(var_pct) / 100 * np.sqrt(252)
                else:
                    sigma_annual = window.std() / 100 * np.sqrt(252)

                preds.append(sigma_annual)
                idx.append(returns.index[t + horizon - 1])

            except Exception:
                preds.append(preds[-1] if preds else 0.0)
                idx.append(returns.index[t + horizon - 1])

        print(f"[GARCH] Forecast concluido. {len(preds)} previsoes geradas.")
        return pd.Series(preds, index=idx, name="garch_forecast")

    def _fit_manual(self, returns):
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

    def get_params(self) -> Dict:
        if self.result is not None:
            return dict(self.result.params)
        return {}