"""
================================================================================
BACKTESTING E AVALIAÇÃO — Comparação GARCH vs LSTM
================================================================================

Métricas utilizadas:
  MAE  = (1/n) · Σ|ŷ_t - y_t|        ← erro médio absoluto (mesma escala)
  RMSE = √[(1/n) · Σ(ŷ_t - y_t)²]   ← penaliza outliers mais severamente
  QLIKE = Σ[y_t/σ̂_t² - ln(σ̂_t²)]   ← loss assimétrica clássica para vol
  DIR   = accuracy de acertar direção (+/-) da variação de vol

Importante: RMSE > MAE indica a presença de previsões muito erradas (spikes).
Em volatilidade, o QLIKE é preferido por penalizar subestimação de risco.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 1e-8
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """QLIKE loss — padrão em avaliação de modelos de volatilidade (Patton, 2011)."""
    pred_sq = np.maximum(y_pred ** 2, 1e-10)
    return float(np.mean(y_true ** 2 / pred_sq - np.log(pred_sq)))

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """% de vezes que acertamos se vol vai subir ou cair."""
    true_dir = np.diff(y_true) > 0
    pred_dir = np.diff(y_pred) > 0
    return float(np.mean(true_dir == pred_dir) * 100)


# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTER
# ─────────────────────────────────────────────────────────────────────────────
class Backtester:
    """
    Compara múltiplos modelos de previsão de volatilidade usando métricas
    padronizadas e visualizações.

    Usage:
        bt = Backtester()
        bt.add_forecast("GARCH(1,1)", dates, y_true, garch_preds)
        bt.add_forecast("LSTM",       dates, y_true, lstm_preds)
        bt.evaluate()
        bt.plot_all()
    """

    def __init__(self):
        self.forecasts: Dict[str, Dict] = {}
        self.colors = {
            "GARCH(1,1)": "#E84855",
            "LSTM":        "#3A86FF",
            "GARCH":       "#E84855",
        }
        self._default_colors = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#A8DADC", "#FF9F43"]

    def add_forecast(
        self,
        name: str,
        dates: pd.DatetimeIndex,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ):
        """Registra as previsões de um modelo."""
        # Alinha pelo menor comprimento
        n = min(len(dates), len(y_true), len(y_pred))
        self.forecasts[name] = {
            "dates":  dates[:n],
            "y_true": np.array(y_true[:n], dtype=float),
            "y_pred": np.array(y_pred[:n], dtype=float),
        }
        print(f"[Backtester] Modelo '{name}' registrado — {n} previsões.")

    # ─────────────────────────────────────────────────────────────────────────
    # AVALIAÇÃO
    # ─────────────────────────────────────────────────────────────────────────
    def evaluate(self) -> pd.DataFrame:
        """
        Calcula métricas para todos os modelos registrados.

        Returns
        -------
        DataFrame com MAE, RMSE, MAPE, QLIKE, DA% por modelo
        """
        results = {}
        for name, data in self.forecasts.items():
            yt = data["y_true"]
            yp = data["y_pred"]
            results[name] = {
                "MAE":   mae(yt, yp),
                "RMSE":  rmse(yt, yp),
                "MAPE%": mape(yt, yp),
                "QLIKE": qlike(yt, yp),
                "DA%":   directional_accuracy(yt, yp),
                "N":     len(yt),
            }

        self.metrics_df = pd.DataFrame(results).T
        print("\n" + "═" * 65)
        print("  RESULTADOS — Comparação de Modelos de Volatilidade")
        print("═" * 65)
        print(self.metrics_df.to_string(float_format="%.5f"))
        print("═" * 65)

        # Destacar vencedor por métrica
        for metric in ["MAE", "RMSE", "MAPE%", "QLIKE"]:
            best = self.metrics_df[metric].idxmin()
            print(f"  Melhor {metric:6s}: {best}")
        best_da = self.metrics_df["DA%"].idxmax()
        print(f"  Melhor DA%  : {best_da}")
        print("═" * 65 + "\n")

        return self.metrics_df

    # ─────────────────────────────────────────────────────────────────────────
    # VISUALIZAÇÕES
    # ─────────────────────────────────────────────────────────────────────────
    def plot_all(self, save_path: Optional[str] = None, figsize: Tuple = (18, 22)):
        """
        Gera dashboard completo com 5 subplots:
          1. Volatilidade real vs. prevista (linha do tempo)
          2. Scatter: previsto vs. real por modelo
          3. Distribuição dos resíduos
          4. ACF dos resíduos (verifica se resíduos são ruído branco)
          5. Comparativo de métricas (bar chart)
        """
        n_models = len(self.forecasts)
        if n_models == 0:
            print("[Backtester] Nenhum modelo registrado.")
            return

        fig = plt.figure(figsize=figsize, facecolor="#0D1117")
        fig.suptitle(
            "BTC/USDT — Volatilidade Realizada: Análise Comparativa de Modelos",
            fontsize=16, color="white", fontweight="bold", y=0.98
        )

        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

        # ── Subplot 1: Série temporal
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_timeseries(ax1)

        # ── Subplot 2 & 3: Scatter por modelo
        for i, (name, data) in enumerate(self.forecasts.items()):
            ax = fig.add_subplot(gs[1, i % 2])
            self._plot_scatter(ax, name, data)
            if i >= 1:
                break

        # ── Subplot 4: Distribuição de resíduos
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_residuals_dist(ax4)

        # ── Subplot 5: ACF dos resíduos (primeiro modelo)
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_residuals_acf(ax5)

        # ── Subplot 6: Métricas comparativas
        ax6 = fig.add_subplot(gs[3, :])
        self._plot_metrics_bar(ax6)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
            print(f"[Backtester] Gráfico salvo em: {save_path}")
        else:
            plt.tight_layout()
            plt.show()

        return fig

    def _plot_timeseries(self, ax):
        ax.set_facecolor("#161B22")
        ax.set_title("Volatilidade Realizada: Real vs. Prevista", color="white", fontsize=13, pad=12)

        first = True
        for name, data in self.forecasts.items():
            color = self.colors.get(name, self._default_colors[len(self.forecasts)])
            if first:
                ax.plot(data["dates"], data["y_true"],
                        color="#58A6FF", linewidth=1.5, label="Real (RV)", alpha=0.9, zorder=3)
                first = False
            ax.plot(data["dates"], data["y_pred"],
                    color=color, linewidth=1.2, label=f"Previsto — {name}",
                    alpha=0.8, linestyle="--", zorder=2)

        ax.set_xlabel("Data", color="#8B949E")
        ax.set_ylabel("Volatilidade Anualizada", color="#8B949E")
        ax.tick_params(colors="#8B949E")
        ax.spines["bottom"].set_color("#30363D")
        ax.spines["left"].set_color("#30363D")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(facecolor="#21262D", edgecolor="#30363D",
                  labelcolor="white", fontsize=10)
        ax.grid(axis="y", color="#21262D", linewidth=0.5, alpha=0.7)

    def _plot_scatter(self, ax, name, data):
        ax.set_facecolor("#161B22")
        color = self.colors.get(name, "#4ECDC4")
        yt, yp = data["y_true"], data["y_pred"]
        mae_val = mae(yt, yp)
        rmse_val = rmse(yt, yp)

        ax.scatter(yt, yp, color=color, alpha=0.4, s=15, zorder=2)
        lim = [min(yt.min(), yp.min()) * 0.95, max(yt.max(), yp.max()) * 1.05]
        ax.plot(lim, lim, color="white", linewidth=1, linestyle="--", alpha=0.5, zorder=3)

        ax.set_title(f"{name} — Previsto vs. Real\nMAE={mae_val:.4f} | RMSE={rmse_val:.4f}",
                     color="white", fontsize=10)
        ax.set_xlabel("Real", color="#8B949E")
        ax.set_ylabel("Previsto", color="#8B949E")
        ax.tick_params(colors="#8B949E")
        for sp in ax.spines.values():
            sp.set_color("#30363D")
        ax.grid(color="#21262D", linewidth=0.5, alpha=0.7)

    def _plot_residuals_dist(self, ax):
        ax.set_facecolor("#161B22")
        ax.set_title("Distribuição dos Resíduos", color="white", fontsize=11)
        for i, (name, data) in enumerate(self.forecasts.items()):
            residuals = data["y_true"] - data["y_pred"]
            color = self.colors.get(name, self._default_colors[i])
            ax.hist(residuals, bins=40, color=color, alpha=0.6, label=name, density=True)

        ax.axvline(0, color="white", linewidth=1, linestyle="--", alpha=0.7)
        ax.set_xlabel("Resíduos", color="#8B949E")
        ax.set_ylabel("Densidade", color="#8B949E")
        ax.tick_params(colors="#8B949E")
        ax.legend(facecolor="#21262D", labelcolor="white", fontsize=9)
        for sp in ax.spines.values():
            sp.set_color("#30363D")

    def _plot_residuals_acf(self, ax):
        """ACF dos resíduos — resíduos i.i.d. indicam modelo bem especificado."""
        ax.set_facecolor("#161B22")
        ax.set_title("ACF dos Resíduos (1º modelo)", color="white", fontsize=11)

        if not self.forecasts:
            return
        first_name = list(self.forecasts.keys())[0]
        data = self.forecasts[first_name]
        residuals = data["y_true"] - data["y_pred"]
        color = self.colors.get(first_name, "#4ECDC4")

        n_lags = min(30, len(residuals) // 5)
        lags = range(1, n_lags + 1)
        acf_vals = [pd.Series(residuals).autocorr(lag=l) for l in lags]

        # Banda de confiança 95%: ±1.96/√n
        ci = 1.96 / np.sqrt(len(residuals))
        ax.bar(lags, acf_vals, color=color, alpha=0.7, width=0.6)
        ax.axhline(ci,  color="white", linewidth=1, linestyle="--", alpha=0.6)
        ax.axhline(-ci, color="white", linewidth=1, linestyle="--", alpha=0.6)
        ax.axhline(0,   color="#8B949E", linewidth=0.5)

        ax.set_xlabel("Lag", color="#8B949E")
        ax.set_ylabel("Autocorrelação", color="#8B949E")
        ax.tick_params(colors="#8B949E")
        ax.set_xlim(0, n_lags + 1)
        for sp in ax.spines.values():
            sp.set_color("#30363D")

    def _plot_metrics_bar(self, ax):
        ax.set_facecolor("#161B22")
        ax.set_title("Comparativo de Métricas — MAE e RMSE", color="white", fontsize=11)

        if not hasattr(self, "metrics_df"):
            self.evaluate()

        models = list(self.metrics_df.index)
        x = np.arange(len(models))
        width = 0.3

        mae_vals  = self.metrics_df["MAE"].values
        rmse_vals = self.metrics_df["RMSE"].values

        colors_list = [self.colors.get(m, self._default_colors[i]) for i, m in enumerate(models)]

        bars1 = ax.bar(x - width/2, mae_vals,  width, label="MAE",
                       color=[c + "CC" for c in colors_list], zorder=2)
        bars2 = ax.bar(x + width/2, rmse_vals, width, label="RMSE",
                       color=colors_list, alpha=0.9, zorder=2)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f"{bar.get_height():.4f}", ha="center", va="bottom",
                    color="white", fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f"{bar.get_height():.4f}", ha="center", va="bottom",
                    color="white", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(models, color="white", fontsize=11)
        ax.set_ylabel("Erro", color="#8B949E")
        ax.tick_params(colors="#8B949E")
        ax.legend(facecolor="#21262D", labelcolor="white", fontsize=10)
        ax.grid(axis="y", color="#21262D", linewidth=0.5, alpha=0.7)
        for sp in ax.spines.values():
            sp.set_color("#30363D")

    def generate_report(self) -> str:
        """Gera relatório textual com análise dos resultados."""
        if not hasattr(self, "metrics_df"):
            self.evaluate()

        best_mae  = self.metrics_df["MAE"].idxmin()
        best_rmse = self.metrics_df["RMSE"].idxmin()
        best_da   = self.metrics_df["DA%"].idxmax()

        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║      RELATÓRIO — Previsão de Volatilidade BTC/USDT               ║
╚══════════════════════════════════════════════════════════════════╝

MÉTRICAS COMPARATIVAS:
{self.metrics_df.to_string(float_format='%.5f')}

CONCLUSÕES:
  • Menor MAE  → {best_mae}
  • Menor RMSE → {best_rmse}
  • Maior DA%  → {best_da} (melhor precisão direcional)

INTERPRETAÇÃO:
  - MAE e RMSE na mesma escala da volatilidade anualizada
  - DA% = % de dias em que o modelo acertou se vol subiu ou desceu
  - Se RMSE >> MAE: previsões têm erros grandes ocasionais (spikes)
  - ACF dos resíduos próxima de zero → modelo bem especificado

NOTA SOBRE GARCH vs LSTM:
  GARCH(1,1) tende a ser competitivo em horizontes curtos (1-5 dias)
  e mercados eficientes. LSTM pode superar GARCH quando:
    (a) Existem relações não-lineares entre features
    (b) Sentimento tem poder preditivo adicional
    (c) O horizonte de previsão é mais longo (5-22 dias)
"""
        return report