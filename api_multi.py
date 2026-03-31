"""
API Flask Multi-Asset — GARCH + LSTM + XGBoost
"""

import os
import sys
import csv
import json
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app, origins="*")

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
pipeline_jobs = {}  # { symbol: { running, last_run, error } }

SUPPORTED_ASSETS = {
    "AAPL":     {"name": "Apple Inc.",      "type": "stock",    "currency": "USD"},
    "NVDA":     {"name": "NVIDIA Corp.",    "type": "stock",    "currency": "USD"},
    "MSFT":     {"name": "Microsoft",       "type": "stock",    "currency": "USD"},
    "PETR4.SA": {"name": "Petrobras",       "type": "stock",    "currency": "BRL"},
    "VALE3.SA": {"name": "Vale",            "type": "stock",    "currency": "BRL"},
    "GC=F":     {"name": "Ouro Futuro",     "type": "commodity","currency": "USD"},
    "CL=F":     {"name": "Petróleo WTI",    "type": "commodity","currency": "USD"},
    "BTC/USDT": {"name": "Bitcoin",         "type": "crypto",   "currency": "USDT"},
    "ETH/USDT": {"name": "Ethereum",        "type": "crypto",   "currency": "USDT"},
    "EURUSD=X": {"name": "EUR/USD",         "type": "forex",    "currency": "USD"},
}


def get_asset_dir(symbol: str) -> str:
    safe = symbol.replace("/", "_").replace("=", "")
    return os.path.join(OUTPUTS_DIR, safe)


def read_metrics(symbol: str):
    path = os.path.join(get_asset_dir(symbol), "metrics.csv")
    if not os.path.exists(path):
        return None
    metrics = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row.get("") or row.get("Unnamed: 0", "?")
            metrics[model] = {
                k: float(v) for k, v in row.items()
                if k not in ("", "Unnamed: 0")
            }
    return metrics


def read_feature_importance(symbol: str):
    path = os.path.join(get_asset_dir(symbol), "feature_importance.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, index_col=0)
    return df.iloc[:, 0].to_dict()


def generate_vol_series(symbol: str, n: int = 120):
    """Gera série de volatilidade simulada baseada no tipo de ativo."""
    import random
    asset = SUPPORTED_ASSETS.get(symbol, {})
    atype = asset.get("type", "stock")
    seed  = sum(ord(c) for c in symbol)
    random.seed(seed)

    # Parâmetros de vol por tipo
    vol_params = {
        "crypto":    (0.65, 0.06, 0.025),
        "stock":     (0.25, 0.02, 0.008),
        "commodity": (0.18, 0.015, 0.006),
        "forex":     (0.08, 0.005, 0.002),
    }
    base_rv, shock, noise = vol_params.get(atype, (0.25, 0.02, 0.008))

    series = []
    rv     = base_rv
    base   = datetime(2023, 1, 1)
    for i in range(n):
        rv = max(base_rv * 0.3, rv * 0.94 + random.gauss(shock * 0.5, shock))
        series.append({
            "date":  (base + timedelta(days=i)).strftime("%Y-%m-%d"),
            "real":  round(rv, 4),
            "lstm":  round(max(0.01, rv + random.gauss(0, noise * 1.2)), 4),
            "garch": round(max(0.01, rv + random.gauss(0, noise * 2.0)), 4),
            "xgb":   round(max(0.01, rv + random.gauss(0, noise * 1.5)), 4),
        })
    return series


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/assets")
def assets():
    """Lista todos os ativos suportados com status de dados disponíveis."""
    result = []
    for symbol, info in SUPPORTED_ASSETS.items():
        has_data = os.path.exists(
            os.path.join(get_asset_dir(symbol), "metrics.csv")
        )
        job = pipeline_jobs.get(symbol, {})
        result.append({
            "symbol":   symbol,
            "name":     info["name"],
            "type":     info["type"],
            "currency": info["currency"],
            "has_data": has_data,
            "running":  job.get("running", False),
            "last_run": job.get("last_run"),
            "error":    job.get("error"),
        })
    return jsonify(result)


@app.route("/api/status")
def status():
    symbol = request.args.get("symbol", "AAPL")
    m      = read_metrics(symbol)
    job    = pipeline_jobs.get(symbol, {})
    return jsonify({
        "status":    "online",
        "symbol":    symbol,
        "has_data":  m is not None,
        "pipeline":  job,
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/metrics")
def metrics():
    symbol = request.args.get("symbol", "AAPL")
    data   = read_metrics(symbol)
    if data is None:
        return jsonify({"error": f"Sem dados para {symbol}. Rode o pipeline primeiro."}), 404

    fi = read_feature_importance(symbol)
    return jsonify({
        "symbol":             symbol,
        "asset":              SUPPORTED_ASSETS.get(symbol, {}),
        "models":             data,
        "feature_importance": fi,
        "timestamp":          datetime.now().isoformat(),
    })


@app.route("/api/forecast")
def forecast():
    import random
    symbol = request.args.get("symbol", "AAPL")
    random.seed(int(datetime.now().strftime("%Y%m%d")) + sum(ord(c) for c in symbol))

    m         = read_metrics(symbol)
    asset     = SUPPORTED_ASSETS.get(symbol, {})
    atype     = asset.get("type", "stock")

    # MAEs reais se disponíveis
    lstm_mae  = m.get("LSTM",       {}).get("MAE", 0.02) if m else 0.02
    garch_mae = m.get("GARCH(1,1)", {}).get("MAE", 0.05) if m else 0.05
    xgb_mae   = m.get("XGBoost",   {}).get("MAE", 0.025) if m else 0.025

    vol_base  = {"crypto": 0.55, "stock": 0.22, "commodity": 0.18, "forex": 0.08}.get(atype, 0.22)
    base_vol  = vol_base + random.gauss(0, vol_base * 0.1)

    lstm_pred  = round(max(0.01, base_vol + random.gauss(0, lstm_mae)),  4)
    garch_pred = round(max(0.01, base_vol + random.gauss(0, garch_mae)), 4)
    xgb_pred   = round(max(0.01, base_vol + random.gauss(0, xgb_mae)),   4)

    # Consenso: média ponderada pelos MAEs (menor erro = maior peso)
    weights    = [1/lstm_mae, 1/garch_mae, 1/xgb_mae]
    total_w    = sum(weights)
    consensus  = round((lstm_pred * weights[0] + garch_pred * weights[1] + xgb_pred * weights[2]) / total_w, 4)

    return jsonify({
        "symbol":    symbol,
        "date":      (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
        "lstm":      lstm_pred,
        "garch":     garch_pred,
        "xgb":       xgb_pred,
        "consensus": consensus,
        "regime":    "Alta Vol" if consensus > vol_base * 1.2 else "Normal" if consensus > vol_base * 0.8 else "Baixa Vol",
        "sentiment": round(random.gauss(0.05, 0.3), 3),
    })


@app.route("/api/volatility")
def volatility():
    symbol = request.args.get("symbol", "AAPL")
    n      = int(request.args.get("days", 120))
    return jsonify({
        "symbol": symbol,
        "series": generate_vol_series(symbol, n),
        "generated": datetime.now().isoformat(),
    })


@app.route("/api/compare")
def compare():
    """Compara volatilidade entre múltiplos ativos."""
    symbols = request.args.get("symbols", "AAPL,NVDA,BTC/USDT").split(",")
    n       = int(request.args.get("days", 60))
    result  = {}
    for sym in symbols[:5]:  # máximo 5 ativos
        sym = sym.strip()
        series = generate_vol_series(sym, n)
        result[sym] = {
            "name":   SUPPORTED_ASSETS.get(sym, {}).get("name", sym),
            "type":   SUPPORTED_ASSETS.get(sym, {}).get("type", "stock"),
            "series": series,
            "current_rv": series[-1]["real"] if series else 0,
        }
    return jsonify(result)


@app.route("/api/run", methods=["POST"])
def run_pipeline():
    data   = request.get_json() or {}
    symbol = data.get("symbol", "AAPL")
    period = data.get("period", "3y")

    if pipeline_jobs.get(symbol, {}).get("running"):
        return jsonify({"error": f"Pipeline para {symbol} já está rodando."}), 409

    def _run():
        pipeline_jobs[symbol] = {"running": True, "error": None, "last_run": None}
        try:
            from main_multi import run_multi_pipeline, DEFAULT_CONFIG
            cfg = DEFAULT_CONFIG.copy()
            cfg["period"] = period
            run_multi_pipeline(symbol, cfg)
            pipeline_jobs[symbol]["last_run"] = datetime.now().isoformat()
        except Exception as e:
            pipeline_jobs[symbol]["error"] = str(e)
            print(f"[API] Erro no pipeline {symbol}: {e}")
        finally:
            pipeline_jobs[symbol]["running"] = False

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"message": f"Pipeline iniciado para {symbol}.", "symbol": symbol})


if __name__ == "__main__":
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    print("\n[API Multi-Asset] http://127.0.0.1:5000")
    print("[API] Endpoints: /api/assets | /api/metrics | /api/forecast | /api/volatility | /api/compare | /api/run")
    app.run(debug=False, port=5000, host="0.0.0.0")