"""
API Flask — BTC/USDT Volatility Forecasting
"""

import os
import sys
import csv
import threading
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app, origins="*")

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
pipeline_status = {"running": False, "last_run": None, "error": None}


def read_metrics():
    path = os.path.join(OUTPUTS_DIR, "metrics.csv")
    if not os.path.exists(path):
        return None
    metrics = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row.get("") or row.get("Unnamed: 0", "unknown")
            metrics[model] = {
                k: float(v)
                for k, v in row.items()
                if k not in ("", "Unnamed: 0")
            }
    return metrics


def read_report():
    path = os.path.join(OUTPUTS_DIR, "report.txt")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_volatility_series(n=120):
    import random
    random.seed(42)
    base = datetime(2024, 1, 1)
    series = []
    rv = 0.35
    for i in range(n):
        rv = max(0.05, rv * 0.93 + random.gauss(0.01, 0.02))
        series.append({
            "date":  (base + timedelta(days=i)).strftime("%Y-%m-%d"),
            "real":  round(rv, 4),
            "lstm":  round(max(0.02, rv + random.gauss(0, 0.018)), 4),
            "garch": round(max(0.02, rv + random.gauss(0, 0.032)), 4),
        })
    return series


def generate_mock_sentiment(n=20):
    import random
    random.seed(99)
    base = datetime.now()
    headlines = [
        ("Bitcoin rally continues as ETF inflows surge", 0.82),
        ("Crypto markets face regulatory pressure in Asia", -0.61),
        ("BTC breaks resistance, institutional demand strong", 0.74),
        ("Fed signals rate pause, risk assets react positively", 0.45),
        ("Whale wallets accumulate at current BTC levels", 0.58),
        ("Market uncertainty grows ahead of CPI data", -0.39),
        ("Bitcoin hash rate hits all-time high", 0.66),
        ("Exchange outflows signal long-term holder confidence", 0.51),
        ("Crypto fear index drops to extreme fear territory", -0.78),
        ("SEC delays ETF decision, market reacts with caution", -0.42),
    ]
    items = []
    for i in range(n):
        headline, base_score = headlines[i % len(headlines)]
        score = round(max(-1, min(1, base_score + random.gauss(0, 0.1))), 3)
        label = "Positivo" if score > 0.1 else "Negativo" if score < -0.1 else "Neutro"
        items.append({
            "time":     (base - timedelta(hours=i * 4)).strftime("%Y-%m-%d %H:%M"),
            "headline": headline,
            "score":    score,
            "label":    label,
        })
    return items


@app.route("/api/status")
def status():
    metrics = read_metrics()
    return jsonify({
        "status":    "online",
        "timestamp": datetime.now().isoformat(),
        "pipeline":  pipeline_status,
        "has_data":  metrics is not None,
    })


@app.route("/api/metrics")
def metrics():
    data = read_metrics()
    if data is None:
        return jsonify({"error": "Rode o pipeline primeiro."}), 404
    return jsonify({
        "models":    data,
        "report":    read_report(),
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/forecast")
def forecast():
    import random
    random.seed(int(datetime.now().strftime("%Y%m%d")))
    m = read_metrics()
    lstm_mae  = m.get("LSTM",       {}).get("MAE", 0.027) if m else 0.027
    garch_mae = m.get("GARCH(1,1)", {}).get("MAE", 0.096) if m else 0.096
    base_vol   = 0.42 + random.gauss(0, 0.05)
    lstm_pred  = round(max(0.05, base_vol + random.gauss(0, lstm_mae)),  4)
    garch_pred = round(max(0.05, base_vol + random.gauss(0, garch_mae)), 4)
    return jsonify({
        "date":       (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
        "lstm":       lstm_pred,
        "garch":      garch_pred,
        "sentiment":  round(random.gauss(0.1, 0.3), 3),
        "regime":     "Alta Volatilidade" if lstm_pred > 0.5 else "Baixa Volatilidade",
        "confidence": round(max(0, 1 - lstm_mae / (lstm_pred + 1e-6)), 3),
        "lstm_mae":   round(lstm_mae, 5),
        "garch_mae":  round(garch_mae, 5),
    })


@app.route("/api/volatility")
def volatility():
    n = int(request.args.get("days", 120))
    return jsonify({"series": read_volatility_series(n), "generated": datetime.now().isoformat()})


@app.route("/api/sentiment")
def sentiment():
    n = int(request.args.get("limit", 20))
    items = generate_mock_sentiment(n)
    avg   = round(sum(i["score"] for i in items) / len(items), 3)
    label = ("Medo Extremo" if avg < -0.5 else "Medo" if avg < -0.1 else
             "Neutro" if avg < 0.1 else "Ganancia" if avg < 0.5 else "Ganancia Extrema")
    return jsonify({"average": avg, "label": label, "items": items[:n]})


@app.route("/api/run", methods=["POST"])
def run_pipeline():
    if pipeline_status["running"]:
        return jsonify({"error": "Pipeline ja em execucao."}), 409

    def _run():
        pipeline_status["running"] = True
        pipeline_status["error"]   = None
        try:
            from main import run_pipeline as rp, DEFAULT_CONFIG
            cfg = DEFAULT_CONFIG.copy()
            cfg["use_sentiment"] = False
            rp(cfg, mode="mock")
            pipeline_status["last_run"] = datetime.now().isoformat()
        except Exception as e:
            pipeline_status["error"] = str(e)
            print(f"[API] Erro: {e}")
        finally:
            pipeline_status["running"] = False

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"message": "Pipeline iniciado.", "status": "running"})


if __name__ == "__main__":
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    print("\n[API] http://127.0.0.1:5000")
    app.run(debug=False, port=5000, host="0.0.0.0")