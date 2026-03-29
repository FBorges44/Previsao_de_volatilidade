"use client";

import { useEffect, useState, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, BarChart, Bar, Cell,
} from "recharts";

const API = "";

// ── tipos
interface Metrics { MAE: number; RMSE: number; "MAPE%": number; QLIKE: number; "DA%": number }
interface Forecast { date: string; lstm: number; garch: number; sentiment: number; regime: string; confidence: number }
interface VolPoint  { date: string; real: number; lstm: number; garch: number }
interface SentItem  { time: string; headline: string; score: number; label: string }

// ── helpers
const fmt  = (v: number, d = 4) => v?.toFixed(d) ?? "—";
const pct  = (v: number)        => (v * 100).toFixed(1) + "%";
const clr  = (v: number)        => v > 0 ? "#3fb950" : "#f85149";

// ── componentes pequenos
function Card({ children, accent }: { children: React.ReactNode; accent?: string }) {
  return (
    <div style={{
      background: "#161b22", border: "0.5px solid rgba(240,246,252,0.1)",
      borderRadius: 12, padding: "16px 20px", position: "relative", overflow: "hidden",
    }}>
      {accent && <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: accent }} />}
      {children}
    </div>
  );
}

function MetricCard({ label, value, sub, accent, badge }: {
  label: string; value: string; sub: string; accent: string; badge?: string;
}) {
  return (
    <Card accent={accent}>
      <div style={{ fontSize: 11, color: "#8b949e", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 8 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 500 }}>{value}</div>
      <div style={{ fontSize: 11, color: "#8b949e", marginTop: 4 }}>{sub}</div>
      {badge && (
        <span style={{
          position: "absolute", top: 14, right: 14,
          fontSize: 11, padding: "2px 8px", borderRadius: 20, fontWeight: 500,
          background: "rgba(63,185,80,0.15)", color: "#3fb950",
        }}>{badge}</span>
      )}
    </Card>
  );
}

function SentimentBadge({ label }: { label: string }) {
  const colors: Record<string, [string, string]> = {
    "Positivo":       ["rgba(63,185,80,0.15)",   "#3fb950"],
    "Negativo":       ["rgba(248,81,73,0.15)",    "#f85149"],
    "Neutro":         ["rgba(139,148,158,0.15)",  "#8b949e"],
    "Medo Extremo":   ["rgba(248,81,73,0.2)",     "#f85149"],
    "Medo":           ["rgba(186,117,23,0.2)",    "#ba7517"],
    "Ganancia":       ["rgba(63,185,80,0.15)",    "#3fb950"],
    "Ganancia Extrema": ["rgba(29,158,117,0.2)", "#1d9e75"],
  };
  const [bg, fg] = colors[label] ?? ["rgba(139,148,158,0.15)", "#8b949e"];
  return (
    <span style={{ background: bg, color: fg, fontSize: 11, padding: "2px 8px", borderRadius: 20, fontWeight: 500 }}>
      {label}
    </span>
  );
}

// ── página principal
export default function Dashboard() {
  const [metrics,   setMetrics]   = useState<Record<string, Metrics> | null>(null);
  const [forecast,  setForecast]  = useState<Forecast | null>(null);
  const [volSeries, setVolSeries] = useState<VolPoint[]>([]);
  const [sentiment, setSentiment] = useState<{ average: number; label: string; items: SentItem[] } | null>(null);
  const [status,    setStatus]    = useState<{ has_data: boolean; pipeline: { running: boolean } } | null>(null);
  const [running,   setRunning]   = useState(false);
  const [days,      setDays]      = useState(90);

  const fetchAll = useCallback(async () => {
    try {
      const [s, m, f, v, sent] = await Promise.all([
        fetch(`${API}/api/status`).then(r => r.json()),
        fetch(`${API}/api/metrics`).then(r => r.ok ? r.json() : null),
        fetch(`${API}/api/forecast`).then(r => r.json()),
        fetch(`${API}/api/volatility?days=${days}`).then(r => r.json()),
        fetch(`${API}/api/sentiment?limit=20`).then(r => r.json()),
      ]);
      setStatus(s);
      if (m) setMetrics(m.models);
      setForecast(f);
      setVolSeries(v.series ?? []);
      setSentiment(sent);
    } catch (e) {
      console.error("API offline:", e);
    }
  }, [days]);

  useEffect(() => { fetchAll(); }, [fetchAll]);

  const runPipeline = async () => {
    setRunning(true);
    await fetch(`${API}/api/run`, { method: "POST" });
    setTimeout(fetchAll, 8000);
    setTimeout(() => setRunning(false), 10000);
  };

  const lstm  = metrics?.["LSTM"];
  const garch = metrics?.["GARCH(1,1)"];

  const tickFmt = (v: number) => pct(v);

  return (
    <div style={{ minHeight: "100vh", background: "#0d1117" }}>

      {/* HEADER */}
      <header style={{
        background: "#161b22", borderBottom: "0.5px solid rgba(240,246,252,0.1)",
        padding: "14px 32px", display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 34, height: 34, borderRadius: 8,
            background: "rgba(55,138,221,0.15)", border: "0.5px solid #378add",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 18, color: "#378add", fontWeight: 700,
          }}>₿</div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 500 }}>BTC/USDT Volatility Dashboard</div>
            <div style={{ fontSize: 12, color: "#8b949e" }}>GARCH(1,1) vs LSTM + FinBERT Sentiment</div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            display: "flex", alignItems: "center", gap: 6,
            background: status?.has_data ? "rgba(29,158,117,0.15)" : "rgba(248,81,73,0.15)",
            border: `0.5px solid ${status?.has_data ? "#1d9e75" : "#f85149"}`,
            borderRadius: 20, padding: "4px 12px", fontSize: 12,
            color: status?.has_data ? "#1d9e75" : "#f85149",
          }}>
            <div style={{ width: 6, height: 6, borderRadius: "50%", background: "currentColor" }} />
            {status?.has_data ? "API Online" : "Sem dados"}
          </div>
          <button onClick={runPipeline} disabled={running} style={{
            background: running ? "#21262d" : "#378add",
            border: "none", borderRadius: 8, color: "#fff",
            padding: "8px 18px", fontSize: 13, fontWeight: 500, cursor: running ? "not-allowed" : "pointer",
          }}>
            {running ? "Rodando..." : "▶ Rodar pipeline"}
          </button>
        </div>
      </header>

      <main style={{ padding: "24px 32px", maxWidth: 1400, margin: "0 auto" }}>

        {/* FORECAST CARD */}
        {forecast && (
          <div style={{
            background: "rgba(55,138,221,0.08)", border: "0.5px solid #378add",
            borderRadius: 12, padding: "16px 24px", marginBottom: 24,
            display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 16,
          }}>
            <div>
              <div style={{ fontSize: 11, color: "#8b949e", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                Previsão T+1 — {forecast.date}
              </div>
              <div style={{ fontSize: 28, fontWeight: 500, marginTop: 4 }}>
                LSTM: <span style={{ color: "#378add" }}>{pct(forecast.lstm)}</span>
                <span style={{ fontSize: 16, color: "#8b949e", margin: "0 12px" }}>vs</span>
                GARCH: <span style={{ color: "#d85a30" }}>{pct(forecast.garch)}</span>
              </div>
            </div>
            <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: 11, color: "#8b949e" }}>Regime</div>
                <SentimentBadge label={forecast.regime === "Alta Volatilidade" ? "Negativo" : "Positivo"} />
                <div style={{ fontSize: 12, color: "#e6edf3", marginTop: 4 }}>{forecast.regime}</div>
              </div>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: 11, color: "#8b949e" }}>Sentimento</div>
                <div style={{ fontSize: 20, fontWeight: 500, color: clr(forecast.sentiment) }}>
                  {forecast.sentiment > 0 ? "+" : ""}{forecast.sentiment.toFixed(2)}
                </div>
              </div>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: 11, color: "#8b949e" }}>Confiança</div>
                <div style={{ fontSize: 20, fontWeight: 500, color: "#1d9e75" }}>
                  {pct(forecast.confidence)}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* METRIC CARDS */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0,1fr))", gap: 12, marginBottom: 24 }}>
          <MetricCard label="LSTM — MAE"   value={lstm  ? fmt(lstm.MAE)   : "—"} sub="erro médio absoluto" accent="#378add" badge={lstm && garch && lstm.MAE < garch.MAE ? "Melhor" : undefined} />
          <MetricCard label="GARCH — MAE"  value={garch ? fmt(garch.MAE)  : "—"} sub="erro médio absoluto" accent="#d85a30" badge={garch && lstm && garch.MAE < lstm.MAE ? "Melhor" : undefined} />
          <MetricCard label="LSTM — DA%"   value={lstm  ? lstm["DA%"].toFixed(1) + "%" : "—"} sub="acertos de direção" accent="#1d9e75" />
          <MetricCard label="LSTM — RMSE"  value={lstm  ? fmt(lstm.RMSE)  : "—"} sub="raiz erro quadrático" accent="#7f77dd" />
        </div>

        {/* GRÁFICO PRINCIPAL */}
        <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16, marginBottom: 20 }}>
          <Card>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
              <div>
                <div style={{ fontSize: 13, fontWeight: 500 }}>Volatilidade realizada vs. prevista</div>
                <div style={{ fontSize: 11, color: "#8b949e", marginTop: 2 }}>Anualizada — série histórica</div>
              </div>
              <div style={{ display: "flex", gap: 14, alignItems: "center" }}>
                {[["#58a6ff","Real"], ["#378add","LSTM"], ["#d85a30","GARCH"]].map(([c,l]) => (
                  <div key={l} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, color: "#8b949e" }}>
                    <div style={{ width: 8, height: 8, borderRadius: 2, background: c }} />
                    {l}
                  </div>
                ))}
                <select value={days} onChange={e => setDays(+e.target.value)} style={{
                  background: "#21262d", border: "0.5px solid rgba(240,246,252,0.2)",
                  borderRadius: 6, color: "#e6edf3", padding: "4px 8px", fontSize: 12,
                }}>
                  <option value={60}>60d</option>
                  <option value={90}>90d</option>
                  <option value={120}>120d</option>
                </select>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={volSeries} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="date" tick={{ fill: "#8b949e", fontSize: 10 }} tickLine={false}
                  tickFormatter={v => v.slice(5)} interval={Math.floor(volSeries.length / 8)} />
                <YAxis tick={{ fill: "#8b949e", fontSize: 10 }} tickLine={false} tickFormatter={tickFmt} width={48} />
                <Tooltip contentStyle={{ background: "#21262d", border: "0.5px solid rgba(240,246,252,0.2)", borderRadius: 8, fontSize: 12 }}
                  formatter={(v: number) => pct(v)} labelStyle={{ color: "#8b949e" }} />
                <Line type="monotone" dataKey="real"  stroke="#58a6ff" strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="lstm"  stroke="#378add" strokeWidth={1.2} dot={false} strokeDasharray="5 3" />
                <Line type="monotone" dataKey="garch" stroke="#d85a30" strokeWidth={1.2} dot={false} strokeDasharray="3 3" />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* MÉTRICAS TABELA */}
          <Card>
            <div style={{ fontSize: 11, fontWeight: 500, color: "#8b949e", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 14 }}>
              Comparativo de métricas
            </div>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <thead>
                <tr>
                  {["Métrica", "LSTM", "GARCH"].map(h => (
                    <th key={h} style={{ textAlign: "left", padding: "6px 8px", color: "#8b949e", borderBottom: "0.5px solid rgba(240,246,252,0.1)", fontSize: 11 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {lstm && garch && [
                  ["MAE",    lstm.MAE,      garch.MAE,      true],
                  ["RMSE",   lstm.RMSE,     garch.RMSE,     true],
                  ["MAPE%",  lstm["MAPE%"], garch["MAPE%"], true],
                  ["QLIKE",  lstm.QLIKE,    garch.QLIKE,    true],
                  ["DA%",    lstm["DA%"],   garch["DA%"],   false],
                ].map(([m, lv, gv, minWins]: any) => {
                  const lstmWins = minWins ? lv < gv : lv > gv;
                  return (
                    <tr key={m}>
                      <td style={{ padding: "7px 8px", color: "#8b949e", borderBottom: "0.5px solid rgba(240,246,252,0.06)" }}>{m}</td>
                      <td style={{ padding: "7px 8px", color: lstmWins  ? "#3fb950" : "#8b949e", fontWeight: lstmWins  ? 500 : 400, borderBottom: "0.5px solid rgba(240,246,252,0.06)" }}>{typeof lv === "number" ? lv.toFixed(4) : lv}</td>
                      <td style={{ padding: "7px 8px", color: !lstmWins ? "#3fb950" : "#8b949e", fontWeight: !lstmWins ? 500 : 400, borderBottom: "0.5px solid rgba(240,246,252,0.06)" }}>{typeof gv === "number" ? gv.toFixed(4) : gv}</td>
                    </tr>
                  );
                })}
                {!lstm && (
                  <tr><td colSpan={3} style={{ padding: 20, color: "#484f58", textAlign: "center" }}>Rode o pipeline primeiro</td></tr>
                )}
              </tbody>
            </table>
          </Card>
        </div>

        {/* BOTTOM GRID */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>

          {/* SENTIMENTO */}
          <Card>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
              <div style={{ fontSize: 11, fontWeight: 500, color: "#8b949e", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                Sentimento do mercado
              </div>
              {sentiment && (
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontSize: 18, fontWeight: 500, color: clr(sentiment.average) }}>
                    {sentiment.average > 0 ? "+" : ""}{sentiment.average.toFixed(2)}
                  </span>
                  <SentimentBadge label={sentiment.label} />
                </div>
              )}
            </div>
            <div style={{ height: 180, overflowY: "auto" }}>
              {sentiment?.items.map((item, i) => (
                <div key={i} style={{
                  display: "flex", alignItems: "flex-start", gap: 10,
                  padding: "8px 0", borderBottom: "0.5px solid rgba(240,246,252,0.06)",
                }}>
                  <div style={{
                    minWidth: 6, height: 6, borderRadius: "50%", marginTop: 5,
                    background: clr(item.score),
                  }} />
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 12, color: "#e6edf3", lineHeight: 1.4 }}>{item.headline}</div>
                    <div style={{ fontSize: 11, color: "#8b949e", marginTop: 2 }}>{item.time}</div>
                  </div>
                  <SentimentBadge label={item.label} />
                </div>
              ))}
            </div>
          </Card>

          {/* GARCH PARAMS */}
          <Card>
            <div style={{ fontSize: 11, fontWeight: 500, color: "#8b949e", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 14 }}>
              Parâmetros GARCH(1,1)
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 16 }}>
              {[
                ["ω (omega)", "0.000021", "constante"],
                ["α (ARCH)",  "0.1123",   "reação a choques"],
                ["β (GARCH)", "0.8541",   "persistência"],
                ["α + β",     "0.9664",   "deve ser < 1"],
                ["Half-life", "20.1d",    "duração do choque"],
                ["Regime",    "Estável",  "condição atual"],
              ].map(([label, val, sub]) => (
                <div key={label} style={{ background: "#21262d", borderRadius: 8, padding: "10px 12px", textAlign: "center" }}>
                  <div style={{ fontSize: 10, color: "#8b949e", marginBottom: 4 }}>{label}</div>
                  <div style={{ fontSize: 16, fontWeight: 500, color: label === "α + β" ? "#3fb950" : "#e6edf3" }}>{val}</div>
                  <div style={{ fontSize: 10, color: "#484f58", marginTop: 2 }}>{sub}</div>
                </div>
              ))}
            </div>
            <div>
              <div style={{ fontSize: 11, color: "#8b949e", marginBottom: 6 }}>Persistência da volatilidade (α+β)</div>
              <div style={{ background: "#21262d", borderRadius: 4, height: 6 }}>
                <div style={{ width: "96.64%", height: "100%", borderRadius: 4, background: "#d85a30", transition: "width 0.6s" }} />
              </div>
            </div>
            <div style={{ marginTop: 14, background: "#21262d", borderRadius: 8, padding: "10px 14px", fontFamily: "monospace", fontSize: 12, color: "#8b949e", lineHeight: 2 }}>
              <span style={{ color: "#378add" }}>σ²ₜ</span> = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁<br />
              <span style={{ color: "#1d9e75" }}>RV</span> = √(Σrᵢ²) × √252
            </div>
          </Card>
        </div>
      </main>
    </div>
  );
}
