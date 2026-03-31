"use client";

import { useEffect, useState, useCallback } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

const API = "";

const ASSET_COLORS: Record<string, string> = {
  "AAPL": "#58a6ff", "NVDA": "#3fb950", "MSFT": "#7f77dd",
  "PETR4.SA": "#f0e040", "VALE3.SA": "#1d9e75", "GC=F": "#ba7517",
  "CL=F": "#d85a30", "BTC/USDT": "#f7931a", "ETH/USDT": "#627eea", "EURUSD=X": "#378add",
};

const MODEL_COLORS: Record<string, string> = {
  "GARCH(1,1)": "#d85a30", "LSTM": "#378add", "XGBoost": "#3fb950",
};

const TYPE_LABELS: Record<string, string> = {
  stock: "Ação", crypto: "Cripto", commodity: "Commodity", forex: "Forex",
};

const SUPPORTED_ASSETS: Record<string, any> = {
  "AAPL":     { name: "Apple",     type: "stock" },
  "NVDA":     { name: "NVIDIA",    type: "stock" },
  "MSFT":     { name: "Microsoft", type: "stock" },
  "PETR4.SA": { name: "Petrobras", type: "stock" },
  "VALE3.SA": { name: "Vale",      type: "stock" },
  "GC=F":     { name: "Ouro",      type: "commodity" },
  "CL=F":     { name: "Petróleo",  type: "commodity" },
  "BTC/USDT": { name: "Bitcoin",   type: "crypto" },
  "ETH/USDT": { name: "Ethereum",  type: "crypto" },
  "EURUSD=X": { name: "EUR/USD",   type: "forex" },
};

const pct  = (v: number) => (v * 100).toFixed(1) + "%";
const fmt  = (v: number, d = 4) => v?.toFixed(d) ?? "—";

function Tag({ label, type }: { label: string; type: string }) {
  const colors: Record<string, [string, string]> = {
    stock:     ["rgba(55,138,221,0.15)",  "#378add"],
    crypto:    ["rgba(247,147,26,0.15)",  "#f7931a"],
    commodity: ["rgba(186,117,23,0.15)",  "#ba7517"],
    forex:     ["rgba(127,119,221,0.15)", "#7f77dd"],
  };
  const [bg, fg] = colors[type] ?? ["rgba(139,148,158,0.15)", "#8b949e"];
  return <span style={{ background: bg, color: fg, fontSize: 10, padding: "2px 7px", borderRadius: 20, fontWeight: 500 }}>{label}</span>;
}

function Card({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div style={{ background: "#161b22", border: "0.5px solid rgba(240,246,252,0.1)", borderRadius: 12, padding: "16px 20px", ...style }}>
      {children}
    </div>
  );
}

export default function MultiDashboard() {
  const [assets,    setAssets]    = useState<any[]>([]);
  const [selected,  setSelected]  = useState("AAPL");
  const [metrics,   setMetrics]   = useState<Record<string, any> | null>(null);
  const [forecast,  setForecast]  = useState<any>(null);
  const [volSeries, setVolSeries] = useState<any[]>([]);
  const [compareData, setCompare] = useState<Record<string, any>>({});
  const [featImp,   setFeatImp]   = useState<Record<string, number>>({});
  const [days,      setDays]      = useState(90);
  const [running,   setRunning]   = useState(false);
  const [activeTab, setActiveTab] = useState<"single" | "compare">("single");

  const fetchAssets = useCallback(async () => {
    const data = await fetch(`${API}/api/assets`).then(r => r.json()).catch(() => []);
    setAssets(data.length ? data : Object.entries(SUPPORTED_ASSETS).map(([symbol, info]) => ({
      symbol, ...info, has_data: false, running: false,
    })));
  }, []);

  const fetchAssetData = useCallback(async (sym: string) => {
    try {
      const [m, f, v] = await Promise.all([
        fetch(`${API}/api/metrics?symbol=${encodeURIComponent(sym)}`).then(r => r.ok ? r.json() : null),
        fetch(`${API}/api/forecast?symbol=${encodeURIComponent(sym)}`).then(r => r.json()),
        fetch(`${API}/api/volatility?symbol=${encodeURIComponent(sym)}&days=${days}`).then(r => r.json()),
      ]);
      setMetrics(m?.models ?? null);
      setFeatImp(m?.feature_importance ?? {});
      setForecast(f);
      setVolSeries(v.series ?? []);
    } catch (e) { console.error(e); }
  }, [days]);

  const fetchCompare = useCallback(async () => {
    const syms = "AAPL,NVDA,BTC/USDT,GC=F,PETR4.SA";
    const data = await fetch(`${API}/api/compare?symbols=${encodeURIComponent(syms)}&days=60`)
      .then(r => r.json()).catch(() => ({}));
    setCompare(data);
  }, []);

  useEffect(() => { fetchAssets(); fetchCompare(); }, [fetchAssets, fetchCompare]);
  useEffect(() => { fetchAssetData(selected); }, [fetchAssetData, selected]);

  const runPipeline = async () => {
    setRunning(true);
    await fetch(`${API}/api/run`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol: selected, period: "3y" }),
    });
    setTimeout(() => { fetchAssetData(selected); fetchAssets(); setRunning(false); }, 30000);
  };

  const selAsset   = assets.find(a => a.symbol === selected) ?? SUPPORTED_ASSETS[selected];
  const selColor   = ASSET_COLORS[selected] ?? "#378add";
  const compareLines = Object.entries(compareData).map(([sym, d]: [string, any]) => ({
    sym, color: ASSET_COLORS[sym] ?? "#8b949e", data: d.series ?? [],
  }));

  const bestMAE = metrics ? Math.min(...Object.values(metrics).map((m: any) => m.MAE)) : null;
  const bestDA  = metrics ? Math.max(...Object.values(metrics).map((m: any) => m["DA%"])) : null;

  return (
    <div style={{ minHeight: "100vh", background: "#0d1117", color: "#e6edf3", fontFamily: "'Segoe UI',system-ui,sans-serif", fontSize: 14 }}>

      {/* HEADER */}
      <header style={{ background: "#161b22", borderBottom: "0.5px solid rgba(240,246,252,0.1)", padding: "14px 32px", display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 34, height: 34, borderRadius: 8, background: "rgba(55,138,221,0.15)", border: "0.5px solid #378add", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, color: "#378add", fontWeight: 700 }}>Q</div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 500 }}>Quant Volatility Dashboard</div>
            <div style={{ fontSize: 12, color: "#8b949e" }}>GARCH · LSTM · XGBoost — Multi-Asset</div>
          </div>
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
          {["single", "compare"].map(tab => (
            <button key={tab} onClick={() => setActiveTab(tab as any)} style={{
              background: activeTab === tab ? "rgba(55,138,221,0.2)" : "transparent",
              border: `0.5px solid ${activeTab === tab ? "#378add" : "rgba(240,246,252,0.15)"}`,
              borderRadius: 8, color: activeTab === tab ? "#378add" : "#8b949e",
              padding: "6px 14px", fontSize: 12, cursor: "pointer",
            }}>
              {tab === "single" ? "Ativo único" : "Comparar"}
            </button>
          ))}
          <button onClick={runPipeline} disabled={running} style={{
            background: running ? "#21262d" : selColor, border: "none",
            borderRadius: 8, color: "#fff", padding: "8px 18px", fontSize: 13, fontWeight: 500, cursor: running ? "not-allowed" : "pointer",
          }}>
            {running ? "Rodando..." : `▶ Rodar ${selected}`}
          </button>
        </div>
      </header>

      <main style={{ padding: "20px 32px", maxWidth: 1400, margin: "0 auto" }}>

        {/* ASSET SELECTOR */}
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 20 }}>
          {assets.map((a: any) => (
            <button key={a.symbol} onClick={() => setSelected(a.symbol)} style={{
              background: selected === a.symbol ? `${ASSET_COLORS[a.symbol] ?? "#378add"}22` : "#161b22",
              border: `0.5px solid ${selected === a.symbol ? (ASSET_COLORS[a.symbol] ?? "#378add") : "rgba(240,246,252,0.1)"}`,
              borderRadius: 10, padding: "8px 14px", cursor: "pointer",
              display: "flex", alignItems: "center", gap: 8,
            }}>
              <div style={{ width: 7, height: 7, borderRadius: "50%", background: a.has_data ? "#3fb950" : "#484f58" }} />
              <div style={{ textAlign: "left" }}>
                <div style={{ fontSize: 12, fontWeight: 500, color: selected === a.symbol ? (ASSET_COLORS[a.symbol] ?? "#378add") : "#e6edf3" }}>{a.symbol}</div>
                <div style={{ fontSize: 10, color: "#8b949e" }}>{a.name}</div>
              </div>
              <Tag label={TYPE_LABELS[a.type] ?? a.type} type={a.type} />
            </button>
          ))}
        </div>

        {activeTab === "single" ? (
          <>
            {/* FORECAST BANNER */}
            {forecast && (
              <div style={{ background: `${selColor}11`, border: `0.5px solid ${selColor}`, borderRadius: 12, padding: "16px 24px", marginBottom: 20, display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 16 }}>
                <div>
                  <div style={{ fontSize: 11, color: "#8b949e", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                    Previsão T+1 — {selAsset?.name ?? selected} — {forecast.date}
                  </div>
                  <div style={{ fontSize: 22, fontWeight: 500, marginTop: 6, display: "flex", gap: 20, flexWrap: "wrap" }}>
                    <span>LSTM: <span style={{ color: MODEL_COLORS["LSTM"] }}>{pct(forecast.lstm)}</span></span>
                    <span style={{ color: "#484f58" }}>·</span>
                    <span>GARCH: <span style={{ color: MODEL_COLORS["GARCH(1,1)"] }}>{pct(forecast.garch)}</span></span>
                    <span style={{ color: "#484f58" }}>·</span>
                    <span>XGB: <span style={{ color: MODEL_COLORS["XGBoost"] }}>{pct(forecast.xgb)}</span></span>
                  </div>
                </div>
                <div style={{ display: "flex", gap: 28 }}>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 11, color: "#8b949e", marginBottom: 4 }}>Consenso</div>
                    <div style={{ fontSize: 24, fontWeight: 500, color: selColor }}>{pct(forecast.consensus)}</div>
                  </div>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 11, color: "#8b949e", marginBottom: 4 }}>Regime</div>
                    <div style={{ fontSize: 13, fontWeight: 500, marginTop: 4 }}>{forecast.regime}</div>
                  </div>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 11, color: "#8b949e", marginBottom: 4 }}>Sentimento</div>
                    <div style={{ fontSize: 22, fontWeight: 500, color: forecast.sentiment >= 0 ? "#3fb950" : "#f85149" }}>
                      {forecast.sentiment > 0 ? "+" : ""}{forecast.sentiment.toFixed(2)}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* MODEL CARDS */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,minmax(0,1fr))", gap: 12, marginBottom: 20 }}>
              {["GARCH(1,1)", "LSTM", "XGBoost"].map(model => {
                const m     = metrics?.[model];
                const color = MODEL_COLORS[model];
                const wins  = m && bestMAE !== null && m.MAE === bestMAE;
                return (
                  <Card key={model} style={{ borderTop: `2px solid ${color}` }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                      <span style={{ fontSize: 13, fontWeight: 500, color }}>{model}</span>
                      {wins && <span style={{ fontSize: 10, background: "rgba(63,185,80,0.15)", color: "#3fb950", padding: "2px 8px", borderRadius: 20 }}>Melhor MAE</span>}
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                      {[["MAE", m?.MAE, 4], ["RMSE", m?.RMSE, 4], ["MAPE%", m?.["MAPE%"], 1], ["DA%", m?.["DA%"], 1]].map(([label, val, d]) => (
                        <div key={String(label)}>
                          <div style={{ fontSize: 10, color: "#8b949e", marginBottom: 2 }}>{String(label)}</div>
                          <div style={{ fontSize: 16, fontWeight: 500 }}>{val != null ? Number(val).toFixed(Number(d)) : "—"}</div>
                        </div>
                      ))}
                    </div>
                  </Card>
                );
              })}
            </div>

            {/* CHARTS */}
            <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16, marginBottom: 16 }}>
              <Card>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 500 }}>Volatilidade — {selAsset?.name ?? selected}</div>
                    <div style={{ fontSize: 11, color: "#8b949e" }}>Anualizada</div>
                  </div>
                  <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                    {[["#58a6ff","Real"],["#378add","LSTM"],["#d85a30","GARCH"],["#3fb950","XGB"]].map(([c,l]) => (
                      <div key={l} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 11, color: "#8b949e" }}>
                        <div style={{ width: 8, height: 8, borderRadius: 2, background: c }} />{l}
                      </div>
                    ))}
                    <select value={days} onChange={e => setDays(+e.target.value)} style={{ background: "#21262d", border: "0.5px solid rgba(240,246,252,0.2)", borderRadius: 6, color: "#e6edf3", padding: "4px 8px", fontSize: 12 }}>
                      {[60, 90, 120].map(d => <option key={d} value={d}>{d}d</option>)}
                    </select>
                  </div>
                </div>
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={volSeries}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                    <XAxis dataKey="date" tick={{ fill: "#8b949e", fontSize: 10 }} tickLine={false} tickFormatter={v => v.slice(5)} interval={Math.floor(volSeries.length / 8)} />
                    <YAxis tick={{ fill: "#8b949e", fontSize: 10 }} tickLine={false} tickFormatter={v => (v * 100).toFixed(0) + "%"} width={44} />
                    <Tooltip contentStyle={{ background: "#21262d", border: "0.5px solid rgba(240,246,252,0.15)", borderRadius: 8, fontSize: 12 }} formatter={(v: number) => pct(v)} labelStyle={{ color: "#8b949e" }} />
                    <Line type="monotone" dataKey="real"  stroke="#58a6ff" strokeWidth={1.8} dot={false} name="Real" />
                    <Line type="monotone" dataKey="lstm"  stroke="#378add" strokeWidth={1.2} dot={false} strokeDasharray="5 3" name="LSTM" />
                    <Line type="monotone" dataKey="garch" stroke="#d85a30" strokeWidth={1.2} dot={false} strokeDasharray="3 3" name="GARCH" />
                    <Line type="monotone" dataKey="xgb"   stroke="#3fb950" strokeWidth={1.2} dot={false} strokeDasharray="2 4" name="XGBoost" />
                  </LineChart>
                </ResponsiveContainer>
              </Card>

              <Card>
                <div style={{ fontSize: 11, fontWeight: 500, color: "#8b949e", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 12 }}>
                  Feature Importance — XGBoost
                </div>
                {Object.keys(featImp).length > 0 ? (
                  <div style={{ height: 210, overflowY: "auto" }}>
                    {Object.entries(featImp).slice(0, 10).map(([feat, score]) => {
                      const maxScore = Math.max(...Object.values(featImp));
                      return (
                        <div key={feat} style={{ marginBottom: 9 }}>
                          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 3 }}>
                            <span style={{ color: "#e6edf3" }}>{feat}</span>
                            <span style={{ color: "#8b949e" }}>{Number(score).toFixed(0)}</span>
                          </div>
                          <div style={{ background: "#21262d", borderRadius: 3, height: 4 }}>
                            <div style={{ width: `${(score / maxScore) * 100}%`, height: "100%", borderRadius: 3, background: "#3fb950" }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div style={{ color: "#484f58", fontSize: 12, textAlign: "center", paddingTop: 60 }}>
                    Rode o pipeline para ver feature importance
                  </div>
                )}
              </Card>
            </div>

            {/* METRICS TABLE */}
            <Card>
              <div style={{ fontSize: 11, fontWeight: 500, color: "#8b949e", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 12 }}>
                Comparativo completo
              </div>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <thead>
                  <tr>
                    {["Modelo", "MAE", "RMSE", "MAPE%", "QLIKE", "DA%", "N"].map(h => (
                      <th key={h} style={{ textAlign: "left", padding: "7px 10px", color: "#8b949e", borderBottom: "0.5px solid rgba(240,246,252,0.1)", fontSize: 11 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {metrics ? Object.entries(metrics).map(([model, m]: [string, any]) => (
                    <tr key={model}>
                      <td style={{ padding: "8px 10px", borderBottom: "0.5px solid rgba(240,246,252,0.06)" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                          <div style={{ width: 8, height: 8, borderRadius: 2, background: MODEL_COLORS[model] ?? "#8b949e" }} />
                          <span style={{ color: MODEL_COLORS[model] ?? "#e6edf3", fontWeight: 500 }}>{model}</span>
                        </div>
                      </td>
                      <td style={{ padding: "8px 10px", borderBottom: "0.5px solid rgba(240,246,252,0.06)", color: m.MAE === bestMAE ? "#3fb950" : "#e6edf3", fontWeight: m.MAE === bestMAE ? 500 : 400 }}>{fmt(m.MAE)}</td>
                      <td style={{ padding: "8px 10px", borderBottom: "0.5px solid rgba(240,246,252,0.06)" }}>{fmt(m.RMSE)}</td>
                      <td style={{ padding: "8px 10px", borderBottom: "0.5px solid rgba(240,246,252,0.06)" }}>{fmt(m["MAPE%"], 2)}</td>
                      <td style={{ padding: "8px 10px", borderBottom: "0.5px solid rgba(240,246,252,0.06)" }}>{fmt(m.QLIKE, 3)}</td>
                      <td style={{ padding: "8px 10px", borderBottom: "0.5px solid rgba(240,246,252,0.06)", color: m["DA%"] === bestDA ? "#3fb950" : "#e6edf3", fontWeight: m["DA%"] === bestDA ? 500 : 400 }}>{fmt(m["DA%"], 1)}%</td>
                      <td style={{ padding: "8px 10px", borderBottom: "0.5px solid rgba(240,246,252,0.06)", color: "#8b949e" }}>{m.N?.toFixed(0)}</td>
                    </tr>
                  )) : (
                    <tr><td colSpan={7} style={{ padding: 24, textAlign: "center", color: "#484f58" }}>Rode o pipeline para {selected}</td></tr>
                  )}
                </tbody>
              </table>
            </Card>
          </>
        ) : (
          /* COMPARE TAB */
          <>
            <Card style={{ marginBottom: 16 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 500 }}>Volatilidade comparada — múltiplos ativos</div>
                  <div style={{ fontSize: 11, color: "#8b949e" }}>60 dias</div>
                </div>
                <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                  {compareLines.map(({ sym, color }) => (
                    <div key={sym} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 11, color: "#8b949e" }}>
                      <div style={{ width: 8, height: 8, borderRadius: 2, background: color }} />{sym}
                    </div>
                  ))}
                </div>
              </div>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis dataKey="date" tick={{ fill: "#8b949e", fontSize: 10 }} tickLine={false} tickFormatter={v => v?.slice(5)} interval={8} />
                  <YAxis tick={{ fill: "#8b949e", fontSize: 10 }} tickLine={false} tickFormatter={v => (v * 100).toFixed(0) + "%"} width={44} />
                  <Tooltip contentStyle={{ background: "#21262d", border: "0.5px solid rgba(240,246,252,0.15)", borderRadius: 8, fontSize: 12 }} formatter={(v: number) => pct(v)} labelStyle={{ color: "#8b949e" }} />
                  {compareLines.map(({ sym, color, data }) => (
                    <Line key={sym} data={data} type="monotone" dataKey="real" stroke={color} strokeWidth={1.5} dot={false} name={sym} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </Card>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(5,minmax(0,1fr))", gap: 12 }}>
              {compareLines.map(({ sym, color, data }) => {
                const current = data[data.length - 1]?.real ?? 0;
                const prev    = data[data.length - 8]?.real ?? current;
                const change  = prev ? ((current - prev) / prev * 100) : 0;
                const atype   = SUPPORTED_ASSETS[sym]?.type ?? "stock";
                return (
                  <Card key={sym} style={{ borderTop: `2px solid ${color}` }}>
                    <div style={{ fontSize: 11, color: "#8b949e", marginBottom: 6 }}>{SUPPORTED_ASSETS[sym]?.name ?? sym}</div>
                    <div style={{ fontSize: 20, fontWeight: 500, color }}>{pct(current)}</div>
                    <div style={{ fontSize: 11, marginTop: 4, color: change >= 0 ? "#f85149" : "#3fb950" }}>
                      {change >= 0 ? "▲" : "▼"} {Math.abs(change).toFixed(1)}% (7d)
                    </div>
                    <div style={{ marginTop: 6 }}>
                      <Tag label={TYPE_LABELS[atype] ?? atype} type={atype} />
                    </div>
                  </Card>
                );
              })}
            </div>
          </>
        )}
      </main>
    </div>
  );
}