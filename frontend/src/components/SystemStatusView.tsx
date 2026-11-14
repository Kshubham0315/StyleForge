import React, { useEffect, useState } from "react";
import { fetchPrometheusMetrics } from "../api";
import { HealthInfo, ModelInfo } from "../types";

interface SystemStatusViewProps {
  health: HealthInfo | null;
  model: ModelInfo | null;
  onRefreshHealth: () => void;
}

export const SystemStatusView: React.FC<SystemStatusViewProps> = ({
  health,
  model,
  onRefreshHealth,
}) => {
  const [metricsText, setMetricsText] = useState<string>("");
  const [loadingMetrics, setLoadingMetrics] = useState<boolean>(false);

  const loadMetrics = async () => {
    setLoadingMetrics(true);
    try {
      const text = await fetchPrometheusMetrics();
      setMetricsText(text);
    } catch (e) {
      setMetricsText("# Failed to scrape Prometheus metrics endpoint.");
    } finally {
      setLoadingMetrics(false);
    }
  };

  useEffect(() => {
    loadMetrics();
  }, []);

  const vram = model?.vram_telemetry;
  const vramPct = vram && vram.total_mb > 0 ? (vram.allocated_mb / vram.total_mb) * 100 : 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.75rem" }}>
      
      {/* Top Telemetry Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem" }}>
        <div>
          <h2 style={{ fontSize: "1.25rem", fontWeight: 700, marginBottom: "0.25rem" }}>System Health & Telemetry</h2>
          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
            Live hardware profiling, VRAM gauges, and Prometheus exporter inspection
          </p>
        </div>
        <button onClick={() => { onRefreshHealth(); loadMetrics(); }} className="btn-secondary">
          🔄 Refresh Telemetry
        </button>
      </div>

      {/* Hardware & VRAM Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1.25rem" }}>
        
        {/* API Health Card */}
        <div className="glass-panel" style={{ padding: "1.5rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
            <span style={{ fontSize: "0.85rem", fontWeight: 700 }}>FastAPI Platform Service</span>
            <span className="badge badge-emerald">{health?.status || "ONLINE"}</span>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", fontSize: "0.85rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Platform Version:</span>
              <span>{health?.version || "v2.0.0"}</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>System Uptime:</span>
              <span>{health ? `${(health.uptime_seconds / 60).toFixed(1)} minutes` : "N/A"}</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Database Connected:</span>
              <span style={{ color: "#34d399" }}>{health?.database_connected ? "Yes (Active)" : "Disconnected"}</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Inference State:</span>
              <span style={{ color: health?.models_loaded ? "#34d399" : "#fbbf24" }}>
                {health?.models_loaded ? "Generators Pre-loaded" : "Awaiting Weights"}
              </span>
            </div>
          </div>
        </div>

        {/* GPU VRAM Telemetry Card */}
        <div className="glass-panel" style={{ padding: "1.5rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
            <span style={{ fontSize: "0.85rem", fontWeight: 700 }}>GPU Memory Footprint</span>
            <span className={health?.cuda_available ? "badge badge-emerald" : "badge badge-amber"}>
              {health?.cuda_available ? "CUDA ACCELERATED" : "CPU FALLBACK"}
            </span>
          </div>

          {health?.cuda_available && vram ? (
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", marginBottom: "0.375rem" }}>
                <span style={{ color: "var(--text-muted)" }}>VRAM Allocated</span>
                <span style={{ fontWeight: 700 }}>{vram.allocated_mb.toFixed(1)} / {vram.total_mb.toFixed(0)} MB ({vramPct.toFixed(1)}%)</span>
              </div>
              <div style={{ width: "100%", height: "8px", background: "rgba(255,255,255,0.1)", borderRadius: "4px", overflow: "hidden", marginBottom: "1rem" }}>
                <div style={{ width: `${Math.min(100, Math.max(2, vramPct))}%`, height: "100%", background: "linear-gradient(90deg, #6366f1, #06b6d4)", borderRadius: "4px" }} />
              </div>

              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem" }}>
                <span style={{ color: "var(--text-muted)" }}>Device Name:</span>
                <span style={{ fontWeight: 600 }}>{vram.device_name}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", marginTop: "0.25rem" }}>
                <span style={{ color: "var(--text-muted)" }}>Free VRAM:</span>
                <span style={{ color: "#34d399", fontWeight: 600 }}>{vram.free_mb.toFixed(0)} MB</span>
              </div>
            </div>
          ) : (
            <div style={{ color: "var(--text-muted)", fontSize: "0.85rem", padding: "1rem 0" }}>
              CUDA is not active. The inference engine is operating on host CPU memory with automatic thread scaling.
            </div>
          )}
        </div>

      </div>

      {/* Multi-Service Docker Architecture Status */}
      <div className="glass-panel" style={{ padding: "1.75rem" }}>
        <h3 style={{ fontSize: "1.05rem", fontWeight: 700, marginBottom: "0.75rem" }}>Production Architecture Services</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "0.75rem" }}>
          {[
            { name: "Frontend", tech: "React + Vite (Port 3000)", status: "Active" },
            { name: "Backend API", tech: "FastAPI + Uvicorn (Port 8000)", status: "Active" },
            { name: "Database", tech: "PostgreSQL / SQLite (Port 5432)", status: "Active" },
            { name: "Artifacts", tech: "Local Disk / S3 Adapter", status: "Active" },
            { name: "Prometheus", tech: "Scrape Engine (Port 9090)", status: "Configured" },
            { name: "Grafana", tech: "Telemetry Visualizer (Port 3001)", status: "Configured" },
          ].map((svc, i) => (
            <div key={i} style={{ background: "rgba(255,255,255,0.02)", border: "1px solid var(--border-subtle)", borderRadius: "0.5rem", padding: "0.75rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.25rem" }}>
                <span style={{ fontSize: "0.85rem", fontWeight: 600 }}>{svc.name}</span>
                <span className="badge badge-emerald" style={{ fontSize: "0.6rem", padding: "0.1rem 0.35rem" }}>{svc.status}</span>
              </div>
              <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>{svc.tech}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Live Prometheus Metrics Exporter Feed */}
      <div className="glass-panel" style={{ padding: "1.75rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
          <h3 style={{ fontSize: "1.05rem", fontWeight: 700 }}>Live Prometheus Exporter Stream</h3>
          <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontFamily: "JetBrains Mono, monospace" }}>
            GET /api/v1/metrics
          </span>
        </div>
        <pre style={{
          background: "#030712",
          border: "1px solid var(--border-subtle)",
          borderRadius: "0.5rem",
          padding: "1rem",
          fontSize: "0.75rem",
          fontFamily: "JetBrains Mono, monospace",
          color: "#93c5fd",
          maxHeight: "260px",
          overflowY: "auto",
          lineHeight: 1.5,
        }}>
          {loadingMetrics ? "Scraping Prometheus feed..." : metricsText || "No metrics returned."}
        </pre>
      </div>

    </div>
  );
};
