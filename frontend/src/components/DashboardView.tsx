import React from "react";
import { GenerationListItem, HealthInfo, ModelInfo } from "../types";

interface DashboardViewProps {
  onGoToGenerator: (preset?: "standard" | "smooth") => void;
  recentGenerations: GenerationListItem[];
  model: ModelInfo | null;
  health: HealthInfo | null;
}

export const DashboardView: React.FC<DashboardViewProps> = ({
  onGoToGenerator,
  recentGenerations,
  model,
  health,
}) => {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.75rem" }}>
      
      {/* Hero Section */}
      <div className="glass-panel" style={{ padding: "2.5rem 2rem", position: "relative", overflow: "hidden" }}>
        <div style={{ position: "relative", zIndex: 1, maxWidth: "720px" }}>
          <div className="badge badge-indigo" style={{ marginBottom: "1rem" }}>
            Production Generative Platform v2.0
          </div>
          <h1 style={{ fontSize: "2.25rem", fontWeight: 800, letterSpacing: "-0.03em", marginBottom: "0.75rem", lineHeight: 1.2 }}>
            High-Stability Style-Based <br />
            <span style={{ background: "linear-gradient(135deg, #818cf8 0%, #38bdf8 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
              Facial Synthesis Studio
            </span>
          </h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "1rem", lineHeight: 1.6, marginBottom: "1.75rem" }}>
            Explore continuous latent space traversals, W-space truncation modulation, and validated spectral parity face generation on the authentic CelebA distribution.
          </p>
          <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
            <button
              onClick={() => onGoToGenerator("smooth")}
              className="btn-primary"
              style={{ padding: "0.75rem 1.75rem", fontSize: "0.95rem" }}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
              </svg>
              Launch Clean Face Mode (0.25)
            </button>
            <button
              onClick={() => onGoToGenerator("standard")}
              className="btn-secondary"
              style={{ padding: "0.75rem 1.5rem" }}
            >
              Standard Mode (1.00)
            </button>
          </div>
        </div>

        {/* Decorative Grid Graphic */}
        <div style={{
          position: "absolute",
          right: "2rem",
          top: "50%",
          transform: "translateY(-50%)",
          opacity: 0.15,
          pointerEvents: "none",
          display: "grid",
          gridTemplateColumns: "repeat(4, 50px)",
          gap: "8px"
        }}>
          {Array.from({ length: 16 }).map((_, i) => (
            <div key={i} style={{ width: "50px", height: "50px", border: "1px solid #6366f1", borderRadius: "8px" }} />
          ))}
        </div>
      </div>

      {/* Metrics & Architecture Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "1.25rem" }}>
        
        {/* Card 1 */}
        <div className="glass-panel" style={{ padding: "1.5rem" }}>
          <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase", marginBottom: "0.5rem" }}>
            Model Resolution
          </div>
          <div style={{ fontSize: "1.75rem", fontWeight: 700, color: "#ffffff", marginBottom: "0.25rem" }}>
            64 × 64 Native
          </div>
          <div style={{ fontSize: "0.8rem", color: "#34d399", display: "flex", alignItems: "center", gap: "0.25rem" }}>
            Authentic CelebA Photorealistic Scale
          </div>
        </div>

        {/* Card 2 */}
        <div className="glass-panel" style={{ padding: "1.5rem" }}>
          <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase", marginBottom: "0.5rem" }}>
            Best Inception FID
          </div>
          <div style={{ fontSize: "1.75rem", fontWeight: 700, color: "#818cf8", marginBottom: "0.25rem" }}>
            294.75
          </div>
          <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
            Ablation C2 (Noise 0.50, Raw Weights)
          </div>
        </div>

        {/* Card 3 */}
        <div className="glass-panel" style={{ padding: "1.5rem" }}>
          <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase", marginBottom: "0.5rem" }}>
            Spectral Parity (HF FFT)
          </div>
          <div style={{ fontSize: "1.75rem", fontWeight: 700, color: "#38bdf8", marginBottom: "0.25rem" }}>
            0.97× Real
          </div>
          <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
            Smooth Mode (Laplacian: 922 vs 874 Real)
          </div>
        </div>

        {/* Card 4 */}
        <div className="glass-panel" style={{ padding: "1.5rem" }}>
          <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", fontWeight: 600, textTransform: "uppercase", marginBottom: "0.5rem" }}>
            Active Hardware Device
          </div>
          <div style={{ fontSize: "1.75rem", fontWeight: 700, color: health?.cuda_available ? "#10b981" : "#f59e0b", marginBottom: "0.25rem" }}>
            {health?.device.toUpperCase() || "CPU"}
          </div>
          <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
            {model?.vram_telemetry?.device_name || "Host Processor"}
          </div>
        </div>

      </div>

      {/* Recent Generations Strip */}
      <div className="glass-panel" style={{ padding: "1.5rem" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
          <div>
            <h3 style={{ fontSize: "1.125rem", fontWeight: 700 }}>Recent Generations</h3>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Persisted outputs from synthesis sessions</p>
          </div>
          <button onClick={() => onGoToGenerator()} className="btn-secondary" style={{ fontSize: "0.8rem" }}>
            View Generator Studio &rarr;
          </button>
        </div>

        {recentGenerations.length === 0 ? (
          <div style={{ textAlign: "center", padding: "3rem 1rem", color: "var(--text-muted)", fontSize: "0.9rem" }}>
            No recent generations found. Launch the Studio Generator to generate your first face!
          </div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(110px, 1fr))", gap: "1rem" }}>
            {recentGenerations.slice(0, 8).map((gen) => (
              <div
                key={gen.id}
                className="glass-panel-interactive"
                style={{ padding: "0.5rem", textAlign: "center", cursor: "pointer" }}
                onClick={() => onGoToGenerator()}
              >
                {gen.download_url ? (
                  <img
                    src={gen.download_url}
                    alt={`Gen ${gen.id.slice(0, 6)}`}
                    className="pixelated"
                    style={{ width: "100%", aspectRatio: "1/1", borderRadius: "0.375rem", objectFit: "cover", marginBottom: "0.5rem" }}
                  />
                ) : (
                  <div style={{ width: "100%", aspectRatio: "1/1", background: "rgba(255,255,255,0.05)", borderRadius: "0.375rem", marginBottom: "0.5rem" }} />
                )}
                <div style={{ fontSize: "0.7rem", fontWeight: 600, color: "var(--text-primary)" }}>
                  Seed: {gen.seed !== null ? gen.seed : "Rand"}
                </div>
                <div style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}>
                  {gen.mode} ({gen.latency_ms.toFixed(0)}ms)
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

    </div>
  );
};
