import React from "react";
import { ModelInfo } from "../types";

interface ModelInfoViewProps {
  model: ModelInfo | null;
}

export const ModelInfoView: React.FC<ModelInfoViewProps> = ({ model }) => {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.75rem" }}>
      
      {/* Model Overview & Governance Card */}
      <div className="glass-panel" style={{ padding: "2rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "1rem", marginBottom: "1.5rem" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.5rem" }}>
              <h2 style={{ fontSize: "1.5rem", fontWeight: 800 }}>StyleForge v1.0.0 Production Model</h2>
              <span className="badge badge-emerald">Verified Official Checkpoint</span>
            </div>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
              StyleGAN-inspired generative adversarial architecture trained on authentic photographic CelebA human faces.
            </p>
          </div>

          <div style={{ textAlign: "right" }}>
            <span className="badge badge-indigo" style={{ fontSize: "0.85rem", padding: "0.35rem 0.75rem" }}>
              Native Resolution: 64×64
            </span>
          </div>
        </div>

        {/* SHA-256 Card */}
        <div style={{ background: "rgba(0,0,0,0.4)", borderRadius: "0.5rem", padding: "0.875rem 1rem", border: "1px solid var(--border-subtle)", marginBottom: "1.5rem" }}>
          <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 600, marginBottom: "0.25rem" }}>
            Cryptographic SHA-256 Checksum
          </div>
          <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.85rem", color: "#38bdf8", wordBreak: "break-all" }}>
            {model?.checkpoint_sha256 || "ec8e4bf1e4f21f240f7793a1b3922a4510394831d0c39de521c4cd456c9eed64"}
          </div>
        </div>

        {/* Key Model Specifications */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem" }}>
          <div style={{ background: "rgba(255,255,255,0.03)", padding: "1rem", borderRadius: "0.5rem" }}>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Latent Dimension (Z)</div>
            <div style={{ fontSize: "1.25rem", fontWeight: 700 }}>512-dim Standard Gaussian</div>
          </div>
          <div style={{ background: "rgba(255,255,255,0.03)", padding: "1rem", borderRadius: "0.5rem" }}>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Style Manifold (W)</div>
            <div style={{ fontSize: "1.25rem", fontWeight: 700 }}>512-dim Disentangled</div>
          </div>
          <div style={{ background: "rgba(255,255,255,0.03)", padding: "1rem", borderRadius: "0.5rem" }}>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Mapping Layers</div>
            <div style={{ fontSize: "1.25rem", fontWeight: 700 }}>8 Equalized MLP (lr_mul=0.01)</div>
          </div>
          <div style={{ background: "rgba(255,255,255,0.03)", padding: "1rem", borderRadius: "0.5rem" }}>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Synthesis Stages</div>
            <div style={{ fontSize: "1.25rem", fontWeight: 700 }}>4×4 to 64×64 (AdaIN)</div>
          </div>
        </div>
      </div>

      {/* Explicit Scientific Limitations Notice */}
      <div style={{ background: "rgba(245, 158, 11, 0.08)", border: "1px solid rgba(245, 158, 11, 0.25)", borderRadius: "1rem", padding: "1.5rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.5rem" }}>
          <span style={{ fontSize: "1.25rem" }}>⚠️</span>
          <h3 style={{ fontSize: "1.05rem", fontWeight: 700, color: "#fbbf24" }}>Model Architecture & Operational Scope</h3>
        </div>
        <ul style={{ paddingLeft: "1.5rem", fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.6, display: "flex", flexDirection: "column", gap: "0.375rem" }}>
          <li>
            <strong>Resolution Reality:</strong> StyleForge is natively trained at <strong>64×64 pixel resolution</strong> on CelebA. The platform intentionally outputs the native resolution and does not apply generative upscaling or hallucinated interpolation filters.
          </li>
          <li>
            <strong>Domain Boundaries:</strong> The model generates centered, photographic human faces. Arbitrary prompts, full-body poses, or non-facial objects are out-of-distribution.
          </li>
          <li>
            <strong>Inference Optimization:</strong> Clean Face Mode (noise scale 0.25) aligns high-frequency Fourier spectral energy with real CelebA images (0.97× parity), but does not alter the underlying model resolution.
          </li>
        </ul>
      </div>

      {/* Master Empirical Benchmarks */}
      <div className="glass-panel" style={{ padding: "2rem" }}>
        <h3 style={{ fontSize: "1.125rem", fontWeight: 700, marginBottom: "0.25rem" }}>Controlled Inference Ablation Benchmarks</h3>
        <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "1.25rem" }}>
          Empirically measured on N=614 authentic photographic validation partition using TorchMetrics Inception-v3 pool3.
        </p>

        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem", textAlign: "left" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border-subtle)", color: "var(--text-muted)" }}>
                <th style={{ padding: "0.75rem 0.5rem" }}>Condition</th>
                <th style={{ padding: "0.75rem 0.5rem" }}>Weights</th>
                <th style={{ padding: "0.75rem 0.5rem" }}>Noise Scale</th>
                <th style={{ padding: "0.75rem 0.5rem" }}>Inception FID ↓</th>
                <th style={{ padding: "0.75rem 0.5rem" }}>Laplacian Var (Sharpness)</th>
                <th style={{ padding: "0.75rem 0.5rem" }}>HF PSD Ratio</th>
                <th style={{ padding: "0.75rem 0.5rem" }}>Diversity (L1)</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ borderBottom: "1px solid var(--border-subtle)", background: "rgba(255,255,255,0.02)", fontWeight: 600 }}>
                <td style={{ padding: "0.75rem 0.5rem", color: "#34d399" }}>Real CelebA Validation</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>Photographic</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>N/A</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>0.00</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>874.23 ± 277</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>0.000046 (1.00×)</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>N/A</td>
              </tr>
              <tr style={{ borderBottom: "1px solid var(--border-subtle)" }}>
                <td style={{ padding: "0.75rem 0.5rem" }}>C1 Baseline (v1.0.0 default)</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>Raw</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>1.00</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>298.86</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>4057.01</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>0.000090 (1.95×)</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>0.5862</td>
              </tr>
              <tr style={{ borderBottom: "1px solid var(--border-subtle)", background: "rgba(99, 102, 241, 0.05)" }}>
                <td style={{ padding: "0.75rem 0.5rem", color: "#818cf8", fontWeight: 600 }}>C2 Best Inception FID</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>Raw</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>0.50</td>
                <td style={{ padding: "0.75rem 0.5rem", fontWeight: 700, color: "#818cf8" }}>294.75</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>1561.56</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>0.000054 (1.17×)</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>0.5813</td>
              </tr>
              <tr style={{ borderBottom: "1px solid var(--border-subtle)", background: "rgba(56, 189, 248, 0.05)" }}>
                <td style={{ padding: "0.75rem 0.5rem", color: "#38bdf8", fontWeight: 600 }}>C3 Clean Mode (Parity)</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>Raw</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>0.25</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>304.76</td>
                <td style={{ padding: "0.75rem 0.5rem", fontWeight: 700, color: "#38bdf8" }}>922.43 (Parity)</td>
                <td style={{ padding: "0.75rem 0.5rem", fontWeight: 700, color: "#38bdf8" }}>0.000045 (0.97×)</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>0.5801</td>
              </tr>
              <tr style={{ borderBottom: "1px solid var(--border-subtle)" }}>
                <td style={{ padding: "0.75rem 0.5rem" }}>C4 Fully Denoised</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>Raw</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>0.00</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>311.89</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>707.67</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>0.000042 (0.91×)</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>0.5797</td>
              </tr>
              <tr>
                <td style={{ padding: "0.75rem 0.5rem" }}>C5 AdaIN Clamped (Rejected)</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>Raw</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>0.00</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>312.03</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>911.77</td>
                <td style={{ padding: "0.75rem 0.5rem" }}>0.000055 (1.19×)</td>
                <td style={{ padding: "0.75rem 0.5rem", color: "#f87171" }}>0.5207 (-10.2%)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
};
