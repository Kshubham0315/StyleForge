import React, { useState } from "react";
import { generateFacesApi } from "../api";
import { GenerateResponse, ImageItem } from "../types";

interface GeneratorViewProps {
  initialMode?: "standard" | "smooth";
  onGenerationComplete: () => void;
}

export const GeneratorView: React.FC<GeneratorViewProps> = ({
  initialMode = "smooth",
  onGenerationComplete,
}) => {
  // Config state
  const [mode, setMode] = useState<"standard" | "smooth" | "custom">(initialMode);
  const [noiseScale, setNoiseScale] = useState<number>(initialMode === "smooth" ? 0.25 : 1.0);
  const [numSamples, setNumSamples] = useState<number>(4);
  const [truncationPsi, setTruncationPsi] = useState<number>(0.8);
  const [seed, setSeed] = useState<number | "">("");
  const [useEma, setUseEma] = useState<boolean>(false);
  const [outputFormat, setOutputFormat] = useState<string>("png");
  const [pixelatedView, setPixelatedView] = useState<boolean>(true);

  // Execution state
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [selectedImage, setSelectedImage] = useState<ImageItem | null>(null);

  // Comparison state
  const [compareResult, setCompareResult] = useState<{ standard: ImageItem[]; smooth: ImageItem[] } | null>(null);
  const [isComparing, setIsComparing] = useState<boolean>(false);

  const handleModeChange = (newMode: "standard" | "smooth" | "custom") => {
    setMode(newMode);
    if (newMode === "standard") setNoiseScale(1.0);
    if (newMode === "smooth") setNoiseScale(0.25);
  };

  const handleRandomizeSeed = () => {
    setSeed(Math.floor(Math.random() * 2147483647));
  };

  const handleGenerate = async () => {
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const activeSeed = seed === "" ? null : Number(seed);
      const res = await generateFacesApi({
        seed: activeSeed,
        num_samples: numSamples,
        truncation_psi: truncationPsi,
        mode: mode,
        noise_scale: noiseScale,
        use_ema: useEma,
        clamp_adain: false,
        output_format: outputFormat,
      });
      setResult(res);
      setSelectedImage(res.images[0] || null);
      onGenerationComplete();
    } catch (err: any) {
      setErrorMsg(err.message || "Face synthesis failed.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunSideBySideComparison = async () => {
    setIsComparing(true);
    setErrorMsg(null);
    try {
      const fixedSeed = seed === "" ? 2026 : Number(seed);
      // 1. Run Standard Mode
      const stdRes = await generateFacesApi({
        seed: fixedSeed,
        num_samples: 2,
        truncation_psi: truncationPsi,
        mode: "standard",
        noise_scale: 1.0,
        use_ema: useEma,
        clamp_adain: false,
        output_format: "png",
      });

      // 2. Run Smooth Mode
      const smoothRes = await generateFacesApi({
        seed: fixedSeed,
        num_samples: 2,
        truncation_psi: truncationPsi,
        mode: "smooth",
        noise_scale: 0.25,
        use_ema: useEma,
        clamp_adain: false,
        output_format: "png",
      });

      setCompareResult({
        standard: stdRes.images,
        smooth: smoothRes.images,
      });
      onGenerationComplete();
    } catch (err: any) {
      setErrorMsg(err.message || "Comparison execution failed.");
    } finally {
      setIsComparing(false);
    }
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "360px 1fr", gap: "1.5rem", alignItems: "start" }}>
      
      {/* Left Column: Control Panel */}
      <div className="glass-panel" style={{ padding: "1.5rem", display: "flex", flexDirection: "column", gap: "1.25rem" }}>
        
        <div>
          <h2 style={{ fontSize: "1.25rem", fontWeight: 700, marginBottom: "0.25rem" }}>Synthesis Controls</h2>
          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Configure latent modulation & noise parameters</p>
        </div>

        {/* Mode Selector Presets */}
        <div>
          <label style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--text-secondary)", display: "block", marginBottom: "0.5rem" }}>
            Inference Mode Preset
          </label>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem", marginBottom: "0.5rem" }}>
            <button
              onClick={() => handleModeChange("smooth")}
              style={{
                background: mode === "smooth" ? "rgba(99, 102, 241, 0.25)" : "rgba(255, 255, 255, 0.04)",
                border: mode === "smooth" ? "1px solid #6366f1" : "1px solid var(--border-subtle)",
                borderRadius: "0.5rem",
                padding: "0.6rem 0.5rem",
                color: mode === "smooth" ? "#ffffff" : "var(--text-secondary)",
                fontSize: "0.8rem",
                fontWeight: mode === "smooth" ? 600 : 500,
                cursor: "pointer",
                textAlign: "center"
              }}
            >
              <div style={{ color: "#38bdf8", fontWeight: 700 }}>Clean Mode</div>
              <div style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}>Noise 0.25 (Parity)</div>
            </button>

            <button
              onClick={() => handleModeChange("standard")}
              style={{
                background: mode === "standard" ? "rgba(99, 102, 241, 0.25)" : "rgba(255, 255, 255, 0.04)",
                border: mode === "standard" ? "1px solid #6366f1" : "1px solid var(--border-subtle)",
                borderRadius: "0.5rem",
                padding: "0.6rem 0.5rem",
                color: mode === "standard" ? "#ffffff" : "var(--text-secondary)",
                fontSize: "0.8rem",
                fontWeight: mode === "standard" ? 600 : 500,
                cursor: "pointer",
                textAlign: "center"
              }}
            >
              <div>Standard</div>
              <div style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}>Noise 1.00 (v1.0.0)</div>
            </button>
          </div>
        </div>

        {/* Noise Scale Slider */}
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", marginBottom: "0.375rem" }}>
            <span style={{ fontWeight: 600, color: "var(--text-secondary)" }}>Noise Scale Multiplier</span>
            <span style={{ fontWeight: 700, color: "#818cf8" }}>{noiseScale.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0.0"
            max="1.0"
            step="0.05"
            value={noiseScale}
            onChange={(e) => {
              setNoiseScale(parseFloat(e.target.value));
              setMode("custom");
            }}
          />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.65rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
            <span>0.0 (Smooth)</span>
            <span style={{ color: "#38bdf8" }}>0.25 (Parity)</span>
            <span>1.0 (Default)</span>
          </div>
        </div>

        {/* Truncation Psi */}
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", marginBottom: "0.375rem" }}>
            <span style={{ fontWeight: 600, color: "var(--text-secondary)" }}>Truncation Psi (ψ)</span>
            <span style={{ fontWeight: 700, color: "#818cf8" }}>{truncationPsi.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0.0"
            max="1.0"
            step="0.05"
            value={truncationPsi}
            onChange={(e) => setTruncationPsi(parseFloat(e.target.value))}
          />
          <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
            ψ &lt; 1.0 pulls style towards average face w̄, reducing artifacts.
          </div>
        </div>

        {/* Number of Samples */}
        <div>
          <label style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--text-secondary)", display: "block", marginBottom: "0.375rem" }}>
            Sample Batch Count
          </label>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "0.375rem" }}>
            {[1, 2, 4, 8].map((n) => (
              <button
                key={n}
                onClick={() => setNumSamples(n)}
                style={{
                  background: numSamples === n ? "var(--primary)" : "rgba(255,255,255,0.05)",
                  color: numSamples === n ? "#ffffff" : "var(--text-secondary)",
                  border: numSamples === n ? "1px solid var(--primary)" : "1px solid var(--border-subtle)",
                  borderRadius: "0.375rem",
                  padding: "0.4rem 0",
                  fontWeight: 600,
                  fontSize: "0.8rem",
                  cursor: "pointer",
                }}
              >
                {n}
              </button>
            ))}
          </div>
        </div>

        {/* Seed Input */}
        <div>
          <label style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--text-secondary)", display: "block", marginBottom: "0.375rem" }}>
            Identity Random Seed
          </label>
          <div style={{ display: "flex", gap: "0.5rem" }}>
            <input
              type="number"
              placeholder="Random (blank)"
              value={seed}
              onChange={(e) => setSeed(e.target.value === "" ? "" : parseInt(e.target.value))}
            />
            <button onClick={handleRandomizeSeed} className="btn-secondary" title="Pick Random Seed" style={{ flexShrink: 0 }}>
              🎲
            </button>
          </div>
        </div>

        {/* Weights Selection (Raw vs EMA) */}
        <div>
          <label style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--text-secondary)", display: "block", marginBottom: "0.375rem" }}>
            Generator Weights
          </label>
          <div style={{ display: "flex", gap: "0.5rem" }}>
            <button
              onClick={() => setUseEma(false)}
              style={{
                flex: 1,
                padding: "0.45rem",
                borderRadius: "0.375rem",
                fontSize: "0.75rem",
                fontWeight: 600,
                background: !useEma ? "rgba(99, 102, 241, 0.2)" : "transparent",
                border: !useEma ? "1px solid #6366f1" : "1px solid var(--border-subtle)",
                color: !useEma ? "#ffffff" : "var(--text-muted)",
                cursor: "pointer",
              }}
            >
              Raw Weights (FID 294.75)
            </button>
            <button
              onClick={() => setUseEma(true)}
              style={{
                flex: 1,
                padding: "0.45rem",
                borderRadius: "0.375rem",
                fontSize: "0.75rem",
                fontWeight: 600,
                background: useEma ? "rgba(99, 102, 241, 0.2)" : "transparent",
                border: useEma ? "1px solid #6366f1" : "1px solid var(--border-subtle)",
                color: useEma ? "#ffffff" : "var(--text-muted)",
                cursor: "pointer",
              }}
            >
              EMA Weights
            </button>
          </div>
        </div>

        {/* Action Button */}
        <button
          onClick={handleGenerate}
          disabled={isLoading || isComparing}
          className="btn-primary"
          style={{ width: "100%", padding: "0.875rem", marginTop: "0.5rem" }}
        >
          {isLoading ? (
            <span>Synthesizing ({numSamples} faces)...</span>
          ) : (
            <>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
              </svg>
              <span>Generate Faces</span>
            </>
          )}
        </button>

        {/* Side by side comparison shortcut */}
        <button
          onClick={handleRunSideBySideComparison}
          disabled={isLoading || isComparing}
          className="btn-secondary"
          style={{ width: "100%", justifyContent: "center", fontSize: "0.8rem" }}
        >
          {isComparing ? "Comparing Modes..." : "⚡ Quick Compare: Standard vs Clean"}
        </button>

        {errorMsg && (
          <div style={{ background: "rgba(239, 68, 68, 0.15)", border: "1px solid rgba(239, 68, 68, 0.3)", color: "#f87171", padding: "0.75rem", borderRadius: "0.5rem", fontSize: "0.8rem" }}>
            {errorMsg}
          </div>
        )}

      </div>

      {/* Right Column: Output Showcase & Telemetry */}
      <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
        
        {/* Comparison Showcase (if triggered) */}
        {compareResult && (
          <div className="glass-panel" style={{ padding: "1.5rem", border: "1px solid rgba(56, 189, 248, 0.3)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <span className="badge badge-emerald">Ablation Mode Comparison</span>
                <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Identical Latent Seed</span>
              </div>
              <button onClick={() => setCompareResult(null)} className="btn-secondary" style={{ fontSize: "0.75rem", padding: "0.25rem 0.5rem" }}>
                Close
              </button>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
              {/* Standard */}
              <div style={{ textAlign: "center", background: "rgba(0,0,0,0.25)", padding: "1rem", borderRadius: "0.75rem" }}>
                <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "var(--text-secondary)", marginBottom: "0.5rem" }}>
                  Standard Mode (Noise 1.0)
                </div>
                <div style={{ display: "flex", gap: "0.5rem", justifyContent: "center" }}>
                  {compareResult.standard.map((img, i) => (
                    <img key={i} src={img.base64_data} alt="std" className="pixelated" style={{ width: "128px", height: "128px", borderRadius: "0.375rem" }} />
                  ))}
                </div>
                <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: "0.5rem" }}>
                  Laplacian Var: ~4,057 (Harsh Grain)
                </div>
              </div>

              {/* Smooth */}
              <div style={{ textAlign: "center", background: "rgba(99, 102, 241, 0.08)", padding: "1rem", borderRadius: "0.75rem", border: "1px solid rgba(99, 102, 241, 0.25)" }}>
                <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "#38bdf8", marginBottom: "0.5rem" }}>
                  Clean Mode (Noise 0.25 - Parity)
                </div>
                <div style={{ display: "flex", gap: "0.5rem", justifyContent: "center" }}>
                  {compareResult.smooth.map((img, i) => (
                    <img key={i} src={img.base64_data} alt="smooth" className="pixelated" style={{ width: "128px", height: "128px", borderRadius: "0.375rem" }} />
                  ))}
                </div>
                <div style={{ fontSize: "0.7rem", color: "#34d399", marginTop: "0.5rem" }}>
                  Laplacian Var: ~922 (0.97× Real Match)
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Output Gallery Panel */}
        <div className="glass-panel" style={{ padding: "1.5rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem", flexWrap: "wrap", gap: "0.5rem" }}>
            <div>
              <h3 style={{ fontSize: "1.125rem", fontWeight: 700 }}>Synthesis Output Gallery</h3>
              <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                Native 64×64 Pixel Canvas ({result ? `${result.num_samples} faces generated` : "Awaiting execution"})
              </p>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <button
                onClick={() => setPixelatedView(!pixelatedView)}
                className="btn-secondary"
                style={{ fontSize: "0.75rem", padding: "0.3rem 0.6rem" }}
                title="Toggle nearest-neighbor pixel rendering"
              >
                {pixelatedView ? "Pixelated View: ON" : "Interpolated: ON"}
              </button>
            </div>
          </div>

          {!result ? (
            <div style={{ textAlign: "center", padding: "4rem 1rem", border: "2px dashed var(--border-subtle)", borderRadius: "0.75rem" }}>
              <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>✨</div>
              <div style={{ fontSize: "0.95rem", fontWeight: 600, color: "var(--text-secondary)" }}>
                Ready to generate human faces
              </div>
              <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
                Select parameters on the left and click "Generate Faces"
              </div>
            </div>
          ) : (
            <div>
              {/* Image Grid */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: "1rem", marginBottom: "1.5rem" }}>
                {result.images.map((img) => {
                  const isSelected = selectedImage?.index === img.index;
                  return (
                    <div
                      key={img.index}
                      className="glass-panel-interactive"
                      style={{
                        padding: "0.5rem",
                        textAlign: "center",
                        cursor: "pointer",
                        border: isSelected ? "2px solid #6366f1" : "1px solid var(--border-subtle)",
                      }}
                      onClick={() => setSelectedImage(img)}
                    >
                      <img
                        src={img.base64_data}
                        alt={`Face ${img.index}`}
                        className={pixelatedView ? "pixelated" : ""}
                        style={{ width: "100%", aspectRatio: "1/1", borderRadius: "0.375rem", objectFit: "cover", display: "block" }}
                      />
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "0.5rem", padding: "0 0.25rem" }}>
                        <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>Face #{img.index + 1}</span>
                        <a
                          href={img.download_url}
                          download={`face_${result.seed || "rand"}_${img.index}.png`}
                          className="btn-secondary"
                          style={{ padding: "0.15rem 0.4rem", fontSize: "0.65rem" }}
                          onClick={(e) => e.stopPropagation()}
                        >
                          ↓
                        </a>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Focused Preview & Telemetry Strip */}
              {selectedImage && (
                <div style={{ background: "rgba(0,0,0,0.3)", borderRadius: "0.75rem", padding: "1rem", display: "flex", alignItems: "center", gap: "1.5rem", flexWrap: "wrap" }}>
                  <img
                    src={selectedImage.base64_data}
                    alt="Selected"
                    className={pixelatedView ? "pixelated" : ""}
                    style={{ width: "96px", height: "96px", borderRadius: "0.5rem", border: "1px solid var(--border-focus)" }}
                  />
                  <div style={{ flex: 1, display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: "0.75rem" }}>
                    <div>
                      <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>Total Latency</div>
                      <div style={{ fontSize: "0.9rem", fontWeight: 700 }}>{result.latency_ms.toFixed(1)} ms</div>
                    </div>
                    <div>
                      <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>Throughput</div>
                      <div style={{ fontSize: "0.9rem", fontWeight: 700 }}>{result.latency_per_face_ms.toFixed(1)} ms/face</div>
                    </div>
                    <div>
                      <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>Mode / Noise</div>
                      <div style={{ fontSize: "0.9rem", fontWeight: 700 }}>{result.mode} ({result.noise_scale})</div>
                    </div>
                    <div>
                      <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>Seed</div>
                      <div style={{ fontSize: "0.9rem", fontWeight: 700 }}>{result.seed !== null ? result.seed : "Stochastic"}</div>
                    </div>
                  </div>
                  <a
                    href={selectedImage.download_url}
                    download={`styleforge_face_${selectedImage.index}.png`}
                    className="btn-primary"
                    style={{ padding: "0.5rem 1rem", fontSize: "0.85rem" }}
                  >
                    Download PNG
                  </a>
                </div>
              )}

            </div>
          )}
        </div>

      </div>

    </div>
  );
};
