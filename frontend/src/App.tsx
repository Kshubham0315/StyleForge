import React, { useEffect, useState } from "react";
import { fetchCurrentModel, fetchGenerationsHistory, fetchHealth } from "./api";
import { DashboardView } from "./components/DashboardView";
import { GeneratorView } from "./components/GeneratorView";
import { Header } from "./components/Header";
import { HistoryView } from "./components/HistoryView";
import { ModelInfoView } from "./components/ModelInfoView";
import { SystemStatusView } from "./components/SystemStatusView";
import { GenerationListItem, HealthInfo, ModelInfo } from "./types";

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>("dashboard");
  const [initialGeneratorPreset, setInitialGeneratorPreset] = useState<"standard" | "smooth">("smooth");
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [recentGenerations, setRecentGenerations] = useState<GenerationListItem[]>([]);
  const [showSettingsModal, setShowSettingsModal] = useState<boolean>(false);
  const [customApiKey, setCustomApiKey] = useState<string>(
    localStorage.getItem("styleforge_api_key") || "styleforge-dev-key-2026"
  );

  const loadData = async () => {
    try {
      const h = await fetchHealth();
      setHealth(h);
    } catch (e) {
      console.warn("Health check error:", e);
    }

    try {
      const m = await fetchCurrentModel();
      setModel(m);
    } catch (e) {
      console.warn("Model check error:", e);
    }

    try {
      const hist = await fetchGenerationsHistory(8, 0);
      setRecentGenerations(hist.items);
    } catch (e) {
      console.warn("History check error:", e);
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 15000); // 15s telemetry polling
    return () => clearInterval(interval);
  }, []);

  const handleGoToGenerator = (preset: "standard" | "smooth" = "smooth") => {
    setInitialGeneratorPreset(preset);
    setActiveTab("generator");
  };

  const handleSaveSettings = () => {
    localStorage.setItem("styleforge_api_key", customApiKey);
    setShowSettingsModal(false);
    loadData();
  };

  return (
    <div style={{ position: "relative", minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      {/* Ambient background mesh */}
      <div className="bg-mesh" />

      {/* Header bar */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        health={health}
        model={model}
        onOpenSettings={() => setShowSettingsModal(true)}
      />

      {/* Main Content Area */}
      <main style={{ flex: 1, padding: "0 1.5rem 3rem 1.5rem", maxWidth: "1400px", margin: "0 auto", width: "100%", position: "relative", zIndex: 1 }}>
        {activeTab === "dashboard" && (
          <DashboardView
            onGoToGenerator={handleGoToGenerator}
            recentGenerations={recentGenerations}
            model={model}
            health={health}
          />
        )}

        {activeTab === "generator" && (
          <GeneratorView
            initialMode={initialGeneratorPreset}
            onGenerationComplete={loadData}
          />
        )}

        {activeTab === "history" && <HistoryView />}

        {activeTab === "model-info" && <ModelInfoView model={model} />}

        {activeTab === "system-status" && (
          <SystemStatusView
            health={health}
            model={model}
            onRefreshHealth={loadData}
          />
        )}
      </main>

      {/* Settings Modal */}
      {showSettingsModal && (
        <div className="modal-overlay" onClick={() => setShowSettingsModal(false)}>
          <div className="glass-panel" style={{ padding: "2rem", maxWidth: "440px", width: "100%", background: "#0d1322" }} onClick={(e) => e.stopPropagation()}>
            <h3 style={{ fontSize: "1.125rem", fontWeight: 700, marginBottom: "0.5rem" }}>Platform Settings</h3>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "1.25rem" }}>
              Configure your client API key for protected generation requests.
            </p>

            <div style={{ marginBottom: "1.5rem" }}>
              <label style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--text-secondary)", display: "block", marginBottom: "0.5rem" }}>
                Client API Key
              </label>
              <input
                type="text"
                value={customApiKey}
                onChange={(e) => setCustomApiKey(e.target.value)}
                placeholder="styleforge-dev-key-2026"
                style={{ fontFamily: "JetBrains Mono, monospace" }}
              />
              <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
                Default: <code>styleforge-dev-key-2026</code>
              </div>
            </div>

            <div style={{ display: "flex", gap: "0.75rem" }}>
              <button onClick={handleSaveSettings} className="btn-primary" style={{ flex: 1 }}>
                Save & Connect
              </button>
              <button onClick={() => setShowSettingsModal(false)} className="btn-secondary" style={{ flex: 1, justifyContent: "center" }}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer style={{ borderTop: "1px solid var(--border-subtle)", padding: "1.5rem", textAlign: "center", fontSize: "0.75rem", color: "var(--text-muted)", zIndex: 1 }}>
        StyleForge Platform v2.0 • StyleGAN-Inspired Architecture • Native 64×64 CelebA Checkpoint • Open Source AI Research
      </footer>
    </div>
  );
};
