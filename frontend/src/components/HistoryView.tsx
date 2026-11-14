import React, { useEffect, useState } from "react";
import { fetchGenerationsHistory } from "../api";
import { GenerationListItem, GenerationListResponse } from "../types";

export const HistoryView: React.FC = () => {
  const [historyData, setHistoryData] = useState<GenerationListResponse | null>(null);
  const [offset, setOffset] = useState<number>(0);
  const [filterMode, setFilterMode] = useState<string>("all");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [selectedItem, setSelectedItem] = useState<GenerationListItem | null>(null);

  const limit = 12;

  const loadHistory = async () => {
    setIsLoading(true);
    try {
      const res = await fetchGenerationsHistory(limit, offset, filterMode);
      setHistoryData(res);
    } catch (e) {
      console.error("Failed to load history:", e);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, [offset, filterMode]);

  return (
    <div className="glass-panel" style={{ padding: "1.75rem" }}>
      
      {/* Header & Filter Controls */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.5rem", flexWrap: "wrap", gap: "1rem" }}>
        <div>
          <h2 style={{ fontSize: "1.25rem", fontWeight: 700, marginBottom: "0.25rem" }}>Generation Audit History</h2>
          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
            Persisted multi-tenant generation log with artifact references
          </p>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <select
            value={filterMode}
            onChange={(e) => {
              setFilterMode(e.target.value);
              setOffset(0);
            }}
            style={{ width: "160px" }}
          >
            <option value="all">All Modes</option>
            <option value="standard">Standard (Noise 1.0)</option>
            <option value="smooth">Clean Mode (Noise 0.25)</option>
            <option value="custom">Custom</option>
          </select>

          <button onClick={() => loadHistory()} className="btn-secondary" style={{ padding: "0.6rem 0.8rem" }}>
            🔄 Refresh
          </button>
        </div>
      </div>

      {/* Grid of Past Generations */}
      {isLoading ? (
        <div style={{ textAlign: "center", padding: "4rem 1rem", color: "var(--text-muted)" }}>
          Loading generation history...
        </div>
      ) : !historyData || historyData.items.length === 0 ? (
        <div style={{ textAlign: "center", padding: "4rem 1rem", color: "var(--text-muted)" }}>
          No records match the current filter.
        </div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: "1.25rem", marginBottom: "1.5rem" }}>
          {historyData.items.map((item) => (
            <div
              key={item.id}
              className="glass-panel-interactive"
              style={{ padding: "0.75rem", cursor: "pointer", display: "flex", flexDirection: "column" }}
              onClick={() => setSelectedItem(item)}
            >
              {item.download_url ? (
                <img
                  src={item.download_url}
                  alt={item.id}
                  className="pixelated"
                  style={{ width: "100%", aspectRatio: "1/1", borderRadius: "0.375rem", objectFit: "cover", marginBottom: "0.5rem" }}
                />
              ) : (
                <div style={{ width: "100%", aspectRatio: "1/1", background: "rgba(255,255,255,0.05)", borderRadius: "0.375rem", marginBottom: "0.5rem" }} />
              )}
              
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "0.75rem", fontWeight: 600 }}>
                <span>Seed: {item.seed !== null ? item.seed : "Rand"}</span>
                <span className={item.mode === "smooth" ? "badge badge-indigo" : "badge"} style={{ fontSize: "0.65rem", padding: "0.1rem 0.3rem" }}>
                  {item.mode}
                </span>
              </div>
              <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
                {new Date(item.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} • {item.latency_ms.toFixed(0)}ms
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Pagination Footer */}
      {historyData && historyData.total > limit && (
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderTop: "1px solid var(--border-subtle)", paddingTop: "1rem" }}>
          <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
            Showing {offset + 1}–{Math.min(offset + limit, historyData.total)} of {historyData.total} items
          </div>
          <div style={{ display: "flex", gap: "0.5rem" }}>
            <button
              onClick={() => setOffset(Math.max(0, offset - limit))}
              disabled={offset === 0}
              className="btn-secondary"
              style={{ fontSize: "0.8rem", padding: "0.4rem 0.8rem" }}
            >
              &larr; Previous
            </button>
            <button
              onClick={() => setOffset(offset + limit)}
              disabled={offset + limit >= historyData.total}
              className="btn-secondary"
              style={{ fontSize: "0.8rem", padding: "0.4rem 0.8rem" }}
            >
              Next &rarr;
            </button>
          </div>
        </div>
      )}

      {/* Modal Detail View */}
      {selectedItem && (
        <div className="modal-overlay" onClick={() => setSelectedItem(null)}>
          <div className="glass-panel" style={{ padding: "2rem", maxWidth: "420px", width: "100%", background: "#0e1424" }} onClick={(e) => e.stopPropagation()}>
            <h3 style={{ fontSize: "1.125rem", fontWeight: 700, marginBottom: "1rem" }}>Generation Record</h3>
            
            {selectedItem.download_url && (
              <div style={{ textAlign: "center", marginBottom: "1rem" }}>
                <img
                  src={selectedItem.download_url}
                  alt="detail"
                  className="pixelated"
                  style={{ width: "160px", height: "160px", borderRadius: "0.5rem", border: "1px solid var(--border-focus)" }}
                />
              </div>
            )}

            <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", fontSize: "0.85rem", marginBottom: "1.5rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "var(--text-muted)" }}>Job UUID:</span>
                <span style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "0.75rem" }}>{selectedItem.id.slice(0, 16)}...</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "var(--text-muted)" }}>Identity Seed:</span>
                <span style={{ fontWeight: 600 }}>{selectedItem.seed !== null ? selectedItem.seed : "Random"}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "var(--text-muted)" }}>Mode:</span>
                <span>{selectedItem.mode} (noise: {selectedItem.noise_scale})</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "var(--text-muted)" }}>Inference Latency:</span>
                <span>{selectedItem.latency_ms.toFixed(1)} ms</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "var(--text-muted)" }}>Timestamp:</span>
                <span>{new Date(selectedItem.created_at).toLocaleString()}</span>
              </div>
            </div>

            <div style={{ display: "flex", gap: "0.75rem" }}>
              {selectedItem.download_url && (
                <a
                  href={selectedItem.download_url}
                  download={`styleforge_${selectedItem.id}.png`}
                  className="btn-primary"
                  style={{ flex: 1, textDecoration: "none" }}
                >
                  Download PNG
                </a>
              )}
              <button onClick={() => setSelectedItem(null)} className="btn-secondary" style={{ flex: 1, justifyContent: "center" }}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
};
