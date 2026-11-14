import {
  GenerateResponse,
  GenerationListResponse,
  HealthInfo,
  ModelInfo,
} from "./types";

const API_BASE = "";
const DEFAULT_API_KEY = "styleforge-dev-key-2026";

export const getHeaders = (): HeadersInit => {
  const customKey = localStorage.getItem("styleforge_api_key") || DEFAULT_API_KEY;
  return {
    "Content-Type": "application/json",
    "X-API-Key": customKey,
  };
};

export async function fetchHealth(): Promise<HealthInfo> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) {
    throw new Error(`Health check failed: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchCurrentModel(): Promise<ModelInfo> {
  const res = await fetch(`${API_BASE}/api/v1/models/current`, {
    headers: getHeaders(),
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch model info: ${res.statusText}`);
  }
  return res.json();
}

export async function generateFacesApi(payload: {
  seed?: number | null;
  num_samples: number;
  truncation_psi: number;
  mode: string;
  noise_scale: number;
  use_ema: boolean;
  clamp_adain: boolean;
  enhance?: boolean;
  output_format: string;
}): Promise<GenerateResponse> {
  const res = await fetch(`${API_BASE}/api/v1/generate`, {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    let errorMsg = `Generation error (${res.status})`;
    try {
      const errJson = await res.json();
      errorMsg = errJson.error?.message || errJson.detail?.message || errorMsg;
    } catch {
      // fallback
    }
    throw new Error(errorMsg);
  }

  return res.json();
}

export async function fetchGenerationsHistory(
  limit: number = 20,
  offset: number = 0,
  mode?: string
): Promise<GenerationListResponse> {
  const params = new URLSearchParams({
    limit: limit.toString(),
    offset: offset.toString(),
  });
  if (mode && mode !== "all") {
    params.append("mode", mode);
  }

  const res = await fetch(`${API_BASE}/api/v1/generations?${params.toString()}`, {
    headers: getHeaders(),
  });
  if (!res.ok) {
    throw new Error(`Failed to load history: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchPrometheusMetrics(): Promise<string> {
  const res = await fetch(`${API_BASE}/api/v1/metrics`);
  if (!res.ok) {
    throw new Error("Failed to fetch Prometheus metrics");
  }
  return res.text();
}
