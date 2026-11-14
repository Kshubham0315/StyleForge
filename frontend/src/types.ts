export type InferenceMode = "standard" | "smooth" | "custom";
export type OutputFormat = "png" | "jpeg" | "webp";

export interface ImageItem {
  index: number;
  base64_data: string;
  download_url: string;
  artifact_path: string;
}

export interface GenerateResponse {
  generation_id: string;
  status: string;
  num_samples: number;
  seed: number | null;
  mode: InferenceMode;
  noise_scale: number;
  use_ema: boolean;
  latency_ms: number;
  latency_per_face_ms: number;
  resolution: string;
  images: ImageItem[];
  created_at: string;
}

export interface GenerationListItem {
  id: string;
  created_at: string;
  num_samples: number;
  seed: number | null;
  mode: string;
  noise_scale: number;
  use_ema: boolean;
  latency_ms: number;
  download_url: string | null;
}

export interface GenerationListResponse {
  total: number;
  limit: number;
  offset: number;
  items: GenerationListItem[];
}

export interface ModelInfo {
  name: string;
  version: string;
  checkpoint_sha256: string;
  resolution: number;
  latent_dim: number;
  style_dim: number;
  device: string;
  is_official: boolean;
  modes_supported: string[];
  vram_telemetry: {
    cuda_available: boolean;
    device_name: string;
    allocated_mb: number;
    reserved_mb: number;
    total_mb: number;
    free_mb: number;
  };
}

export interface HealthInfo {
  status: string;
  uptime_seconds: number;
  version: string;
  device: string;
  cuda_available: boolean;
  database_connected: boolean;
  models_loaded: boolean;
}
