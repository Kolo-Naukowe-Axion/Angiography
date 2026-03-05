import type {
  GroundTruthBoxInput,
  HealthResponse,
  InferFrameResponse,
  LabelsResponse,
  ModelCard,
  PatientSummary,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {}),
      },
    });
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Unknown network error";
    throw new Error(`Cannot reach backend at ${API_BASE}${path}. ${detail}`);
  }

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }

  return (await response.json()) as T;
}

export function resolveApiUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) {
    return path;
  }
  if (path.startsWith("/")) {
    return `${API_BASE}${path}`;
  }
  return `${API_BASE}/${path}`;
}

export function getFrameUrl(patientId: string, frameIndex: number): string {
  return `${API_BASE}/api/patients/${patientId}/frames/${frameIndex}`;
}

export function getMaskUrl(patientId: string, frameIndex: number, source: "prediction" | "ground_truth"): string {
  return `${API_BASE}/api/patients/${patientId}/frames/${frameIndex}/masks/${source}`;
}

export function getHealth(): Promise<HealthResponse> {
  return fetchJson<HealthResponse>("/api/health");
}

export function getModels(): Promise<ModelCard[]> {
  return fetchJson<ModelCard[]>("/api/models");
}

export function selectModel(modelId: ModelCard["id"]): Promise<ModelCard[]> {
  return fetchJson<ModelCard[]>("/api/models/select", {
    method: "POST",
    body: JSON.stringify({ modelId }),
  });
}

export function getPatients(): Promise<PatientSummary[]> {
  return fetchJson<PatientSummary[]>("/api/patients");
}

export function inferFrame(patientId: string, frameIndex: number): Promise<InferFrameResponse> {
  return fetchJson<InferFrameResponse>("/api/infer/frame", {
    method: "POST",
    body: JSON.stringify({ patientId, frameIndex }),
  });
}

export function getLabels(patientId: string, frameIndex: number): Promise<LabelsResponse> {
  return fetchJson<LabelsResponse>(`/api/labels/${patientId}/${frameIndex}`);
}

export function saveLabels(
  patientId: string,
  frameIndex: number,
  boxes: GroundTruthBoxInput[],
): Promise<LabelsResponse> {
  return fetchJson<LabelsResponse>(`/api/labels/${patientId}/${frameIndex}`, {
    method: "PUT",
    body: JSON.stringify({ boxes }),
  });
}

export function prefetchFrames(patientId: string, startFrame: number, endFrame: number): Promise<void> {
  return fetchJson("/api/infer/prefetch", {
    method: "POST",
    body: JSON.stringify({ patientId, startFrame, endFrame }),
  }).then(() => undefined);
}
