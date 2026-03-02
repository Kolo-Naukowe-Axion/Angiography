export type ModelCard = {
  id: "yolo26s" | "yolo26m" | "yolov8" | "sam_vmnet";
  name: string;
  active: boolean;
  status: "ready" | "coming_soon";
  notes: string;
};

export type PatientSummary = {
  id: string;
  displayName: string;
  frameCount: number;
  hasLabels: boolean;
  defaultFps: number;
};

export type Box = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  classId: number;
  className: "stenosis";
};

export type InferFrameResponse = {
  patientId: string;
  frameIndex: number;
  boxes: Box[];
  cached: boolean;
  inferenceMs: number;
};

export type HealthResponse = {
  status: "ok";
  modelLoaded: boolean;
  modelPath: string;
  device: string;
  cacheSize: number;
  cacheEntries: number;
  prefetchQueueSize: number;
  prefetchQueued: number;
};

export type LabelsResponse = {
  patientId: string;
  frameIndex: number;
  hasLabels: boolean;
  boxes: Box[];
};
