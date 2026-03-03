from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ModelCard(BaseModel):
    id: Literal["yolo26s", "yolo26n"]
    name: str
    active: bool
    status: Literal["ready", "unavailable"]
    notes: str


class PatientSummary(BaseModel):
    id: str
    displayName: str
    frameCount: int
    hasLabels: bool
    defaultFps: int


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    classId: int
    className: Literal["stenosis"] = "stenosis"


class InferFrameRequest(BaseModel):
    patientId: str
    frameIndex: int = Field(ge=0)


class InferFrameResponse(BaseModel):
    patientId: str
    frameIndex: int
    boxes: list[Box]
    cached: bool
    inferenceMs: float


class PrefetchRequest(BaseModel):
    patientId: str
    startFrame: int = Field(ge=0)
    endFrame: int = Field(ge=0)


class PrefetchResponse(BaseModel):
    patientId: str
    queued: int
    startFrame: int
    endFrame: int


class ModelSelectRequest(BaseModel):
    modelId: Literal["yolo26s", "yolo26n"]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    modelLoaded: bool
    modelPath: str
    device: str
    cacheSize: int
    cacheEntries: int
    prefetchQueueSize: int
    prefetchQueued: int
