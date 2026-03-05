from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .data import PatientStore
from .schemas import Box, InferFrameResponse, MaskPayload


def select_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def ensure_ultralytics_loss_compatibility() -> None:
    """Backfill classes needed only for deserializing older/newer checkpoints."""
    import torch
    from ultralytics.utils import loss as ultralytics_loss

    if hasattr(ultralytics_loss, "E2ELoss"):
        return

    class E2ELoss(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.args = args
            self.kwargs = kwargs

        def forward(self, *args, **kwargs):
            raise RuntimeError("E2ELoss compatibility shim should not be used for runtime loss computation.")

    ultralytics_loss.E2ELoss = E2ELoss


@dataclass
class CachedInference:
    response: InferFrameResponse


class LRUFrameCache:
    def __init__(self, max_entries: int):
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[str, int], CachedInference] = OrderedDict()

    def get(self, key: tuple[str, int]) -> CachedInference | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry

    def set(self, key: tuple[str, int], value: CachedInference) -> None:
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def stats(self) -> tuple[int, int]:
        return self.max_entries, len(self._entries)


class YOLOInferenceService:
    def __init__(
        self,
        model: Any,
        model_path: Path,
        device: str,
        cache_size: int,
        min_infer_confidence: float,
        prefetch_queue_size: int,
    ):
        self.model = model
        self.model_path = model_path
        self.device = device
        self.min_infer_confidence = min_infer_confidence
        self.cache = LRUFrameCache(cache_size)
        self._infer_lock = asyncio.Lock()
        self._prefetch_queue: asyncio.Queue[tuple[str, int, Path] | None] = asyncio.Queue(maxsize=prefetch_queue_size)
        self._prefetch_pending: set[tuple[str, int]] = set()
        self._worker_task: asyncio.Task[None] | None = None

    @classmethod
    async def from_weights(
        cls,
        model_path: Path,
        cache_size: int,
        min_infer_confidence: float,
        prefetch_queue_size: int,
    ) -> "YOLOInferenceService":
        def _load_model() -> Any:
            from ultralytics import YOLO

            ensure_ultralytics_loss_compatibility()
            return YOLO(str(model_path))

        model = await asyncio.to_thread(_load_model)
        device = select_device()
        return cls(
            model=model,
            model_path=model_path,
            device=device,
            cache_size=cache_size,
            min_infer_confidence=min_infer_confidence,
            prefetch_queue_size=prefetch_queue_size,
        )

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._prefetch_worker(), name="prefetch-worker")

    async def stop(self) -> None:
        if self._worker_task is not None:
            await self._prefetch_queue.put(None)
            await self._worker_task
            self._worker_task = None

    def _extract_boxes(self, yolo_result: Any) -> list[Box]:
        boxes: list[Box] = []
        raw_boxes = yolo_result.boxes
        for idx in range(len(raw_boxes)):
            xyxy = raw_boxes.xyxy[idx].tolist()
            conf = float(raw_boxes.conf[idx].item())
            class_id = int(raw_boxes.cls[idx].item())
            boxes.append(
                Box(
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                    confidence=conf,
                    classId=class_id,
                    className="stenosis",
                )
            )
        return boxes

    async def infer_frame(self, patient_id: str, frame_index: int, frame_path: Path) -> InferFrameResponse:
        cache_key = (patient_id, frame_index)
        cached = self.cache.get(cache_key)
        if cached:
            return cached.response.model_copy(update={"cached": True})

        start = time.perf_counter()
        async with self._infer_lock:
            result = await asyncio.to_thread(
                self.model.predict,
                str(frame_path),
                conf=self.min_infer_confidence,
                device=self.device,
                verbose=False,
            )
        inference_ms = (time.perf_counter() - start) * 1000.0

        boxes = self._extract_boxes(result[0])
        response = InferFrameResponse(
            patientId=patient_id,
            frameIndex=frame_index,
            outputType="bbox",
            boxes=boxes,
            mask=None,
            stenosisDetected=bool(boxes),
            cached=False,
            inferenceMs=round(inference_ms, 3),
        )
        self.cache.set(cache_key, CachedInference(response=response))
        return response

    async def queue_prefetch(self, patient_id: str, frame_index: int, frame_path: Path) -> bool:
        key = (patient_id, frame_index)
        if self.cache.get(key):
            return False
        if key in self._prefetch_pending:
            return False
        if self._prefetch_queue.full():
            return False
        self._prefetch_pending.add(key)
        await self._prefetch_queue.put((patient_id, frame_index, frame_path))
        return True

    async def _prefetch_worker(self) -> None:
        while True:
            item = await self._prefetch_queue.get()
            if item is None:
                self._prefetch_queue.task_done()
                return

            patient_id, frame_index, frame_path = item
            try:
                await self.infer_frame(patient_id, frame_index, frame_path)
            except Exception:
                pass
            finally:
                self._prefetch_pending.discard((patient_id, frame_index))
                self._prefetch_queue.task_done()

    def health(self) -> dict[str, Any]:
        cache_size, cache_entries = self.cache.stats()
        return {
            "modelLoaded": True,
            "modelPath": str(self.model_path),
            "device": self.device,
            "cacheSize": cache_size,
            "cacheEntries": cache_entries,
            "prefetchQueueSize": self._prefetch_queue.maxsize,
            "prefetchQueued": self._prefetch_queue.qsize(),
        }


class PrecomputedMaskInferenceService:
    def __init__(
        self,
        model_id: str,
        model_path: Path,
        patient_store: PatientStore,
        cache_size: int,
        prefetch_queue_size: int,
    ):
        self.model_id = model_id
        self.model_path = model_path
        self.patient_store = patient_store
        self.cache = LRUFrameCache(cache_size)
        self._prefetch_queue: asyncio.Queue[tuple[str, int, Path] | None] = asyncio.Queue(maxsize=prefetch_queue_size)
        self._prefetch_pending: set[tuple[str, int]] = set()
        self._worker_task: asyncio.Task[None] | None = None

    @classmethod
    async def create(
        cls,
        model_id: str,
        model_path: Path,
        patient_store: PatientStore,
        cache_size: int,
        prefetch_queue_size: int,
    ) -> "PrecomputedMaskInferenceService":
        return cls(
            model_id=model_id,
            model_path=model_path,
            patient_store=patient_store,
            cache_size=cache_size,
            prefetch_queue_size=prefetch_queue_size,
        )

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._prefetch_worker(), name="prefetch-worker-mask")

    async def stop(self) -> None:
        if self._worker_task is not None:
            await self._prefetch_queue.put(None)
            await self._worker_task
            self._worker_task = None

    def _build_mask_payload(self, patient_id: str, frame_index: int, mask_path: Path) -> MaskPayload:
        with Image.open(mask_path) as mask_image:
            mask_gray = np.array(mask_image.convert("L"), dtype=np.uint8)
            width, height = mask_image.size

        positive_pixel_ratio = float(np.count_nonzero(mask_gray > 127) / max(1, mask_gray.size))
        return MaskPayload(
            url=f"/api/patients/{patient_id}/frames/{frame_index}/masks/prediction",
            width=width,
            height=height,
            positivePixelRatio=round(positive_pixel_ratio, 6),
        )

    async def infer_frame(self, patient_id: str, frame_index: int, frame_path: Path) -> InferFrameResponse:
        cache_key = (patient_id, frame_index)
        cached = self.cache.get(cache_key)
        if cached:
            return cached.response.model_copy(update={"cached": True})

        start = time.perf_counter()
        mask_path = self.patient_store.get_mask_path(
            patient_id,
            frame_index,
            source="prediction",
            model_id=self.model_id,
        )
        if mask_path is None or not mask_path.exists():
            raise FileNotFoundError(f"Prediction mask not found for {patient_id} frame {frame_index} ({self.model_id})")

        payload = await asyncio.to_thread(self._build_mask_payload, patient_id, frame_index, mask_path)
        inference_ms = (time.perf_counter() - start) * 1000.0
        response = InferFrameResponse(
            patientId=patient_id,
            frameIndex=frame_index,
            outputType="mask",
            boxes=[],
            mask=payload,
            stenosisDetected=payload.positivePixelRatio > 0,
            cached=False,
            inferenceMs=round(inference_ms, 3),
        )
        self.cache.set(cache_key, CachedInference(response=response))
        return response

    async def queue_prefetch(self, patient_id: str, frame_index: int, frame_path: Path) -> bool:
        key = (patient_id, frame_index)
        if self.cache.get(key):
            return False
        if key in self._prefetch_pending:
            return False
        if self._prefetch_queue.full():
            return False
        self._prefetch_pending.add(key)
        await self._prefetch_queue.put((patient_id, frame_index, frame_path))
        return True

    async def _prefetch_worker(self) -> None:
        while True:
            item = await self._prefetch_queue.get()
            if item is None:
                self._prefetch_queue.task_done()
                return

            patient_id, frame_index, frame_path = item
            try:
                await self.infer_frame(patient_id, frame_index, frame_path)
            except Exception:
                pass
            finally:
                self._prefetch_pending.discard((patient_id, frame_index))
                self._prefetch_queue.task_done()

    def health(self) -> dict[str, Any]:
        cache_size, cache_entries = self.cache.stats()
        return {
            "modelLoaded": True,
            "modelPath": str(self.model_path),
            "device": "cpu",
            "cacheSize": cache_size,
            "cacheEntries": cache_entries,
            "prefetchQueueSize": self._prefetch_queue.maxsize,
            "prefetchQueued": self._prefetch_queue.qsize(),
        }


class MockInferenceService(YOLOInferenceService):
    @classmethod
    async def create(
        cls,
        model_path: Path,
        cache_size: int,
        min_infer_confidence: float,
        prefetch_queue_size: int,
    ) -> "MockInferenceService":
        return cls(
            model=object(),
            model_path=model_path,
            device="cpu",
            cache_size=cache_size,
            min_infer_confidence=min_infer_confidence,
            prefetch_queue_size=prefetch_queue_size,
        )

    async def infer_frame(self, patient_id: str, frame_index: int, frame_path: Path) -> InferFrameResponse:
        cache_key = (patient_id, frame_index)
        cached = self.cache.get(cache_key)
        if cached:
            return cached.response.model_copy(update={"cached": True})

        box = Box(
            x1=64,
            y1=64,
            x2=196,
            y2=196,
            confidence=0.75 if frame_index % 2 == 0 else 0.25,
            classId=0,
            className="stenosis",
        )
        response = InferFrameResponse(
            patientId=patient_id,
            frameIndex=frame_index,
            outputType="bbox",
            boxes=[box],
            mask=None,
            stenosisDetected=True,
            cached=False,
            inferenceMs=3.0,
        )
        self.cache.set(cache_key, CachedInference(response=response))
        return response
