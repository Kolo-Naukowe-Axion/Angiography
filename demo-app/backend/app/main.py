from __future__ import annotations

import asyncio
import math
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

from .classification import has_stenosis
from .config import Settings
from .data import ManifestValidationError, PatientRecord, PatientStore
from .inference import MockInferenceService, PrecomputedMaskInferenceService, YOLOInferenceService
from .label_utils import parse_yolo_labels_to_boxes, write_yolo_labels_from_boxes
from .models_registry import (
    MODEL_PATHS,
    SAM_VMNET_MODEL_ID,
    get_model_cards,
    get_model_dataset_id,
    get_model_id_for_path,
    get_model_inference_mode,
    get_model_output_type,
    get_model_path,
)
from .schemas import (
    GroundTruthBoxInput,
    HealthResponse,
    InferFrameRequest,
    InferFrameResponse,
    LabelsResponse,
    MaskPayload,
    ModelSelectRequest,
    PrefetchRequest,
    PrefetchResponse,
    SaveLabelsRequest,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or Settings.from_env()

    def _validate_ground_truth_boxes(boxes: list[GroundTruthBoxInput]) -> None:
        for index, box in enumerate(boxes):
            coordinates = (box.x1, box.y1, box.x2, box.y2)
            if not all(math.isfinite(value) for value in coordinates):
                raise HTTPException(status_code=400, detail=f"Invalid box at index {index}: coordinates must be finite.")
            if box.x2 <= box.x1 or box.y2 <= box.y1:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid box at index {index}: expected x2>x1 and y2>y1.",
                )

    def _active_dataset_id() -> str:
        return get_model_dataset_id(app.state.active_model_id)

    def _get_patient_for_active_dataset(patient_id: str) -> PatientRecord:
        dataset_id = _active_dataset_id()
        try:
            return app.state.patient_store.get_patient(patient_id, dataset_id=dataset_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=f"Unknown patient for active dataset '{dataset_id}': {patient_id}") from error

    def _build_mask_payload(patient_id: str, frame_index: int, source: str, mask_path: Path) -> MaskPayload:
        with Image.open(mask_path) as image:
            mask_gray = np.array(image.convert("L"), dtype=np.uint8)
            width, height = image.size

        positive_ratio = float(np.count_nonzero(mask_gray > 127) / max(1, mask_gray.size))
        return MaskPayload(
            url=f"/api/patients/{patient_id}/frames/{frame_index}/masks/{source}",
            width=width,
            height=height,
            positivePixelRatio=round(positive_ratio, 6),
        )

    async def _create_model_service(model_id: str, model_path: Path, patient_store: PatientStore):
        output_type = get_model_output_type(model_id)
        inference_mode = get_model_inference_mode(model_id)

        if output_type == "mask" and inference_mode == "precomputed":
            return await PrecomputedMaskInferenceService.create(
                model_id=model_id,
                model_path=model_path,
                patient_store=patient_store,
                cache_size=app_settings.cache_size,
                prefetch_queue_size=app_settings.prefetch_queue_size,
            )

        if app_settings.use_mock_model:
            return await MockInferenceService.create(
                model_path=model_path,
                cache_size=app_settings.cache_size,
                min_infer_confidence=app_settings.min_infer_confidence,
                prefetch_queue_size=app_settings.prefetch_queue_size,
            )
        return await YOLOInferenceService.from_weights(
            model_path=model_path,
            cache_size=app_settings.cache_size,
            min_infer_confidence=app_settings.min_infer_confidence,
            prefetch_queue_size=app_settings.prefetch_queue_size,
        )

    async def _probe_model(model_id: str, model_path: Path, patient_store: PatientStore) -> tuple[bool, str | None]:
        output_type = get_model_output_type(model_id)
        inference_mode = get_model_inference_mode(model_id)

        if output_type == "mask" and inference_mode == "precomputed":
            is_ready, reason = patient_store.is_model_prediction_ready(
                model_id=model_id,
                dataset_id=get_model_dataset_id(model_id),
            )
            if not is_ready:
                return False, reason
            return True, None

        if not app_settings.use_mock_model and not model_path.exists():
            return False, f"weights not found at {model_path}"

        probe_service = None
        try:
            probe_service = await _create_model_service(model_id, model_path, patient_store)
            await probe_service.start()
            return True, None
        except Exception as error:
            return False, str(error)
        finally:
            if probe_service is not None:
                try:
                    await probe_service.stop()
                except Exception:
                    pass

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            patient_store = PatientStore(app_settings.data_dir)
        except ManifestValidationError as error:
            raise RuntimeError(f"Manifest validation failed: {error}") from error

        active_model_id = get_model_id_for_path(app_settings.model_path)
        model_path = get_model_path(active_model_id) or app_settings.model_path
        model_service = await _create_model_service(active_model_id, model_path, patient_store)

        await model_service.start()
        app.state.settings = replace(app_settings, model_path=model_path)
        app.state.model_switch_lock = asyncio.Lock()
        app.state.patient_store = patient_store
        app.state.model_service = model_service
        app.state.active_model_id = active_model_id

        model_availability: dict[str, bool] = {}
        model_availability_reasons: dict[str, str] = {}
        for model_id, candidate_model_path in MODEL_PATHS.items():
            if model_id == active_model_id:
                model_availability[model_id] = True
                continue
            is_ready, reason = await _probe_model(model_id, candidate_model_path, patient_store)
            model_availability[model_id] = is_ready
            if reason:
                model_availability_reasons[model_id] = reason
        app.state.model_availability = model_availability
        app.state.model_availability_reasons = model_availability_reasons
        yield
        await app.state.model_service.stop()

    app = FastAPI(title="Angiography Demo API", version="1.0.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[app_settings.frontend_origin, "http://localhost:5173", "http://127.0.0.1:5173"],
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        health_payload = app.state.model_service.health()
        return HealthResponse(status="ok", **health_payload)

    @app.get("/api/models")
    async def models():
        return get_model_cards(
            app.state.settings.model_path,
            app.state.model_availability,
            app.state.model_availability_reasons,
        )

    @app.post("/api/models/select")
    async def select_model(payload: ModelSelectRequest):
        model_path = get_model_path(payload.modelId)
        if model_path is None:
            raise HTTPException(status_code=400, detail=f"Unsupported model id: {payload.modelId}")

        async with app.state.model_switch_lock:
            current_model_id = app.state.active_model_id
            if current_model_id == payload.modelId:
                return get_model_cards(
                    app.state.settings.model_path,
                    app.state.model_availability,
                    app.state.model_availability_reasons,
                )

            previous_service = app.state.model_service
            try:
                next_service = await _create_model_service(payload.modelId, model_path, app.state.patient_store)
                await next_service.start()
            except Exception as error:
                app.state.model_availability[payload.modelId] = False
                app.state.model_availability_reasons[payload.modelId] = str(error)
                raise HTTPException(status_code=500, detail=f"Failed to load selected model: {error}") from error

            app.state.model_service = next_service
            app.state.settings = replace(app.state.settings, model_path=model_path)
            app.state.active_model_id = payload.modelId
            app.state.model_availability[payload.modelId] = True
            app.state.model_availability_reasons.pop(payload.modelId, None)
            await previous_service.stop()
            return get_model_cards(
                model_path,
                app.state.model_availability,
                app.state.model_availability_reasons,
            )

    @app.get("/api/patients")
    async def patients():
        return app.state.patient_store.summaries(dataset_id=_active_dataset_id())

    @app.get("/api/patients/{patient_id}/frames/{frame_index}")
    async def frame(patient_id: str, frame_index: int):
        _get_patient_for_active_dataset(patient_id)
        try:
            frame_path = app.state.patient_store.get_frame_path(patient_id, frame_index, dataset_id=_active_dataset_id())
        except IndexError as error:
            raise HTTPException(status_code=404, detail=f"Frame index out of range: {frame_index}") from error

        return FileResponse(frame_path)

    @app.get("/api/patients/{patient_id}/frames/{frame_index}/masks/{source}")
    async def frame_mask(patient_id: str, frame_index: int, source: str):
        _get_patient_for_active_dataset(patient_id)
        if source not in {"prediction", "ground_truth"}:
            raise HTTPException(status_code=400, detail=f"Unsupported mask source: {source}")

        model_id = app.state.active_model_id if source == "prediction" else None
        try:
            mask_path = app.state.patient_store.get_mask_path(
                patient_id,
                frame_index,
                source=source,
                model_id=model_id,
                dataset_id=_active_dataset_id(),
            )
        except IndexError as error:
            raise HTTPException(status_code=404, detail=f"Frame index out of range: {frame_index}") from error

        if mask_path is None or not mask_path.exists():
            raise HTTPException(status_code=404, detail=f"Mask not found for source '{source}'.")

        return FileResponse(mask_path)

    @app.post("/api/infer/frame", response_model=InferFrameResponse)
    async def infer_frame(payload: InferFrameRequest) -> InferFrameResponse:
        _get_patient_for_active_dataset(payload.patientId)
        try:
            frame_path = app.state.patient_store.get_frame_path(
                payload.patientId,
                payload.frameIndex,
                dataset_id=_active_dataset_id(),
            )
        except IndexError as error:
            raise HTTPException(status_code=404, detail=f"Frame index out of range: {payload.frameIndex}") from error

        try:
            return await app.state.model_service.infer_frame(payload.patientId, payload.frameIndex, frame_path)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/infer/prefetch", response_model=PrefetchResponse)
    async def prefetch(payload: PrefetchRequest) -> PrefetchResponse:
        patient = _get_patient_for_active_dataset(payload.patientId)

        start = min(payload.startFrame, payload.endFrame)
        end = max(payload.startFrame, payload.endFrame)
        end = min(end, len(patient.frames) - 1)

        queued = 0
        for frame_index in range(start, end + 1):
            frame_path = patient.frames[frame_index]
            if await app.state.model_service.queue_prefetch(payload.patientId, frame_index, frame_path):
                queued += 1

        return PrefetchResponse(
            patientId=payload.patientId,
            queued=queued,
            startFrame=start,
            endFrame=end,
        )

    @app.get("/api/labels/{patient_id}/{frame_index}", response_model=LabelsResponse)
    async def labels(patient_id: str, frame_index: int) -> LabelsResponse:
        patient = _get_patient_for_active_dataset(patient_id)

        try:
            frame_path = app.state.patient_store.get_frame_path(patient_id, frame_index, dataset_id=_active_dataset_id())
        except IndexError as error:
            raise HTTPException(status_code=404, detail=f"Frame index out of range: {frame_index}") from error

        if patient.label_type == "mask":
            mask_path = app.state.patient_store.get_mask_path(
                patient_id,
                frame_index,
                source="ground_truth",
                dataset_id=_active_dataset_id(),
            )
            if mask_path is None or not mask_path.exists():
                return LabelsResponse(
                    patientId=patient_id,
                    frameIndex=frame_index,
                    hasLabels=False,
                    labelType="mask",
                    boxes=[],
                    mask=None,
                )

            payload = _build_mask_payload(patient_id, frame_index, "ground_truth", mask_path)
            return LabelsResponse(
                patientId=patient_id,
                frameIndex=frame_index,
                hasLabels=True,
                labelType="mask",
                boxes=[],
                mask=payload,
            )

        label_path = app.state.patient_store.get_label_path(patient_id, frame_index, dataset_id=_active_dataset_id())
        if label_path is None:
            return LabelsResponse(patientId=patient_id, frameIndex=frame_index, hasLabels=False, labelType="bbox", boxes=[])

        boxes = parse_yolo_labels_to_boxes(label_path, frame_path)
        return LabelsResponse(patientId=patient_id, frameIndex=frame_index, hasLabels=True, labelType="bbox", boxes=boxes)

    @app.put("/api/labels/{patient_id}/{frame_index}", response_model=LabelsResponse)
    async def save_labels(patient_id: str, frame_index: int, payload: SaveLabelsRequest) -> LabelsResponse:
        _validate_ground_truth_boxes(payload.boxes)

        patient = _get_patient_for_active_dataset(patient_id)
        if patient.label_type != "bbox":
            raise HTTPException(
                status_code=409,
                detail="Mask-labeled frames are read-only in this demo; bbox label editing is not supported for this dataset.",
            )

        try:
            frame_path = app.state.patient_store.get_frame_path(patient_id, frame_index, dataset_id=_active_dataset_id())
            label_path = app.state.patient_store.get_writable_label_path(patient_id, frame_index, dataset_id=_active_dataset_id())
        except IndexError as error:
            raise HTTPException(status_code=404, detail=f"Frame index out of range: {frame_index}") from error

        try:
            write_yolo_labels_from_boxes(label_path, frame_path, payload.boxes)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        app.state.patient_store.mark_label_saved(patient_id, frame_index, dataset_id=_active_dataset_id())
        boxes = parse_yolo_labels_to_boxes(label_path, frame_path)
        return LabelsResponse(patientId=patient_id, frameIndex=frame_index, hasLabels=True, labelType="bbox", boxes=boxes)

    @app.get("/api/classification/{patient_id}/{frame_index}")
    async def classification(patient_id: str, frame_index: int, threshold: float = 0.5):
        _get_patient_for_active_dataset(patient_id)
        try:
            frame_path = app.state.patient_store.get_frame_path(patient_id, frame_index, dataset_id=_active_dataset_id())
        except IndexError as error:
            raise HTTPException(status_code=404, detail=f"Frame index out of range: {frame_index}") from error

        inference = await app.state.model_service.infer_frame(patient_id, frame_index, frame_path)
        if inference.outputType == "bbox":
            detected = has_stenosis(inference.boxes, threshold)
        else:
            detected = bool(inference.mask and inference.mask.positivePixelRatio > 0)

        return {
            "patientId": patient_id,
            "frameIndex": frame_index,
            "threshold": threshold,
            "outputType": inference.outputType,
            "stenosisDetected": detected,
            "boxCount": len(inference.boxes),
            "maskPositivePixelRatio": inference.mask.positivePixelRatio if inference.mask else None,
        }

    return app


app = create_app()
