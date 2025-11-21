"""FastAPI service for the Interface-Final interactive station."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ConfigCreateRequest,
    ConfigCreateResponse,
    ConfigReadResponse,
    ConfigScanRequest,
    ConfigScanResponse,
    DownloadResponse,
    LoadStudyRequest,
    PreviewRequest,
    PreviewResponse,
    PreviewClearRequest,
    PreviewClearResponse,
    PreviewDownloadRequest,
    PreviewDownloadResponse,
    PixelSizeUpdateRequest,
    PixelSizeUpdateResponse,
    RatioUpdateRequest,
    RatioUpdateResponse,
    RunStatus,
    StatisticsRequest,
    StatisticsResponse,
    ThresholdRunRequest,
    UploadResponse,
)
from .services.configurator import create_config, read_config, scan_input_directory
from .services.studies import (
    analyze_study,
    generate_downloads,
    generate_previews,
    clear_preview_cache,
    load_study,
    perform_statistics,
    resolve_download_path,
    resolve_preview_path,
    list_ratio_definitions,
    update_ratio_definitions,
    update_pixel_size,
    render_preview_panel,
)
from .services.uploads import UploadCategory, store_upload
from .services.threshold_runner import describe_run, launch_threshold_run
from .state import STATE


def create_app() -> FastAPI:
    app = FastAPI(title="ND2 Interface-Final Station", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/status")
    async def status() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/config/scan", response_model=ConfigScanResponse)
    async def config_scan(request: ConfigScanRequest) -> ConfigScanResponse:
        return scan_input_directory(request)

    @app.post("/config/create", response_model=ConfigCreateResponse)
    async def config_create(request: ConfigCreateRequest) -> ConfigCreateResponse:
        return create_config(request)

    @app.get("/config/read", response_model=ConfigReadResponse)
    async def config_read(path: str) -> ConfigReadResponse:
        return read_config(path)

    @app.post("/runs/threshold", response_model=RunStatus)
    async def run_threshold(request: ThresholdRunRequest, background: BackgroundTasks) -> RunStatus:
        return launch_threshold_run(request, background)

    @app.get("/runs/{job_id}", response_model=RunStatus)
    async def get_run(job_id: str) -> RunStatus:
        return describe_run(job_id)

    @app.post("/studies/load")
    async def load_threshold_results_endpoint(request: LoadStudyRequest) -> Dict[str, object]:
        record = load_study(request)
        return {
            "study_id": record.study_id,
            "study_name": record.results.study_name,
            "source_path": str(record.source_path),
            "groups": list(record.results.group_info.keys()),
            "mice_count": len({img.mouse_id for img in record.results.image_data}),
            "image_count": len(record.results.image_data),
            "nd2_root": str(record.input_dir),
            "nd2_available": record.input_dir.exists(),
            "ratio_definitions": record.ratio_definitions,
            "pixel_size_um": record.pixel_size_um,
        }

    @app.get("/studies")
    async def list_loaded_studies() -> Dict[str, Dict[str, object]]:
        return {
            study_id: {
                "study_name": record.results.study_name,
                "source_path": str(record.source_path),
                "loaded_at": record.loaded_at.isoformat() + "Z",
                "groups": list(record.results.group_info.keys()),
            }
            for study_id, record in STATE.iter_studies().items()
        }

    @app.post("/studies/{study_id}/analyze", response_model=AnalyzeResponse)
    async def analyze_endpoint(study_id: str, request: AnalyzeRequest) -> AnalyzeResponse:
        return analyze_study(study_id, request)

    @app.post("/studies/{study_id}/statistics", response_model=StatisticsResponse)
    async def statistics_endpoint(study_id: str, request: StatisticsRequest) -> StatisticsResponse:
        return perform_statistics(study_id, request)

    @app.post("/studies/{study_id}/previews", response_model=PreviewResponse)
    async def previews_endpoint(study_id: str, request: PreviewRequest) -> PreviewResponse:
        return generate_previews(study_id, request)

    @app.post("/studies/{study_id}/previews/clear", response_model=PreviewClearResponse)
    async def preview_cache_clear_endpoint(study_id: str, request: PreviewClearRequest) -> PreviewClearResponse:
        return clear_preview_cache(study_id, request.scope, request.threshold_key)

    @app.post("/studies/{study_id}/previews/render", response_model=PreviewDownloadResponse)
    async def preview_render_endpoint(study_id: str, request: PreviewDownloadRequest) -> PreviewDownloadResponse:
        return render_preview_panel(study_id, request)

    @app.post("/studies/{study_id}/downloads/current", response_model=DownloadResponse)
    async def download_endpoint(study_id: str, request: AnalyzeRequest) -> DownloadResponse:
        return generate_downloads(study_id, request.thresholds)

    @app.get("/studies/{study_id}/ratios", response_model=RatioUpdateResponse)
    async def list_ratio_defs(study_id: str) -> RatioUpdateResponse:
        ratios = list_ratio_definitions(study_id)
        return RatioUpdateResponse(ratios=ratios)

    @app.post("/studies/{study_id}/ratios", response_model=RatioUpdateResponse)
    async def update_ratio_defs(study_id: str, request: RatioUpdateRequest) -> RatioUpdateResponse:
        ratios = update_ratio_definitions(study_id, [entry.dict() for entry in request.ratios])
        return RatioUpdateResponse(ratios=ratios)

    @app.post("/studies/{study_id}/pixel-size", response_model=PixelSizeUpdateResponse)
    async def update_pixel_size_endpoint(study_id: str, request: PixelSizeUpdateRequest) -> PixelSizeUpdateResponse:
        value = update_pixel_size(study_id, request.pixel_size_um)
        return PixelSizeUpdateResponse(pixel_size_um=value)

    @app.get("/studies/{study_id}/preview-file")
    async def preview_file_endpoint(study_id: str, path: str) -> FileResponse:
        file_path = resolve_preview_path(study_id, path)
        return FileResponse(file_path)

    @app.get("/studies/{study_id}/download-file")
    async def download_file_endpoint(study_id: str, path: str) -> FileResponse:
        file_path = resolve_download_path(study_id, path)
        return FileResponse(file_path, filename=Path(file_path).name)

    @app.post("/uploads", response_model=UploadResponse)
    async def upload_file_endpoint(category: UploadCategory = Form(...), file: UploadFile = File(...)) -> UploadResponse:
        stored_path = store_upload(file, category)
        return UploadResponse(
            stored_path=str(stored_path),
            original_name=file.filename or "",
            category=category,
        )

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)
