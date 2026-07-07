# File: server.py
# Main FastAPI application for the TTS Server.
# Handles API requests for text-to-speech generation, UI serving,
# configuration management, and file uploads.

import os
import io
import logging
import logging.handlers  # For RotatingFileHandler
import shutil
import time
import uuid
import yaml  # For loading presets
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Literal, Iterator
import webbrowser  # For automatic browser opening
import threading  # For automatic browser opening


from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from fastapi import Depends, Security, Header
import hmac

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    File,
    UploadFile,
    Form,
    BackgroundTasks,
)
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
    FileResponse,
)
from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates  # Not used, serving static HTML
from fastapi.middleware.cors import CORSMiddleware

# --- Internal Project Imports ---
from config import (
    config_manager,
    get_host,
    get_port,
    get_open_browser,
    get_log_file_path,
    get_output_path,
    get_ui_title,
    get_gen_default_speed,
    get_gen_default_language,
    get_audio_sample_rate,
    get_full_config_for_template,
    get_audio_output_format,
)

import engine  # TTS Engine interface
from models import (  # Pydantic models
    CustomTTSRequest,
    ErrorResponse,
    UpdateStatusResponse,
)
import utils  # Utility functions

from pydantic import BaseModel, Field


class OpenAISpeechRequest(BaseModel):
    model: str
    input_: str = Field(..., alias="input", min_length=1)
    voice: str
    response_format: Literal["wav", "opus", "mp3"] = "wav"
    speed: float = 1.0
    seed: Optional[int] = None


# --- Logging Configuration ---
log_file_path_obj = get_log_file_path()
log_file_max_size_mb = config_manager.get_int("server.log_file_max_size_mb", 10)
log_backup_count = config_manager.get_int("server.log_file_backup_count", 5)

log_file_path_obj.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.handlers.RotatingFileHandler(
            str(log_file_path_obj),
            maxBytes=log_file_max_size_mb * 1024 * 1024,
            backupCount=log_backup_count,
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _is_performance_monitor_enabled() -> bool:
    """Return whether concise request performance logging is enabled."""
    env_value = os.getenv("ENABLE_PERFORMANCE_MONITOR")
    if env_value is not None:
        return env_value.strip().lower() in ("true", "1", "t", "yes", "y")
    return config_manager.get_bool("server.enable_performance_monitor", False)


def _format_duration_ms(start_time: float, end_time: Optional[float] = None) -> str:
    """Format an elapsed duration in milliseconds for compact logs."""
    end = time.perf_counter() if end_time is None else end_time
    return f"{(end - start_time) * 1000:.1f}ms"

# --- Global Variables & Application Setup ---
startup_complete_event = threading.Event()  # For coordinating browser opening
app_start_time = time.monotonic()



def _delayed_browser_open(host: str, port: int):
    """
    Waits for the startup_complete_event, then opens the web browser
    to the server's main page after a short delay.
    """
    try:
        startup_complete_event.wait(timeout=30)
        if not startup_complete_event.is_set():
            logger.warning(
                "Server startup did not signal completion within timeout. Browser will not be opened automatically."
            )
            return

        time.sleep(1.5)
        display_host = "localhost" if host == "0.0.0.0" else host
        browser_url = f"http://{display_host}:{port}/"
        logger.info(f"Attempting to open web browser to: {browser_url}")
        webbrowser.open(browser_url)
    except Exception as e:
        logger.error(f"Failed to open browser automatically: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown events."""
    logger.info("TTS Server: Initializing application...")
    try:
        logger.info(f"Configuration loaded. Log file at: {get_log_file_path()}")

        paths_to_ensure = [
            get_output_path(),
            Path("ui"),
            config_manager.get_path(
                "paths.model_cache", "./model_cache", ensure_absolute=True
            ),
        ]
        for p in paths_to_ensure:
            p.mkdir(parents=True, exist_ok=True)

        if not engine.load_model():
            logger.critical(
                "CRITICAL: TTS Model failed to load on startup. Server might not function correctly."
            )
        else:
            logger.info("TTS Model loaded successfully via engine.")
            engine.warmup(text="Hello")
            if get_open_browser():
                host_address = get_host()
                server_port = get_port()
                browser_thread = threading.Thread(
                    target=lambda: _delayed_browser_open(host_address, server_port),
                    daemon=True,
                )
                browser_thread.start()
            else:
                logger.info(
                    "Browser auto-open is disabled. Set server.open_browser=true "
                    "or OPEN_BROWSER=true to enable it for local UI development."
                )

        logger.info("Application startup sequence complete.")
        startup_complete_event.set()
        yield
    except Exception as e_startup:
        logger.error(
            f"FATAL ERROR during application startup: {e_startup}", exc_info=True
        )
        startup_complete_event.set()
        yield
    finally:
        logger.info("TTS Server: Application shutdown sequence initiated...")
        logger.info("TTS Server: Application shutdown complete.")



security = HTTPBearer(auto_error=False)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    x_api_key: str = Security(api_key_header)
):
    api_key = os.environ.get("TTS_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="TTS_API_KEY is not configured on the server. Access denied.")

    token = x_api_key
    if not token and credentials and credentials.credentials:
        token = credentials.credentials

    if not token:
        raise HTTPException(status_code=401, detail="Missing API Key")

    if not hmac.compare_digest(token, api_key):
        raise HTTPException(status_code=403, detail="Invalid API Key")

    return token

def verify_management_access(api_key = Depends(verify_api_key)):
    # Block if ENABLE_MANAGEMENT_ENDPOINTS is false
    if os.environ.get("ENABLE_MANAGEMENT_ENDPOINTS", "false").lower() == "false":
        raise HTTPException(status_code=403, detail="Management endpoints are disabled")
    return api_key

def verify_ui_access():
    if os.environ.get("ENABLE_WEB_UI", "false").lower() == "false":
        raise HTTPException(status_code=403, detail="Web UI is disabled")

# --- FastAPI Application Instance ---
app = FastAPI(
    title=get_ui_title(),
    description="Text-to-Speech server with advanced UI and API capabilities.",
    version="3.0.0",  # Multi-model support
    lifespan=lifespan,
)

# --- CORS Middleware ---
allowed_origins_str = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173,http://localhost:8000")
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]

# If no origins are configured or it's a wildcard, we must disable credentials
allow_credentials = "*" not in allowed_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["http://localhost:8000"],
    allow_credentials=allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Static Files and HTML Templates ---
ui_static_path = Path(__file__).parent / "ui"
if ui_static_path.is_dir():
    app.mount("/ui", StaticFiles(directory=ui_static_path), name="ui_static_assets")
else:
    logger.warning(
        f"UI static assets directory not found at '{ui_static_path}'. UI may not load correctly."
    )

# This will serve files from 'ui_static_path/vendor' when requests come to '/vendor/*'
if (ui_static_path / "vendor").is_dir():
    app.mount(
        "/vendor", StaticFiles(directory=ui_static_path / "vendor"), name="vendor_files"
    )
else:
    logger.warning(
        f"Vendor directory not found at '{ui_static_path}' /vendor. Wavesurfer might not load."
    )


@app.get("/styles.css", include_in_schema=False)
async def get_main_styles():
    styles_file = ui_static_path / "styles.css"
    if styles_file.is_file():
        return FileResponse(styles_file)
    raise HTTPException(status_code=404, detail="styles.css not found")


@app.get("/script.js", include_in_schema=False)
async def get_main_script():
    script_file = ui_static_path / "script.js"
    if script_file.is_file():
        return FileResponse(script_file)
    raise HTTPException(status_code=404, detail="script.js not found")


outputs_static_path = get_output_path(ensure_absolute=True)
try:
    app.mount(
        "/outputs",
        StaticFiles(directory=str(outputs_static_path)),
        name="generated_outputs",
    )
except RuntimeError as e_mount_outputs:
    logger.error(
        f"Failed to mount /outputs directory '{outputs_static_path}': {e_mount_outputs}. "
        "Output files may not be accessible via URL."
    )

# templates removed - serving index.html as static file

# --- API Endpoints ---


@app.get("/health", tags=["Health"])
@app.get("/api/tts/health", tags=["Health"])
async def health_check():
    """Reports process, model, ONNX provider, and readiness health."""
    model_loaded = bool(engine.MODEL_LOADED)
    loading = engine.is_loading()
    download_status = engine.get_download_status()
    load_error = download_status.get("error")
    warmup_completed = startup_complete_event.is_set() and model_loaded
    warmup_failed = (
        startup_complete_event.is_set()
        and not model_loaded
        and not loading
    )
    provider_info = engine.get_onnx_provider_info()
    model_info = engine.get_model_info()
    configured_selector = engine.resolve_selector(
        config_manager.get_string("model.repo_id", "KittenML/kitten-tts-nano-0.1")
    )
    model_repo_id = (
        model_info.get("repo_id")
        or engine.MODEL_REGISTRY[configured_selector]["repo_id"]
    )

    if model_loaded:
        status = "ok"
        status_code = 200
    elif loading:
        status = "model_loading"
        status_code = 503
    elif load_error:
        status = f"model_load_failed: {load_error}"
        status_code = 503
    else:
        status = "model_load_failed" if warmup_failed else "starting"
        status_code = 503

    payload = {
        "status": status,
        "model_loaded": model_loaded,
        "warmup_completed": warmup_completed,
        "warmup_failed": warmup_failed,
        "available_voices_count": len(engine.get_available_voices()),
        "provider": provider_info.get("active"),
        "uptime_seconds": round(time.monotonic() - app_start_time, 3),
        "model_repo_id": model_repo_id,
        "device": model_info.get("device"),
    }
    return JSONResponse(content=payload, status_code=status_code)


# --- Main UI Route ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_web_ui(request: Request, _: None = Depends(verify_ui_access)):
    """Serves the main web interface (index.html)."""
    logger.info("Request received for main UI page ('/').")
    try:
        index_path = ui_static_path / "index.html"
        if index_path.is_file():
            return FileResponse(index_path, media_type="text/html")
        return HTMLResponse(
            "<html><body><h1>Not Found</h1><p>index.html not found.</p></body></html>",
            status_code=404,
        )
    except Exception as e_render:
        logger.error(f"Error rendering main UI page: {e_render}", exc_info=True)
        return HTMLResponse(
            "<html><body><h1>Internal Server Error</h1><p>Could not load the TTS interface. "
            "Please check server logs for more details.</p></body></html>",
            status_code=500,
        )


# --- API Endpoint for Model Information ---
@app.get("/api/model-info", tags=["Model Information"])
async def get_model_info_endpoint():
    """Returns detailed information about the currently loaded TTS model."""
    logger.debug("Request received for /api/model-info")
    try:
        return engine.get_model_info()
    except Exception as e:
        logger.error(f"Error getting model info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")


@app.get("/api/model-registry", tags=["Model Information"])
async def get_model_registry_endpoint():
    """Returns the full list of available models for the UI dropdown."""
    return engine.get_model_registry()


@app.get("/api/model-status", tags=["Model Information"])
async def get_model_status_endpoint():
    """Returns the current download/loading progress for model switching."""
    return engine.get_download_status()


@app.get("/health", tags=["Health"])
@app.get("/api/tts/health", tags=["Health"])
async def health_endpoint():
    """Returns service, model, and warmup health state."""
    return {
        "status": "ok" if engine.MODEL_LOADED else "degraded",
        "model_loaded": engine.MODEL_LOADED,
        "model": engine.get_model_info(),
        "warmup": engine.get_warmup_info(),
    }


# --- API Endpoint for Initial UI Data ---
@app.get("/api/ui/initial-data", tags=["UI Helpers"])
async def get_ui_initial_data(_: None = Depends(verify_ui_access), __: None = Depends(verify_api_key)):
    """
    Provides all necessary initial data for the UI to render,
    including configuration, file lists, presets, and model information.
    """
    logger.info("Request received for /api/ui/initial-data.")
    try:
        full_config = get_full_config_for_template()

        # Get model information for UI
        model_info = engine.get_model_info()
        model_registry = engine.get_model_registry()

        loaded_presets = []
        presets_file = ui_static_path / "presets.yaml"
        if presets_file.exists():
            with open(presets_file, "r", encoding="utf-8") as f:
                yaml_content = yaml.safe_load(f)
                if isinstance(yaml_content, list):
                    loaded_presets = yaml_content
                else:
                    logger.warning(
                        f"Invalid format in {presets_file}. Expected a list, got {type(yaml_content)}."
                    )
        else:
            logger.info(
                f"Presets file not found: {presets_file}. No presets will be loaded for initial data."
            )

        initial_gen_result_placeholder = {
            "outputUrl": None,
            "filename": None,
            "genTime": None,
            "submittedVoice": None,
        }

        sanitized_config = {
            "ui": full_config.get("ui", {}),
            "ui_state": full_config.get("ui_state", {}),
            "generation_defaults": full_config.get("generation_defaults", {}),
            "audio_output": full_config.get("audio_output", {}),
        }
        return {
            "config": sanitized_config,
            "presets": loaded_presets,
            "initial_gen_result": initial_gen_result_placeholder,
            "model_info": model_info,
            "model_registry": model_registry,
            "available_voices": engine.get_available_voices(),
        }
    except Exception as e:
        logger.error(f"Error preparing initial UI data for API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to load initial data for UI."
        )


# --- Configuration Management API Endpoints ---
@app.post("/save_settings", response_model=UpdateStatusResponse, tags=["Configuration"])
async def save_settings_endpoint(request: Request, _: None = Depends(verify_management_access)):
    """
    Saves partial configuration updates to the config.yaml file.
    Merges the update with the current configuration.
    """
    logger.info("Request received for /save_settings.")
    try:
        partial_update = await request.json()
        if not isinstance(partial_update, dict):
            raise ValueError("Request body must be a JSON object for /save_settings.")
        logger.debug(f"Received partial config data to save: {partial_update}")

        if config_manager.update_and_save(partial_update):
            restart_needed = any(
                key in partial_update
                for key in ["server", "tts_engine", "paths", "model"]
            )
            message = "Settings saved successfully."
            if restart_needed:
                message += " A server restart may be required for some changes to take full effect."
            return UpdateStatusResponse(message=message, restart_needed=restart_needed)
        else:
            logger.error(
                "Failed to save configuration via config_manager.update_and_save."
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to save configuration file due to an internal error.",
            )
    except ValueError as ve:
        logger.error(f"Invalid data format for /save_settings: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid request data: {str(ve)}")
    except Exception as e:
        logger.error(f"Error processing /save_settings request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during settings save: {str(e)}",
        )


@app.post(
    "/reset_settings", response_model=UpdateStatusResponse, tags=["Configuration"]
)
async def reset_settings_endpoint(_: None = Depends(verify_management_access)):
    """Resets the configuration in config.yaml back to hardcoded defaults."""
    logger.warning("Request received to reset all configurations to default values.")
    try:
        if config_manager.reset_and_save():
            logger.info("Configuration successfully reset to defaults and saved.")
            return UpdateStatusResponse(
                message="Configuration reset to defaults. Please reload the page. A server restart may be beneficial.",
                restart_needed=True,
            )
        else:
            logger.error("Failed to reset and save configuration via config_manager.")
            raise HTTPException(
                status_code=500, detail="Failed to reset and save configuration file."
            )
    except Exception as e:
        logger.error(f"Error processing /reset_settings request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during settings reset: {str(e)}",
        )


@app.post(
    "/restart_server", response_model=UpdateStatusResponse, tags=["Configuration"]
)
async def restart_server_endpoint(_: None = Depends(verify_management_access)):
    """
    Triggers an async hot-swap of the TTS model engine.
    Returns immediately while the model downloads and loads in the background.
    The UI polls /api/model-status to track progress.
    """
    logger.info("Request received for /restart_server (Async Model Hot-Swap).")

    try:
        engine.reload_model_async()
        return UpdateStatusResponse(
            message="Model reload initiated in background. Poll /api/model-status for progress.",
            restart_needed=False,
        )
    except Exception as e:
        logger.error(f"Error initiating model hot-swap: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initiate model reload: {str(e)}",
        )


@app.post("/api/cancel-loading", tags=["Configuration"])
async def cancel_loading_endpoint(_: None = Depends(verify_management_access)):
    """Cancels any in-progress model loading."""
    logger.info("Request received for /api/cancel-loading.")
    cancelled = engine.cancel_loading()
    if cancelled:
        return {"message": "Model loading cancellation requested."}
    return {"message": "No model loading in progress."}


@app.post("/api/unload", tags=["Configuration"])
async def unload_model_endpoint(_: None = Depends(verify_management_access)):
    """
    Unloads the TTS model and releases all resources.
    The model will need to be reloaded (via /restart_server) before TTS requests can be processed.
    """
    logger.info("Request received for /api/unload (Model Unload).")
    try:
        success = engine.unload_model()
        if success:
            return {"message": "Model unloaded successfully. Resources released."}
        else:
            raise HTTPException(status_code=500, detail="Failed to unload model.")
    except Exception as e:
        logger.error(f"Error during model unload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# --- TTS Generation Endpoint ---


@app.post("/api/tts/speak", tags=["TTS Generation"])
@app.post(
    "/tts",
    tags=["TTS Generation"],
    summary="Generate speech with custom parameters",
    responses={
        200: {
            "content": {"audio/wav": {}, "audio/opus": {}},
            "description": "Successful audio generation.",
        },
        400: {
            "model": ErrorResponse,
            "description": "Invalid request parameters or input.",
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal server error during generation.",
        },
        503: {
            "model": ErrorResponse,
            "description": "TTS engine not available or model not loaded.",
        },
    },
)
async def custom_tts_endpoint(
    request: CustomTTSRequest, background_tasks: BackgroundTasks, _: None = Depends(verify_api_key)
):
    """
    Generates speech audio from text using specified parameters.
    Returns audio as a stream (WAV or Opus).
    """
    perf_monitor = utils.PerformanceMonitor(
        enabled=config_manager.get_bool("server.enable_performance_monitor", False)
    )
    perf_monitor.record("TTS request received")

    output_format_str = (
        request.output_format if request.output_format else get_audio_output_format()
    )
    if output_format_str == "mp3":
        raise HTTPException(status_code=400, detail="MP3 output is not supported in this deployment. Use wav.")
    if output_format_str not in ["wav", "opus"]:
        raise HTTPException(status_code=400, detail=f"Invalid format: {output_format_str}. Only 'wav' or 'opus' are supported.")


    if not engine.MODEL_LOADED:
        logger.error("TTS request failed: Model not loaded.")
        raise HTTPException(
            status_code=503,
            detail="TTS engine model is not currently loaded or available.",
        )

    logger.info(
        f"Received /tts request: voice='{request.voice}', format='{request.output_format}', stream={request.stream}"
    )
    logger.debug(
        f"TTS params: speed={request.speed}, split={request.split_text}, chunk_size={request.chunk_size}"
    )
    logger.debug(f"Input text (first 100 chars): '{request.text[:100]}...'")

    perf_monitor.record("Parameters resolved")

    final_output_sample_rate = get_audio_sample_rate()

    output_format_str = (
        request.output_format if request.output_format else get_audio_output_format()
    )


    if request.split_text and len(request.text) > (
        request.chunk_size * 1.5 if request.chunk_size else 120 * 1.5
    ):
        chunk_size_to_use = (
            request.chunk_size if request.chunk_size is not None else 120
        )
        logger.info(f"Splitting text into chunks of size ~{chunk_size_to_use}.")
        text_chunks = utils.chunk_text_by_sentences(request.text, chunk_size_to_use)
        perf_monitor.record(f"Text split into {len(text_chunks)} chunks")
    else:
        text_chunks = [request.text]
        logger.info(
            "Processing text as a single chunk (splitting not enabled or text too short)."
        )

    if not text_chunks:
        raise HTTPException(
            status_code=400, detail="Text processing resulted in no usable chunks."
        )

    if request.stream:
        media_type = f"audio/{output_format_str}"

        def stream_encoded_chunks() -> Iterator[bytes]:
            engine_output_sample_rate: Optional[int] = None
            chunks_yielded = 0

            for i, chunk in enumerate(text_chunks):
                logger.info(f"Streaming synthesis for chunk {i+1}/{len(text_chunks)}...")
                try:
                    chunk_audio_np, chunk_sr_from_engine = engine.synthesize(
                        text=chunk,
                        voice=request.voice,
                        speed=(
                            request.speed
                            if request.speed is not None
                            else get_gen_default_speed()
                        ),
                    )
                    perf_monitor.record(f"Engine synthesized streaming chunk {i+1}")

                    if chunk_audio_np is None or chunk_sr_from_engine is None:
                        raise RuntimeError(
                            f"TTS engine failed to synthesize audio for chunk {i+1}."
                        )

                    if engine_output_sample_rate is None:
                        engine_output_sample_rate = chunk_sr_from_engine
                    elif engine_output_sample_rate != chunk_sr_from_engine:
                        logger.warning(
                            f"Inconsistent sample rate from engine: chunk {i+1} ({chunk_sr_from_engine}Hz) "
                            f"differs from previous ({engine_output_sample_rate}Hz). Encoding chunk with its own SR."
                        )

                    encoded_chunk = utils.encode_audio(
                        audio_array=chunk_audio_np,
                        sample_rate=chunk_sr_from_engine,
                        output_format=output_format_str,
                        target_sample_rate=final_output_sample_rate,
                    )
                    perf_monitor.record(
                        f"Streaming chunk {i+1} encoded to {output_format_str} "
                        f"(target SR: {final_output_sample_rate}Hz from engine SR: {chunk_sr_from_engine}Hz)"
                    )

                    if encoded_chunk is None or len(encoded_chunk) < 100:
                        raise RuntimeError(
                            f"Failed to encode audio chunk {i+1} to {output_format_str} "
                            f"or generated invalid audio."
                        )

                    if output_format_str == "wav" and chunks_yielded > 0:
                        encoded_chunk = encoded_chunk[44:]
                    chunks_yielded += 1
                    yield encoded_chunk
                except Exception as e_chunk:
                    logger.error(
                        f"Error processing streaming audio chunk {i+1}: {e_chunk}",
                        exc_info=True,
                    )
                    raise

            if chunks_yielded == 0:
                logger.error("Streaming audio generation yielded no chunks.")
                raise RuntimeError("Audio generation resulted in no output.")

            logger.info(
                f"Successfully streamed {chunks_yielded} encoded audio chunks as {media_type}."
            )
            logger.debug(perf_monitor.report())

        headers = {"X-TTS-Stream-Mode": "chunked-encoded-audio"}
        if output_format_str == "wav":
            headers["X-TTS-WAV-Streaming-Note"] = (
                "Each chunk is a standalone WAV with its own header; clients requiring one WAV file may be incompatible."
            )

        return StreamingResponse(
            stream_encoded_chunks(), media_type=media_type, headers=headers
        )

    all_audio_segments_np: List[np.ndarray] = []
    engine_output_sample_rate: Optional[int] = None

    for i, chunk in enumerate(text_chunks):
        logger.info(f"Synthesizing chunk {i+1}/{len(text_chunks)}...")
        try:
            chunk_audio_np, chunk_sr_from_engine = engine.synthesize(
                text=chunk,
                voice=request.voice,
                speed=(
                    request.speed
                    if request.speed is not None
                    else get_gen_default_speed()
                ),
            )
            perf_monitor.record(f"Engine synthesized chunk {i+1}")

            if chunk_audio_np is None or chunk_sr_from_engine is None:
                error_detail = f"TTS engine failed to synthesize audio for chunk {i+1}."
                logger.error(error_detail)
                raise HTTPException(status_code=500, detail=error_detail)

            if engine_output_sample_rate is None:
                engine_output_sample_rate = chunk_sr_from_engine
            elif engine_output_sample_rate != chunk_sr_from_engine:
                logger.warning(
                    f"Inconsistent sample rate from engine: chunk {i+1} ({chunk_sr_from_engine}Hz) "
                    f"differs from previous ({engine_output_sample_rate}Hz). Using first chunk's SR."
                )

            # The speed factor is now handled by the engine directly, so no post-processing for speed is needed here.

            all_audio_segments_np.append(chunk_audio_np)

        except HTTPException as http_exc:
            raise http_exc
        except Exception as e_chunk:
            error_detail = f"Error processing audio chunk {i+1}: {str(e_chunk)}"
            logger.error(error_detail, exc_info=True)
            raise HTTPException(status_code=500, detail=error_detail)

    if not all_audio_segments_np:
        logger.error("No audio segments were successfully generated.")
        raise HTTPException(
            status_code=500, detail="Audio generation resulted in no output."
        )

    if engine_output_sample_rate is None:
        logger.error("Engine output sample rate could not be determined.")
        raise HTTPException(
            status_code=500, detail="Failed to determine engine sample rate."
        )

    try:
        if len(all_audio_segments_np) > 1:
            # Add silence between chunks for natural pauses
            silence_duration_ms = 200  # silence between chunks
            silence_samples = int(
                silence_duration_ms / 1000 * engine_output_sample_rate
            )
            silence_array = np.zeros(silence_samples, dtype=np.float32)

            # Apply crossfade and add silence between chunks
            crossfade_samples = int(0.01 * engine_output_sample_rate)  # 10ms crossfade

            merged_audio = []
            for i, chunk in enumerate(all_audio_segments_np):
                if i == 0:
                    merged_audio.append(chunk)
                else:
                    # Add silence gap between chunks
                    merged_audio.append(silence_array)

                    # Then add the next chunk with optional crossfade
                    if (
                        len(merged_audio[-2]) >= crossfade_samples
                        and len(chunk) >= crossfade_samples
                    ):
                        # Apply fade out to end of previous audio (before silence)
                        fade_out = np.linspace(1, 0, crossfade_samples)
                        merged_audio[-2][-crossfade_samples:] *= fade_out

                        # Apply fade in to start of current chunk
                        fade_in = np.linspace(0, 1, crossfade_samples)
                        chunk_copy = chunk.copy()
                        chunk_copy[:crossfade_samples] *= fade_in
                        merged_audio.append(chunk_copy)
                    else:
                        merged_audio.append(chunk)

            final_audio_np = np.concatenate(merged_audio)
            logger.debug(
                f"Added {silence_duration_ms}ms silence between {len(all_audio_segments_np)} chunks"
            )
        else:
            final_audio_np = all_audio_segments_np[0]

        perf_monitor.record("All audio chunks processed and concatenated")

    except ValueError as e_concat:
        logger.error(f"Audio concatenation failed: {e_concat}", exc_info=True)
        for idx, seg in enumerate(all_audio_segments_np):
            logger.error(f"Segment {idx} shape: {seg.shape}, dtype: {seg.dtype}")
        raise HTTPException(
            status_code=500, detail=f"Audio concatenation error: {e_concat}"
        )

    encoded_audio_bytes = utils.encode_audio(
        audio_array=final_audio_np,
        sample_rate=engine_output_sample_rate,
        output_format=output_format_str,
        target_sample_rate=final_output_sample_rate,
    )
    perf_monitor.record(
        f"Final audio encoded to {output_format_str} (target SR: {final_output_sample_rate}Hz from engine SR: {engine_output_sample_rate}Hz)"
    )

    if encoded_audio_bytes is None or len(encoded_audio_bytes) < 100:
        logger.error(
            f"Failed to encode final audio to format: {output_format_str} or output is too small ({len(encoded_audio_bytes or b'')} bytes)."
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to encode audio to {output_format_str} or generated invalid audio.",
        )

    media_type = f"audio/{output_format_str}"
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    suggested_filename_base = f"tts_output_{timestamp_str}"
    download_filename = utils.sanitize_filename(
        f"{suggested_filename_base}.{output_format_str}"
    )
    headers = {"Content-Disposition": f'attachment; filename="{download_filename}"'}

    logger.info(
        f"Successfully generated audio: {download_filename}, {len(encoded_audio_bytes)} bytes, type {media_type}."
    )
    logger.debug(perf_monitor.report())

    return StreamingResponse(
        io.BytesIO(encoded_audio_bytes), media_type=media_type, headers=headers
    )


@app.post("/v1/audio/speech", tags=["OpenAI Compatible"])
async def openai_speech_endpoint(request: OpenAISpeechRequest, _: None = Depends(verify_api_key)):
    # Check if the TTS model is loaded
    if not engine.MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="TTS engine model is not currently loaded or available.",
        )

    if request.response_format == "mp3":
        raise HTTPException(status_code=400, detail="MP3 output is not supported in this deployment. Use wav.")
    request_start = time.perf_counter()
    text_length = len(request.input_ or "")
    chunk_size = 300
    should_stream_chunks = text_length > chunk_size and request.response_format == "wav"
    media_type = f"audio/{request.response_format}"

    def log_timing(event_name: str, start_time: float = request_start) -> None:
        elapsed = time.perf_counter() - start_time
        total_elapsed = time.perf_counter() - request_start
        logger.info(
            "OpenAI speech timing: %s took %.3fs (total %.3fs)",
            event_name,
            elapsed,
            total_elapsed,
        )

    def synthesize_and_encode_chunk(chunk_text: str, chunk_index: int) -> bytes:
        chunk_start = time.perf_counter()
        audio_np, sr = engine.synthesize(
            text=chunk_text,
            voice=request.voice,
            speed=request.speed,
        )
        log_timing(f"chunk {chunk_index} synthesis", chunk_start)

        if audio_np is None or sr is None:
            raise HTTPException(
                status_code=500,
                detail=f"TTS engine failed to synthesize audio for chunk {chunk_index}.",
            )

        # Ensure it's 1D
        if audio_np.ndim == 2:
            audio_np = audio_np.squeeze()

        encode_start = time.perf_counter()
        encoded_audio = utils.encode_audio(
            audio_array=audio_np,
            sample_rate=sr,
            output_format=request.response_format,
            target_sample_rate=get_audio_sample_rate(),
        )
        log_timing(
            f"chunk {chunk_index} {request.response_format} encoding", encode_start
        )

        if encoded_audio is None:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to encode audio for chunk {chunk_index}.",
            )

        return encoded_audio

    try:
        if not should_stream_chunks:
            if text_length > chunk_size and request.response_format != "wav":
                logger.info(
                    "OpenAI speech request uses response_format='%s'; using buffered compatibility path. "
                    "Chunked low-latency streaming is currently enabled for WAV responses only.",
                    request.response_format,
                )
            encoded_audio = synthesize_and_encode_chunk(request.input_, 1)
            log_timing("buffered response ready")
            return StreamingResponse(io.BytesIO(encoded_audio), media_type=media_type)

        logger.info(
            "OpenAI speech request length %d exceeds %d characters; streaming WAV sentence chunks.",
            text_length,
            chunk_size,
        )

        def audio_chunk_generator():
            first_chunk_logged = False
            first_yield_logged = False

            try:
                for chunk_index, text_chunk in enumerate(
                    utils.chunk_text_by_sentences(request.input_, chunk_size), start=1
                ):
                    if not text_chunk.strip():
                        continue

                    chunk_start = time.perf_counter()
                    encoded_audio = synthesize_and_encode_chunk(text_chunk, chunk_index)
                    if not first_chunk_logged:
                        log_timing("first chunk generated", chunk_start)
                        first_chunk_logged = True

                    if not first_yield_logged:
                        log_timing("first chunk yielded")
                        first_yield_logged = True
                    if request.response_format == "wav" and chunk_index > 1:
                        encoded_audio = encoded_audio[44:]
                    yield encoded_audio

                log_timing("streaming response completed")
            except Exception as e:
                logger.error(
                    f"Error while streaming openai_speech_endpoint: {e}", exc_info=True
                )
                raise

        return StreamingResponse(audio_chunk_generator(), media_type=media_type)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in openai_speech_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# --- Main Execution ---
if __name__ == "__main__":
    server_host = get_host()
    server_port = get_port()

    logger.info(f"Starting TTS Server directly on http://{server_host}:{server_port}")
    logger.info(
        f"API documentation will be available at http://{server_host}:{server_port}/docs"
    )
    logger.info(f"Web UI will be available at http://{server_host}:{server_port}/")

    import uvicorn

    uvicorn.run(
        "server:app",
        host=server_host,
        port=server_port,
        log_level="info",
        workers=1,
        reload=False,
    )
