# File: engine.py
# Core TTS model loading and speech generation logic for KittenTTS ONNX.
# Supports multiple KittenTTS model variants with hot-swappable model switching.

import ctypes.util
import gc
import os
import json
import logging
import threading
import time
import numpy as np
import onnxruntime as ort
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
from huggingface_hub import hf_hub_download
import phonemizer

# This loader can be problematic on Linux, we will bypass it with system-installed eSpeak.
# We still import it as it's a dependency, but we will avoid calling it directly where possible.
import espeakng_loader

# Import the singleton config_manager
from config import config_manager

logger = logging.getLogger(__name__)

# --- Model Registry ---
# Maps friendly selector names to HuggingFace repo IDs, voice lists, and metadata.
# Voice lists for v0.1/v0.2 models use expr-voice-* format.
# Voice lists for v0.8 models use named voices.

# v0.1/v0.2 models: named voices mapped to internal expr-voice-* keys
_V01_VOICES = ["Amber", "Felix", "Clara", "Marcus", "Ivy", "Oscar", "Nora", "Reed"]

# Mapping from v0.1 display names to internal voice keys
_V01_VOICE_ALIASES = {
    "Amber": "expr-voice-2-f",
    "Felix": "expr-voice-2-m",
    "Clara": "expr-voice-3-f",
    "Marcus": "expr-voice-3-m",
    "Ivy": "expr-voice-4-f",
    "Oscar": "expr-voice-4-m",
    "Nora": "expr-voice-5-f",
    "Reed": "expr-voice-5-m",
}

# v0.8 models: named voices (aliases come from model config.json)
_V08_VOICES = [
    "Bella", "Jasper", "Luna", "Bruno",
    "Rosie", "Hugo", "Kiki", "Leo",
]

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "kitten-nano-0.1": {
        "repo_id": "KittenML/kitten-tts-nano-0.1",
        "display_name": "Nano 0.1 (15M, Original)",
        "voices": _V01_VOICES,
        "default_voice": "Reed",
        "params": "15M",
        "size": "<25MB",
        "version": "0.1",
    },
    "kitten-nano-0.2": {
        "repo_id": "KittenML/kitten-tts-nano-0.2",
        "display_name": "Nano 0.2 (15M, Preview)",
        "voices": _V01_VOICES,
        "default_voice": "Reed",
        "params": "15M",
        "size": "<25MB",
        "version": "0.2",
    },
    "kitten-mini-0.1": {
        "repo_id": "KittenML/kitten-tts-mini-0.1",
        "display_name": "Mini 0.1 (80M, Original)",
        "voices": _V01_VOICES,
        "default_voice": "Reed",
        "params": "80M",
        "size": "~170MB",
        "version": "0.1",
    },
    "kitten-nano-0.8-int8": {
        "repo_id": "KittenML/kitten-tts-nano-0.8-int8",
        "display_name": "Nano 0.8 INT8 (15M, Quantized)",
        "voices": _V08_VOICES,
        "default_voice": "Jasper",
        "params": "15M",
        "size": "~25MB",
        "version": "0.8",
    },
    "kitten-nano-0.8-fp32": {
        "repo_id": "KittenML/kitten-tts-nano-0.8-fp32",
        "display_name": "Nano 0.8 FP32 (15M, Full Precision)",
        "voices": _V08_VOICES,
        "default_voice": "Jasper",
        "params": "15M",
        "size": "~50MB",
        "version": "0.8",
    },
    "kitten-micro-0.8": {
        "repo_id": "KittenML/kitten-tts-micro-0.8",
        "display_name": "Micro 0.8 (40M, Balanced)",
        "voices": _V08_VOICES,
        "default_voice": "Jasper",
        "params": "40M",
        "size": "~40MB",
        "version": "0.8",
    },
    "kitten-mini-0.8": {
        "repo_id": "KittenML/kitten-tts-mini-0.8",
        "display_name": "Mini 0.8 (80M, Highest Quality)",
        "voices": _V08_VOICES,
        "default_voice": "Jasper",
        "params": "80M",
        "size": "~79MB",
        "version": "0.8",
    },
}

# Build a reverse lookup: repo_id -> selector key
_REPO_TO_SELECTOR = {}
for _sel, _info in MODEL_REGISTRY.items():
    _REPO_TO_SELECTOR[_info["repo_id"]] = _sel


def resolve_selector(config_value: str) -> str:
    """
    Resolves a config model.repo_id value to a registry selector key.
    Accepts either a selector key directly or a full HuggingFace repo_id.
    Returns the selector key, or falls back to 'kitten-nano-0.1' if unknown.
    """
    # Direct selector match
    if config_value in MODEL_REGISTRY:
        return config_value
    # Full repo_id match
    if config_value in _REPO_TO_SELECTOR:
        return _REPO_TO_SELECTOR[config_value]
    # Try case-insensitive match
    lower = config_value.lower().strip()
    for sel in MODEL_REGISTRY:
        if sel.lower() == lower:
            return sel
    for repo_id, sel in _REPO_TO_SELECTOR.items():
        if repo_id.lower() == lower:
            return sel
    logger.warning(
        f"Unknown model selector '{config_value}'. "
        f"Valid selectors: {list(MODEL_REGISTRY.keys())}. "
        f"Defaulting to 'kitten-nano-0.1'."
    )
    return "kitten-nano-0.1"


# --- Global Module Variables ---
onnx_session: Optional[ort.InferenceSession] = None
voices_data: Optional[dict] = None
phonemizer_backend: Optional[phonemizer.backend.EspeakBackend] = None
text_cleaner: Optional["TextCleaner"] = None
MODEL_LOADED: bool = False
WARMUP_COMPLETED: bool = False
WARMUP_FAILED: bool = False
WARMUP_DURATION_SECONDS: Optional[float] = None

# Track which model is loaded
loaded_model_selector: Optional[str] = None
loaded_model_repo_id: Optional[str] = None
loaded_model_device: Optional[str] = None

# Voice alias mapping for v0.8 models (e.g. "Jasper" -> "expr-voice-2-m")
_voice_aliases: Dict[str, str] = {}
# Speed priors per voice from model config (per-voice speed correction factors)
_speed_priors: Dict[str, float] = {}

# --- Async Loading & Cancellation ---
_load_lock = threading.Lock()
_cancel_event = threading.Event()
_load_thread: Optional[threading.Thread] = None

# Download progress tracking for UI status modal
_download_status: Dict[str, Any] = {
    "active": False,
    "phase": "",
    "detail": "",
    "progress_pct": 0,
    "error": None,
}


def _check_cancelled():
    """Raises RuntimeError if model loading has been cancelled."""
    if _cancel_event.is_set():
        raise RuntimeError("Model loading cancelled by user.")


def _update_download_status(phase: str, detail: str = "", progress_pct: int = 0, error: str = None):
    """Update the download status for the UI to poll."""
    global _download_status
    _download_status = {
        "active": error is None and phase != "complete",
        "phase": phase,
        "detail": detail,
        "progress_pct": progress_pct,
        "error": error,
    }


def get_download_status() -> Dict[str, Any]:
    """Returns the current download/loading status for UI polling."""
    return dict(_download_status)


def is_loading() -> bool:
    """Returns True if a model is currently being loaded in the background."""
    return _load_thread is not None and _load_thread.is_alive()


def get_available_voices() -> List[str]:
    """Returns the named voice list for the currently loaded model (for UI dropdown)."""
    if loaded_model_selector and loaded_model_selector in MODEL_REGISTRY:
        return list(MODEL_REGISTRY[loaded_model_selector]["voices"])
    # Fallback
    return list(_V01_VOICES)


def get_all_accepted_voices() -> List[str]:
    """Returns all accepted voice identifiers (named + internal) for API validation."""
    named = get_available_voices()
    # Also accept raw internal voice keys (expr-voice-*) for API compatibility
    if voices_data is not None:
        internal = list(voices_data.keys())
        return list(dict.fromkeys(named + internal))  # deduplicated, order preserved
    return named


def get_onnx_provider_info() -> Dict[str, Any]:
    """Returns ONNX Runtime provider information for the active session."""
    active_providers: List[str] = []
    if onnx_session is not None:
        try:
            active_providers = list(onnx_session.get_providers())
        except Exception as e:
            logger.warning(f"Failed to read active ONNX providers: {e}")

    return {
        "active": active_providers[0] if active_providers else None,
        "active_providers": active_providers,
        "available_providers": list(ort.get_available_providers()),
    }


def get_model_info() -> Dict[str, Any]:
    """
    Returns information about the currently loaded model.
    Used by the API to expose model details to the UI.
    """
    if loaded_model_selector and loaded_model_selector in MODEL_REGISTRY:
        reg = MODEL_REGISTRY[loaded_model_selector]
        return {
            "loaded": MODEL_LOADED,
            "selector": loaded_model_selector,
            "repo_id": loaded_model_repo_id,
            "display_name": reg["display_name"],
            "voices": reg["voices"],
            "default_voice": reg["default_voice"],
            "params": reg["params"],
            "size": reg["size"],
            "version": reg["version"],
            "device": loaded_model_device,
        }
    return {
        "loaded": MODEL_LOADED,
        "selector": loaded_model_selector,
        "repo_id": loaded_model_repo_id,
        "display_name": None,
        "voices": list(_V01_VOICES),
        "default_voice": "expr-voice-5-m",
        "params": None,
        "size": None,
        "version": None,
        "device": loaded_model_device,
    }


def get_default_voice() -> str:
    """Returns the production default voice for the currently loaded model."""
    if loaded_model_selector and loaded_model_selector in MODEL_REGISTRY:
        return MODEL_REGISTRY[loaded_model_selector]["default_voice"]
    return "expr-voice-5-m"


def get_warmup_info() -> Dict[str, Any]:
    """Returns the current module-level warmup state."""
    return {
        "completed": WARMUP_COMPLETED,
        "failed": WARMUP_FAILED,
        "duration_seconds": WARMUP_DURATION_SECONDS,
    }


def get_model_registry() -> Dict[str, Dict[str, Any]]:
    """Returns the full model registry for the UI dropdown."""
    return {k: {
        "display_name": v["display_name"],
        "voices": v["voices"],
        "default_voice": v["default_voice"],
        "params": v["params"],
        "size": v["size"],
        "version": v["version"],
    } for k, v in MODEL_REGISTRY.items()}


class TextCleaner:
    """Text cleaner for KittenTTS - converts text to token indices."""

    def __init__(self):
        _pad = "$"
        _punctuation = ';:,.!?¡¿—…"«»"" '
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

        symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

        self.word_index_dictionary = {}
        for i in range(len(symbols)):
            self.word_index_dictionary[symbols[i]] = i

    def __call__(self, text: str):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                pass
        return indexes


def basic_english_tokenize(text: str):
    """Basic English tokenizer that splits on whitespace and punctuation."""
    import re

    tokens = re.findall(r"\w+|[^\w\s]", text)
    return tokens


def _init_espeak():
    """Initialize eSpeak for phonemization. Only needs to be done once."""
    global phonemizer_backend, text_cleaner

    if phonemizer_backend is not None:
        return  # Already initialized

    # --- Cross-Platform eSpeak Configuration ---
    if os.name == "nt":  # Windows
        logger.info("Checking for eSpeak NG on Windows...")
        possible_paths = [
            Path(r"C:\Program Files\eSpeak NG"),
            Path(r"C:\Program Files (x86)\eSpeak NG"),
            Path(r"C:\eSpeak NG"),
            Path(os.environ.get("ProgramFiles", "")) / "eSpeak NG",
            Path(os.environ.get("ProgramFiles(x86)", "")) / "eSpeak NG",
        ]
        espeak_found = False
        for espeak_path in possible_paths:
            if espeak_path.exists():
                dll_path = espeak_path / "libespeak-ng.dll"
                if dll_path.exists():
                    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(dll_path)
                    from phonemizer.backend.espeak.wrapper import (
                        EspeakWrapper as PhonemizeEspeakWrapper,
                    )
                    PhonemizeEspeakWrapper.set_library(str(dll_path))
                    logger.info(f"Auto-configured eSpeak from: {espeak_path}")
                    espeak_found = True
                    break
        if not espeak_found:
            logger.warning("eSpeak NG not found in common Windows locations.")

    elif os.name == "posix":  # Linux/macOS
        logger.info("Checking for system-installed eSpeak NG on POSIX...")
        candidate_paths = [
            "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
            "/usr/lib/x86_64-linux-gnu/libespeak-ng.so",
            "/usr/lib/libespeak-ng.so.1",
            "/usr/lib/libespeak-ng.so",
        ]
        espeak_lib_path = next(
            (path for path in candidate_paths if Path(path).exists()),
            None,
        )

        if espeak_lib_path is None:
            espeak_lib_path = ctypes.util.find_library("espeak-ng")

        if espeak_lib_path:
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeak_lib_path
            from phonemizer.backend.espeak.wrapper import (
                EspeakWrapper as PhonemizeEspeakWrapper,
            )

            if hasattr(PhonemizeEspeakWrapper, "set_library"):
                PhonemizeEspeakWrapper.set_library(espeak_lib_path)

            logger.info(f"Found and configured eSpeak NG library: {espeak_lib_path}")
        else:
            raise RuntimeError(
                "Could not find the eSpeak NG shared library. Please install both "
                "'espeak-ng' and 'libespeak-ng1' using your package manager."
            )

    # Initialize phonemizer
    try:
        import logging as log_module
        phonemizer_logger = log_module.getLogger("phonemizer")
        original_level = phonemizer_logger.level
        phonemizer_logger.setLevel(log_module.ERROR)

        phonemizer_backend = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )

        phonemizer_logger.setLevel(original_level)
        logger.info("Phonemizer backend initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize phonemizer: {e}")
        logger.error(
            "Please ensure eSpeak NG is installed:\n"
            "  Windows: Download from https://github.com/espeak-ng/espeak-ng/releases\n"
            "  Linux: Run 'sudo apt install espeak-ng'"
        )
        raise

    # Initialize text cleaner
    text_cleaner = TextCleaner()


def _is_truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _repo_baked_model_dir(repo_id: str) -> Path:
    baked_root = Path(os.environ.get("KITTEN_TTS_BAKED_MODEL_ROOT", "/app/model_cache/baked"))
    return baked_root / repo_id.rstrip("/").split("/")[-1]


def _load_baked_model_files(repo_id: str) -> Optional[Tuple[Path, Path, Path, Dict[str, Any]]]:
    baked_dir = _repo_baked_model_dir(repo_id)
    config_path = baked_dir / "config.json"
    if not config_path.exists():
        logger.info(f"No baked model config found at {config_path}; using Hugging Face fallback.")
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    model_filename = model_config.get("model_file")
    voices_filename = model_config.get("voices")
    if not model_filename or not voices_filename:
        raise RuntimeError(f"Baked model config at {config_path} is missing model_file or voices.")

    model_path = baked_dir / model_filename
    voices_path = baked_dir / voices_filename
    missing = [str(path) for path in (model_path, voices_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Baked model directory is incomplete. Missing file(s): " + ", ".join(missing)
        )

    logger.info(f"Loading KittenTTS model files from baked local path: {baked_dir}")
    return config_path, model_path, voices_path, model_config

def load_model() -> bool:
    """
    Loads the KittenTTS model from Hugging Face Hub and initializes ONNX session.
    Automatically downloads the model if not already cached.
    Updates global variables for model components.

    Returns:
        bool: True if the model was loaded successfully, False otherwise.
    """
    global onnx_session, voices_data, MODEL_LOADED
    global WARMUP_COMPLETED, WARMUP_FAILED, WARMUP_DURATION_SECONDS
    global loaded_model_selector, loaded_model_repo_id, loaded_model_device

    if MODEL_LOADED:
        logger.info("KittenTTS model is already loaded.")
        return True

    try:
        # Resolve the model selector from config
        config_value = config_manager.get_string(
            "model.repo_id", "KittenML/kitten-tts-nano-0.1"
        )
        selector = resolve_selector(config_value)
        reg = MODEL_REGISTRY[selector]
        model_repo_id = reg["repo_id"]

        model_cache_path = config_manager.get_path(
            "paths.model_cache", "./model_cache", ensure_absolute=True
        )

        logger.info(f"Loading KittenTTS model: {reg['display_name']} ({model_repo_id})")
        logger.info(f"Using cache directory: {model_cache_path}")

        # Ensure cache directory exists
        model_cache_path.mkdir(parents=True, exist_ok=True)

        remote_fallback_disabled = _is_truthy_env("KITTEN_TTS_DISABLE_HF_FALLBACK")

        try:
            baked_files = _load_baked_model_files(model_repo_id)
        except Exception as baked_error:
            if remote_fallback_disabled:
                raise RuntimeError(
                    "Baked local model files are missing or invalid, and Hugging Face fallback "
                    "is disabled with KITTEN_TTS_DISABLE_HF_FALLBACK. "
                    f"Expected baked directory: {_repo_baked_model_dir(model_repo_id)}. "
                    f"Details: {baked_error}"
                ) from baked_error
            logger.warning(
                "Baked local model files are missing or invalid; falling back to Hugging Face. "
                f"Details: {baked_error}"
            )
            baked_files = None

        if baked_files is not None:
            config_path, model_path, voices_path, model_config = baked_files
            _update_download_status("loading", "Loading baked local model files...", 60)
        else:
            if remote_fallback_disabled:
                raise RuntimeError(
                    "No baked local model files were found and Hugging Face fallback is disabled "
                    "with KITTEN_TTS_DISABLE_HF_FALLBACK. "
                    f"Expected baked directory: {_repo_baked_model_dir(model_repo_id)}."
                )

            logger.info("Loading KittenTTS model files from Hugging Face fallback.")

            # Phase 1: Download config.json
            _check_cancelled()
            _update_download_status("downloading", f"Downloading config for {reg['display_name']}...", 10)

            config_path = hf_hub_download(
                repo_id=model_repo_id,
                filename="config.json",
                cache_dir=str(model_cache_path),
            )

            # Load config to get model filenames
            with open(config_path, "r", encoding="utf-8") as f:
                model_config = json.load(f)

            # Phase 2: Download model file (can be large - check cancellation)
            _check_cancelled()
            model_filename = model_config["model_file"]
            _update_download_status(
                "downloading",
                f"Downloading model file ({reg['size']})...",
                30,
            )

            model_path = hf_hub_download(
                repo_id=model_repo_id,
                filename=model_filename,
                cache_dir=str(model_cache_path),
            )

            # Phase 3: Download voices file
            _check_cancelled()
            _update_download_status("downloading", "Downloading voice data...", 60)

            voices_path = hf_hub_download(
                repo_id=model_repo_id,
                filename=model_config["voices"],
                cache_dir=str(model_cache_path),
            )

        model_type = model_config.get("type", "")
        if model_type not in ("ONNX1", "ONNX2"):
            raise ValueError(f"Unsupported model type '{model_type}'. Expected ONNX1 or ONNX2.")

        # Phase 4: Load voices data and voice aliases
        _update_download_status("loading", "Loading voice embeddings...", 70)
        voices_data = np.load(voices_path)
        logger.info(f"Loaded voices data with keys: {list(voices_data.keys())}")

        # Load voice aliases - v0.8 models have these in config.json,
        # v0.1/v0.2 models use our own named aliases
        global _voice_aliases, _speed_priors
        _voice_aliases = {}
        _speed_priors = {}
        if "voice_aliases" in model_config:
            _voice_aliases = dict(model_config["voice_aliases"])
            logger.info(f"Loaded voice aliases from model config: {_voice_aliases}")
        elif reg["version"] in ("0.1", "0.2"):
            _voice_aliases = dict(_V01_VOICE_ALIASES)
            logger.info(f"Applied v0.1 named voice aliases: {_voice_aliases}")

        # Load speed priors (per-voice speed correction factors from model config)
        if "speed_priors" in model_config:
            _speed_priors = dict(model_config["speed_priors"])
            logger.info(f"Loaded speed priors: {_speed_priors}")

        # Phase 5: Initialize ONNX session
        _check_cancelled()
        _update_download_status("loading", "Initializing ONNX inference session...", 80)

        device_setting = config_manager.get_string("tts_engine.device", "auto").lower()
        available_providers = ort.get_available_providers()
        logger.info(f"Available ONNX Runtime providers: {available_providers}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        intra_op_threads_raw = os.getenv("ORT_INTRA_OP_NUM_THREADS", "1")
        try:
            intra_op_threads = int(intra_op_threads_raw)
            if intra_op_threads < 1:
                raise ValueError
        except ValueError:
            logger.warning(
                "Invalid ORT_INTRA_OP_NUM_THREADS value '%s'; defaulting to 1.",
                intra_op_threads_raw,
            )
            intra_op_threads = 1
        sess_options.intra_op_num_threads = intra_op_threads
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        providers = []
        provider_options = []

        attempt_gpu = device_setting in ["auto", "cuda", "gpu"]
        is_gpu_available = "CUDAExecutionProvider" in available_providers

        if attempt_gpu and is_gpu_available:
            logger.info(f"'{device_setting}' mode selected and CUDAExecutionProvider is available.")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options = [{"device_id": "0"}, {}]
            loaded_model_device = "cuda"
        else:
            if device_setting in ["cuda", "gpu"] and not is_gpu_available:
                logger.warning(
                    f"Configuration explicitly requests GPU ('{device_setting}'), "
                    "but CUDAExecutionProvider is NOT available."
                )
            logger.info("Defaulting to CPUExecutionProvider.")
            providers = ["CPUExecutionProvider"]
            loaded_model_device = "cpu"

        logger.info(f"Selected ONNX Runtime providers: {providers}")
        logger.info(
            "ONNX Runtime graph optimization level: %s",
            sess_options.graph_optimization_level,
        )
        logger.info(
            "ONNX Runtime intra-op thread count: %s",
            sess_options.intra_op_num_threads,
        )
        logger.info("ONNX Runtime execution mode: %s", sess_options.execution_mode)
        logger.info(f"Initializing ONNX InferenceSession with providers: {providers}")

        if "CUDAExecutionProvider" in providers:
            onnx_session = ort.InferenceSession(
                str(model_path), sess_options,
                providers=providers, provider_options=provider_options,
            )
        else:
            onnx_session = ort.InferenceSession(
                str(model_path), sess_options, providers=providers,
            )

        # Phase 6: Initialize phonemizer (shared across models, only once)
        _check_cancelled()
        _update_download_status("loading", "Initializing phonemizer...", 90)
        _init_espeak()

        # Done
        loaded_model_selector = selector
        loaded_model_repo_id = model_repo_id
        MODEL_LOADED = True
        WARMUP_COMPLETED = False
        WARMUP_FAILED = False
        WARMUP_DURATION_SECONDS = None

        _update_download_status("complete", f"{reg['display_name']} loaded successfully!", 100)
        logger.info(f"KittenTTS model loaded successfully: {reg['display_name']}")
        return True

    except RuntimeError as e:
        if "cancelled" in str(e).lower():
            logger.info(f"Model loading cancelled: {e}")
            onnx_session = None
            voices_data = None
            MODEL_LOADED = False
            loaded_model_selector = None
            loaded_model_repo_id = None
            loaded_model_device = None
            _update_download_status("cancelled", "Model loading was cancelled.", 0)
            return False
        else:
            logger.error(f"Error loading KittenTTS model: {e}", exc_info=True)
            onnx_session = None
            voices_data = None
            MODEL_LOADED = False
            loaded_model_selector = None
            loaded_model_repo_id = None
            loaded_model_device = None
            _update_download_status("error", "", 0, str(e))
            raise RuntimeError(f"Failed to load KittenTTS model '{loaded_model_repo_id or 'unknown'}': {e}") from e
    except Exception as e:
        logger.error(f"Error loading KittenTTS model: {e}", exc_info=True)
        onnx_session = None
        voices_data = None
        MODEL_LOADED = False
        loaded_model_selector = None
        loaded_model_repo_id = None
        loaded_model_device = None
        _update_download_status("error", "", 0, str(e))
        raise RuntimeError(f"Failed to load KittenTTS model: {e}") from e


def unload_model() -> bool:
    """
    Unloads the current model and releases resources.
    Does NOT reload - use reload_model() for hot-swap.

    Returns:
        bool: True if the model was unloaded successfully.
    """
    global onnx_session, voices_data, MODEL_LOADED
    global WARMUP_COMPLETED, WARMUP_FAILED, WARMUP_DURATION_SECONDS
    global loaded_model_selector, loaded_model_repo_id, loaded_model_device
    global _voice_aliases, _speed_priors

    logger.info("Initiating model unload sequence...")

    if onnx_session is not None:
        logger.info("Unloading ONNX session from memory...")
        del onnx_session
        onnx_session = None

    if voices_data is not None:
        del voices_data
        voices_data = None

    MODEL_LOADED = False
    WARMUP_COMPLETED = False
    WARMUP_FAILED = False
    WARMUP_DURATION_SECONDS = None
    loaded_model_selector = None
    loaded_model_repo_id = None
    loaded_model_device = None
    _voice_aliases = {}
    _speed_priors = {}

    # Force garbage collection
    gc.collect()
    logger.info("Python garbage collection completed.")

    # ONNX Runtime owns GPU allocations for inference. Avoid importing PyTorch here so
    # CPU-only installations do not need torch just to unload the model.

    logger.info("Model unloaded and resources released.")
    return True


def _reload_model_worker():
    """
    Background worker that performs the actual model reload.
    Runs in a separate thread so the FastAPI server stays responsive.
    """
    try:
        _update_download_status("unloading", "Unloading current model...", 5)
        unload_model()

        _check_cancelled()
        logger.info("Resources cleared. Loading model from updated config...")
        load_model()
    except RuntimeError as e:
        if "cancelled" in str(e).lower():
            logger.info(f"Reload worker cancelled: {e}")
            _update_download_status("cancelled", "Model loading was cancelled.", 0)
        else:
            logger.error(f"Reload worker error: {e}", exc_info=True)
            _update_download_status("error", "", 0, str(e))
    except Exception as e:
        logger.error(f"Reload worker error: {e}", exc_info=True)
        _update_download_status("error", "", 0, str(e))


def reload_model_async():
    """
    Initiates a model hot-swap in a background thread.
    Returns immediately so the server can continue serving status polls.
    If a load is already in progress, cancels it first.
    """
    global _load_thread

    logger.info("Initiating async model hot-swap/reload sequence...")

    # Cancel any in-progress load
    if is_loading():
        logger.info("Cancelling in-progress model load...")
        _cancel_event.set()
        _load_thread.join(timeout=15)
        logger.info("Previous load thread finished.")

    # Reset cancel flag for new load
    _cancel_event.clear()

    # Start new background load
    _load_thread = threading.Thread(target=_reload_model_worker, daemon=True)
    _load_thread.start()
    logger.info("Background model reload thread started.")


def cancel_loading():
    """Cancels any in-progress model loading."""
    if is_loading():
        logger.info("Cancelling model loading by user request...")
        _cancel_event.set()
        _update_download_status("cancelling", "Cancelling model load...", 0)
        return True
    return False


def reload_model() -> bool:
    """
    Synchronous reload (used for startup only).
    Unloads current model, clears resources, reloads from config.
    """
    logger.info("Initiating synchronous model reload...")
    _cancel_event.clear()
    _update_download_status("unloading", "Unloading current model...", 5)
    unload_model()
    logger.info("Resources cleared. Loading model from updated config...")
    return load_model()


def synthesize(
    text: str, voice: str, speed: float = 1.0
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Synthesizes audio from text using the loaded KittenTTS model.

    Args:
        text: The text to synthesize.
        voice: Voice identifier (model-dependent).
        speed: Speech speed factor (1.0 is normal speed).

    Returns:
        A tuple containing the audio waveform (numpy array) and the sample rate (int),
        or (None, None) if synthesis fails.
    """
    global onnx_session, voices_data, phonemizer_backend, text_cleaner

    if not MODEL_LOADED or onnx_session is None:
        logger.error("KittenTTS model is not loaded. Cannot synthesize audio.")
        return None, None

    # Validate voice against all accepted identifiers (named + internal)
    accepted = get_all_accepted_voices()
    if voice not in accepted:
        logger.error(
            f"Voice '{voice}' not available for current model. Available voices: {accepted}"
        )
        return None, None

    # Resolve voice aliases (named voices -> internal expr-voice keys)
    internal_voice = _voice_aliases.get(voice, voice)
    if internal_voice != voice:
        logger.debug(f"Resolved voice alias '{voice}' -> '{internal_voice}'")

    # Apply speed prior for this voice if available (from model config)
    prior = _speed_priors.get(internal_voice, 1.0)
    if prior != 1.0:
        speed = speed * prior
        logger.debug(f"Applied speed prior {prior} for voice '{internal_voice}', effective speed: {speed}")

    try:
        logger.debug(f"Synthesizing with voice='{voice}' (internal='{internal_voice}'), speed={speed}")

        # Get voice embedding and ensure correct shape [1, 256]
        voice_embedding = voices_data[internal_voice]
        if voice_embedding.ndim == 2 and voice_embedding.shape[0] > 1:
            # ONNX2: multiple reference embeddings, select based on text length
            # for varied prosody across different text lengths
            text_length = len(text)
            ref_id = min(text_length, voice_embedding.shape[0] - 1)
            voice_embedding = voice_embedding[ref_id:ref_id + 1]
            logger.debug(f"Selected ONNX2 embedding row {ref_id} (text_len={text_length}), shape {voice_embedding.shape}")
        elif voice_embedding.ndim == 1:
            voice_embedding = voice_embedding.reshape(1, -1)
        voice_embedding = voice_embedding.astype(np.float32)
        logger.debug(f"Input text (first 100 chars): '{text[:100]}...'")

        # Phonemize the input text
        import logging as log_module
        phonemizer_logger = log_module.getLogger("phonemizer")
        original_level = phonemizer_logger.level
        phonemizer_logger.setLevel(log_module.ERROR)

        phonemes_list = phonemizer_backend.phonemize([text])
        phonemizer_logger.setLevel(original_level)

        # Process phonemes to get token IDs
        phonemes = basic_english_tokenize(phonemes_list[0])
        phonemes = " ".join(phonemes)
        tokens = text_cleaner(phonemes)

        # Add start and end tokens
        tokens.insert(0, 0)
        tokens.append(0)

        # Determine the execution device
        provider = onnx_session.get_providers()[0]

        if provider == "CUDAExecutionProvider":
            # GPU inference with I/O Binding
            input_ids_np = np.array([tokens], dtype=np.int64)
            ref_s_np = voice_embedding
            speed_array_np = np.array([speed], dtype=np.float32)

            input_ids_ort = ort.OrtValue.ortvalue_from_numpy(input_ids_np, "cuda", 0)
            ref_s_ort = ort.OrtValue.ortvalue_from_numpy(ref_s_np, "cuda", 0)
            speed_array_ort = ort.OrtValue.ortvalue_from_numpy(speed_array_np, "cuda", 0)

            io_binding = onnx_session.io_binding()
            io_binding.bind_ortvalue_input("input_ids", input_ids_ort)
            io_binding.bind_ortvalue_input("style", ref_s_ort)
            io_binding.bind_ortvalue_input("speed", speed_array_ort)

            output_name = onnx_session.get_outputs()[0].name
            io_binding.bind_output(output_name, "cuda")

            onnx_session.run_with_iobinding(io_binding)
            output_ortvalue = io_binding.get_outputs()[0]
            audio = output_ortvalue.numpy()

        else:
            # CPU inference
            input_ids = np.array([tokens], dtype=np.int64)
            ref_s = voice_embedding
            speed_array = np.array([speed], dtype=np.float32)

            onnx_inputs = {
                "input_ids": input_ids,
                "style": ref_s,
                "speed": speed_array,
            }
            outputs = onnx_session.run(None, onnx_inputs)
            audio = outputs[0]

        sample_rate = 24000

        logger.info(
            f"Successfully generated {len(audio)} audio samples at {sample_rate}Hz"
        )
        return audio, sample_rate

    except Exception as e:
        logger.error(f"Error during KittenTTS synthesis: {e}", exc_info=True)
        return None, None


def warmup(text: str = "Hello", voice: Optional[str] = None) -> bool:
    """
    Warms up the loaded TTS model by synthesizing audio without saving it.

    Args:
        text: Text to synthesize for warmup.
        voice: Voice identifier to use. Defaults to the loaded model's production
            default voice.

    Returns:
        True if warmup synthesis completed successfully, False otherwise.
    """
    global WARMUP_COMPLETED, WARMUP_FAILED, WARMUP_DURATION_SECONDS

    selected_voice = voice or get_default_voice()
    logger.info("Starting KittenTTS warmup synthesis.")
    logger.info(f"Warmup selected voice: {selected_voice}")

    WARMUP_COMPLETED = False
    WARMUP_FAILED = False
    WARMUP_DURATION_SECONDS = None

    start_time = time.perf_counter()
    try:
        audio, sample_rate = synthesize(text=text, voice=selected_voice)
        WARMUP_DURATION_SECONDS = time.perf_counter() - start_time

        if audio is None or sample_rate is None:
            raise RuntimeError("Warmup synthesis returned no audio.")

        WARMUP_COMPLETED = True
        WARMUP_FAILED = False
        logger.info(
            f"KittenTTS warmup completed successfully in {WARMUP_DURATION_SECONDS:.3f}s."
        )
        return True
    except Exception as e:
        WARMUP_DURATION_SECONDS = time.perf_counter() - start_time
        WARMUP_COMPLETED = False
        WARMUP_FAILED = True
        logger.error(
            f"KittenTTS warmup failed after {WARMUP_DURATION_SECONDS:.3f}s: {e}",
            exc_info=True,
        )
        return False


# --- End File: engine.py ---
