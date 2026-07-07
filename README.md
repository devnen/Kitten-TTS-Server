# Kitten TTS Server: High-Performance, Lightweight TTS with API and GPU Acceleration

**Self-host the ultra-lightweight [KittenTTS model](https://github.com/KittenML/KittenTTS) with this enhanced API server. Now supports all 7 KittenTTS models — from the tiny 15M Nano to the 80M Mini — with hot-swappable model switching, an intuitive Web UI, a flexible API, large text processing for audiobooks, and high-performance GPU acceleration.**

This server provides a robust, user-friendly, and powerful interface for the KittenTTS engine family, an open-source, realistic text-to-speech system. This project significantly enhances the original model by adding a full-featured server, an easy-to-use UI, and an optimized inference pipeline for hardware ranging from NVIDIA GPUs to CPUs and even the Raspberry Pi 5.

[![Project Link](https://img.shields.io/badge/GitHub-devnen/Kitten--TTS--Server-blue?style=for-the-badge&logo=github)](https://github.com/devnen/Kitten-TTS-Server)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-green.svg?style=for-the-badge)](https://fastapi.tiangolo.com/)
[![Model Source](https://img.shields.io/badge/Model-KittenML/KittenTTS-orange.svg?style=for-the-badge)](https://github.com/KittenML/KittenTTS)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg?style=for-the-badge)](https://www.docker.com/)
[![Web UI](https://img.shields.io/badge/Web_UI-Included-4285F4?style=for-the-badge&logo=googlechrome&logoColor=white)](#)
[![CUDA Compatible](https://img.shields.io/badge/NVIDIA_CUDA-Compatible-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![API](https://img.shields.io/badge/OpenAI_Compatible_API-Ready-000000?style=for-the-badge&logo=openai&logoColor=white)](https://platform.openai.com/docs/api-reference)

<div align="center">
  <img src="static/screenshot-d.png" alt="Kitten TTS Server Web UI - Dark Mode" width="33%" />
  <img src="static/screenshot-l.png" alt="Kitten TTS Server Web UI - Light Mode" width="33%" />
</div>

---

## 🆕 What's New

### 🎯 Complete KittenTTS model family support (new)

- Added full support for **all 7 KittenTTS models** across three model sizes (Nano, Micro, Mini) and two generations (v0.1/v0.2 and v0.8).
- Models range from the ultra-compact **15M-parameter Nano** to the high-quality **80M-parameter Mini**, all running on ONNX for maximum portability.
- v0.8 models feature improved expressivity, quantized INT8 variants for minimal footprint, and named voices (Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo).

### 🔁 Hot-swappable model switching (new)

- Added a new **model selector** dropdown at the top of the Web UI.
- All 7 models are **hot-swappable** — select from the dropdown, click "Apply & Restart", and the backend automatically downloads (if needed), unloads the current model, and loads your choice. No server restart required.
- A **progress modal** with real-time status updates shows download and loading progress, so you always know what's happening.
- Models are **downloaded automatically** from Hugging Face on first use and cached locally in the project's `model_cache` directory for instant subsequent loads.
- **Cancellation support** — if you change your mind during a download, select a different model and the current load is cancelled automatically.

### 🎤 Named voices for all models (new)

- All models now use **human-friendly voice names** instead of technical identifiers.
- v0.1/v0.2 models: **Amber**, **Felix**, **Clara**, **Marcus**, **Ivy**, **Oscar**, **Nora**, **Reed** (4 female, 4 male).
- v0.8 models: **Bella**, **Jasper**, **Luna**, **Bruno**, **Rosie**, **Hugo**, **Kiki**, **Leo** (4 female, 4 male).
- The voice dropdown automatically updates when you switch models.

### ✅ Complete model lineup

**You now have access to the entire KittenTTS family:**

| Model | Parameters | Size | Voices | Notes |
|-------|-----------|------|--------|-------|
| **Nano 0.1** | 15M | <25MB | Amber, Felix, Clara, Marcus, Ivy, Oscar, Nora, Reed | Original release |
| **Nano 0.2** | 15M | <25MB | Amber, Felix, Clara, Marcus, Ivy, Oscar, Nora, Reed | Developer preview |
| **Mini 0.1** | 80M | ~170MB | Amber, Felix, Clara, Marcus, Ivy, Oscar, Nora, Reed | Highest quality (v0.1) |
| **Nano 0.8 INT8** | 15M | ~25MB | Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo | Quantized, SOTA expressivity |
| **Nano 0.8 FP32** | 15M | ~50MB | Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo | Full precision |
| **Micro 0.8** | 40M | ~40MB | Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo | Balanced quality/speed |
| **Mini 0.8** | 80M | ~79MB | Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo | Highest quality |

**Switching models is effortless:** Simply select your preferred model from the dropdown at the top of the Web UI. The server handles downloading, loading, and voice list updates automatically — just click and go.

---

## 🗣️ Overview: Enhanced KittenTTS Generation

The [KittenTTS model by KittenML](https://github.com/KittenML/KittenTTS) provides a foundation for generating high-quality speech from models smaller than 80MB. This project elevates that foundation into a production-ready service by providing a robust [FastAPI](https://fastapi.tiangolo.com/) server that makes KittenTTS significantly easier to use, more powerful, and drastically faster.

We solve the complexity of setting up and running the model by offering:

*   A **modern Web UI** for easy experimentation, preset loading, and speed adjustment.
*   **Hot-swappable model switching** between all 7 KittenTTS models with automatic download.
*   **True GPU Acceleration** for NVIDIA GPUs, a feature not present in the original implementation.
*   **Large Text Handling & Audiobook Generation:** Intelligently splits long texts into manageable chunks, processes them sequentially, and seamlessly concatenates the audio. Perfect for creating complete audiobooks.
*   **A flexible, dual-API system** including a simple endpoint and an OpenAI-compatible endpoint for easy integration.
*   **Named Voices:** Up to 8 named voices per model for consistent and reliable output.
*   **Cross-platform support** for Windows and Linux, with clear setup instructions.
*   **Docker support** for easy, reproducible containerized deployment.

## 🍓 Raspberry Pi & Edge Device Support

The ultra-lightweight nature of the KittenTTS model and the efficiency of this server make it a perfect candidate for running on single-board computers (SBCs) and other edge devices.

*   ✅ **Raspberry Pi 5 (RP5):** Confirmed to run with **excellent performance**. The server is fast and responsive, easily handling requests from other devices on the same local network (LAN). This makes it ideal for local network services, home automation, and other DIY projects.

To install, simply follow the standard **Linux installation guide** provided in this README.

## 🔥 GPU Acceleration included

A standout feature of this server is the implementation of **high-performance GPU acceleration**, a capability not available in the original KittenTTS project. While the base model is CPU-only, this server unlocks the full potential of your hardware.

*   **Optimized ONNX Runtime Pipeline:** We leverage `onnxruntime-gpu` to move the entire inference process to your NVIDIA graphics card.
*   **Eliminated I/O Bottlenecks:** The server uses advanced **I/O Binding**. This technique pre-allocates memory directly on the GPU for both model inputs and outputs, drastically reducing the latency caused by copying data between system RAM and the GPU's VRAM.
*   **True Performance Gains:** This isn't just running the model on the GPU; it's an optimized pipeline designed to minimize latency and maximize throughput, making real-time generation significantly faster than on CPU.

This enhancement transforms KittenTTS from a lightweight-but-modest engine into a high-speed synthesis powerhouse.

## 🔄 Alternative to Piper TTS

The [KittenTTS model](https://github.com/KittenML/KittenTTS) serves as an excellent alternative to [Piper TTS](https://github.com/rhasspy/piper) for fast generation on limited compute and edge devices like Raspberry Pi 5.

**KittenTTS Model Advantages:**
- **Extreme Efficiency**: 15M to 80M parameters, with the smallest models under 25MB — significantly smaller than most Piper models
- **Universal Compatibility**: CPU-optimized to run without GPU on any device and "works literally everywhere"
- **Real-time Performance**: Optimized for real-time speech synthesis even on resource-constrained hardware
- **Multiple Model Sizes**: Choose the right balance of quality vs speed for your use case

**This Server Project's Enhancement:**
While KittenTTS provides the ultra-lightweight foundation, this server transforms it into a production-ready Piper replacement by adding GPU acceleration (unavailable in the base model), hot-swappable multi-model support, modern REST/OpenAI APIs, audiobook processing capabilities, and an intuitive web interface — all while maintaining the model's edge device compatibility.

Perfect for users seeking Piper's offline capabilities with better performance on limited hardware and modern server infrastructure.

## ✨ Key Features of This Server

*   🔁 **Multi-Model Support:** All 7 KittenTTS models (Nano, Micro, Mini across v0.1/v0.2/v0.8) with hot-swappable switching from the UI.
*   📦 **Automatic Model Download:** Models are downloaded from Hugging Face on first use and cached locally.
*   ⚡ **True GPU Acceleration:** Full support for **NVIDIA (CUDA)** via an optimized `onnxruntime-gpu` pipeline with I/O Binding for maximum performance.
*   📚 **Large Text & Audiobook Generation:**
    *   Automatically handles long texts by intelligently splitting them based on sentence boundaries.
    *   Processes each chunk individually and seamlessly concatenates the resulting audio.
    *   **Ideal for audiobooks** - paste entire books and get professional-quality audio.
*   🖥️ **Modern Web Interface:**
    *   Intuitive UI for text input, model selection, voice selection, and parameter adjustment.
    *   Real-time waveform visualization of generated audio.
    *   Progress modal for model downloads with real-time status updates.
*   🎤 **Named Voices:**
    *   Up to 8 named voices per model (4 male, 4 female).
    *   Voice list updates automatically when switching models.
*   ⚙️ **Dual API Endpoints:**
    *   A primary `/tts` endpoint offering full control over all generation parameters.
    *   An OpenAI-compatible `/v1/audio/speech` endpoint for seamless integration into existing workflows.
*   🔧 **Easy Configuration:**
    *   All settings are managed through a single `config.yaml` file.
    *   The server automatically creates a default config on the first run.
*   💾 **UI State Persistence:** The web interface remembers your last-used text, voice, and settings to streamline your workflow.
*   🐳 **Docker Support:** Easy, reproducible deployment for both CPU and GPU via Docker Compose.

---

## 🔩 System Prerequisites

*   **Operating System:** Windows 10/11 (64-bit) or Linux (Debian/Ubuntu recommended).
*   **Python:** Version 3.10 or later.
*   **Git:** For cloning the repository.
*   **eSpeak NG:** This is a **required** dependency for text phonemization.
    *   **Windows:** See installation guide below.
    *   **Linux:** `sudo apt install espeak-ng`
*   **Raspberry Pi:**
    *   Raspberry Pi 5
*   **(For GPU Acceleration):**
    *   An **NVIDIA GPU** with CUDA support.
*   **(For Linux Only):**
    *   `libsndfile1`: Audio library needed by `soundfile`. Install via `sudo apt install libsndfile1`.
    *   `ffmpeg`: For robust audio operations. Install via `sudo apt install ffmpeg`.

## 💻 Installation and Setup

This project uses specific dependency files and a clear process to ensure a smooth, one-command installation for your hardware.

**1. Clone the Repository**
```bash
git clone https://github.com/devnen/Kitten-TTS-Server.git
cd Kitten-TTS-Server
```

**2. Create and Activate a Python Virtual Environment**
This is crucial to avoid conflicts with other Python projects.

*   **Windows (PowerShell):**
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    ```
    If you see an error about execution policies, run:
    `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` and try activating again.

*   **Linux (Bash):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    Your command prompt should now start with `(venv)`.

**3. Install eSpeak NG (Required)**

*   **Windows:**
    1.  Download the installer from the [eSpeak NG Releases page](https://github.com/espeak-ng/espeak-ng/releases/latest). Look for the file named `espeak-ng-X.XX-x64.msi`.
    2.  Run the installer with default settings.
    3.  **Important:** Restart your terminal (PowerShell/CMD) after installation for the changes to take effect.

*   **Linux (Ubuntu/Debian):**
    ```bash
    sudo apt update && sudo apt install -y espeak-ng
    ```

**4. Install Python Dependencies**

Choose one of the following paths based on your hardware.

---

### **Option 1: CPU-Only Installation**
This is the simplest path and works on any machine.

```bash
# Make sure your (venv) is active
pip install --upgrade pip
pip install -r requirements.txt
```

---

### **Option 2: NVIDIA GPU Installation (Recommended for Performance)**
This method ensures all necessary CUDA libraries are correctly installed within your virtual environment for a hassle-free setup.

```bash
# Make sure your (venv) is active
pip install --upgrade pip

# Step 1: Install the GPU-enabled ONNX Runtime
pip install onnxruntime-gpu

# Step 2: Install PyTorch with CUDA support. This command also brings the
# necessary CUDA and cuDNN .dll files that onnxruntime-gpu needs.
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 3: Install the remaining dependencies from the requirements file
pip install -r requirements-nvidia.txt
```

**After installation, verify that PyTorch can see your GPU:**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```
If `CUDA available:` shows `True`, your setup is correct!

---

### **Option 3: Upgrading from CPU to GPU**

If you initially installed the server for CPU-only usage and now want to enable GPU acceleration, follow these steps to upgrade your environment safely.

```bash
# Make sure your (venv) is active
pip install --upgrade pip

# Step 1: Uninstall the CPU-only versions of onnxruntime and torch.
# This is critical to prevent conflicts with the GPU packages.
pip uninstall onnxruntime torch torchaudio -y

# Step 2: Install the GPU-enabled ONNX Runtime.
pip install onnxruntime-gpu

# Step 3: Install PyTorch with CUDA support. This command also brings the
# necessary CUDA and cuDNN .dll files that onnxruntime-gpu needs.
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 4: Re-install from the nvidia requirements file to ensure all other
# dependencies are correct and up to date.
pip install -r requirements-nvidia.txt
```

**After upgrading, do the following:**

1.  **Verify the installation** by running the same check from Option 2:
    ```bash
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    ```
    The output must be `CUDA available: True`.

2.  **Update your configuration** by editing the `config.yaml` file:
    ```yaml
    tts_engine:
      device: auto  # Or "cuda", or "gpu"
    ```

3.  **Restart the server** for the changes to take effect. It will now use your NVIDIA GPU.

---

## ▶️ Running the Server

**Important: First-Run Model Download**
The first time you start the server, it will automatically download the default KittenTTS Nano model (~25MB) from Hugging Face. This is a one-time process. Subsequent launches will be instant. Additional models are downloaded automatically when selected from the Web UI.

1.  **Activate the virtual environment** (if not already active).
    *   Windows: `.\venv\Scripts\activate`
    *   Linux: `source venv/bin/activate`

2.  **Run the server:**
    ```bash
    python server.py
    ```

3.  Open the Web UI in your browser. Browser auto-open is disabled by default so hosted or containerized deployments do not launch a browser. For local UI development, set `OPEN_BROWSER=true` or `server.open_browser: true`.
    *   **Web UI:** `http://localhost:8005`
    *   **API Docs:** `http://localhost:8005/docs`

4.  **To stop the server:** Press `CTRL+C` in the terminal.

### **Raspberry Pi 5 Installation (CPU-Only)**

KittenTTS runs excellently on Raspberry Pi 5, making it ideal for local network services and DIY projects.

#### **Raspberry Pi 5 - Full Support ✅**

**Raspberry Pi 5 works out-of-the-box** with the standard Linux installation guide above. No special steps required!

**Tested Configuration:**
- **Hardware:** Raspberry Pi 5 Model B Rev 1.0
- **OS:** Debian GNU/Linux 12 (bookworm) 64-bit
- **Architecture:** aarch64 (ARM64)
- **Python:** 3.11
- **Memory:** 4GB RAM
- **Installation:** Follow the standard Linux installation guide exactly

**Installation Steps:**
```bash
# Step 1: Install system dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install -y espeak-ng libsndfile1 ffmpeg python3-pip python3-venv git

# Step 2: Set up Python environment
python -m venv venv
source venv/bin/activate

# Step 3: Install Python dependencies
pip install -r requirements.txt

# Step 4: Start the server
python server.py
```

> **Important:** During the `pip install -r requirements.txt` step, some Python packages (especially audio processing libraries like `librosa`, `praat-parselmouth`, and others) may need to be compiled from source on ARM architecture. This process can take **15-30 minutes** depending on your SD card speed and system load. This is normal - let it complete without interruption.

## 🐳 Docker Installation

Run Kitten-TTS-Server easily using Docker. The recommended method uses Docker Compose, which is pre-configured for both CPU and NVIDIA GPU deployment.

### Prerequisites

*   [Docker](https://docs.docker.com/get-docker/) installed.
*   [Docker Compose](https://docs.docker.com/compose/install/) installed (usually included with Docker Desktop).
*   **(For GPU acceleration)**
    *   An NVIDIA GPU.
    *   Up-to-date NVIDIA drivers for your host operating system.
    *   The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.

### Using Docker Compose (Recommended)

This method uses the provided `docker-compose.yml` files to automatically build the correct image and manage the container, volumes, and configuration.

**1. Clone the Repository**
```bash
git clone https://github.com/devnen/Kitten-TTS-Server.git
cd Kitten-TTS-Server
```

**2. Start the Container Based on Your Hardware**

Choose one of the following commands:

#### **For NVIDIA GPU (Recommended for Performance):**
The default `docker-compose.yml` is configured for NVIDIA GPUs. It will build the image with full CUDA support.

```bash
docker compose up -d --build
```

#### **For CPU-only:**
This uses a dedicated compose file that builds the image without GPU dependencies.

```bash
docker compose -f docker-compose-cpu.yml up -d --build
```

**Note:** The first time you run this, Docker will build the image and the server will download the KittenTTS model, which can take a few minutes. Subsequent starts will be much faster.

### 3. Access and Manage the Application

*   **Access the Web UI:** Open your browser to `http://localhost:8005`
*   **Access the API Docs:** `http://localhost:8005/docs`

*   **View Logs:**
    ```bash
    # For GPU or CPU version
    docker compose logs -f
    ```

*   **Stop the Container:**
    ```bash
    # This stops and removes the container but keeps your data volumes
    docker compose down
    ```

### How It Works

*   **Build-time Argument:** The `Dockerfile` uses a `RUNTIME` argument (`nvidia` or `cpu`) to conditionally install the correct Python packages, creating an optimized image for your hardware.
*   **Persistent Data:** The `docker-compose` files use Docker volumes to persist your important data on your host machine, even if the container is removed:
    *   `./config.yaml`: Your main server configuration file.
    *   `./outputs`: All generated audio files are saved here.
    *   `./logs`: Server log files for troubleshooting.
    *   `hf_cache` (Named Volume): Persists the downloaded Hugging Face models, saving significant time on rebuilds.

### Verify GPU Access (for NVIDIA users)

After starting the GPU container, you can verify that Docker and the application can see your graphics card.

```bash
# Check if the container can see the NVIDIA GPU
docker compose exec kitten-tts-server nvidia-smi

# Check if PyTorch inside the container can access CUDA
docker compose exec kitten-tts-server python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```
If `CUDA available:` prints `True`, your GPU setup is working correctly

## 💡 Usage Guide

### Generate Your First Audio

1.  Start the server and open the Web UI (`http://localhost:8005`).
2.  Select a model from the dropdown at the top (or use the default Nano 0.1).
3.  Select a voice from the voice dropdown.
4.  Type or paste your text into the input box.
5.  Adjust the speech speed if desired.
6.  Click **"Generate Speech"**.
7.  The audio will play automatically and be available for download.

### Switch Models

1.  Select a different model from the **Active Model** dropdown at the top of the page.
2.  Click **"Apply & Restart"**.
3.  A progress modal will show the download and loading status.
4.  Once complete, the voice dropdown updates automatically with the new model's voices.

### Generate an Audiobook

1.  Copy the entire plain text of your book or chapter.
2.  Paste it into the text area.
3.  Ensure **"Split text into chunks"** is enabled.
4.  Set a **Chunk Size** between 300 and 500 characters for natural pauses.
5.  Click **"Generate Speech"**. The server will process the entire text and stitch the audio together seamlessly.
6.  Download your complete audiobook file.

## 📖 API Documentation

The server exposes two main endpoints for TTS. See `http://localhost:8005/docs` for an interactive playground.

### Primary Endpoint: `/tts`

This endpoint offers the most control.

*   **Method:** `POST`
*   **Body:**
    ```json
    {
      "text": "Hello from the KittenTTS API!",
      "voice": "Reed",
      "speed": 1.0,
      "output_format": "mp3",
      "split_text": true,
      "chunk_size": 300
    }
    ```
*   **Response:** Streaming audio file (`audio/wav`, `audio/mp3`, etc.).

### OpenAI-Compatible Endpoint: `/v1/audio/speech`

Use this for drop-in compatibility with scripts expecting OpenAI's TTS API structure.

*   **Method:** `POST`
*   **Body:**
    ```json
    {
      "model": "kitten-tts",
      "input": "This is an OpenAI-compatible request.",
      "voice": "Ivy",
      "response_format": "wav",
      "speed": 0.9
    }
    ```
*   **Latency check:** WAV responses for medium/long text stream sentence chunks as they are generated. MP3 and Opus may still use a buffered compatibility path.
    ```bash
    curl -s -o /tmp/out.wav -w "TTFB=%{time_starttransfer}s TOTAL=%{time_total}s STATUS=%{http_code}\n" -X POST "$BASE_URL/v1/audio/speech" -H "Content-Type: application/json" -d '{"model":"kitten-tts-nano-0.1","input":"Welcome to Pathwisse","voice":"expr-voice-2-f","response_format":"wav"}'
    ```

### Model Management Endpoints

*   `GET /api/model-info` — Returns details about the currently loaded model.
*   `GET /api/model-registry` — Returns all available models for the UI dropdown.
*   `GET /api/model-status` — Returns download/loading progress during model switching.
*   `POST /restart_server` — Triggers an async model hot-swap based on current config.
*   `POST /api/cancel-loading` — Cancels an in-progress model download/load.

## ⚙️ Configuration

All server settings are managed in the `config.yaml` file. It's created automatically on first launch if it doesn't exist.

**Key Settings:**
*   `server.host`, `server.port`: Network settings.
*   `model.repo_id`: The active model selector (e.g., `kitten-nano-0.8-int8` or a HuggingFace repo ID).
*   `tts_engine.device`: Set to `auto`, `cuda`, or `cpu`. The server will use your GPU if set to `auto` or `cuda` and a compatible environment is found.
*   `generation_defaults.speed`: Default speech speed (1.0 is normal).
*   `audio_output.format`: Default audio format (`wav`, `mp3`, `opus`).

## 🛠️ Troubleshooting

*   **Phonemizer / eSpeak Errors:**
    *   This is the most common issue. Ensure you have installed **eSpeak NG** correctly for your OS and **restarted your terminal** afterward. The server includes auto-detection logic for common install paths.
*   **GPU Not Used / Falls Back to CPU:**
    *   Follow the **NVIDIA GPU Installation** steps exactly. The most common cause is `torch` being installed without CUDA support.
    *   Run the verification command from the installation guide to confirm `torch.cuda.is_available()` is `True`.
*   **"No module named 'soundfile'" or Audio Errors on Linux:**
    *   The underlying system library is likely missing. Run `sudo apt install libsndfile1`.
*   **"Port already in use" Error:**
    *   Another application is using port 8005. Stop that application or change the port in `config.yaml` (e.g., `port: 8006`) and restart the server.
*   **Model download hangs or fails:**
    *   Check your internet connection. Models are downloaded from Hugging Face Hub.
    *   Try clearing the `model_cache` directory and restarting.
    *   Large models (Mini 0.1 at ~170MB) may take several minutes on slower connections.

## 🙏 Acknowledgements & Credits

*   **Core Model:** This project is powered by the **[KittenTTS model](https://github.com/KittenML/KittenTTS)** created by **[KittenML](https://github.com/KittenML)**. Our work adds a high-performance server and UI layer on top of their excellent lightweight model.
*   **Core Libraries:** FastAPI, Uvicorn, ONNX Runtime, PyTorch, Hugging Face Hub, Phonemizer.
*   **UI Inspiration:** The UI/server architecture is inspired by our previous work on the [Chatterbox-TTS-Server](https://github.com/devnen/Chatterbox-TTS-Server).

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Please feel free to open an issue or submit a pull request.


### API Authentication and Security Defaults
For safe deployment, the following security defaults are enabled:
- `ENABLE_WEB_UI` is `false` by default.
- `ENABLE_MANAGEMENT_ENDPOINTS` is `false` by default.
- API endpoints (`/tts`, `/api/tts/speak`, `/v1/audio/speech`, `/api/unload`, etc) require a configured `TTS_API_KEY`. If it is unset, these endpoints fail securely.

Provide your configured key via `Authorization: Bearer <your_api_key>` or `X-API-Key: <your_api_key>`.
Only `/health` and `/api/tts/health` are public without auth.

### CORS Configuration
CORS origins should be specified via the `ALLOWED_ORIGINS` environment variable (comma-separated, e.g. `http://localhost:3000,http://localhost:5173`). Wildcard `*` origins are no longer permitted with credentials in production.

### MP3 Support
To ensure compatibility in lightweight server environments (like Railway), MP3 generation is **intentionally unsupported** in this deployment. Please request `wav` (default) or `opus`.

The `/tts` and `/api/tts/speak` endpoints, as well as the OpenAI compatible endpoint `/v1/audio/speech`, now require authentication via an API Key when `TTS_API_KEY` environment variable is set.
Provide the key in the `Authorization` header:
`Authorization: Bearer <your_api_key>`

### Dependency Footprint Note (PR #16)
The CPU image size is significantly large (~2.5GB+) because `misaki[en]>=0.9.4` automatically pulls `spacy-curated-transformers` and `torch`. This violates the lightweight footprint requirement for small Railway instances.
*Recommendation for future:* Suggest moving to a lighter G2P/phonemizer pipeline, such as `gruut` or `phonemizer` with standard `espeak-ng` directly without transformer models to keep the image size small.
