# Use the official NVIDIA CUDA runtime as the base image
# This provides the necessary CUDA libraries for GPU support and works for CPU too.
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Define a build-time argument to switch between CPU and GPU installation
ARG RUNTIME=nvidia

# Set environment variables for Python and Hugging Face
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
# Set the Hugging Face home directory to a path inside the container for better caching
ENV HF_HOME=/app/hf_cache

# Install system dependencies required for the application
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    software-properties-common \
    espeak-ng \
    libespeak-ng1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update --yes --quiet
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    python3.10 \
    pip

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 999 \
    && update-alternatives --config python3 && ln -s /usr/bin/python3 /usr/bin/python

RUN pip install --upgrade pip

# Set the working directory inside the container
WORKDIR /app

# Copy requirements files first to leverage Docker's layer caching
COPY requirements.txt .
COPY requirements-nvidia.txt .

# Upgrade pip and install the base Python dependencies from requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# --- Conditionally Install GPU Dependencies ---
# If the RUNTIME argument is 'nvidia', install the specific GPU packages
# This mirrors the robust manual installation process.
RUN if [ "$RUNTIME" = "nvidia" ]; then \
    echo "RUNTIME=nvidia, installing GPU dependencies..."; \
    pip install --no-cache-dir onnxruntime-gpu; \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121; \
    pip install --no-cache-dir -r requirements-nvidia.txt; \
    else \
    echo "RUNTIME=cpu, skipping GPU dependencies."; \
    fi

# Copy the rest of the application code into the container
COPY . .

# Create required directories for the application data
RUN mkdir -p model_cache outputs logs hf_cache

# Expose the port the application will run on (aligned with docker-compose.yml)
EXPOSE 8005

# The command to run when the container starts
CMD ["python", "server.py"]
