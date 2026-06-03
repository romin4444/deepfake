# Dockerfile — build a container image of the detector
# Usage:
#   docker build -t dfvideo-detector .
#   docker run --gpus all -v /data:/workspace/data -v /outputs:/workspace/outputs dfvideo-detector

FROM pytorch/pytorch:2.1.0-cuda12.1-runtime-ubuntu22.04

WORKDIR /workspace

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget zip \
    libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir opencv-python-headless

# Copy the detector code
COPY . .

# Default command: training
ENTRYPOINT ["python", "-m", "src.train"]
CMD ["--config", "configs/default.yaml"]
