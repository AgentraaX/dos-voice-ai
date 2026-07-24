# Runpod GPU image: CUDA base + ffmpeg (for faster-whisper's audio decode)
# + this service's real deps + the Sesame CSM repo (not on PyPI).
# For local/mock-mode use, requirements.txt alone (no GPU) is enough --
# see README.md "Running locally in mock mode".
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-gpu.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-gpu.txt

# Sesame CSM isn't a pip package -- clone it so tts/sesame_engine.py's
# `from generator import load_csm_1b` resolves.
RUN git clone --depth 1 https://github.com/SesameAILabs/csm.git csm_repo \
    && pip install --no-cache-dir -r csm_repo/requirements.txt
ENV PYTHONPATH="/app/csm_repo:${PYTHONPATH}"

COPY . .
RUN chmod +x start.sh

ENV MOCK_MODE=false
EXPOSE 8000 8001
CMD ["./start.sh"]
