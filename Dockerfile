# RTX 5090 (Blackwell sm_120, 32GB) needs CUDA 12.8+ torch (cu128).
# Do NOT use the old 2.4.1-cuda12.1 base — it has no sm_120 kernels and fails on a 5090.
# Build once, push to GHCR, launch Vast pods from it (~1-min starts).
# Weights (~12GB) download on first run into /root/.cache/huggingface — mount a
# network volume there to persist them across sessions.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
RUN apt-get update -qq && apt-get install -y -qq git ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir hf_transfer
COPY . .
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENTRYPOINT ["python", "batch.py"]
CMD ["--help"]
