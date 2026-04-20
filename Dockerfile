# CPU-only Docker image for AstroMAE. Rebuild from pytorch/pytorch for CUDA.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
COPY astromae ./astromae

RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    pip install -e .

COPY configs ./configs
COPY README.md ./

ENTRYPOINT ["astromae"]
CMD ["--help"]
