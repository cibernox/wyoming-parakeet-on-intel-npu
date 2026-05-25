# wyoming-parakeet-on-intel-npu
# Runs Parakeet TDT 0.6B v3 multilingual ASR on Intel NPU via OpenVINO,
# exposed over the Wyoming protocol for Home Assistant / other clients.
#
# Image is amd64 only (Intel NPU is x86 by definition).
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_CACHE=/data/huggingface \
    TZ=Etc/UTC \
    DATA_DIR=/data \
    LANGUAGE=en \
    DEVICE=NPU \
    ENCODER_BUCKETS=5 \
    ENCODER_LAZY_BUCKETS=20

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl tzdata \
        python3 python3-pip python3-venv \
        libtbb12 libtbbmalloc2 ocl-icd-libopencl1 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Level Zero loader >= 1.17 (Ubuntu 24.04 ships 1.16, which can't discover
# the modern intel-level-zero-npu library)
# ---------------------------------------------------------------------------
WORKDIR /tmp/lz
RUN curl -fL -O \
        "https://github.com/oneapi-src/level-zero/releases/download/v1.28.6/libze1_1.28.6+u24.04_amd64.deb" \
    && apt-get update \
    && apt-get install -y --no-install-recommends ./libze1*.deb \
    && rm -rf /var/lib/apt/lists/* /tmp/lz

# ---------------------------------------------------------------------------
# Intel NPU userspace driver + firmware + compiler
# ---------------------------------------------------------------------------
WORKDIR /tmp/npu
RUN curl -fL -o npu.tar.gz \
        "https://github.com/intel/linux-npu-driver/releases/download/v1.32.1/linux-npu-driver-v1.32.1.20260422-24767473183-ubuntu2404.tar.gz" \
    && tar xzf npu.tar.gz \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ./intel-driver-compiler-npu_*.deb \
        ./intel-fw-npu_*.deb \
        ./intel-level-zero-npu_*.deb \
    && rm -rf /var/lib/apt/lists/* /tmp/npu

# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir . \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

# Persistent volume for model files + OV cache
VOLUME ["/data"]

EXPOSE 10300

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python3 -c "import socket,sys; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1', 10300)); s.close()" || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
