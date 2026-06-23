# ── Build stage ────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    curl cmake pkg-config libssl-dev build-essential \
    clang libclang-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_ROOT=/usr/local/cuda

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app
COPY Cargo.toml Cargo.lock ./

RUN mkdir -p src/bin/server && \
    echo 'fn main() {}' > src/bin/server/main.rs && \
    cargo build --release --features server,cuda && \
    rm -f target/release/server target/release/deps/server*

COPY src ./src
RUN cargo build --release --features server,cuda

# ── Model download stage ────────────────────────────────────────────────────────
FROM python:3.11-slim AS downloader

ARG HF_MODEL=Alibaba-NLP/gte-multilingual-base
RUN pip install --no-cache-dir "huggingface_hub[cli]"
ARG HF_REVISION=main
RUN hf download ${HF_MODEL} \
    tokenizer.json onnx/model.onnx \
    --revision ${HF_REVISION} \
    --local-dir /model

# ── Runtime stage ───────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    libssl3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder    /app/target/release/server  /app/server
COPY --from=downloader /model                      /app/models/gte-multilingual-base
COPY docker/server.toml /app/server.toml

ENV PORT=80
EXPOSE 80

CMD ["/app/server", "/app/server.toml"]
