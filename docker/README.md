# Docker

## Architecture

The image is built in three stages:

**`builder`** — Compiles the `server` binary from source inside `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`. The Rust toolchain is installed at build time. Dependencies are compiled in a separate layer so that code changes don't trigger a full dependency rebuild.

**`downloader`** — Downloads the model files from HuggingFace Hub using a minimal Python image. This stage is independent of the build stage and its layer is cached separately — the model is only re-downloaded when `HF_MODEL` changes.

**`runtime`** — The final image, based on `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`. It receives the binary from `builder` and the model files from `downloader`. No compiler, no Python, no build tools.

### CUDA version

The image targets **CUDA 12.4** via the `cuda-12040` feature of `cudarc`. This must match the CUDA version available on the RunPod worker. The `tract-cuda` dependency is declared with `default-features = false` to ensure exactly one CUDA version feature is active at compile time (activating two simultaneously causes duplicate type definition errors).

---

## Build

```bash
docker buildx build -f docker/Dockerfile \
  --build-arg HF_REVISION=refs/pr/23 \
  -t tractile:local .
```

Two build arguments control which model is downloaded:

| Argument | Default | Description |
|---|---|---|
| `HF_MODEL` | `Alibaba-NLP/gte-multilingual-base` | HuggingFace repo |
| `HF_REVISION` | `main` | Branch, tag, or PR ref (`refs/pr/N`) |

> **Note:** The ONNX export for `gte-multilingual-base` currently lives in [PR #23](https://huggingface.co/Alibaba-NLP/gte-multilingual-base/tree/refs%2Fpr%2F23) which has not been merged yet, hence the `--build-arg HF_REVISION=refs/pr/23`. Once merged, the default `main` revision will work without extra arguments.

The image bundles `docker/server.toml` (absolute paths under `/app/models/`) as its config. The root `server.toml` (relative paths) is used for development outside Docker and is unaffected.

---

## Test locally

### Health check lifecycle (no GPU needed)

Start the container without GPU — the HTTP server comes up immediately, but model loading fails silently (no CUDA device). This lets you verify the `/ping` 204 response and the HTTP surface:

```bash
docker run --rm -p 8080:8080 -e PORT=8080 tractile:local

curl -i http://localhost:8080/ping     # 204 — server up, model not ready
curl -i http://localhost:8080/embed    # 503 — model not ready
```

### Full test (Linux + NVIDIA GPU required)

```bash
docker run --rm --gpus all -p 8080:8080 tractile:local

# Poll until ready (204 → 200)
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8080/ping)" = "200" ]; do
  echo "waiting..."; sleep 1
done && echo "ready!"

# Test embedding
curl -s -X POST http://localhost:8080/embed \
  -H 'Content-Type: application/json' \
  -d '{"texts":["hello world","bonjour le monde"]}' | python3 -m json.tool
```

### Test without Docker (CPU, macOS)

The fastest way to validate the full 204 → 200 lifecycle and the `/embed` endpoint:

```bash
cargo run --release --features server

until [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:12345/ping)" = "200" ]; do
  echo "waiting..."; sleep 1
done && echo "ready!"

curl -s -X POST http://localhost:12345/embed \
  -H 'Content-Type: application/json' \
  -d '{"texts":["hello world","bonjour le monde"]}' | python3 -m json.tool
```

---

## Publishing to GitHub Container Registry

**Authenticate once** — generate a token at GitHub → Settings → Developer settings → Personal access tokens with the `write:packages` scope:

```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u fbilhaut --password-stdin
```

**Build and push:**

```bash
docker buildx build -f docker/Dockerfile --platform linux/amd64 \
  --build-arg HF_REVISION=refs/pr/23 \
  -t ghcr.io/fbilhaut/tractile:latest --push .
```

**Pull on another machine:**

```bash
docker pull ghcr.io/fbilhaut/tractile:latest
```

> By default the package is private. To make it public: GitHub → repository → Packages → tractile → Package settings → Change visibility.

---

## RunPod deployment

1. Push the image to GHCR (see above).
2. In the RunPod console, create a **Load Balancer** endpoint.
3. Set the container image to `ghcr.io/fbilhaut/tractile:latest` and configure the environment:
   - `PORT=80`
4. Deploy. RunPod will poll `GET /ping` on port 80 — the worker appears as **Active** once the model finishes loading and `/ping` returns 200.
