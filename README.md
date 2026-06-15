# AMPARA Model Service

FastAPI service that exposes ONNX inference for the AMPARA legal RAG system:

- `POST /embed` — text → 1024-dimensional L2-normalized embedding (BGE-M3 fine-tuned)
- `POST /rerank` — query + documents → relevance scores (BGE-reranker-v2-m3)
- `GET /health` — readiness check

All inference runs on CPU via ONNX Runtime. No GPU required.

---

## Installation

```bash
uv sync
```

---

## Configuration

Copy `.env.example` to `.env` and edit as needed:

```bash
cp .env.example .env
```

Key variable:

| Variable | Values | Effect |
|---|---|---|
| `MODEL_PRECISION` | `fp32` \| `int8` | Which model variant to load |

All other variables have sensible defaults. See `.env.example` for the full list with comments.

---

## Running

**Development** (auto-reload on file changes):

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

**Production** (no reload):

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001
```

The service starts on port `8001` by default (matches the backend client configuration).

---

## Switching between FP32 and INT8

Edit `.env`:

```env
MODEL_PRECISION=int8   # or fp32
```

Then restart the service. No code changes needed.

This is the intended workflow for RAGAS benchmarking:
1. Set `MODEL_PRECISION=int8`, start service, run RAGAS evaluation.
2. Set `MODEL_PRECISION=fp32`, restart, run RAGAS evaluation again.
3. Compare results.

**Note:** FP32 models (~2.1 GB each) use ONNX external data files (`model.onnx_data`). ONNX Runtime loads them automatically — startup is slower and memory usage is higher than INT8 (~540 MB each).

---

## Smoke test

With the service running:

```bash
uv run python scripts/smoke_test.py
```

Verifies `/health`, `/embed`, `/rerank`, and all error codes. Prints latencies for each call.

---

## API Contract

### `POST /embed`

```json
// Request
{ "text": "¿Cuáles son las medidas de protección según la Ley 30364?" }

// Response 200
{ "embedding": [0.0231, -0.1847, ..., 0.0412] }  // 1024 floats
```

Errors: `400` (empty text), `500` (model error).

### `POST /rerank`

```json
// Request
{
  "query": "medidas de protección Ley 30364 artículo 22",
  "documents": ["doc1...", "doc2...", "doc3..."]
}

// Response 200
{
  "results": [
    { "index": 0, "score": 0.94, "text": "doc1..." },
    { "index": 2, "score": 0.61, "text": "doc3..." },
    { "index": 1, "score": 0.23, "text": "doc2..." }
  ]
}
```

- `index`: original position in the `documents` array (0-based).
- `score`: float in `[0, 1]` (sigmoid-normalized relevance logit).
- Results are sorted by `score` descending.

Errors: `400` (empty query or documents), `500` (model error).

### `GET /health`

```json
{
  "status": "ok",
  "mode": "int8",
  "embedding_model": "modelo_onnx_int8",
  "reranker_model": "reranker_onnx_int8",
  "embedding_dim": 1024
}
```
