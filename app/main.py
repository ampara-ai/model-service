import logging
import time
from contextlib import asynccontextmanager

import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings, validate_model_dir
from app.embedder import AmparaEmbedder
from app.reranker import AmparaReranker
from app.schemas import EmbedRequest, EmbedResponse, RerankRequest, RerankResponse, RerankResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


def _build_session_options() -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if settings.onnx_intra_op_num_threads != 0:
        so.intra_op_num_threads = settings.onnx_intra_op_num_threads
    if settings.onnx_inter_op_num_threads != 0:
        so.inter_op_num_threads = settings.onnx_inter_op_num_threads
    return so


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Model Service — precision mode: %s", settings.model_precision)

    emb_dir = settings.embedding_model_dir
    base_emb_dir = settings.base_embedding_model_dir
    rer_dir = settings.reranker_model_dir
    validate_model_dir(emb_dir, "embedding")
    validate_model_dir(base_emb_dir, "base_embedding")
    validate_model_dir(rer_dir, "reranker")

    so = _build_session_options()

    t0 = time.perf_counter()
    embedder = AmparaEmbedder(emb_dir, settings.embedding_max_length, so)
    logger.info("Embedder loaded from '%s' in %.2fs — dim=%d", emb_dir, time.perf_counter() - t0, embedder.dimension)

    t0 = time.perf_counter()
    base_embedder = AmparaEmbedder(base_emb_dir, settings.embedding_base_max_length, so)
    logger.info("Base embedder loaded from '%s' in %.2fs — dim=%d", base_emb_dir, time.perf_counter() - t0, base_embedder.dimension)

    t0 = time.perf_counter()
    reranker = AmparaReranker(rer_dir, settings.reranker_max_length, settings.reranker_batch_size, so)
    logger.info("Reranker loaded from '%s' in %.2fs", rer_dir, time.perf_counter() - t0)

    app.state.embedder = embedder
    app.state.base_embedder = base_embedder
    app.state.reranker = reranker
    app.state.settings = settings

    logger.info("Model Service ready — embedding_dir=%s base_embedding_dir=%s reranker_dir=%s", emb_dir, base_emb_dir, rer_dir)
    yield
    logger.info("Model Service shutting down")


app = FastAPI(title="AMPARA Model Service", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest, req: Request) -> EmbedResponse:
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="text vacío o ausente")
    t0 = time.perf_counter()
    try:
        vector = req.app.state.embedder.encode(request.text)
    except Exception:
        logger.exception("Error en /embed")
        raise HTTPException(status_code=500, detail="Error interno del modelo")
    logger.info("/embed — %.0f chars — %.3fs", len(request.text), time.perf_counter() - t0)
    return EmbedResponse(embedding=vector.tolist())


@app.post("/embed_base", response_model=EmbedResponse)
async def embed_base(request: EmbedRequest, req: Request) -> EmbedResponse:
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="text vacío o ausente")
    t0 = time.perf_counter()
    try:
        vector = req.app.state.base_embedder.encode(request.text)
    except Exception:
        logger.exception("Error en /embed_base")
        raise HTTPException(status_code=500, detail="Error interno del modelo")
    logger.info("/embed_base — %.0f chars — %.3fs", len(request.text), time.perf_counter() - t0)
    return EmbedResponse(embedding=vector.tolist())


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest, req: Request) -> RerankResponse:
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query vacío o ausente")
    if not request.documents:
        raise HTTPException(status_code=400, detail="documents vacío o ausente")
    t0 = time.perf_counter()
    try:
        scores = req.app.state.reranker.compute_scores(request.query, request.documents)
    except Exception:
        logger.exception("Error en /rerank")
        raise HTTPException(status_code=500, detail="Error interno del modelo")
    logger.info("/rerank — %d docs — %.3fs", len(request.documents), time.perf_counter() - t0)
    results = [
        RerankResult(index=i, score=float(scores[i]), text=request.documents[i])
        for i in range(len(request.documents))
    ]
    results.sort(key=lambda r: r.score, reverse=True)
    return RerankResponse(results=results)


@app.get("/health")
async def health(req: Request):
    s = req.app.state.settings
    return {
        "status": "ok",
        "mode": s.model_precision,
        "embedding_model": str(s.embedding_model_dir),
        "base_embedding_model": str(s.base_embedding_model_dir),
        "reranker_model": str(s.reranker_model_dir),
        "embedding_dim": req.app.state.embedder.dimension,
        "base_embedding_dim": req.app.state.base_embedder.dimension,
    }
