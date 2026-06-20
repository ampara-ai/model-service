"""
Smoke test for the AMPARA Model Service.
Assumes the service is already running at http://localhost:8001.

Usage:
    uv run python scripts/smoke_test.py
"""
import sys
import time

import httpx

BASE_URL = "http://localhost:8001"

EMBED_TEXT = "¿Cuáles son las medidas de protección según la Ley 30364?"

RERANK_QUERY = "medidas de protección Ley 30364 artículo 22"
RERANK_DOCS = [
    "El juzgado de familia puede dictar medidas de protección inmediatas cuando existe riesgo de violencia familiar, según el artículo 22 de la Ley 30364.",
    "Las lesiones graves se tipifican en el artículo 121-B del Código Penal y tienen penas de privación de libertad de entre 6 y 12 años.",
    "El proceso de alimentos permite al alimentista solicitar una pensión proporcional a las necesidades del menor y las posibilidades del obligado.",
    "La Ley 30364 establece un enfoque de género para la prevención, sanción y erradicación de la violencia contra las mujeres y los integrantes del grupo familiar.",
]


def check(condition: bool, msg: str) -> None:
    if not condition:
        print(f"  FAIL: {msg}", file=sys.stderr)
        sys.exit(1)
    print(f"  OK  : {msg}")


def main() -> None:
    client = httpx.Client(base_url=BASE_URL, timeout=120.0)

    # --- /health ---
    print("\n[GET /health]")
    t0 = time.perf_counter()
    resp = client.get("/health")
    latency = time.perf_counter() - t0
    check(resp.status_code == 200, f"status 200 (got {resp.status_code})")
    data = resp.json()
    check(data.get("status") == "ok", "status == 'ok'")
    check(data.get("mode") in ("fp32", "int8"), f"mode is fp32 or int8 (got {data.get('mode')})")
    print(f"  mode={data['mode']}  embedding_dim={data.get('embedding_dim')}  latency={latency:.3f}s")

    # --- /embed ---
    print("\n[POST /embed]")
    t0 = time.perf_counter()
    resp = client.post("/embed", json={"text": EMBED_TEXT})
    latency = time.perf_counter() - t0
    check(resp.status_code == 200, f"status 200 (got {resp.status_code})")
    embedding = resp.json().get("embedding")
    check(isinstance(embedding, list), "embedding is a list")
    check(len(embedding) == 1024, f"embedding length == 1024 (got {len(embedding)})")
    check(all(isinstance(v, float) for v in embedding), "all embedding values are float")
    print(f"  dim={len(embedding)}  latency={latency:.3f}s")

    # --- /embed 400 on empty text ---
    print("\n[POST /embed — empty text → 400]")
    resp = client.post("/embed", json={"text": "   "})
    check(resp.status_code == 400, f"status 400 (got {resp.status_code})")

    # --- /embed_base ---
    print("\n[POST /embed_base]")
    t0 = time.perf_counter()
    resp = client.post("/embed_base", json={"text": EMBED_TEXT})
    latency = time.perf_counter() - t0
    check(resp.status_code == 200, f"status 200 (got {resp.status_code})")
    embedding_base = resp.json().get("embedding")
    check(isinstance(embedding_base, list), "embedding is a list")
    check(len(embedding_base) == 1024, f"embedding length == 1024 (got {len(embedding_base)})")
    check(all(isinstance(v, float) for v in embedding_base), "all embedding values are float")
    print(f"  dim={len(embedding_base)}  latency={latency:.3f}s")
    # Cosine similarity vs fine-tuned embedding (should differ since models differ)
    import numpy as np
    v1 = np.array(embedding)
    v2 = np.array(embedding_base)
    cosine_sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    print(f"  cosine similarity vs /embed (finetuned): {cosine_sim:.4f}")

    # --- /embed_base 400 on empty text ---
    print("\n[POST /embed_base — empty text → 400]")
    resp = client.post("/embed_base", json={"text": "   "})
    check(resp.status_code == 400, f"status 400 (got {resp.status_code})")

    # --- /rerank ---
    print("\n[POST /rerank]")
    t0 = time.perf_counter()
    resp = client.post("/rerank", json={"query": RERANK_QUERY, "documents": RERANK_DOCS})
    latency = time.perf_counter() - t0
    check(resp.status_code == 200, f"status 200 (got {resp.status_code})")
    results = resp.json().get("results", [])
    check(len(results) == len(RERANK_DOCS), f"results length == {len(RERANK_DOCS)} (got {len(results)})")
    indices = {r["index"] for r in results}
    check(indices == set(range(len(RERANK_DOCS))), f"all indices present (got {indices})")
    check(all(0.0 <= r["score"] <= 1.0 for r in results), "all scores in [0, 1]")
    check(all("text" in r for r in results), "all results have 'text' field")
    print(f"  latency={latency:.3f}s")
    print("  Top results by score:")
    for r in results:
        print(f"    [idx={r['index']} score={r['score']:.4f}] {r['text'][:70]}...")

    # --- /rerank 400 on empty documents ---
    print("\n[POST /rerank — empty documents → 400]")
    resp = client.post("/rerank", json={"query": RERANK_QUERY, "documents": []})
    check(resp.status_code == 400, f"status 400 (got {resp.status_code})")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
