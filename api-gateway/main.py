# api-gateway/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
import httpx, os, time

app = FastAPI(title="AI Platform API Gateway")
Instrumentator().instrument(app).expose(app)  # Integration 9: Prometheus

VLLM_URL = os.environ.get("VLLM_URL", "")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")


class ChatRequest(BaseModel):
    query: str
    embedding: list[float] = [0.0] * 384


@app.post("/api/v1/chat")
async def chat(req: ChatRequest):
    start = time.time()

    # 1. Vector search (best-effort — collection may not exist yet)
    context = []
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            search_resp = await client.post(
                f"{QDRANT_URL}/collections/documents/points/search",
                json={"vector": req.embedding, "limit": 3},
            )
            if search_resp.status_code == 200:
                context = search_resp.json().get("result", [])
    except Exception:
        pass

    # 2. LLM inference (fall back to mock when vLLM is unavailable)
    answer = None
    model = "mock"
    if VLLM_URL:
        try:
            prompt = f"Context: {context}\n\nQuery: {req.query}"
            async with httpx.AsyncClient(timeout=30) as client:
                llm_resp = await client.post(
                    f"{VLLM_URL}/v1/chat/completions",
                    json={
                        "model": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
            if llm_resp.status_code == 200:
                result = llm_resp.json()
                answer = result["choices"][0]["message"]["content"]
                model = result.get("model", "unknown")
        except Exception:
            pass

    if answer is None:
        answer = f"Mock answer for: {req.query}"

    latency = (time.time() - start) * 1000
    return {"answer": answer, "latency_ms": round(latency, 2), "model": model}


@app.get("/health")
def health():
    return {"status": "ok"}
