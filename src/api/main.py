"""FastAPI app for intent prediction, RAG retrieval, and agentic routing."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.intent.models import predict_baseline, predict_transformer
from src.rag.agent import agent_answer, route
from src.rag.retriever import retrieve

ROOT = Path(__file__).resolve().parents[2]


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512)
    model: str = Field("baseline", pattern="^(baseline|distilbert)$")


class PredictResponse(BaseModel):
    intent: str
    confidence: float
    model: str


class RAGRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512)
    k: int = Field(3, ge=1, le=10)


class AgentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512)


class AgentResponse(BaseModel):
    action: str
    response: str
    intent: str | None = None
    source: str = "agent"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Smoke-check that model artifacts are present.
    baseline_files = [
        ROOT / "models" / "baseline_clf.joblib",
        ROOT / "models" / "tfidf_vectorizer.joblib",
        ROOT / "models" / "label_encoder.joblib",
    ]
    missing = [p.name for p in baseline_files if not p.exists()]
    if missing:
        print(f"WARNING: missing baseline artifacts: {missing}. Run `make train` first.")
    yield


app = FastAPI(
    title="Customer Intent Classifier + RAG Agent",
    description="NLP and Generative AI pipeline for banking customer support queries.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        if req.model == "distilbert":
            result = predict_transformer([req.text])[0]
        else:
            result = predict_baseline([req.text])[0]
        return PredictResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model artifacts not found: {exc}. Run `make train` first.",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/rag")
def rag(req: RAGRequest):
    try:
        return {"query": req.text, "results": retrieve(req.text, k=req.k)}
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="FAISS index not found. Run `make index` first.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/agent", response_model=AgentResponse)
def agent(req: AgentRequest):
    try:
        result = agent_answer(req.text)
        return AgentResponse(**result)
    except FileNotFoundError as exc:
        # Fall back to rule-based routing if models are missing.
        result = route(req.text)
        return AgentResponse(**result, source="rule_fallback")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
