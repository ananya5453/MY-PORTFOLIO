"""FastAPI embedding backend for the portfolio playground.

Default model: all-MiniLM-L6-v2  (fast, no auth required, 384-dim)
To use a larger model override with the env var, e.g.:
  Windows PowerShell:
      $env:EMBED_MODEL = "google/embeddinggemma-2"
      uvicorn main:app --port 8000
  Mac/Linux:
      EMBED_MODEL=google/embeddinggemma-2 uvicorn main:app --port 8000

Optional env vars:
    ALLOW_ORIGIN  comma-separated origins allowed by CORS (default: * = any)
"""
import os, json
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_ID = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")

# EmbeddingGemma 2 is multimodal — pass special kwargs only for that model.
# Standard models (e.g. all-MiniLM-L6-v2) do not accept these kwargs.
IS_GEMMA = "embeddinggemma" in MODEL_ID.lower() or "embedding-gemma" in MODEL_ID.lower()

if IS_GEMMA:
    # float16 gives NaN/garbage on EmbeddingGemma 2, so use bfloat16 (GPU) or float32.
    DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    model = SentenceTransformer(
        MODEL_ID,
        config_kwargs={"vision_config": None, "audio_config": None},  # text-only
        model_kwargs={"torch_dtype": DTYPE},
    )
else:
    model = SentenceTransformer(MODEL_ID)

# Use the task prefixes only if this model defines them (EmbeddingGemma does).
PROMPTS = getattr(model, "prompts", None) or {}
QUERY_PROMPT = "SearchQuery" if "SearchQuery" in PROMPTS else None
DOC_PROMPT = "Document" if "Document" in PROMPTS else None

def encode(texts, prompt_name=None, **kw):
    if prompt_name:
        kw["prompt_name"] = prompt_name
    return model.encode(texts, **kw)

app = FastAPI(title="Portfolio embedding backend")
origins = [o.strip() for o in os.environ.get("ALLOW_ORIGIN", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

# Optional knowledge base: chunks.json = ["chunk text", ...] made by YOUR chunking code
CHUNKS = json.load(open("chunks.json", encoding="utf-8")) if os.path.exists("chunks.json") else []
CV = encode(CHUNKS, DOC_PROMPT, normalize_embeddings=True) if CHUNKS else None

class Q(BaseModel):
    text: str
    k: int = 3

@app.get("/")
def health():
    dim = model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension") else model.get_sentence_embedding_dimension()
    return {"status": "ok", "model": MODEL_ID, "dimensions": dim,
            "knowledge_base_chunks": len(CHUNKS), "docs": "/docs"}

@app.post("/embed")
def embed(q: Q):
    if not q.text.strip():
        raise HTTPException(400, "Text is empty")
    v = encode(q.text, QUERY_PROMPT)
    return {"model": MODEL_ID, "embedding": [float(x) for x in v]}

@app.post("/search")
def search(q: Q):
    if CV is None:
        raise HTTPException(404, "No chunks.json knowledge base loaded")
    s = CV @ encode(q.text, QUERY_PROMPT, normalize_embeddings=True)  # cosine similarity
    return {"results": [{"text": CHUNKS[i], "score": float(s[i])} for i in np.argsort(-s)[:q.k]]}
