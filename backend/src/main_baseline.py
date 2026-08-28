"""
Pre-optimisation /search, kept only so the latency benchmark has a baseline.

This is the search endpoint exactly as it stood at commit 14a07ab, ported onto
the current database and index so the benchmark varies one thing — the endpoint
implementation — and not the surrounding infrastructure.

Two deliberate flaws are preserved:
  1. gen_embeddings runs inline in an `async def`, so the CPU-bound forward pass
     blocks the asyncio event loop and no other request can even be parsed.
  2. one DB query per result (N+1), each loading the embedding blob it discards.

Served by bench_latency.py; not part of the application.
"""

from io import BytesIO
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from fastapi import FastAPI, File, Query, UploadFile
from PIL import Image
from pydantic import BaseModel

from database import SessionLocal
from embeddings import gen_embeddings
from models import Product

app = FastAPI(title="DormDeals (pre-optimisation baseline)")

DATA_DIR = Path(__file__).parent.parent / "data"

faiss_index = None
id_map = None


@app.on_event("startup")
def startup():
    global faiss_index, id_map
    faiss_index = faiss.read_index(str(DATA_DIR / "products.index"))
    id_map = np.load(str(DATA_DIR / "id_map.npy"))
    print(f"FAISS index loaded ({faiss_index.ntotal} vectors)")


class ProductResult(BaseModel):
    id: int
    title: str
    price: Optional[float]
    image_url: Optional[str]
    category: Optional[str]
    score: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search", response_model=List[ProductResult])
async def search(file: UploadFile = File(...), k: int = Query(5, ge=1, le=50)):
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")

    #inline on the event loop — the flaw being measured
    query = gen_embeddings(img)[np.newaxis, :]
    scores, positions = faiss_index.search(query, k=k)
    product_ids = [int(id_map[pos]) for pos in positions[0]]

    db = SessionLocal()
    try:
        results = []
        for i, pid in enumerate(product_ids):
            #one round trip per result, embedding blob included
            product = db.query(Product).filter(Product.id == pid).first()
            if product:
                results.append(ProductResult(
                    id=product.id,
                    title=product.title,
                    price=product.price,
                    image_url=product.image_url,
                    category=product.category,
                    score=float(scores[0][i]),
                ))
        return results
    finally:
        db.close()
