from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from typing import List, Optional
import numpy as np
import faiss
import torch
from PIL import Image
from io import BytesIO
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import defer

from auth import hash_password, verify_password, create_access_token
from database import SessionLocal, init_db
from models import User, Product
from embeddings import gen_embeddings

app = FastAPI(
    title = "DormDeals",
    version = "0.1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#One inference thread per request, parallelism across requests rather than
#inside them. torch defaults to 4 intra-op threads on this box; combined with
#run_in_threadpool that is up to 4x the concurrency in compute threads, which
#at 16 concurrent requests thrashes 8 cores and sent p95 to 7.8s — worse than
#doing the work inline. Serving wants many small inferences, not few wide ones.
torch.set_num_threads(1)

DATA_DIR = Path(__file__).parent.parent / "data"
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"

faiss_index = None
id_map = None

@app.on_event("startup")
def startup():
    global faiss_index, id_map

    init_db()
    print("Database initialized!")

    #CLIP is already loaded at module level by embeddings
    print("CLIP model ready!")

    #load FAISS index and id map
    print("Loading FAISS index...")
    faiss_index = faiss.read_index(str(DATA_DIR / "products.index"))
    id_map = np.load(str(DATA_DIR / "id_map.npy"))
    print(f"FAISS index loaded ({faiss_index.ntotal} vectors)")

#response models
class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str

class LoginRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    access_token: str
    user_id: int
    email: str
    name: str

class ProductResult(BaseModel):
    id: int
    title: str
    price: Optional[float]
    image_url: Optional[str]
    category: Optional[str]
    score: float

#endpoints!!!
@app.get("/health")
def health():
    return {"status": "ok", "indexed": int(faiss_index.ntotal) if faiss_index else 0}

@app.post("/register", response_model=AuthResponse)
def register(request: RegisterRequest):
    #register new account
    db = SessionLocal()
    try:
        #checks for email use under existing user
        existing_user = db.query(User).filter(User.email == request.email).first()
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="Email already registered"
            )
        hashed_password = hash_password(request.password)

        new_user = User(
            email=request.email,
            password=hashed_password,
            name=request.name
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        #creates JWT token
        access_token = create_access_token(data={"sub": str(new_user.id)})

        return AuthResponse(
            access_token=access_token,
            user_id=new_user.id,
            email=new_user.email,
            name=new_user.name
        )
    finally:
        db.close()

#handles logging in
@app.post("/login", response_model=AuthResponse)
def login(request: LoginRequest):
    db = SessionLocal()
    try:
        #identifies user using email
        user = db.query(User).filter(User.email == request.email).first()

        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )

        #checks password
        if not verify_password(request.password, user.password):
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )

        #create JWT token
        access_token = create_access_token(data={"sub": str(user.id)})

        return AuthResponse(
            access_token=access_token,
            user_id=user.id,
            email=user.email,
            name=user.name
        )
    finally:
        db.close()

@app.post("/search", response_model=List[ProductResult])
async def search(file: UploadFile = File(...), k: int = Query(5, ge=1, le=50)):
    #open uploaded image
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")

    # CLIP inference is CPU-heavy; run it off the event loop so other requests aren't blocked
    embedding = await run_in_threadpool(gen_embeddings, img)
    query = embedding[np.newaxis, :]
    scores, positions = faiss_index.search(query, k=k)

    #FAISS pads positions with -1 when it can't fill k results
    ranked = [(int(id_map[pos]), float(scores[0][i]))
              for i, pos in enumerate(positions[0]) if pos != -1]
    product_ids = [pid for pid, _ in ranked]

    #fetch products from DB and build response
    db = SessionLocal()
    try:
        #one IN query instead of k queries; defer skips the unused embedding blob
        rows = db.query(Product)\
            .options(defer(Product.embedding))\
            .filter(Product.id.in_(product_ids))\
            .all()
        #IN returns rows in arbitrary order — restore FAISS rank order
        by_id = {p.id: p for p in rows}

        results = []
        for pid, score in ranked:
            product = by_id.get(pid)
            if product:
                results.append(ProductResult(
                    id = product.id,
                    title = product.title,
                    price = product.price,
                    image_url = product.image_url,
                    category = product.category,
                    score = score
                ))
        return results
    finally:
        db.close()


#serve the frontend from the same origin as the API — dashboard.js uses a
#relative API_URL, so a separate static server can't reach /search at all.
#Registered last: routes declared above take precedence over the mount.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
