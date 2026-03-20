from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from typing import List, Optional
import numpy as np
import faiss
from PIL import Image
from io import BytesIO
from pathlib import Path
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel

from auth import hash_password, verify_password, create_access_token, get_current_user_id
from database import SessionLocal, init_db
from models import User, Product
from generate_embeddings import gen_embeddings

load_dotenv()
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

DATA_DIR = Path(__file__).parent.parent / "data"

faiss_index = None
id_map = None

@app.on_event("startup")
def startup():
    global faiss_index, id_map

    init_db()
    print("Database initialized!")

    # CLIP is already loaded at module level by generate_embeddings
    print("CLIP model ready!")

    # load FAISS index and id map
    print("Loading FAISS index...")
    faiss_index = faiss.read_index(str(DATA_DIR / "products.index"))
    id_map = np.load(str(DATA_DIR / "id_map.npy"))
    print(f"FAISS index loaded ({faiss_index.ntotal} vectors)")

#response models
class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    created_at: str

    class Config:
        from_attributes = True

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
@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/register", response_model=AuthResponse)
def register(request: RegisterRequest):
    #register new account
    db = SessionLocal()
    try:
        # checks to see if email is already used
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

        # creates JWT token
        access_token = create_access_token(data={"sub": str(new_user.id)})

        return AuthResponse(
            access_token=access_token,
            user_id=new_user.id,
            email=new_user.email,
            name=new_user.name
        )
    finally:
        db.close()

@app.post("/login", response_model=AuthResponse)
def login(request: LoginRequest):
    #handles logging in needs email and password
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

        # create JWT token
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
async def search(file: UploadFile = File(...), k = 5):
    # open uploaded image
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")

    #generate query embedding
    #search FAISS
    #map FAISS positions to product DB ids
    query = gen_embeddings(img)[np.newaxis, :]
    scores, positions = faiss_index.search(query, k=k)
    product_ids = [int(id_map[pos]) for pos in positions[0]]

    #fetch products from DB and build response
    db = SessionLocal()
    try:
        results = []
        for i, pid in enumerate(product_ids):
            product = db.query(Product).filter(Product.id == pid).first()
            if product:
                results.append(ProductResult(
                    id = product.id,
                    title = product.title,
                    price = product.price,
                    image_url = product.image_url,
                    category = product.category,
                    score = float(scores[0][i])
                ))
        return results
    finally:
        db.close()
