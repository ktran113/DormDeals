"""
Builds a FAISS index from product embeddings stored in the DB.
Run this after generate_embeddings.py.

Outputs two files to backend/data/:
  - products.index  : the FAISS index
  - id_map.npy      : maps FAISS position → product DB id
"""

import sys
import numpy as np
import faiss
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  #allow flat imports from src/

from database import SessionLocal
from models import Product

OUTPUT_DIR = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

EMBEDDING_DIM = 512

db = SessionLocal()

# load all products w embeddings
print("Loading embeddings")
products = db.query(Product).filter(
    Product.embedding != None,
    Product.embedding != b''
).all()
db.close()

print(f"Found {len(products)} products with embeddings")

# build matrix and id map in the same pass
id_map = []
vectors = []

for product in products:
    vec = np.frombuffer(product.embedding, dtype=np.float32)
    if vec.shape[0] != EMBEDDING_DIM:
        print(f"  Skipping {product.id} — unexpected shape {vec.shape}")
        continue
    vectors.append(vec)
    id_map.append(product.id)

matrix = np.stack(vectors).astype(np.float32)  # shape: (N, 512)
id_map = np.array(id_map, dtype=np.int64)

print(f"Matrix shape: {matrix.shape}")

# build FAISS index
# IndexFlatIP = exact inner product search (cosine similarity on L2-normalized vectors)
print("Building FAISS index...")
index = faiss.IndexFlatIP(EMBEDDING_DIM)
index.add(matrix)
print(f"Index contains {index.ntotal} vectors")

# save to disk
index_path = OUTPUT_DIR / "products.index"
id_map_path = OUTPUT_DIR / "id_map.npy"

faiss.write_index(index, str(index_path))
np.save(str(id_map_path), id_map)

print(f"\nSaved index to {index_path}")
print(f"Saved id map to {id_map_path}")
