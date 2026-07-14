"""
Batch-embeds product images: fetches each product's image URL and stores
the CLIP embedding bytes in products.embedding. Run from backend/src/:
    python scripts/generate_embeddings.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  #allow flat imports from src/

import requests
from PIL import Image
from io import BytesIO

from database import SessionLocal
from models import Product
from embeddings import gen_embeddings

BATCH_SIZE = 50


def fetch_img(url):
    #uses URL to return PIL image
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        print(f"Fetching image failed: {e}")
        return None


if __name__ == "__main__":
    db = SessionLocal()
    total_processed = 0

    while True:
        batch = db.query(Product)\
            .filter(Product.embedding == None)\
            .limit(BATCH_SIZE)\
            .all()

        if not batch:
            break

        for product in batch:
            #checks to see if there is a img url
            img = fetch_img(product.image_url) if product.image_url else None
            if img is None:
                product.embedding = b''  #mark with empty bytes
                continue

            embedding = gen_embeddings(img)
            product.embedding = embedding.tobytes()
            print(f"  Embedded: {product.title[:60]}")

        db.commit()
        total_processed += len(batch)
        print(f"[{total_processed}] Batch committed")

    print(f"\nProcessed {total_processed} products.")
    db.close()
