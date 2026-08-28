"""
Second encoder for the CLIP-vs-SigLIP comparison.

Embeds every indexed product image with SigLIP and stores the bytes in
products.embedding_siglip, leaving the CLIP column untouched so both indexes
can be built from the same catalog and scored on the same queries.

Resumable: rows that already have a SigLIP embedding are skipped, so an
interrupted run picks up where it stopped.

    python scripts/generate_embeddings_siglip.py
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  #allow flat imports from src/

import torch
from sqlalchemy import text
from transformers import AutoModel, AutoProcessor

from database import SessionLocal, engine
from imgcache import fetch
from models import Product

MODEL_ID = "google/siglip-base-patch16-224"
BATCH = 32
FETCH_WORKERS = 24

print(f"Loading {MODEL_ID}...")
model = AutoModel.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model.eval()


def ensure_column():
    #same hand-rolled migration as fetch_alt_images: no migration tooling here,
    #and create_all() will not alter an existing table
    with engine.connect() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(products)"))}
        if "embedding_siglip" not in cols:
            conn.execute(text("ALTER TABLE products ADD COLUMN embedding_siglip BLOB"))
            conn.commit()
            print("Added products.embedding_siglip")


def embed(images):
    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt")
        out = model.get_image_features(**inputs)
    #transformers returns a bare tensor for some encoders and an output object
    #for others; normalise to a tensor either way
    feats = getattr(out, "pooler_output", out)
    return torch.nn.functional.normalize(feats, p=2, dim=1).numpy()


def main():
    ensure_column()
    db = SessionLocal()
    done = 0
    failed = 0
    try:
        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
            while True:
                #only products already in the CLIP index, so both indexes cover
                #exactly the same catalog
                batch = db.query(Product).filter(
                    Product.embedding != None,        # noqa: E711
                    Product.embedding_siglip == None,  # noqa: E711
                ).limit(BATCH).all()
                if not batch:
                    break

                imgs = list(pool.map(lambda p: fetch(p.image_url), batch))
                keep = [(p, im) for p, im in zip(batch, imgs) if im is not None]
                for p, im in zip(batch, imgs):
                    if im is None:
                        p.embedding_siglip = b""  #mark so the row is not retried forever
                        failed += 1

                if keep:
                    vecs = embed([im for _, im in keep])
                    for (p, _), v in zip(keep, vecs):
                        p.embedding_siglip = v.astype("float32").tobytes()

                db.commit()
                done += len(batch)
                print(f"[{done}] committed ({failed} image fetches failed)", flush=True)
    finally:
        db.close()
    print(f"\nEmbedded {done - failed} products with SigLIP ({failed} failures).")


if __name__ == "__main__":
    main()
