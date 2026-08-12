"""
Captures the alternate listing images that download_products discards.

get_best_image() keeps one URL per product (in practice the MAIN catalog shot,
since it is first in the list) and drops the rest. Those dropped images show
the same product, were never embedded, and their correct answer is known — the
ASIN they came from — which makes them held-out eval queries for free.

Re-streams each category file, matches ASINs already in the DB, and stores the
non-MAIN images in products.alt_image_urls. Stops streaming a category once
every product from it has been matched.

    python scripts/fetch_alt_images.py
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  #allow flat imports from src/

from sqlalchemy import text

from database import SessionLocal, engine
from models import Product
from download_products import BASE_URL, CATEGORIES, stream_jsonl

COMMIT_EVERY = 500


def ensure_column():
    #no migration tooling in this project, and create_all() won't alter an
    #existing table, so add the column by hand if it isn't there yet
    with engine.connect() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(products)"))}
        if "alt_image_urls" not in cols:
            conn.execute(text("ALTER TABLE products ADD COLUMN alt_image_urls TEXT"))
            conn.commit()
            print("Added products.alt_image_urls")


def alt_images(images):
    #every non-MAIN image, best available resolution per variant
    out = []
    for img in images or []:
        if not isinstance(img, dict):
            continue
        variant = img.get("variant")
        if not variant or variant == "MAIN":
            continue
        url = img.get("hi_res") or img.get("large") or img.get("thumb")
        if url:
            out.append({"variant": variant, "url": url})
    return out


def fetch_alts():
    ensure_column()
    db = SessionLocal()
    total_products = 0
    total_images = 0

    try:
        for category in CATEGORIES:
            #only the products this category actually supplied, so the stream
            #can stop as soon as they're all accounted for
            wanted = {
                asin: pid for pid, asin in
                db.query(Product.id, Product.asin)
                  .filter(Product.category == category, Product.alt_image_urls == None)
                  .all()
            }
            if not wanted:
                print(f"\n{category}: nothing outstanding, skipping")
                continue

            print(f"\n{category}: looking for {len(wanted)} products")
            matched = 0
            pending = 0

            try:
                for item in stream_jsonl(f"{BASE_URL}/meta_{category}.jsonl"):
                    asin = item.get("parent_asin") or item.get("asin")
                    pid = wanted.pop(asin, None) if asin else None
                    if pid is None:
                        continue

                    alts = alt_images(item.get("images"))
                    #store [] rather than leaving NULL so single-image products
                    #aren't re-searched on the next run
                    db.query(Product).filter(Product.id == pid).update(
                        {"alt_image_urls": json.dumps(alts)}, synchronize_session=False
                    )
                    matched += 1
                    pending += 1
                    total_images += len(alts)

                    if pending >= COMMIT_EVERY:
                        db.commit()
                        pending = 0
                        print(f"  [{matched}/{matched + len(wanted)}] {total_images} alt images so far")

                    if not wanted:
                        break

                db.commit()
            except Exception as e:
                db.commit()
                print(f"  Error streaming {category}: {e}")

            total_products += matched
            print(f"  {category}: matched {matched}, {len(wanted)} not found in stream")

        print(f"\nStored alt images for {total_products} products ({total_images} images total)")

    finally:
        db.close()


if __name__ == "__main__":
    fetch_alts()
