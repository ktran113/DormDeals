"""
Targeted top-up of decoration products.

The main download caps each category at TARGET_COUNT_CAT, and inside that cap
decor competes against every other keyword, so it ends up underrepresented.
This re-streams Home_and_Kitchen matching decoration keywords only, skipping
ASINs already stored, until TARGET new rows are added.

    python scripts/backfill_decor.py
"""

import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  #allow flat imports from src/

from database import SessionLocal
from models import Product
from download_products import BASE_URL, BATCH_SIZE, stream_jsonl, get_best_image, parse_price, process_batch

SOURCE_CATEGORY = "Home_and_Kitchen"
TARGET = 772

#the decorations block from download_products.KEYWORDS, matched on word
#boundaries: bare "light" would hit "lightweight" and bare "frame" would hit
#"bed frame", which is furniture rather than decor
DECOR_KEYWORDS = [
    "rug", "carpet", "floor mat", "bath mat",
    "lamp", "desk lamp", "floor lamp", "table lamp",
    "led strip", "string lights", "fairy lights",
    "mirror", "wall mirror",
    "curtain", "curtains", "blinds",
    "poster", "wall art", "tapestry",
    "picture frame", "photo frame",
]
DECOR_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in DECOR_KEYWORDS) + r")\b", re.I)


def matches_decor(title: str) -> bool:
    return bool(title) and bool(DECOR_RE.search(title))


def backfill():
    db = SessionLocal()
    added = 0
    try:
        url = f"{BASE_URL}/meta_{SOURCE_CATEGORY}.jsonl"
        batch = []
        for item in stream_jsonl(url):
            if added >= TARGET:
                break
            title = item.get("title", "")
            if not matches_decor(title):
                continue

            asin = item.get("parent_asin") or item.get("asin")
            if not asin:
                continue

            image_url = get_best_image(item.get("images", []))
            if not image_url:
                continue

            batch.append({
                "asin": asin,
                "title": title,
                "price": parse_price(item.get("price")),
                "image_url": image_url,
            })

            if len(batch) >= BATCH_SIZE:
                print(f"\n[decor: {added}/{TARGET}] Processing batch of {len(batch)}...")
                added += len(process_batch(batch, db, SOURCE_CATEGORY))
                batch = []

        if batch and added < TARGET:
            print(f"\n[decor: {added}/{TARGET}] Processing final batch of {len(batch)}...")
            added += len(process_batch(batch, db, SOURCE_CATEGORY))

        print(f"\nAdded {added} decoration products")
    finally:
        db.close()


if __name__ == "__main__":
    backfill()
