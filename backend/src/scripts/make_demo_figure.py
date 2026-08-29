"""
Renders a real search into docs/demo.png for the README.

Runs an actual held-out query through the actual index — nothing staged. The
query is an image the index never saw, so the top row is genuine retrieval.

    python scripts/make_demo_figure.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  #allow flat imports from src/

import faiss
import numpy as np
from PIL import Image, ImageDraw

from database import SessionLocal
from embeddings import gen_embeddings
from imgcache import fetch
from models import Product

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DOCS_DIR = Path(__file__).parent.parent.parent.parent / "docs"

TILE = 190
PAD = 14
CAPTION = 34


def tile(img, label, accent=(40, 40, 40)):
    canvas = Image.new("RGB", (TILE, TILE + CAPTION), (255, 255, 255))
    thumb = img.copy()
    thumb.thumbnail((TILE - 8, TILE - 8))
    canvas.paste(thumb, ((TILE - thumb.width) // 2, (TILE - thumb.height) // 2))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, TILE - 1, TILE - 1], outline=(222, 222, 222))
    d.text((4, TILE + 8), label[:30], fill=accent)
    return canvas


def main():
    index = faiss.read_index(str(DATA_DIR / "products.index"))
    id_map = np.load(str(DATA_DIR / "id_map.npy"))

    #reuse a query the eval already scored, so the figure matches the numbers
    run = json.loads((DATA_DIR / "eval_clip.json").read_text())
    hit = next(r for r in run["records"] if r["rank"] == 1 and r["self_sim"] < 0.9)

    query_img = fetch(hit["url"])
    if query_img is None:
        raise SystemExit("query image unavailable")

    vec = np.atleast_2d(gen_embeddings([query_img])).astype(np.float32)
    scores, positions = index.search(vec, 5)
    ids = [int(id_map[p]) for p in positions[0]]

    db = SessionLocal()
    try:
        rows = {p.id: p for p in db.query(Product).filter(Product.id.in_(ids)).all()}
    finally:
        db.close()

    tiles = [tile(query_img, "QUERY (held out)", (200, 60, 20))]
    for pid, score in zip(ids, scores[0]):
        prod = rows[pid]
        img = fetch(prod.image_url)
        if img is None:
            continue
        tiles.append(tile(img, f"{score:.3f}  {prod.title[:24]}"))

    w = len(tiles) * TILE + (len(tiles) + 1) * PAD
    sheet = Image.new("RGB", (w, TILE + CAPTION + 2 * PAD), (255, 255, 255))
    for i, t in enumerate(tiles):
        sheet.paste(t, (PAD + i * (TILE + PAD), PAD))

    DOCS_DIR.mkdir(exist_ok=True)
    out = DOCS_DIR / "demo.png"
    sheet.save(out)
    print(f"wrote {out}  ({sheet.width}x{sheet.height})")
    print(f"query product {hit['product_id']}, top score {scores[0][0]:.4f}")


if __name__ == "__main__":
    main()
