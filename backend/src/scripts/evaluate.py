"""
Retrieval evaluation over held-out listing images.

download_products keeps one image per listing (in practice the MAIN catalog
shot) and discards the rest; fetch_alt_images recovered those discards into
products.alt_image_urls. They show the same product, their correct answer is
known, and they were never embedded — so they are held-out queries.

Listings mix real product photography with infographics, spec diagrams and
text banners. Retrieving a product from an advertising banner is not the task
this system performs, so queries are restricted to genuine photographs before
scoring. Reports recall@1/5/10 and MRR.

    python scripts/evaluate.py
    python scripts/evaluate.py --n 200                 # quick pass
    python scripts/evaluate.py --index ../data/products_siglip.index \
        --id-map ../data/id_map_siglip.npy --label siglip
"""

import argparse
import json
import re
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import faiss
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))  #allow flat imports from src/

from database import SessionLocal
from download_products import KEYWORDS
from embeddings import gen_embeddings, model, processor
from imgcache import fetch
from models import Product

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "results"

#a query image this close to the indexed image is the same photo re-cropped;
#retrieving it is trivially correct and would inflate the score
NEAR_DUP_MAX = 0.98

#zero-shot split of photography from listing graphics. This classifies image
#*type*, which is a different task from the retrieval being scored, so it does
#not leak the answer — but it is a filter applied by the same model, and the
#reported numbers are conditional on it.
PHOTO_PROMPT = "a photograph of a physical product"
GRAPHIC_PROMPT = "a promotional graphic with large text, labels and diagrams"

SIGLIP_ID = "google/siglip-base-patch16-224"

#The photo filter and the near-duplicate guard always run on CLIP, whatever
#encoder is being scored. They define WHICH queries are in the eval set, so
#letting them vary with the encoder would score CLIP and SigLIP on two
#different sets and make the comparison meaningless.

#the CDN throttles per connection, so throughput comes from width, not patience:
#one fetch can take 2 minutes while 16 in parallel finish in 5 seconds
FETCH_WORKERS = 24
CHUNK = 96


def photo_filter_vectors():
    with torch.no_grad():
        inputs = processor(text=[PHOTO_PROMPT, GRAPHIC_PROMPT],
                           return_tensors="pt", padding=True)
        #transformers 5 returns an output object here, same as get_image_features
        feats = model.get_text_features(**inputs).pooler_output
    return torch.nn.functional.normalize(feats, p=2, dim=1).numpy().astype(np.float32)


def load_encoder(name):
    if name == "clip":
        return gen_embeddings

    from transformers import AutoModel, AutoProcessor
    print(f"Loading {SIGLIP_ID}...")
    m = AutoModel.from_pretrained(SIGLIP_ID)
    proc = AutoProcessor.from_pretrained(SIGLIP_ID)
    m.eval()

    def embed(images):
        with torch.no_grad():
            out = m.get_image_features(**proc(images=images, return_tensors="pt"))
        feats = getattr(out, "pooler_output", out)
        return torch.nn.functional.normalize(feats, p=2, dim=1).numpy()

    return embed


def build_sample(db, n, seed):
    """One query image per product, so queries stay independent.

    Products that are easy tend to be easy in all of their images, so drawing
    several images from one product would correlate the trials and overstate
    the effective sample size.
    """
    rows = db.query(Product.id, Product.alt_image_urls).filter(
        Product.embedding != None,          # noqa: E711 — SQLAlchemy needs ==/!=
        Product.alt_image_urls != None,     # noqa: E711
        Product.alt_image_urls != "[]",
    ).all()

    rng = random.Random(seed)
    rng.shuffle(rows)

    sample = []
    for i, (pid, raw) in enumerate(rows):
        if len(sample) >= n:
            break
        try:
            alts = json.loads(raw)
        except (TypeError, ValueError):
            continue
        alts = [a for a in alts if a.get("url")]
        if not alts:
            continue
        alts.sort(key=lambda a: a.get("variant") or "")
        sample.append((pid, alts[i % len(alts)]["variant"], alts[i % len(alts)]["url"]))
    return sample


def evaluate(n, seed, k, index_path, id_map_path, encoder_name):
    index = faiss.read_index(str(index_path))
    id_map = np.load(str(id_map_path))
    print(f"Index: {index.ntotal} vectors, dim {index.d}")

    text_vecs = photo_filter_vectors()
    encode = load_encoder(encoder_name)

    db = SessionLocal()
    try:
        sample = build_sample(db, n, seed)
        print(f"Sampled {len(sample)} candidate images (one per product, seed={seed})")

        own = {}
        ids = [pid for pid, _, _ in sample]
        for i in range(0, len(ids), 900):  #stay under SQLite's variable limit
            for pid, blob in db.query(Product.id, Product.embedding).filter(
                Product.id.in_(ids[i:i + 900])
            ).all():
                own[pid] = np.frombuffer(blob, dtype=np.float32)
    finally:
        db.close()

    records = []
    skipped = {"fetch_failed": 0, "graphic": 0, "near_dup": 0}

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        for start in range(0, len(sample), CHUNK):
            chunk = sample[start:start + CHUNK]
            images = list(pool.map(lambda s: fetch(s[2]), chunk))

            keep, imgs = [], []
            for meta, img in zip(chunk, images):
                if img is None:
                    skipped["fetch_failed"] += 1
                    continue
                keep.append(meta)
                imgs.append(img)
            if not imgs:
                continue

            #CLIP vectors decide membership in the eval set
            clip_vecs = np.atleast_2d(gen_embeddings(imgs)).astype(np.float32)

            #filter before searching so graphics cost nothing downstream
            is_photo = (clip_vecs @ text_vecs.T).argmax(axis=1) == 0
            skipped["graphic"] += int((~is_photo).sum())
            rows = [i for i, ok in enumerate(is_photo) if ok]
            if not rows:
                continue

            #retrieval vectors come from the encoder under test
            if encoder_name == "clip":
                sub = clip_vecs[rows]
            else:
                sub = np.atleast_2d(
                    encode([imgs[i] for i in rows])).astype(np.float32)
            scores, positions = index.search(sub, k)

            for row, src in enumerate(rows):
                pid, variant, url = keep[src]
                own_vec = own.get(pid)
                #self-similarity against the indexed image, not the top hit —
                #measures whether this query IS the photo already in the index
                self_sim = float(clip_vecs[src] @ own_vec) if own_vec is not None else 0.0
                if self_sim > NEAR_DUP_MAX:
                    skipped["near_dup"] += 1
                    continue

                hits = [int(id_map[p]) for p in positions[row] if p != -1]
                records.append({
                    "product_id": pid,
                    "variant": variant,
                    "url": url,
                    "self_sim": round(self_sim, 4),
                    "rank": hits.index(pid) + 1 if pid in hits else None,
                    "hits": hits,
                })

            print(f"  [{start + len(chunk)}/{len(sample)}] scored={len(records)} "
                  f"graphic={skipped['graphic']} dup={skipped['near_dup']} "
                  f"failed={skipped['fetch_failed']}", flush=True)

    skipped.update(candidates=len(sample), k=k, seed=seed, indexed=int(index.ntotal))
    return records, skipped


#Exact-ASIN recall asks whether the one right product came back. That is a
#harder question than the product poses — a user wanting "a lamp like this"
#is served by a different lamp. Type consistency measures that instead, using
#the same keyword vocabulary the catalog was built from.
#
#Earliest mention wins, with longer keywords breaking ties. Listing titles put
#the head noun before the compatibility blurb, so "longest match" would type
#an organizing tray as a refrigerator on the strength of "fits your
#refrigerator". Still approximate: accessories inherit the appliance they
#attach to.
_TYPE_RE = [(t, re.compile(r"\b" + re.escape(t) + r"\b", re.I)) for t in set(KEYWORDS)]


def product_type(title):
    title = title or ""
    best = None
    for t, rx in _TYPE_RE:
        m = rx.search(title)
        if m:
            key = (m.start(), -len(t))
            if best is None or key < best[0]:
                best = (key, t)
    return best[1] if best else None


def type_metrics(records, titles, seed=11):
    """Same-type rate in the top 5, against a random-draw baseline.

    The baseline is not optional. A coarse 6-class category score reads as
    0.89 here and is nearly worthless, because drawing five products at random
    already scores 0.61. Only the lift over chance says anything.
    """
    types = {pid: product_type(t) for pid, t in titles.items()}
    pool = list(titles)
    rng = random.Random(seed)

    def measure(pick):
        fracs, anyhit = [], []
        for r in records:
            want = types.get(r["product_id"])
            if not want:
                continue
            got = [types.get(h) for h in pick(r)]
            if not got:
                continue
            fracs.append(sum(g == want for g in got) / len(got))
            anyhit.append(any(g == want for g in got))
        n = len(fracs) or 1
        return sum(fracs) / n, sum(anyhit) / n, len(fracs)

    mean, any5, n = measure(lambda r: r.get("hits", [])[:5])
    r_mean, r_any5, _ = measure(lambda r: rng.sample(pool, 5))
    return {"n": n, "mean": mean, "any5": any5,
            "rand_mean": r_mean, "rand_any5": r_any5}


def metrics(rows, k):
    ranks = [r["rank"] for r in rows]
    out = {"n": len(rows), "MRR": sum(1.0 / r for r in ranks if r) / len(ranks)}
    for cut in sorted({1, 5, k}):
        out[f"recall@{cut}"] = sum(1 for r in ranks if r and r <= cut) / len(ranks)
    return out


def report(records, meta, label, titles):
    m = metrics(records, meta["k"])
    ci = 1.96 * (m["recall@5"] * (1 - m["recall@5"]) / m["n"]) ** 0.5 * 100

    lines = [
        f"# Retrieval evaluation — {label}", "",
        f"{m['n']} held-out query images, one per product, searched against a "
        f"{meta['indexed']:,}-vector index.",
        "",
        "Query images are listing photos the index never saw: `download_products` "
        "embeds one image per product, and these are the ones it discarded. The "
        "correct answer is the ASIN they came from.",
        "",
        f"From {meta['candidates']:,} candidates, {meta['graphic']:,} were dropped as "
        f"listing graphics rather than product photography, {meta['near_dup']} as "
        f"near-duplicates of the indexed image (cosine > {NEAR_DUP_MAX}), and "
        f"{meta['fetch_failed']} failed to fetch.",
        "", "| metric | value |", "|---|---|",
    ]
    for key in ("n", "recall@1", "recall@5", f"recall@{meta['k']}", "MRR"):
        v = m[key]
        lines.append(f"| {key} | {v:,} |" if key == "n" else f"| {key} | {v:.3f} |")
    lines += ["", f"recall@5 95% CI: ±{ci:.1f} points.", ""]

    tm = type_metrics(records, titles)
    lines += [
        "## Relevance", "",
        "Exact-ASIN recall asks whether one specific product came back. The "
        "product answers a looser question — *show me items like this* — so "
        "results are also scored on whether they are the same kind of thing, "
        "typed with the keyword vocabulary the catalog was built from.",
        "",
        "Scored against drawing five products at random from the same index, "
        "because a same-type rate means nothing without knowing what chance "
        "already gives.",
        "",
        f"Measured over {tm['n']:,} queries whose product type was recognised.",
        "", "| metric | retrieved | random | lift |", "|---|---|---|---|",
        f"| at least one same-type item in top 5 | **{tm['any5']:.3f}** | "
        f"{tm['rand_any5']:.3f} | {tm['any5'] / tm['rand_any5']:.1f}x |",
        f"| mean same-type fraction of top 5 | **{tm['mean']:.3f}** | "
        f"{tm['rand_mean']:.3f} | {tm['mean'] / tm['rand_mean']:.1f}x |",
    ]
    return "\n".join(lines), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="candidate images to sample")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--index", default=str(DATA_DIR / "products.index"))
    ap.add_argument("--id-map", default=str(DATA_DIR / "id_map.npy"))
    ap.add_argument("--label", default="clip")
    ap.add_argument("--encoder", default="clip", choices=["clip", "siglip"],
                    help="encoder used for retrieval; filtering is always CLIP")
    ap.add_argument("--report-only", action="store_true",
                    help="recompute metrics from the saved run, no searching")
    args = ap.parse_args()

    raw = DATA_DIR / f"eval_{args.label}.json"
    if args.report_only:
        saved = json.loads(raw.read_text())
        records, meta = saved["records"], saved["meta"]
        print(f"Reusing {len(records)} scored queries from {raw.name}")
    else:
        records, meta = evaluate(args.n, args.seed, args.k, args.index, args.id_map,
                                 args.encoder)

    db = SessionLocal()
    try:
        titles = dict(db.query(Product.id, Product.title).all())
    finally:
        db.close()

    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    if not args.report_only:
        raw.write_text(json.dumps({"meta": meta, "records": records}, indent=1))

    summary, _ = report(records, meta, args.label, titles)
    (RESULTS_DIR / f"eval_{args.label}.md").write_text(summary + "\n")
    print("\n" + summary)


if __name__ == "__main__":
    main()
