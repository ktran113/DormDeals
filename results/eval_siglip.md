# Retrieval evaluation — siglip

2367 held-out query images, one per product, searched against a 14,873-vector index.

Query images are listing photos the index never saw: `download_products` embeds one image per product, and these are the ones it discarded. The correct answer is the ASIN they came from.

From 4,000 candidates, 1,582 were dropped as listing graphics rather than product photography, 51 as near-duplicates of the indexed image (cosine > 0.98), and 0 failed to fetch.

| metric | value |
|---|---|
| n | 2,367 |
| recall@1 | 0.522 |
| recall@5 | 0.690 |
| recall@10 | 0.740 |
| MRR | 0.594 |

recall@5 95% CI: ±1.9 points.

## Relevance

Exact-ASIN recall asks whether one specific product came back. The product answers a looser question — *show me items like this* — so results are also scored on whether they are the same kind of thing, typed with the keyword vocabulary the catalog was built from.

Scored against drawing five products at random from the same index, because a same-type rate means nothing without knowing what chance already gives.

Measured over 2,014 queries whose product type was recognised.

| metric | retrieved | random | lift |
|---|---|---|---|
| at least one same-type item in top 5 | **0.866** | 0.071 | 12.1x |
| mean same-type fraction of top 5 | **0.451** | 0.015 | 29.5x |
