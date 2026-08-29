# /search latency — before vs after

`baseline` runs CLIP inline in an `async def` and issues one DB query per result. `current` moves CLIP to a threadpool and batches the lookup into a single `IN` query.

Servers were benchmarked sequentially on the same machine, never concurrently, so they never competed for cores.

| concurrency | build | p50 (ms) | p95 (ms) | p99 (ms) | req/s |
|---|---|---|---|---|---|
| 1 | baseline | 64 | 72 | 86 | 15.5 |
| 1 | current | 60 | 66 | 70 | 16.5 |
| 4 | baseline | 256 | 262 | 266 | 15.6 |
| 4 | current | 146 | 199 | 226 | 27.0 |
| 8 | baseline | 524 | 603 | 647 | 15.1 |
| 8 | current | 283 | 416 | 452 | 27.3 |
| 16 | baseline | 1240 | 1887 | 2132 | 12.3 |
| 16 | current | 567 | 759 | 821 | 27.6 |

## p95 improvement

| concurrency | baseline p95 | current p95 | reduction |
|---|---|---|---|
| 1 | 72 ms | 66 ms | 8% |
| 4 | 262 ms | 199 ms | 24% |
| 8 | 603 ms | 416 ms | 31% |
| 16 | 1887 ms | 759 ms | 60% |
