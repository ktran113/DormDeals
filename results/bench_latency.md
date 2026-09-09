# /search latency — before vs after

`baseline` runs CLIP inline in an `async def` and issues one DB query per result. `current` moves CLIP to a threadpool and batches the lookup into a single `IN` query.

Servers were benchmarked sequentially on the same machine, never concurrently, so they never competed for cores.

| concurrency | build | p50 (ms) | p95 (ms) | p99 (ms) | req/s |
|---|---|---|---|---|---|
| 1 | baseline | 112 | 124 | 146 | 8.8 |
| 1 | current | 134 | 145 | 156 | 7.4 |
| 4 | baseline | 459 | 738 | 934 | 7.9 |
| 4 | current | 262 | 434 | 490 | 14.3 |
| 8 | baseline | 920 | 1002 | 1117 | 8.6 |
| 8 | current | 488 | 682 | 778 | 16.1 |
| 16 | baseline | 1873 | 3462 | 3891 | 7.5 |
| 16 | current | 982 | 1231 | 1325 | 15.9 |

## p95 improvement

| concurrency | baseline p95 | current p95 | reduction |
|---|---|---|---|
| 1 | 124 ms | 145 ms | -17% |
| 4 | 738 ms | 434 ms | 41% |
| 8 | 1002 ms | 682 ms | 32% |
| 16 | 3462 ms | 1231 ms | 64% |
