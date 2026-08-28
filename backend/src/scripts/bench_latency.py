"""
Latency benchmark for /search, before vs after the event-loop fix.

Runs the same request grid against two servers:
  baseline  main_baseline:app  — CLIP inline in `async def`, N+1 result queries
  current   main:app           — CLIP via run_in_threadpool, single IN query

Concurrency is the whole point. A blocked event loop is invisible when one
request is in flight and severe when sixteen are, so a single-threaded
benchmark would miss the entire effect.

The two servers run sequentially, never at the same time: side by side they
would compete for the same cores and the comparison would be meaningless.

    python scripts/bench_latency.py
    python scripts/bench_latency.py --requests 60 --concurrency 1 8
"""

import argparse
import io
import json
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))  #allow flat imports from src/

SRC_DIR = Path(__file__).parent.parent
DATA_DIR = SRC_DIR.parent / "data"
RESULTS_DIR = SRC_DIR.parent.parent / "results"

BUILDS = [("baseline", "main_baseline:app", 8801), ("current", "main:app", 8802)]
STARTUP_TIMEOUT = 300  #CLIP load dominates; it is slow on a cold page cache


def query_image() -> bytes:
    """A real catalog image, cached so every run sends identical bytes."""
    cached = DATA_DIR / "bench_query.jpg"
    if cached.exists():
        return cached.read_bytes()

    from database import SessionLocal
    from models import Product

    db = SessionLocal()
    try:
        url = db.query(Product.image_url).filter(
            Product.embedding != None,  # noqa: E711
        ).first()[0]
    finally:
        db.close()

    data = requests.get(url, timeout=30).content
    cached.write_bytes(data)
    return data


def wait_until_up(port: int, proc: subprocess.Popen) -> None:
    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server on :{port} exited with {proc.returncode}")
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            time.sleep(1)
    raise RuntimeError(f"server on :{port} did not come up in {STARTUP_TIMEOUT}s")


def one_request(port: int, blob: bytes) -> float:
    start = time.perf_counter()
    try:
        r = requests.post(
            f"http://127.0.0.1:{port}/search",
            files={"file": ("q.jpg", io.BytesIO(blob), "image/jpeg")},
            timeout=180,
        )
        r.raise_for_status()
    except requests.RequestException:
        return float("nan")
    return (time.perf_counter() - start) * 1000


def run_grid(port: int, blob: bytes, levels, n_requests, warmup):
    out = {}
    for c in levels:
        #warm caches/threadpool so the first level isn't unfairly penalised
        with ThreadPoolExecutor(max_workers=c) as pool:
            list(pool.map(lambda _: one_request(port, blob), range(warmup)))

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=c) as pool:
            times = list(pool.map(lambda _: one_request(port, blob), range(n_requests)))
        wall = time.perf_counter() - start

        ok = sorted(t for t in times if t == t)  # drop NaN failures
        if not ok:
            raise RuntimeError(f"every request failed at concurrency {c}")
        out[c] = {
            "n": len(ok),
            "failed": len(times) - len(ok),
            "p50": statistics.quantiles(ok, n=100)[49],
            "p95": statistics.quantiles(ok, n=100)[94],
            "p99": statistics.quantiles(ok, n=100)[98],
            "throughput_rps": len(ok) / wall,
        }
        print(f"    c={c:<3} p50={out[c]['p50']:7.1f}ms  p95={out[c]['p95']:7.1f}ms  "
              f"p99={out[c]['p99']:7.1f}ms  {out[c]['throughput_rps']:5.1f} req/s", flush=True)
    return out


def bench_build(name, target, port, blob, levels, n_requests, warmup):
    print(f"\n{name} ({target}) on :{port}")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", target, "--port", str(port), "--log-level", "warning"],
        cwd=str(SRC_DIR), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        wait_until_up(port, proc)
        print("  up, running grid")
        return run_grid(port, blob, levels, n_requests, warmup)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def report(results, levels):
    lines = ["# /search latency — before vs after", "",
             "`baseline` runs CLIP inline in an `async def` and issues one DB query per "
             "result. `current` moves CLIP to a threadpool and batches the lookup into a "
             "single `IN` query.", "",
             "Servers were benchmarked sequentially on the same machine, never "
             "concurrently, so they never competed for cores.", "",
             "| concurrency | build | p50 (ms) | p95 (ms) | p99 (ms) | req/s |",
             "|---|---|---|---|---|---|"]
    for c in levels:
        for name in ("baseline", "current"):
            r = results[name][c]
            lines.append(f"| {c} | {name} | {r['p50']:.0f} | {r['p95']:.0f} | "
                         f"{r['p99']:.0f} | {r['throughput_rps']:.1f} |")

    lines += ["", "## p95 improvement", "",
              "| concurrency | baseline p95 | current p95 | reduction |", "|---|---|---|---|"]
    for c in levels:
        b, cur = results["baseline"][c]["p95"], results["current"][c]["p95"]
        lines.append(f"| {c} | {b:.0f} ms | {cur:.0f} ms | {(1 - cur / b) * 100:.0f}% |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=120, help="measured requests per level")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16])
    args = ap.parse_args()

    blob = query_image()
    print(f"Query image: {len(blob)} bytes")

    results = {}
    for name, target, port in BUILDS:
        results[name] = bench_build(name, target, port, blob,
                                    args.concurrency, args.requests, args.warmup)

    RESULTS_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "bench_latency.json").write_text(json.dumps(results, indent=1))
    summary = report(results, args.concurrency)
    (RESULTS_DIR / "bench_latency.md").write_text(summary + "\n")
    print("\n" + summary)


if __name__ == "__main__":
    main()
