"""
On-disk image cache, keyed by URL.

Amazon's CDN throttles per connection: a single fetch can take minutes, while
sixteen in parallel finish in seconds. Caching matters because the same images
are fetched more than once — the SigLIP comparison re-embeds the whole catalog
and re-runs the eval on identical queries — and a cached run costs nothing.

Cache lives in backend/data/, which is gitignored with the other artifacts.
"""

import hashlib
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

CACHE_DIR = Path(__file__).parent.parent / "data" / "img_cache"

#connect timeout is short, read timeout is generous: the CDN trickles bytes
#when throttling, so a slow read is normal and a slow connect is a dead host
TIMEOUT = (5, 25)


def _path(url: str) -> Path:
    h = hashlib.sha256(url.encode()).hexdigest()
    #shard so one directory doesn't hold tens of thousands of files
    return CACHE_DIR / h[:2] / h[2:18]


def fetch(url: str, retries: int = 2):
    """Return a PIL RGB image, or None. Cached on first success."""
    p = _path(url)
    if p.exists():
        try:
            return Image.open(p).convert("RGB")
        except Exception:
            p.unlink(missing_ok=True)  #truncated write from an interrupted run

    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
        except Exception:
            if attempt == retries - 1:
                return None
            continue

        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        try:
            img.save(tmp, format="JPEG", quality=92)
            tmp.replace(p)  #atomic, so a killed run never leaves a partial file
        except Exception:
            tmp.unlink(missing_ok=True)
        return img
    return None
