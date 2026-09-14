"""
Assembles a self-contained deploy bundle in deploy/bundle/.

The bundle differs from this repo in three ways: the Dockerfile sits at the
root, it carries serving dependencies only, and it includes the data artifacts
that are gitignored here.

    python deploy/prepare_deploy.py
    cd deploy/bundle && gcloud run deploy dormdeals --source . ...

See deploy/CLOUD_RUN.md for the full command. JWT_SECRET_KEY must be set on the
service — auth.py raises on import without it, so the container will not start.
"""

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
BUNDLE = ROOT / "deploy" / "bundle"
DATA = ROOT / "backend" / "data"

#datasets is a build-time dependency of the download pipeline and pulls in a
#large tree; the Space only ever serves
SERVE_REQUIREMENTS = """fastapi
uvicorn[standard]
python-dotenv
pydantic
python-multipart
sqlalchemy
python-jose[cryptography]
passlib[bcrypt]
bcrypt==4.0.1
transformers==5.0.0
torch
Pillow
numpy
faiss-cpu
requests
"""

BUNDLE_README = """# DormDeals

Visual search for dorm essentials. Upload a photo and get visually similar
products from a 14,873-item catalog, ranked by SigLIP embedding similarity over
a FAISS index.

Source and evaluation: https://github.com/ktran113/DormDeals

Retrieval was measured on 2,367 held-out listing images the index never saw.
SigLIP reaches recall@5 0.690 and MRR 0.594, against 0.327 and 0.269 for CLIP
ViT-B/32 on the identical query set.
"""

#Without this, gcloud falls back to .gitignore — which excludes backend/data/,
#so the index and database would be silently missing from the build and the
#container would crash on startup.
GCLOUDIGNORE = """.git
__pycache__/
*.pyc
"""


def main():
    required = [DATA / "products_siglip.index",
                DATA / "id_map_siglip.npy",
                DATA / "dorm_deals.slim.db"]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise SystemExit("missing artifacts (run deploy/slim_db.py first):\n  "
                         + "\n  ".join(str(p) for p in missing))

    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)

    shutil.copytree(ROOT / "backend" / "src", BUNDLE / "backend" / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "frontend", BUNDLE / "frontend")
    shutil.copy(ROOT / "deploy" / "Dockerfile", BUNDLE / "Dockerfile")

    data = BUNDLE / "backend" / "data"
    data.mkdir(parents=True)
    shutil.copy(DATA / "products_siglip.index", data / "products_siglip.index")
    shutil.copy(DATA / "id_map_siglip.npy", data / "id_map_siglip.npy")
    #database.py opens dorm_deals.db by name, so the slimmed copy ships as that
    shutil.copy(DATA / "dorm_deals.slim.db", data / "dorm_deals.db")

    (BUNDLE / "requirements.txt").write_text(SERVE_REQUIREMENTS)
    (BUNDLE / "README.md").write_text(BUNDLE_README)
    (BUNDLE / ".gcloudignore").write_text(GCLOUDIGNORE)

    size = sum(f.stat().st_size for f in BUNDLE.rglob("*") if f.is_file())
    print(f"assembled {BUNDLE.relative_to(ROOT)}  ({size / 1048576:.1f} MB)")
    print("\nnext: see deploy/CLOUD_RUN.md")


if __name__ == "__main__":
    main()
