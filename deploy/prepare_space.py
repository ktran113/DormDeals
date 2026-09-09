"""
Assembles a ready-to-push HuggingFace Space in deploy/space/.

The Space needs a slightly different shape from this repo: the Dockerfile at
the root, a README carrying HF's YAML frontmatter, the serving dependencies
only, and the data artifacts that are gitignored here.

    python deploy/prepare_space.py
    cd deploy/space && git init && git lfs install
    git add -A && git commit -m "DormDeals"
    git remote add origin https://huggingface.co/spaces/<user>/dormdeals
    git push -u origin main

Set JWT_SECRET_KEY as a Space secret before the first build — auth.py raises on
import without it, so the container will not start.
"""

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
SPACE = ROOT / "deploy" / "space"
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

SPACE_README = """---
title: DormDeals
emoji: 🛋️
colorFrom: orange
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# DormDeals

Visual search for dorm essentials. Upload a photo and get visually similar
products from a 14,873-item catalog, ranked by SigLIP embedding similarity over
a FAISS index.

Source and evaluation: https://github.com/ktran113/DormDeals

Retrieval was measured on 2,367 held-out listing images the index never saw.
SigLIP reaches recall@5 0.690 and MRR 0.594, against 0.327 and 0.269 for CLIP
ViT-B/32 on the identical query set.
"""

LFS_ATTRS = """*.index filter=lfs diff=lfs merge=lfs -text
*.db filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
"""


def main():
    required = [DATA / "products_siglip.index",
                DATA / "id_map_siglip.npy",
                DATA / "dorm_deals.slim.db"]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise SystemExit("missing artifacts (run deploy/slim_db.py first):\n  "
                         + "\n  ".join(str(p) for p in missing))

    if SPACE.exists():
        shutil.rmtree(SPACE)
    SPACE.mkdir(parents=True)

    shutil.copytree(ROOT / "backend" / "src", SPACE / "backend" / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "frontend", SPACE / "frontend")
    shutil.copy(ROOT / "deploy" / "Dockerfile", SPACE / "Dockerfile")

    data = SPACE / "backend" / "data"
    data.mkdir(parents=True)
    shutil.copy(DATA / "products_siglip.index", data / "products_siglip.index")
    shutil.copy(DATA / "id_map_siglip.npy", data / "id_map_siglip.npy")
    #database.py opens dorm_deals.db by name, so the slimmed copy ships as that
    shutil.copy(DATA / "dorm_deals.slim.db", data / "dorm_deals.db")

    (SPACE / "requirements.txt").write_text(SERVE_REQUIREMENTS)
    (SPACE / "README.md").write_text(SPACE_README)
    (SPACE / ".gitattributes").write_text(LFS_ATTRS)

    size = sum(f.stat().st_size for f in SPACE.rglob("*") if f.is_file())
    print(f"assembled {SPACE.relative_to(ROOT)}  ({size / 1048576:.1f} MB)")
    print("\nnext:")
    print("  1. create a Docker Space at https://huggingface.co/new-space")
    print("  2. add JWT_SECRET_KEY as a Space secret (the app will not boot without it)")
    print("  3. cd deploy/space && git init && git lfs install")
    print("     git add -A && git commit -m 'DormDeals'")
    print("     git remote add origin https://huggingface.co/spaces/<user>/dormdeals")
    print("     git push -u origin main")


if __name__ == "__main__":
    main()
