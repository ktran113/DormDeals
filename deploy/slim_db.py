"""
Writes a deploy-sized copy of the database.

/search never reads the embedding blobs — FAISS holds those, and main.py
defers the column — so they can be dropped from the shipped DB. That takes it
from ~62 MB to single-digit MB, small enough to commit to a Space.

    python deploy/slim_db.py
"""

import shutil
import sqlite3
from pathlib import Path

SRC = Path(__file__).parent.parent / "backend" / "data" / "dorm_deals.db"
DST = Path(__file__).parent.parent / "backend" / "data" / "dorm_deals.slim.db"


def main():
    shutil.copyfile(SRC, DST)
    con = sqlite3.connect(DST)
    con.execute("UPDATE products SET embedding = NULL")
    #the SigLIP column only exists once the comparison has been run
    cols = {r[1] for r in con.execute("PRAGMA table_info(products)")}
    if "embedding_siglip" in cols:
        con.execute("UPDATE products SET embedding_siglip = NULL")
    con.execute("DELETE FROM users")  #never ship real accounts
    con.commit()
    con.execute("VACUUM")
    con.close()
    print(f"{DST.name}: {SRC.stat().st_size / 1e6:.0f} MB -> {DST.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
