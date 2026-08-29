from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from pathlib import Path
from models import Base

#DB lives in backend/data/ with the other generated artifacts, anchored to this file so CWD doesn't matter
DATABASE_URL = f"sqlite:///{Path(__file__).parent.parent / 'data' / 'dorm_deals.db'}"

#need check_same_thread=False to work with FastAPI
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

#columns added to the model after a database already exists. create_all()
#creates missing tables but never alters existing ones, and this project has no
#migration tooling, so without this every query selecting them fails with
#"no such column" — including /search, which selects the whole Product row.
LATE_COLUMNS = {
    "alt_image_urls": "TEXT",
    "embedding_siglip": "BLOB",
}


def init_db():
    Base.metadata.create_all(bind=engine)

    with engine.connect() as conn:
        have = {row[1] for row in conn.execute(text("PRAGMA table_info(products)"))}
        for column, decl in LATE_COLUMNS.items():
            if column not in have:
                conn.execute(text(f"ALTER TABLE products ADD COLUMN {column} {decl}"))
                print(f"Added products.{column}")
        conn.commit()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()