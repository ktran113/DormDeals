from sqlalchemy import create_engine
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

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()