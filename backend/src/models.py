from sqlalchemy import Column, Integer, String, Float, DateTime, LargeBinary, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    asin = Column(String, unique=True, index=True)  #amazon product ID
    title = Column(String, nullable=False)
    category = Column(String, index=True)
    price = Column(Float, nullable=True)
    image_url = Column(String)
    local_image_path = Column(String)  # img path
    embedding = Column(LargeBinary, nullable=True)  # clip embedding (bytes)
    created_at = Column(DateTime, default=datetime.utcnow)
